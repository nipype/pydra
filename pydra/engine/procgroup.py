"""Process groups for process-owning providers (RFC 0005 section 3.1, W2b).

A provider whose ``load`` starts a child (a model server, a worker in another
interpreter) returns a handle over a :class:`ProcessGroup`. ``close`` stops the whole
group and every descendant, waits until they are gone, and waits until the device
memory they held is free, so a restart does not race the dying process.

Only a provider that starts a process imports this module; a plain run does not.

On POSIX the leader of the group is a small shim (``python -m pydra.engine.procgroup``)
that starts the real command in its own session and group and holds the read end of a
pipe whose write end is the parent's. When the parent dies without running ``close``
(SIGKILL, an OOM kill) the pipe reaches end of file and the shim stops the group and
its descendants; on Linux ``PR_SET_PDEATHSIG`` does the same when the parent *thread*
that started the shim exits. On Windows the child is created with
``CREATE_NEW_PROCESS_GROUP`` and stopped with ``taskkill /T /F``; there is no lifeline.

``stop`` snapshots the descendants by parent pid before signalling, signals the group
and each descendant that left it (``setsid``, a daemonising launcher), waits for all of
them, and reaps the leader last, so the pid and group id cannot be reused while they are
signalled. Limit: a descendant that is re-parented to init *before* the first snapshot
(its parent already dead, and the snapshot never saw it through ``measure``) is not
found; per-provider cgroups would close this and are not assumed (often unavailable on
clusters).

Default timeouts: 10 s SIGTERM wait, 10 s after SIGKILL, 30 s for device memory to
settle, so one close is bounded by about 50 s plus the ``nvidia-smi`` calls; Slurm's
default ``KillWait`` is 30 s, after which the step's cgroup teardown reaps leftovers.
"""

import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import time
import typing as ty
from pathlib import Path

from pydra.engine.resident import Device, Footprint

__all__ = [
    "CgroupOomBaseline",
    "ProcessGroup",
    "ProcessHandle",
    "ProcessProvider",
    "gpu_memory_by_pid",
    "wait_device_memory_free",
]

logger = logging.getLogger("pydra.resident")

_POLL = 0.02
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_LOGGED_NO_ACCOUNTING = False


def popen_group_kwargs(platform: str = sys.platform) -> dict[str, ty.Any]:
    """The ``Popen`` keywords that put a child in a group of its own"""
    if platform == "win32":
        return {
            "creationflags": getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", _CREATE_NEW_PROCESS_GROUP
            )
        }
    return {"start_new_session": True}


def windows_kill_tree_command(pid: int) -> list[str]:
    """The command that ends ``pid`` and its descendants on Windows"""
    return ["taskkill", "/T", "/F", "/PID", str(pid)]


# (ppid, pgid, running, start): ``running`` is False for a zombie; ``start`` is the
# process start time where the OS gives it (Linux) and identifies a pid across reuse
_Row = tuple[int, int, bool, ty.Optional[str]]


def _process_table() -> dict[int, _Row]:
    """Every process: from ``/proc`` on Linux, from ``ps`` elsewhere"""
    table: dict[int, _Row] = {}
    if os.path.isdir("/proc/self"):
        try:
            names = os.listdir("/proc")
        except OSError:
            names = []
        for name in names:
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat") as f:
                    raw = f.read()
            except OSError:
                continue
            fields = raw[raw.rindex(")") + 2 :].split()
            try:
                table[int(name)] = (
                    int(fields[1]),
                    int(fields[2]),
                    fields[0] not in ("Z", "X"),
                    fields[19],
                )
            except (IndexError, ValueError):
                continue
        return table
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,stat="],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return table
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                table[int(parts[0])] = (
                    int(parts[1]),
                    int(parts[2]),
                    not parts[3].startswith("Z"),
                    None,
                )
            except ValueError:
                continue
    return table


def _descendants(table: dict[int, _Row], root: int) -> set[int]:
    """``root`` and everything below it by parent pid"""
    children: dict[int, list[int]] = {}
    for pid, (ppid, *_rest) in table.items():
        children.setdefault(ppid, []).append(pid)
    found, todo = set(), [root]
    while todo:
        pid = todo.pop()
        if pid in found:
            continue
        found.add(pid)
        todo.extend(children.get(pid, ()))
    return found


class ProcessGroup:
    """A child process and the group it leads"""

    def __init__(self, proc: subprocess.Popen, lifeline: int | None = None):
        self.proc = proc
        self._lifeline = lifeline
        self._known: dict[int, ty.Optional[str]] = {}  # pid -> start identity
        self._stopped = False

    @classmethod
    def start(
        cls,
        argv: ty.Sequence[str],
        *,
        env: ty.Mapping[str, str] | None = None,
        cwd: str | os.PathLike | None = None,
        **popen_kwargs: ty.Any,
    ) -> "ProcessGroup":
        """Start ``argv`` as the leader of a new process group (on POSIX, under the
        lifeline shim)"""
        common = dict(
            env=None if env is None else dict(env), cwd=cwd, **popen_group_kwargs()
        )
        if sys.platform == "win32":
            return cls(subprocess.Popen(list(argv), **common, **popen_kwargs))
        read_fd, write_fd = os.pipe()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "pydra.engine.procgroup", str(read_fd), *argv],
                pass_fds=(read_fd,),
                **common,
                **popen_kwargs,
            )
        except BaseException:
            os.close(write_fd)
            raise
        finally:
            os.close(read_fd)
        return cls(proc, lifeline=write_fd)

    @property
    def pid(self) -> int:
        return self.proc.pid

    # -- liveness, without reaping -------------------------------------------------

    def leader_running(self) -> bool:
        """Whether the leader has not exited. It is not reaped by looking, so its pid
        (and the group id) stay reserved until :meth:`stop`"""
        if self._stopped or self.proc.returncode is not None:
            return False
        if hasattr(os, "waitid") and sys.platform != "win32":
            try:
                done = os.waitid(
                    os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
            except ChildProcessError:
                return False
            return done is None
        if sys.platform == "win32":
            return self.proc.poll() is None
        row = _process_table().get(self.pid)
        return row is not None and row[2]

    def _refresh(self, table: dict[int, _Row] | None = None) -> dict[int, _Row]:
        """Add the live descendants and group members to what is known"""
        table = _process_table() if table is None else table
        if sys.platform == "win32":
            return table
        mine = _descendants(table, self.pid)
        mine |= {p for p, row in table.items() if row[1] == self.pid}
        for pid in mine:
            row = table.get(pid)
            if row is not None and row[2]:
                self._known.setdefault(pid, row[3])
        return table

    def _live_known(self, table: dict[int, _Row]) -> list[int]:
        live = []
        for pid, start in self._known.items():
            row = table.get(pid)
            if row is None or not row[2]:
                continue
            if start is not None and row[3] != start:
                continue  # the pid was reused
            live.append(pid)
        return live

    def members(self) -> list[int]:
        """Pids of the live (not zombie) members of the group and its descendants"""
        if self._stopped:
            return []
        if sys.platform == "win32":
            return [self.pid] if self.leader_running() else []
        table = self._refresh()
        return sorted(self._live_known(table))

    def known_pids(self) -> list[int]:
        """Every pid seen in the group or below it, live or not"""
        return sorted(self._known)

    def _signal_all(self, sig: int, table: dict[int, _Row]) -> None:
        try:
            if self._group_has_members(table):
                os.killpg(self.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        for pid in self._live_known(table):
            if table[pid][1] == self.pid:
                continue  # reached through the group
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def _group_has_members(self, table: dict[int, _Row]) -> bool:
        return any(row[1] == self.pid and row[2] for row in table.values())

    def _wait_all_gone(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            table = self._refresh()
            if not self._live_known(table) and not self._group_has_members(table):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_POLL)

    def stop(self, term_timeout: float = 10.0, kill_timeout: float = 10.0) -> None:
        """SIGTERM the group and every descendant, wait up to ``term_timeout``, then
        SIGKILL, and reap the leader last. Returns when every member is gone (or, after
        ``kill_timeout`` past the kill, logs that one is unkillable). A second call does
        nothing"""
        if self._stopped:
            return
        try:
            if sys.platform == "win32":
                self._stop_windows(term_timeout, kill_timeout)
                return
            table = self._refresh()
            self._signal_all(signal.SIGTERM, table)
            if not self._wait_all_gone(term_timeout):
                self._signal_all(signal.SIGKILL, self._refresh())
                if not self._wait_all_gone(kill_timeout):
                    logger.error("Process group %s survived SIGKILL", self.pid)
            try:
                self.proc.wait(timeout=1)  # the leader is a zombie or dead: reap it
            except subprocess.TimeoutExpired:
                logger.error("Leader %s of a process group did not exit", self.pid)
        finally:
            self._stopped = True
            if self._lifeline is not None:
                try:
                    os.close(self._lifeline)
                except OSError:
                    pass
                self._lifeline = None

    def _stop_windows(self, term_timeout: float, kill_timeout: float) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                self.proc.wait(timeout=term_timeout)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if self.proc.poll() is None:
            subprocess.run(
                windows_kill_tree_command(self.pid),
                capture_output=True,
                timeout=kill_timeout + 10,
            )
        try:
            self.proc.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            logger.error("Process %s survived taskkill", self.pid)


def _shim_main(args: list[str]) -> int:
    """The group leader: run the command, and stop the group if the parent goes away"""
    lifeline, cmd = int(args[0]), args[1:]
    flags = {"term": False, "orphan": False}
    signal.signal(signal.SIGTERM, lambda *_: flags.__setitem__("term", True))
    signal.signal(signal.SIGUSR1, lambda *_: flags.__setitem__("orphan", True))
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGUSR1))  # PDEATHSIG
        except Exception:  # pragma: no cover
            pass
    child = subprocess.Popen(cmd)
    forwarded = False
    while True:
        rc = child.poll()
        if rc is not None:
            break
        if flags["term"] and not forwarded:
            forwarded = True
            child.terminate()
        ready, _, _ = select.select([lifeline], [], [], 0.1)
        if flags["orphan"] or (ready and os.read(lifeline, 1) == b""):
            _stop_own_tree(child)
            break
    rc = child.returncode
    if rc is not None and rc < 0:
        signal.signal(-rc, signal.SIG_DFL)
        os.kill(os.getpid(), -rc)
    return 0 if rc is None else rc


def _stop_own_tree(child: subprocess.Popen) -> None:
    me = os.getpid()
    table = _process_table()
    victims = _descendants(table, me) - {me}
    victims |= {p for p, row in table.items() if row[1] == me and p != me}

    def _all(sig: int) -> None:
        for pid in victims:
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    _all(signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        child.poll()
        live = [p for p, row in _process_table().items() if p in victims and row[2]]
        if not live or all(p == child.pid and child.poll() is not None for p in live):
            return
        time.sleep(_POLL)
    _all(signal.SIGKILL)
    child.poll()


def gpu_memory_by_pid(
    pids: ty.Iterable[int] | None = None, timeout: float = 20.0
) -> dict[int, int] | None:
    """Bytes of device memory held by each compute process, from ``nvidia-smi
    --query-compute-apps``; ``None`` when ``nvidia-smi`` is absent or fails (there is
    then no GPU accounting), otherwise restricted to ``pids`` when given"""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    wanted = None if pids is None else set(pids)
    held: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, mib = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if wanted is None or pid in wanted:
            held[pid] = held.get(pid, 0) + mib * 1024 * 1024
    return held


def wait_device_memory_free(pids: ty.Iterable[int], timeout: float) -> bool:
    """Wait until none of ``pids`` is listed by ``nvidia-smi``. True when free, or when
    there is no GPU accounting; False on timeout"""
    pids = list(pids)
    deadline = time.monotonic() + timeout
    while True:
        held = gpu_memory_by_pid(pids)
        if not held:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


class CgroupOomBaseline:
    """The ``oom_kill`` count of the process's cgroup, taken as a baseline after each
    provider start so a later kill is charged to the call that ran it.

    The counter is read from cgroup v2 ``memory.events``, or on cgroup v1 from
    ``memory.oom_control`` (kernel 4.13 and later). ``path`` is ``None`` (no accounting;
    :meth:`take` returns ``None``) when neither file is present: a file is read only
    when it exists.
    """

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path is not None else None
        self._base = 0

    @classmethod
    def discover(
        cls,
        proc_cgroup: str | os.PathLike = "/proc/self/cgroup",
        root: str | os.PathLike = "/sys/fs/cgroup",
    ) -> "CgroupOomBaseline":
        """The baseline over the calling process's own cgroup"""
        global _LOGGED_NO_ACCOUNTING
        try:
            lines = Path(proc_cgroup).read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            hierarchy, _, rest = line.partition(":")
            controllers, _, path = rest.partition(":")
            relative = path.lstrip("/")
            if hierarchy == "0":
                candidate = Path(root) / relative / "memory.events"
            elif "memory" in controllers.split(","):
                candidate = Path(root) / "memory" / relative / "memory.oom_control"
            else:
                continue
            if candidate.is_file():
                return cls(candidate)
        if not _LOGGED_NO_ACCOUNTING:
            _LOGGED_NO_ACCOUNTING = True
            logger.debug(
                "No readable cgroup oom_kill counter: OOM accounting for process-owning "
                "providers is off"
            )
        return cls(None)

    def _read(self) -> int | None:
        if self.path is None:
            return None
        try:
            for line in self.path.read_text().splitlines():
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value)
        except (OSError, ValueError):
            pass
        return None

    def reset(self) -> None:
        """Take the baseline now (after a provider start)"""
        current = self._read()
        self._base = current if current is not None else 0

    def take(self) -> int | None:
        """OOM kills since the baseline, which moves up to now; ``None`` without a
        readable counter"""
        current = self._read()
        if current is None:
            return None
        killed, self._base = max(current - self._base, 0), current
        return killed


_BASELINE: CgroupOomBaseline | None = None


def process_baseline() -> CgroupOomBaseline:
    """The one OOM baseline of this process (the worker's), shared by every provider"""
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = CgroupOomBaseline.discover()
    return _BASELINE


class ProcessHandle:
    """What a :class:`ProcessProvider` ``load`` returns: the group and the client built
    over it. The OOM baseline is the process's, not the handle's"""

    def __init__(self, group: ProcessGroup, client: ty.Any) -> None:
        self.group = group
        self.client = client

    @property
    def pid(self) -> int:
        return self.group.pid

    def take_oom(self) -> int | None:
        """OOM kills charged since the baseline or the last call, to whichever call asks
        first (see :class:`CgroupOomBaseline`)"""
        return process_baseline().take()


class ProcessProvider:
    """Base for a provider whose state is a child process.

    Subclasses set ``name`` and implement :meth:`argv` (and optionally :meth:`env`,
    :meth:`connect`, :meth:`footprint`, :meth:`version`).
    """

    name: ty.ClassVar[str]

    def __init__(
        self,
        term_timeout: float = 10.0,
        kill_timeout: float = 10.0,
        gpu_settle_timeout: float = 30.0,
    ):
        self.term_timeout = term_timeout
        self.kill_timeout = kill_timeout
        self.gpu_settle_timeout = gpu_settle_timeout

    def argv(self, spec: ty.Any, device: Device) -> list[str]:
        raise NotImplementedError

    def env(self, spec: ty.Any, device: Device) -> dict[str, str] | None:
        """The child's environment; ``None`` inherits. A CUDA device index is relative
        to this process's own ``CUDA_VISIBLE_DEVICES`` when that is set"""
        if device.kind == "cuda" and device.index is not None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible.strip():
                entries = [e.strip() for e in visible.split(",")]
                if device.index >= len(entries):
                    raise ValueError(
                        f"Device index {device.index} is outside "
                        f"CUDA_VISIBLE_DEVICES={visible!r}"
                    )
                value = entries[device.index]
            else:
                value = str(device.index)
            return {**os.environ, "CUDA_VISIBLE_DEVICES": value}
        return None

    def connect(self, group: ProcessGroup, spec: ty.Any) -> ty.Any:
        """The client of the started process (default: none)"""
        return None

    def load(self, spec: ty.Any, device: Device) -> ProcessHandle:
        group = ProcessGroup.start(self.argv(spec, device), env=self.env(spec, device))
        try:
            client = self.connect(group, spec)
        except BaseException:
            group.stop(self.term_timeout, self.kill_timeout)
            raise
        process_baseline().reset()  # after the start, so a load-time kill is not charged
        return ProcessHandle(group, client)

    def close(self, obj: object) -> None:
        """Stop the group, wait for every member, then for its device memory to be
        free. Errors are logged, not raised"""
        handle = ty.cast(ProcessHandle, obj)
        try:
            handle.group.members()  # snapshot the descendants before signalling
            pids = handle.group.known_pids()
            handle.group.stop(self.term_timeout, self.kill_timeout)
            if not wait_device_memory_free(pids, self.gpu_settle_timeout):
                logger.error(
                    "Device memory of %s still held after %ss",
                    self.name,
                    self.gpu_settle_timeout,
                )
        except Exception as exc:
            logger.error("Stopping process group %s failed: %r", handle.pid, exc)

    def footprint(self, spec: ty.Any) -> Footprint | None:
        return None

    def measure(self, obj: object) -> Footprint | None:
        """Device bytes held by the group's processes, by pid (``None`` without
        ``nvidia-smi``)"""
        handle = ty.cast(ProcessHandle, obj)
        pids = handle.group.members() or [handle.pid]
        held = gpu_memory_by_pid(pids)
        if held is None:
            return None
        return Footprint(device=sum(held.values()))

    def version(self) -> str:
        return type(self).__qualname__


if __name__ == "__main__":  # the lifeline shim
    sys.exit(_shim_main(sys.argv[1:]))
