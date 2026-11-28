"""The group leader of a :class:`pydra.engine.procgroup.ProcessGroup`, and the process table
helpers it shares with it.

Run as a script under ``python -I``, importing nothing from pydra. The parent passes it the
read end of a lifeline pipe (end of file means the parent is gone) and an anonymous file with
the command's JSON ``{"argv", "env", "cwd"}``. The shim runs the command and stays until the
command, its group and everything below it are gone; on Linux it is a child subreaper, so a
daemonised descendant is stopped too. When the lifeline reaches end of file it stops its tree,
TERM then KILL after a grace period; on SIGTERM it forwards TERM once and carries on.
"""

import json
import os
import select
import signal
import subprocess
import sys
import time
import typing as ty

GRACE = 5.0
TEARDOWN_BOUND = 20.0
_POLL = 0.1
_IDLE_POLL = 1.0  # after the command exits with survivors, where nothing signals

# (ppid, pgid, running, start): ``running`` is False for a zombie; ``start`` is a token
# that identifies the process across pid reuse (None if the OS gives none)
_Row = tuple[int, int, bool, ty.Optional[str]]
_USE_PROC = os.path.isdir("/proc/self")
_HELPERS: set[int] = set()


def _process_table() -> dict[int, _Row]:
    """Every process: from ``/proc`` on Linux, from ``ps`` elsewhere"""
    table: dict[int, _Row] = {}
    if _USE_PROC:
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
        ps = subprocess.Popen(
            ["ps", "-axo", "pid=,ppid=,pgid=,stat=,lstart="],
            stdout=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return table
    _HELPERS.add(ps.pid)  # a child of ours that must not count as a descendant
    try:
        out, _ = ps.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        ps.kill()
        ps.communicate()
        return table
    finally:
        _HELPERS.discard(ps.pid)  # reaped; the pid may be reused
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                table[int(parts[0])] = (
                    int(parts[1]),
                    int(parts[2]),
                    not parts[3].startswith("Z"),
                    " ".join(parts[4:]) or None,
                )
            except ValueError:
                continue
    table.pop(ps.pid, None)
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


def _tree(me: int) -> set[int]:
    """The running processes below ``me`` and in its group, not ``me``"""
    table = _process_table()
    found = _descendants(table, me) | {p for p, row in table.items() if row[1] == me}
    found -= {me}
    return {p for p in found if table.get(p, (0, 0, False, None))[2]}


def _signal(pids: ty.Iterable[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _reap(child_pid: int) -> int | None:
    """Reap every exited child (the command and adopted orphans); the command's exit
    code if it was among them"""
    code = None
    while True:
        try:
            pid, wstatus = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return code
        if pid == 0:
            return code
        if pid == child_pid:
            code = os.waitstatus_to_exitcode(wstatus)


def main(args: list[str]) -> int:
    lifeline = int(args[0])
    with os.fdopen(int(args[1]), "r") as cfg_file:
        cfg_file.seek(0)
        cfg = json.load(cfg_file)
    wake_r, wake_w = os.pipe()
    os.set_blocking(wake_r, False)
    os.set_blocking(wake_w, False)
    signal.set_wakeup_fd(wake_w)
    flags = {"term": False}
    signal.signal(signal.SIGCHLD, lambda *_: None)  # so that SIGCHLD feeds the pipe
    signal.signal(signal.SIGTERM, lambda *_: flags.__setitem__("term", True))
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(36, 1)  # PR_SET_CHILD_SUBREAPER
        except Exception:  # pragma: no cover
            pass
    me = os.getpid()
    child = subprocess.Popen(cfg["argv"], env=cfg["env"], cwd=cfg["cwd"])
    status: int | None = None  # the command's exit code, once reaped here
    forwarded = False
    teardown_at: float | None = None
    eof = False
    while True:
        code = _reap(child.pid)
        if code is not None:
            status = code
        elif status is None and child.returncode is not None:
            status = child.returncode  # reaped by Popen itself
        if flags["term"] and not forwarded:
            forwarded = True
            _signal(_tree(me), signal.SIGTERM)
        if status is not None and not _tree(me):
            break
        if teardown_at is not None:
            if time.monotonic() - teardown_at > TEARDOWN_BOUND:
                print(
                    "procshim: tree still alive after teardown bound, exiting",
                    file=sys.stderr,
                )
                break
            fresh = _tree(me)
            _signal(fresh, signal.SIGTERM)
            if time.monotonic() - teardown_at > GRACE:
                _signal(fresh, signal.SIGKILL)
            time.sleep(_POLL)
            continue
        # block: nothing to scan for until the command exits, a signal arrives, or the
        # lifeline closes; with survivors and no command, poll slowly
        fds = [wake_r] if eof else [lifeline, wake_r]
        ready, _, _ = select.select(fds, [], [], None if status is None else _IDLE_POLL)
        if wake_r in ready:
            try:
                os.read(wake_r, 4096)
            except BlockingIOError:
                pass
        if lifeline in ready and os.read(lifeline, 1) == b"":
            eof = True
            teardown_at = time.monotonic()
    signal.set_wakeup_fd(-1)
    if status is not None and status < 0:
        sig = -status
        if sig != signal.SIGKILL:
            try:
                signal.signal(sig, signal.SIG_DFL)
            except (OSError, ValueError):
                sig = signal.SIGKILL
        os.kill(me, sig)
        time.sleep(1)  # the signal is pending delivery; fall through if it was ignored
    return status or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
