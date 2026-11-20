"""The group leader of a :class:`pydra.engine.procgroup.ProcessGroup`, and the process
table helpers it shares with it.

Run as a script under ``python -I`` so that nothing in the provider's environment or
working directory can shadow the standard library, and importing nothing from pydra, so
it costs a few megabytes. The parent passes it the read end of a lifeline pipe (the write
end is the parent's, so end of file means the parent is gone) and an anonymous file with
the real command's JSON ``{"argv", "env", "cwd"}``.

The shim runs the command, then stays until the command, its group and everything below
it are gone: on Linux it is a child subreaper (a process-wide setting, unlike
``PR_SET_PDEATHSIG``), so a daemonised descendant is re-parented to it, reaped and stopped
by it. On other systems there is no subreaper and a descendant that leaves the group and
whose parent dies is re-parented to init and is not found. When the lifeline reaches end
of file the shim stops its whole tree, TERM then, after a grace period, KILL, refreshing
the tree until it is empty. On SIGTERM it forwards TERM to its tree once and carries on.
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
_POLL = 0.1

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
        _HELPERS.add(ps.pid)  # a child of ours that must not count as a descendant
        try:
            out, _ = ps.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            ps.kill()
            ps.communicate()
            return table
    except OSError:
        return table
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
    found -= _HELPERS | {me}
    return {p for p in found if table.get(p, (0, 0, False, None))[2]}


def _signal(pids: ty.Iterable[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def main(args: list[str]) -> int:
    lifeline = int(args[0])
    with os.fdopen(int(args[1]), "r") as cfg_file:
        cfg_file.seek(0)
        cfg = json.load(cfg_file)
    flags = {"term": False}
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
    while True:
        while True:  # reap the command and any adopted orphan
            try:
                pid, wstatus = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                if status is None:
                    status = child.returncode if child.returncode is not None else 0
                break
            if pid == 0:
                break
            if pid == child.pid:
                status = os.waitstatus_to_exitcode(wstatus)
        if flags["term"] and not forwarded:
            forwarded = True
            _signal(_tree(me), signal.SIGTERM)
        tree = _tree(me)
        if status is not None and not tree:
            break
        if teardown_at is None:
            ready, _, _ = select.select([lifeline], [], [], _POLL)
            if ready and os.read(lifeline, 1) == b"":
                teardown_at = time.monotonic()
        if teardown_at is not None:
            # a fresh tree each pass: a descendant that appeared since the last one is
            # found before its parent dies and leaves it to init
            fresh = _tree(me)
            _signal(fresh, signal.SIGTERM)
            if time.monotonic() - teardown_at > GRACE:
                _signal(fresh, signal.SIGKILL)
            time.sleep(_POLL)
    if status is not None and status < 0:
        signal.signal(-status, signal.SIG_DFL)
        os.kill(me, -status)
    return status or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
