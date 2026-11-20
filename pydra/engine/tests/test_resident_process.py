"""Process-owning providers (RFC 0005 section 3.1, W2b)"""

import os
import signal
import stat
import subprocess
import sys
import textwrap
import time

import pytest

from pydra.engine.resident import Device, Footprint, ResidentKey, ResidentSet

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="process groups are POSIX sessions; the Windows path (CREATE_NEW_PROCESS_GROUP "
    "and taskkill /T) is covered by test_windows_stop_command_is_built",
)

# a child that starts a grandchild, records both pids and sleeps
CHILD = textwrap.dedent("""
    import os, subprocess, sys, time
    gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    open(sys.argv[1], "w").write(f"{os.getpid()} {gc.pid}")
    time.sleep(600)
    """)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers signal 0
    try:
        out = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
    except OSError:  # pragma: no cover
        return True
    return bool(out) and not out.startswith("Z")


def _wait_for(path, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().count(" "):
            return [int(p) for p in path.read_text().split()]
        time.sleep(0.05)
    raise AssertionError("child did not start")


def _provider(tmp_path, record, **kw):
    from pydra.engine.procgroup import ProcessProvider

    class Server(ProcessProvider):
        name = "server"

        def argv(self, spec, device):
            return [sys.executable, "-c", CHILD, str(record)]

    return Server(**kw)


@posix_only
def test_provider_load_spawns_child(tmp_path):
    record = tmp_path / "pids"
    provider = _provider(tmp_path, record)
    handle = provider.load({}, Device())
    try:
        child, grandchild = _wait_for(record)
        assert handle.pid == child
        assert _alive(child) and _alive(grandchild)
        assert os.getpgid(child) == child  # its own group
        assert os.getpgid(child) != os.getpgrp()
    finally:
        provider.close(handle)
    assert not _alive(child)


@posix_only
def test_eviction_kills_whole_group_and_waits(tmp_path):
    records = [tmp_path / "a", tmp_path / "b"]
    provider = _provider(tmp_path, records[0])
    rs = ResidentSet(device_budget=10)
    ka, kb = (ResidentKey("provider", "server", s) for s in "ab")
    with rs.use(
        ka,
        lambda: provider.load({}, Device()),
        close=provider.close,
        footprint=Footprint(device=10),
    ):
        pass
    child, grandchild = _wait_for(records[0])
    provider2 = _provider(tmp_path, records[1])
    with rs.use(
        kb,
        lambda: provider2.load({}, Device()),
        close=provider.close,
        footprint=Footprint(device=10),
    ):
        # a is evicted and closed before b loads; by now both are gone, not dying
        assert not _alive(child)
        assert not _alive(grandchild)
        pids_b = _wait_for(records[1])
    rs.close()
    for pid in pids_b:
        assert not _alive(pid)


@posix_only
def test_stop_escalates_to_sigkill_and_reaps(tmp_path):
    from pydra.engine.procgroup import ProcessGroup

    record = tmp_path / "pids"
    script = textwrap.dedent("""
        import signal, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        open(sys.argv[1], "w").write("1 ")
        time.sleep(600)
        """)
    group = ProcessGroup.start([sys.executable, "-c", script, str(record)])
    deadline = time.monotonic() + 20
    while not record.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    t0 = time.monotonic()
    group.stop(term_timeout=0.5, kill_timeout=10)
    assert time.monotonic() - t0 < 8
    assert group.proc.returncode == -signal.SIGKILL  # reaped, not a zombie
    assert not _alive(group.pid)
    group.stop()  # idempotent


@posix_only
def test_stop_of_exited_group_is_quiet(tmp_path):
    from pydra.engine.procgroup import ProcessGroup

    group = ProcessGroup.start([sys.executable, "-c", "pass"])
    group.proc.wait(timeout=20)
    group.stop()
    assert group.proc.returncode == 0


def test_windows_stop_command_is_built(monkeypatch):
    from pydra.engine import procgroup

    assert procgroup.windows_kill_tree_command(1234) == [
        "taskkill",
        "/T",
        "/F",
        "/PID",
        "1234",
    ]
    assert procgroup.popen_group_kwargs("win32") == {
        "creationflags": (
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0x200
        )
    }
    assert procgroup.popen_group_kwargs("linux") == {"start_new_session": True}


def test_child_oom_attributed_to_its_call(tmp_path, monkeypatch):
    """The baseline is taken after the start, so an OOM kill is charged to the call
    that ran when the counter moved, and only once. A fake ``memory.events`` stands in
    where there is no cgroup v2"""
    from pydra.engine.procgroup import CgroupOomBaseline

    events = tmp_path / "memory.events"
    events.write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 3\n")  # earlier siblings
    baseline = CgroupOomBaseline(events)
    baseline.reset()  # after the provider start
    assert baseline.take() == 0  # call 1: nothing
    events.write_text("low 0\nhigh 0\nmax 0\noom 1\noom_kill 4\n")
    assert baseline.take() == 1  # call 2: the kill
    assert baseline.take() == 0  # call 3: not charged again


def test_cgroup_absent_is_no_accounting(tmp_path):
    from pydra.engine.procgroup import CgroupOomBaseline

    baseline = CgroupOomBaseline(tmp_path / "missing")
    baseline.reset()
    assert baseline.take() is None
    assert CgroupOomBaseline.discover(proc_cgroup=tmp_path / "nope").path is None


def test_cgroup_discover_reads_memory_events_only_when_present(tmp_path):
    from pydra.engine.procgroup import CgroupOomBaseline

    root = tmp_path / "cg"
    (root / "job").mkdir(parents=True)
    (root / "job" / "memory.events").write_text("oom_kill 0\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/job\n")
    assert CgroupOomBaseline.discover(proc_cgroup=proc, root=root).path == (
        root / "job" / "memory.events"
    )
    proc.write_text("0::/other\n")
    assert CgroupOomBaseline.discover(proc_cgroup=proc, root=root).path is None


def _fake_nvidia_smi(bindir):
    """Lists a compute app per pid in ``pids`` while that process is alive; ``-q``
    queries are answered from the same file"""
    script = bindir / "nvidia-smi"
    script.write_text(textwrap.dedent(f"""\
            #!{sys.executable}
            import os, sys
            pids = open(os.path.join(os.path.dirname(__file__), "pids")).read().split()
            alive = []
            for p in pids:
                try:
                    os.kill(int(p), 0)
                    alive.append(p)
                except OSError:
                    pass
            if any(a.startswith("--query-compute-apps") for a in sys.argv):
                for p in alive:
                    print(f"{{p}}, 4096")
            """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@posix_only
def test_stopped_provider_device_memory_free_before_next_load(tmp_path, monkeypatch):
    from pydra.engine import procgroup

    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_nvidia_smi(bindir)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    record = tmp_path / "pids"
    provider = _provider(tmp_path, record, gpu_settle_timeout=20)
    handle = provider.load({}, Device("cuda", 0))
    child, grandchild = _wait_for(record)
    (bindir / "pids").write_text(f"{child} {grandchild}")
    # measured by pid, with the whole group counted
    assert provider.measure(handle).device == 2 * 4096 * 1024 * 1024
    provider.close(handle)
    # nothing holds device memory once close returns
    assert procgroup.gpu_memory_by_pid() == {}
    assert provider.measure(handle).device in (0, None)


def test_no_nvidia_smi_means_no_gpu_accounting(tmp_path, monkeypatch):
    from pydra.engine import procgroup

    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on it
    assert procgroup.gpu_memory_by_pid() is None
    assert procgroup.wait_device_memory_free([os.getpid()], timeout=0.1) is True


@posix_only
def test_close_error_is_not_raised_by_eviction(tmp_path):
    """Stopping a group that already exited, twice, does not raise"""
    from pydra.engine.procgroup import ProcessGroup

    group = ProcessGroup.start([sys.executable, "-c", "pass"])
    group.proc.wait(timeout=20)
    group.stop()
    group.stop()


def test_plain_workflows_import_nothing_from_procgroup():
    code = (
        "import sys, pydra.engine.resident, pydra.compose.python, pydra.engine.submitter;"
        "assert 'pydra.engine.procgroup' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
