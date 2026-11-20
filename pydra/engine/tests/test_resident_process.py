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
        assert _alive(child) and _alive(grandchild)
        assert os.getpgid(child) == handle.pid  # the shim's group, not ours
        assert handle.pid != child
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


def _use_fake_cgroup(monkeypatch, tmp_path):
    from pydra.engine import procgroup

    events = tmp_path / "memory.events"
    events.write_text("oom 0\noom_kill 3\n")  # earlier siblings' kills
    monkeypatch.setattr(procgroup, "_BASELINE", None)
    monkeypatch.setattr(
        procgroup.CgroupOomBaseline,
        "discover",
        classmethod(lambda cls, *a, **k: cls(events)),
    )
    return events


def _bump(events, n):
    events.write_text(f"oom 0\noom_kill {n}\n")


def test_child_oom_attributed_to_its_call(tmp_path, monkeypatch):
    """Drives ``load``: a kill during ``connect`` is before the baseline and is not
    charged; one after it is charged to the next call, once. A fake ``memory.events``
    stands in for the cgroup (the real file is covered by the Linux test below)"""
    from pydra.engine.procgroup import ProcessProvider

    events = _use_fake_cgroup(monkeypatch, tmp_path)

    class Server(ProcessProvider):
        name = "server"

        def argv(self, spec, device):
            return [sys.executable, "-c", "import time; time.sleep(600)"]

        def connect(self, group, spec):
            _bump(events, 4)  # killed while starting

    provider = Server()
    handle = provider.load({}, Device())
    try:
        assert handle.take_oom() == 0
        _bump(events, 5)
        assert handle.take_oom() == 1
        assert handle.take_oom() == 0
    finally:
        provider.close(handle)


def test_oom_baseline_is_one_per_process(tmp_path, monkeypatch):
    """A kill is charged once, whichever resident's call asks first"""
    from pydra.engine.procgroup import ProcessProvider

    events = _use_fake_cgroup(monkeypatch, tmp_path)

    class Server(ProcessProvider):
        name = "server"

        def argv(self, spec, device):
            return [sys.executable, "-c", "import time; time.sleep(600)"]

    a, b = Server(), Server()
    ha, hb = a.load({}, Device()), b.load({}, Device())
    try:
        _bump(events, 4)
        assert ha.take_oom() == 1
        assert hb.take_oom() == 0
        _bump(events, 6)
        assert hb.take_oom() == 2
        assert ha.take_oom() == 0
    finally:
        a.close(ha)
        b.close(hb)


def test_starting_a_provider_resets_the_baseline(tmp_path, monkeypatch):
    from pydra.engine.procgroup import ProcessProvider

    events = _use_fake_cgroup(monkeypatch, tmp_path)

    class Server(ProcessProvider):
        name = "server"

        def argv(self, spec, device):
            return [sys.executable, "-c", "import time; time.sleep(600)"]

    a = Server()
    ha = a.load({}, Device())
    _bump(events, 4)  # unclaimed kill, then another provider starts
    hb = a.load({}, Device())
    try:
        assert ha.take_oom() == 0
    finally:
        a.close(ha)
        a.close(hb)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="a real cgroup needs Linux"
)
def test_real_cgroup_baseline_when_present():
    from pydra.engine.procgroup import CgroupOomBaseline

    baseline = CgroupOomBaseline.discover()
    if baseline.path is None:
        pytest.skip("no readable cgroup oom_kill counter on this host")
    baseline.reset()
    assert baseline.take() == 0


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


def test_cgroup_v1_fallback_reads_memory_oom_control(tmp_path):
    from pydra.engine.procgroup import CgroupOomBaseline

    root = tmp_path / "cg"
    (root / "memory" / "job").mkdir(parents=True)
    ctl = root / "memory" / "job" / "memory.oom_control"
    ctl.write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 2\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("4:cpu,cpuacct:/job\n3:memory:/job\n")
    baseline = CgroupOomBaseline.discover(proc_cgroup=proc, root=root)
    assert baseline.path == ctl
    baseline.reset()
    ctl.write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 3\n")
    assert baseline.take() == 1


def test_no_accounting_is_logged_once(tmp_path, caplog, monkeypatch):
    import logging

    from pydra.engine import procgroup

    monkeypatch.setattr(procgroup, "_LOGGED_NO_ACCOUNTING", False)
    with caplog.at_level(logging.DEBUG, logger="pydra.resident"):
        for _ in range(3):
            procgroup.CgroupOomBaseline.discover(proc_cgroup=tmp_path / "nope")
    assert sum("OOM accounting" in r.getMessage() for r in caplog.records) == 1


def test_device_index_is_mapped_through_visible_devices(monkeypatch):
    from pydra.engine.procgroup import ProcessProvider

    class Server(ProcessProvider):
        name = "server"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert Server().env({}, Device("cuda", 0))["CUDA_VISIBLE_DEVICES"] == "2"
    assert Server().env({}, Device("cuda", 1))["CUDA_VISIBLE_DEVICES"] == "5"
    with pytest.raises(ValueError):
        Server().env({}, Device("cuda", 2))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert Server().env({}, Device("cuda", 1))["CUDA_VISIBLE_DEVICES"] == "1"
    assert Server().env({}, Device()) is None


@posix_only
def test_failed_connect_stops_the_leader(tmp_path):
    from pydra.engine.procgroup import ProcessProvider

    seen = []

    class Server(ProcessProvider):
        name = "server"

        def argv(self, spec, device):
            return [sys.executable, "-c", "import time; time.sleep(600)"]

        def connect(self, group, spec):
            seen.append(group)
            raise RuntimeError("no handshake")

    with pytest.raises(RuntimeError):
        Server().load({}, Device())
    assert seen[0].proc.returncode is not None
    assert not _alive(seen[0].pid)


SETSID_CHILD = textwrap.dedent("""
    import os, subprocess, sys, time
    gc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True
    )
    open(sys.argv[1], "w").write(f"{os.getpid()} {gc.pid}")
    time.sleep(600)
    """)

TERM_IGNORING_GRANDCHILD = textwrap.dedent("""
    import os, subprocess, sys, time
    gc = subprocess.Popen([sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"])
    open(sys.argv[1], "w").write(f"{os.getpid()} {gc.pid}")
    sys.exit(0)  # the leader's child exits at once; the grandchild stays
    """)


@posix_only
def test_descendant_that_leaves_the_group_is_stopped_and_waited_for(tmp_path):
    from pydra.engine.procgroup import ProcessGroup

    record = tmp_path / "pids"
    group = ProcessGroup.start([sys.executable, "-c", SETSID_CHILD, str(record)])
    child, escaped = _wait_for(record)
    assert os.getpgid(escaped) != group.pid  # really outside the group
    group.stop(term_timeout=5, kill_timeout=10)
    assert not _alive(child)
    assert not _alive(escaped)


@posix_only
def test_escaped_descendant_is_in_the_gpu_wait(tmp_path, monkeypatch):
    from pydra.engine.procgroup import ProcessProvider

    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_nvidia_smi(bindir)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    record = tmp_path / "pids"

    class Server(ProcessProvider):
        name = "server"

        def argv(self, spec, device):
            return [sys.executable, "-c", SETSID_CHILD, str(record)]

    provider = Server(gpu_settle_timeout=20)
    handle = provider.load({}, Device("cuda", 0))
    child, escaped = _wait_for(record)
    (bindir / "pids").write_text(f"{escaped}")
    assert provider.measure(handle).device == 4096 * 1024 * 1024
    provider.close(handle)
    assert not _alive(escaped)


@posix_only
def test_term_ignoring_grandchild_is_sigkilled(tmp_path):
    from pydra.engine.procgroup import ProcessGroup

    record = tmp_path / "pids"
    group = ProcessGroup.start(
        [sys.executable, "-c", TERM_IGNORING_GRANDCHILD, str(record)]
    )
    _, grandchild = _wait_for(record)
    t0 = time.monotonic()
    group.stop(term_timeout=0.5, kill_timeout=10)
    assert time.monotonic() - t0 < 8
    assert not _alive(grandchild)


@posix_only
def test_crashed_leader_then_stop_does_not_reuse_or_resignal(tmp_path, monkeypatch):
    """A leader that died stays unreaped until ``stop``, so its pid is not reused;
    ``stop`` kills the survivors, reaps once, and a second ``stop`` signals nothing"""
    from pydra.engine import procgroup
    from pydra.engine.procgroup import ProcessGroup

    record = tmp_path / "pids"
    group = ProcessGroup.start([sys.executable, "-c", CHILD, str(record)])
    _, grandchild = _wait_for(record)
    os.kill(group.pid, signal.SIGKILL)  # the leader crashes; its group lives on
    deadline = time.monotonic() + 20
    while group.leader_running() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not group.leader_running()
    assert group.proc.returncode is None  # not reaped by looking
    group.members()
    assert group.proc.returncode is None
    group.stop(term_timeout=5, kill_timeout=10)
    assert not _alive(grandchild)
    assert group.proc.returncode is not None
    calls = []
    monkeypatch.setattr(procgroup.os, "killpg", lambda *a: calls.append(a))
    monkeypatch.setattr(procgroup.os, "kill", lambda *a: calls.append(a))
    group.stop()
    assert calls == []


@posix_only
def test_child_group_is_gone_when_the_parent_is_sigkilled(tmp_path):
    """The lifeline: the parent dies without running ``close`` and the server, its
    descendants (also one that left the group) and the shim go away"""
    record = tmp_path / "pids"
    parent_code = textwrap.dedent(f"""
        import sys, time
        from pydra.engine.procgroup import ProcessGroup
        g = ProcessGroup.start([sys.executable, "-c", {SETSID_CHILD!r}, {str(record)!r}])
        open({str(tmp_path / "shim")!r}, "w").write(str(g.pid))
        time.sleep(600)
        """)
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    try:
        child, escaped = _wait_for(record)
        shim = int((tmp_path / "shim").read_text())
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=20)
        deadline = time.monotonic() + 30
        pids = (shim, child, escaped)
        while any(_alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(_alive(p) for p in pids), pids
    finally:
        parent.kill()
        parent.wait(timeout=20)


@posix_only
def test_provider_loaded_from_a_short_lived_thread_survives_the_thread(tmp_path):
    import threading

    record = tmp_path / "pids"
    provider = _provider(tmp_path, record)
    box = []
    t = threading.Thread(target=lambda: box.append(provider.load({}, Device())))
    t.start()
    t.join()
    handle = box[0]
    try:
        child, grandchild = _wait_for(record)
        time.sleep(1)  # a thread-scoped death signal would have fired by now
        assert _alive(handle.pid) and _alive(child) and _alive(grandchild)
    finally:
        provider.close(handle)
    assert not _alive(child)


linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the shim is a child subreaper only on Linux; elsewhere a daemonised "
    "server is a documented limitation",
)

DAEMONISING_LAUNCHER = textwrap.dedent("""
    import os, subprocess, sys
    srv = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True
    )
    open(sys.argv[1], "w").write(f"{os.getpid()} {srv.pid}")
    """)


@linux_only
def test_daemonised_server_is_stopped_and_counted_in_the_gpu_wait(
    tmp_path, monkeypatch
):
    from pydra.engine.procgroup import ProcessProvider

    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_nvidia_smi(bindir)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    record = tmp_path / "pids"

    class Server(ProcessProvider):
        name = "server"

        def argv(self, spec, device):
            return [sys.executable, "-c", DAEMONISING_LAUNCHER, str(record)]

    provider = Server(gpu_settle_timeout=20)
    handle = provider.load({}, Device("cuda", 0))
    launcher, server = _wait_for(record)
    (bindir / "pids").write_text(str(server))
    time.sleep(0.5)  # the launcher has exited; the shim adopted the server
    assert _alive(server)
    assert provider.measure(handle).device == 4096 * 1024 * 1024
    provider.close(handle)
    assert not _alive(server)


@linux_only
def test_daemonised_server_is_gone_when_the_parent_is_sigkilled(tmp_path):
    record = tmp_path / "pids"
    parent_code = textwrap.dedent(f"""
        import sys, time
        from pydra.engine.procgroup import ProcessGroup
        g = ProcessGroup.start([sys.executable, "-c", {DAEMONISING_LAUNCHER!r}, {str(record)!r}])
        time.sleep(600)
        """)
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    try:
        _, server = _wait_for(record)
        time.sleep(0.5)
        assert _alive(server)
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=20)
        deadline = time.monotonic() + 30
        while _alive(server) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(server)
    finally:
        parent.kill()
        parent.wait(timeout=20)


@posix_only
def test_a_reused_pid_is_not_signalled(monkeypatch):
    """A helper seen at one snapshot exits and its pid goes to an unrelated process:
    the start token differs, so it is dropped from what ``stop`` signals"""
    from types import SimpleNamespace

    from pydra.engine import procgroup

    leader = 100
    seen = {
        leader: (1, leader, True, "t0"),
        999: (leader, leader, True, "helper-start"),
    }
    reused = {
        leader: (1, leader, True, "t0"),
        999: (1, 999, True, "someone-else"),  # same pid, another process
    }
    group = procgroup.ProcessGroup(SimpleNamespace(pid=leader, returncode=None))
    monkeypatch.setattr(procgroup, "_process_table", lambda: seen)
    group._refresh()
    assert 999 in group._known
    assert group._live_known(reused) == [leader]
    group._refresh(reused)
    assert 999 not in group._known  # pruned
    exited = {leader: seen[leader]}
    group._refresh(exited)
    assert list(group._known) == [leader]


@posix_only
def test_ps_table_carries_a_start_time(tmp_path, monkeypatch):
    """Outside Linux the table comes from ``ps`` and includes ``lstart``"""
    from pydra.engine import _procshim

    bindir = tmp_path / "bin"
    bindir.mkdir()
    ps = bindir / "ps"
    ps.write_text(
        "#!/bin/sh\n"
        "echo '  10     1    10 Ss   Mon Nov 20 07:47:20 2026'\n"
        "echo '  11    10    10 Z    Mon Nov 20 07:47:21 2026'\n"
    )
    ps.chmod(ps.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(_procshim, "_USE_PROC", False)
    table = _procshim._process_table()
    assert table[10] == (1, 10, True, "Mon Nov 20 07:47:20 2026")
    assert table[11][2] is False


@posix_only
def test_provider_env_and_cwd_reach_only_the_command(tmp_path):
    """A provider's PYTHONPATH and cwd, each shadowing what the shim imports, do not
    reach the shim; the command gets them"""
    from pydra.engine.procgroup import ProcessProvider

    shadow = tmp_path / "shadow"
    (shadow / "pydra").mkdir(parents=True)
    (shadow / "pydra" / "__init__.py").write_text("raise ImportError('shadowed')\n")
    for name in ("json", "signal", "subprocess"):
        (shadow / f"{name}.py").write_text("raise ImportError('shadowed')\n")
    out = tmp_path / "out"
    code = (
        "import os, sys; "
        f"open({str(out)!r}, 'w').write(os.getcwd() + '|' + os.environ['PYTHONPATH'] "
        "+ '|' + os.environ['MARK'])"
    )

    class Server(ProcessProvider):
        name = "server"

        def argv(self, spec, device):
            return [sys.executable, "-c", code]

        def env(self, spec, device):
            return {**os.environ, "PYTHONPATH": str(shadow), "MARK": "yes"}

    from pydra.engine.procgroup import ProcessGroup

    provider = Server()
    group = ProcessGroup.start(
        provider.argv({}, Device()), env=provider.env({}, Device()), cwd=shadow
    )
    try:
        deadline = time.monotonic() + 20
        while not out.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        cwd, pypath, mark = out.read_text().split("|")
        assert os.path.realpath(cwd) == os.path.realpath(shadow)
        assert pypath == str(shadow) and mark == "yes"
        assert group.proc.wait(timeout=20) == 0  # the shim itself ran
    finally:
        group.stop()
