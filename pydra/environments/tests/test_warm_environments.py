"""Warm environments (RFC 0005 section 3.1, W3): one started container, Apptainer
instance or Lmod resolution serves many jobs. The runtimes are fakes that run each
command on the host, mapping the container paths back: a path outside the bound
roots is a missing file, and a path bound read-only is a read-only copy, so outputs,
modes and isolation are those of real commands."""

import os
import shutil
import signal
import stat
import statistics
import subprocess as sp
import sys
import tempfile
import textwrap
import threading
import time
import typing as ty
import warnings
from pathlib import Path

import attrs
import pytest

from pydra.compose import shell
from pydra.engine.job import Job
from pydra.engine.resident import Device, ResidentKey
from pydra.engine.submitter import Submitter
from pydra.environments import base, docker, lmod, native, singularity
from pydra.environments.base import WarmSession, serving
from pydra.environments.tests.warm_fakes import D1, IMAGE, FakeRuntime

posix_only = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="the fakes run POSIX commands"
)
pytestmark = posix_only
not_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root writes through read-only file modes",
)

Cp = shell.define("cp <in_file> <out|out_file>")
Mutate = shell.define("sh -c <script> <in_file:generic/file>")


def _sub(cache, env):
    return Submitter(
        worker="debug",
        store="pickle",
        environment=env,
        cache_root=cache,
        telemetry=False,
    )


def _input(path, text="hello\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _sh(script):
    return shell.define("sh")(executable="sh", append_args=["-c", script])


def _echo(i):
    return shell.define("echo")(executable="echo", append_args=[str(i)])


ENVIRONMENTS = {
    "docker": lambda **kw: docker.Docker(image=IMAGE, **kw),
    "apptainer": lambda **kw: singularity.Singularity(image=f"docker://{IMAGE}", **kw),
}


@pytest.fixture(autouse=True)
def roots_below(tmp_path, monkeypatch):
    """A binding root is a directory directly under ``tmp_path``, so a test can make
    jobs that need distinct roots; the default scratch is under ``tmp_path``"""
    (tmp_path / "scratch").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
    real = base.top_root

    def top_root(path):
        path = Path(path).absolute()
        try:
            rel = path.relative_to(tmp_path)
        except ValueError:
            return real(path)
        return tmp_path / rel.parts[0] if rel.parts else tmp_path

    monkeypatch.setattr(base, "top_root", top_root)


@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_one_run_n_exec(kind, monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = ENVIRONMENTS[kind](warm=True)
    session = WarmSession(env)
    with serving(session):
        for i in range(5):
            result = _sub(tmp_path / "cache", env)(_echo(i))
            assert result.outputs.stdout.strip() == str(i)
    assert len(fake.starts) == 1
    assert len(fake.execs) == 5
    assert fake.cold == []
    assert len(fake.stopped) == 1
    assert session.starts == 1 and session.cold_runs == 0
    start = fake.starts[0]
    host, pid = base.owner()
    binds = fake.started[list(fake.started)[0]]["binds"]
    # the scratch /tmp lies in the cache root, which is bound writable
    (tmp,) = [h for h, c, _ in binds if c == "/tmp"]
    assert tmp.parent.parent == tmp_path / "cache"
    assert tmp.parent.name.startswith(".pydra-warm-")
    assert not tmp.parent.exists()  # removed on close
    if kind == "docker":
        assert start[:5] == ["docker", "run", "-d", "--rm", "--init"]
        assert start[-3:] == ["sleep", IMAGE, docker.SLEEP_FOREVER]
        assert f"pydra.warm.owner={base.owner_label()}" in start
        assert f"pydra.warm.uid={os.getuid()}" in start
    else:
        # the instance starts from the local SIF, never from the docker:// URI
        assert start[:5] == ["apptainer", "instance", "start", "-c", "--no-home"]
        assert not any("docker://" in a for a in start)
        assert fake.started[start[-1]]["image"].endswith(".sif")
        assert fake.sifs == [f"docker://{IMAGE}"]
        assert start[-1].startswith(f"pydra-warm-{pid}-")
        assert start[-1].endswith(f"-{host}")


def test_one_module_resolution_n_commands(monkeypatch, tmp_path):
    loads = []

    def purge_and_load(names):
        loads.append(list(names))
        env = dict(os.environ, TOOL_VERSION="2.0", LOADEDMODULES="tool/2.0")
        return dict(os.environ), ["tool/2.0"], ["/tree/tool/2.0.lua"], env

    monkeypatch.setenv("MODULESHOME", str(tmp_path / "lmod"))
    monkeypatch.setattr(lmod.Lmod, "_purge_and_load", staticmethod(purge_and_load))
    env = lmod.Lmod(modules=["tool"], warm=True)
    session = WarmSession(env)
    with serving(session):
        for i in range(5):
            result = _sub(tmp_path / "cache", env)(_sh(f"echo $TOOL_VERSION {i}"))
            assert result.outputs.stdout.split() == ["2.0", str(i)]
    assert loads.count(["tool/2.0"]) == 1  # one purge and load of the full names
    assert ["tool"] in loads  # the submitting process resolved the requested names
    assert session.starts == 1 and session.cold_runs == 0
    # cold, each command loads
    loads.clear()
    for i in range(3):
        _sub(tmp_path / "cold", env)(_sh(f"echo $TOOL_VERSION {i}"))
    assert loads.count(["tool/2.0"]) == 3


@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_job_home_and_tmp_not_visible_to_next_job(kind, monkeypatch, tmp_path):
    FakeRuntime(monkeypatch, tmp_path)
    env = ENVIRONMENTS[kind](warm=True)
    session = WarmSession(env)
    with serving(session):
        a = _sub(tmp_path / "cache", env)(
            _sh(
                'echo a > "$HOME/x"; echo b > /tmp/y; mkdir /tmp/d; '
                'echo c > "$TMPDIR/z"; echo "$HOME"'
            )
        )
        b = _sub(tmp_path / "cache", env)(
            _sh(
                'test -e "$HOME/x" && echo seen-x; test -e /tmp/y && echo seen-y;'
                ' test -e /tmp/d && echo seen-d; test -e "$TMPDIR/z" && echo seen-z;'
                ' echo "$HOME"'
            )
        )
        assert session.starts == 1
    home_a = a.outputs.stdout.split()[-1]
    lines = b.outputs.stdout.split()
    assert not [w for w in lines if w.startswith("seen")]
    assert lines[-1] != home_a
    assert not Path(home_a).exists()
    # neither is in the job's own directory, which stays as a cold run leaves it
    assert sorted(p.name for p in a.cache_dir.iterdir()) == sorted(
        p.name for p in b.cache_dir.iterdir()
    )


@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_outputs_owned_by_submitting_user(kind, monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = ENVIRONMENTS[kind](warm=True)
    src = _input(tmp_path / "inputs" / "in.txt")
    with serving(WarmSession(env)):
        result = _sub(tmp_path / "cache", env)(Cp(in_file=src))
    out = Path(result.outputs.out_file)
    assert out.stat().st_uid == os.getuid()
    start = fake.starts[0]
    if kind == "docker":
        i = start.index("--user")
        assert start[i + 1] == f"{os.getuid()}:{os.getgid()}"
    else:
        # Apptainer runs as the calling user; contained, with no home bound
        assert "-c" in start and "--no-home" in start


@not_root
@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_inputs_stay_read_only(kind, monkeypatch, tmp_path):
    """A job that edits its input in place (another job's cached output) fails warm
    as it fails cold, and the other job's entry is unchanged"""
    fake = FakeRuntime(monkeypatch, tmp_path)
    src = _input(tmp_path / "inputs" / "in.txt")
    cold_env, warm_env = ENVIRONMENTS[kind](), ENVIRONMENTS[kind](warm=True)
    a = _sub(tmp_path / "cache", cold_env)(Cp(in_file=src))
    produced = Path(a.outputs.out_file)
    # writable on the host, so only the bind's mode can refuse the edit (the fake's
    # cold cp copied it from a read-only copy of its input)
    produced.chmod(0o644)
    edit = Mutate(script='echo mutated >> "$0"', in_file=produced)
    with pytest.raises(RuntimeError, match="ermission denied"):
        _sub(tmp_path / "cache", cold_env)(edit)
    assert produced.read_text() == "hello\n"
    session = WarmSession(warm_env)
    with serving(session):
        with pytest.raises(RuntimeError, match="ermission denied"):
            _sub(tmp_path / "cache", warm_env)(edit)
        assert session.starts == 1 and session.cold_runs == 0
    assert produced.read_text() == "hello\n"
    # A's entry still serves the bytes A produced, with nothing run
    calls = len(fake.calls)
    again = _sub(tmp_path / "cache", cold_env)(Cp(in_file=src))
    assert len(fake.calls) == calls
    assert Path(again.outputs.out_file).read_text() == "hello\n"
    # every root read-only under the ro prefix; only the cache root writable
    binds = fake.started[list(fake.started)[-1]]["binds"]
    rw = {h for h, c, m in binds if m == "rw" and c != "/tmp"}
    ro = {h for h, c, m in binds if m == "ro"}
    assert rw == {tmp_path / "cache"}
    assert ro == {tmp_path / "cache"}  # the input is A's output, in the cache
    assert all(c.startswith("/mnt/pydra-ro/") for _, c, m in binds if m == "ro")


def test_unmounted_root_one_restart(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=IMAGE, warm=True)
    session = WarmSession(env)
    with serving(session):
        for name in ("in1", "in2", "in1", "in2"):
            src = _input(tmp_path / name / "in.txt", f"{name}\n")
            result = _sub(tmp_path / "cache", env)(Cp(in_file=src))
            assert Path(result.outputs.out_file).read_text() == f"{name}\n"
    assert session.starts == 2  # in2 was not mounted: one restart
    assert session.cold_runs == 0 and fake.cold == []
    assert len(fake.starts) == 2 and len(fake.execs) == 2  # the rest were cached
    first, second = (
        {h for h, _, m in fake.started[n]["binds"] if m == "ro"} for n in ("id0", "id1")
    )
    assert tmp_path / "in2" not in first
    assert {tmp_path / "in1", tmp_path / "in2"} <= second
    rw = {h for h, c, m in fake.started["id1"]["binds"] if m == "rw" and c != "/tmp"}
    assert rw == {tmp_path / "cache"}  # inputs never writable
    assert fake.stopped[0] == "id0"


def test_beyond_max_mount_roots_runs_cold(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    # cache (holding the scratch) and in1 are two roots; in2 would be a third
    env = docker.Docker(image=IMAGE, warm=True, max_mount_roots=2)
    session = WarmSession(env)
    with serving(session):
        outputs = []
        for name in ("in1", "in2"):
            src = _input(tmp_path / name / "in.txt", f"{name}\n")
            result = _sub(tmp_path / "cache", env)(Cp(in_file=src))
            outputs.append(Path(result.outputs.out_file).read_text())
    assert outputs == ["in1\n", "in2\n"]
    assert session.starts == 1 and session.cold_runs == 1
    assert len(fake.starts) == 1 and len(fake.execs) == 1 and len(fake.cold) == 1


def test_scratch_root_counts_toward_max_mount_roots(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=IMAGE, warm=True, max_mount_roots=2)
    # cache, in1 and the scratch's own root are three
    session = WarmSession(env, scratch=tmp_path / "elsewhere")
    with serving(session):
        src = _input(tmp_path / "in1" / "in.txt")
        _sub(tmp_path / "cache", env)(Cp(in_file=src))
    assert session.starts == 0 and session.cold_runs == 1 and fake.starts == []


@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_warm_and_cold_outputs_equal(kind, monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    src = _input(tmp_path / "inputs" / "in.txt")
    cold_env, warm_env = ENVIRONMENTS[kind](), ENVIRONMENTS[kind](warm=True)
    task = Cp(in_file=src)

    # one checksum: warmth is not keyed
    jobs = [
        Job(task=task, name="j", submitter=_sub(tmp_path / "ck", e))
        for e in (cold_env, warm_env)
    ]
    assert jobs[0].checksum == jobs[1].checksum
    assert jobs[0].context == jobs[1].context

    cold = _sub(tmp_path / "cold", cold_env)(task)
    with serving(WarmSession(warm_env)):
        warm = _sub(tmp_path / "warm", warm_env)(task)
    assert len(fake.cold) == 1 and len(fake.execs) == 1
    assert (
        attrs.asdict(cold.outputs, recurse=False).keys()
        == attrs.asdict(warm.outputs, recurse=False).keys()
    )
    assert cold.outputs.stdout == warm.outputs.stdout
    assert cold.outputs.return_code == warm.outputs.return_code
    assert Path(cold.outputs.out_file).name == Path(warm.outputs.out_file).name
    assert (
        Path(cold.outputs.out_file).read_bytes()
        == Path(warm.outputs.out_file).read_bytes()
    )
    assert cold.cache_dir.name == warm.cache_dir.name
    assert sorted(p.name for p in cold.cache_dir.iterdir()) == sorted(
        p.name for p in warm.cache_dir.iterdir()
    )
    # the cold environment finds the warm result: one slot, nothing runs again
    calls = len(fake.calls)
    again = _sub(tmp_path / "warm", cold_env)(task)
    assert len(fake.calls) == calls
    assert again.outputs.stdout == warm.outputs.stdout


def test_image_entrypoint_runs_warm_as_cold(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path, entrypoints={IMAGE: ["env", "ENTRY=1"]})
    task = _sh('echo "entry=$ENTRY"')
    cold = _sub(tmp_path / "cold", docker.Docker(image=IMAGE))(task)
    env = docker.Docker(image=IMAGE, warm=True)
    with serving(WarmSession(env)):
        warm = _sub(tmp_path / "warm", env)(task)
    assert cold.outputs.stdout == warm.outputs.stdout == "entry=1\n"
    (exec_cmd,) = fake.execs
    assert exec_cmd[exec_cmd.index("id0") + 1 :][:2] == ["env", "ENTRY=1"]


def test_xargs_entrypoint_honoured(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path, entrypoints={IMAGE: ["false"]})
    assert docker.split_entrypoint(["--entrypoint=", "-e", "A=1"]) == (
        ["-e", "A=1"],
        [],
    )
    xargs = ["--entrypoint", "env"]
    env = docker.Docker(image=IMAGE, warm=True, xargs=xargs)
    task = _sh('echo "entry=${ENTRY-none}"')
    with serving(WarmSession(env)):
        warm = _sub(tmp_path / "warm", env)(task)
    cold = _sub(tmp_path / "cold", docker.Docker(image=IMAGE, xargs=xargs))(task)
    assert warm.outputs.stdout == cold.outputs.stdout == "entry=none\n"
    assert fake.starts[0].count("--entrypoint") == 1  # the sleeper's only
    assert fake.of("docker", "image", "inspect") == []
    (exec_cmd,) = fake.execs
    assert exec_cmd[exec_cmd.index("id0") + 1] == "env"


def test_native_warm_is_none(monkeypatch, tmp_path):
    assert native.Native().warm() is None
    assert native.Native(warm=True).warm([tmp_path], Device()) is None
    assert docker.Docker(image="busybox").warm([tmp_path]) is None  # not opted in
    env = native.Native(warm=True)
    session = WarmSession(env)
    with serving(session):
        result = _sub(tmp_path / "cache", env)(_echo("hi"))
    assert result.outputs.stdout.strip() == "hi"
    assert session.starts == 0 and session.cold_runs == 1


def test_warm_fields_not_keyed():
    plain = docker.Docker(image="busybox")
    assert (
        plain.checksum()
        == docker.Docker(
            image="busybox", warm=True, max_jobs=3, max_mount_roots=2, read_only=True
        ).checksum()
    )
    assert (
        lmod.Lmod(modules=["a"]).checksum()
        == lmod.Lmod(modules=["a"], warm=True).checksum()
    )
    assert native.Native().checksum() == native.Native(warm=True).checksum()
    assert plain._ref_key() == docker.Docker(image="busybox", warm=True)._ref_key()


def test_resident_key_is_the_resolved_reference(monkeypatch, tmp_path):
    FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=IMAGE, warm=True)
    ref = env.resolve()
    with env.warm([tmp_path], ref=ref) as handle:
        assert handle.key == ResidentKey("environment", "docker", f"docker@1:{D1}")
        assert tmp_path in handle.roots


def test_max_jobs_restarts(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=IMAGE, warm=True, max_jobs=2)
    session = WarmSession(env)
    with serving(session):
        for i in range(5):
            _sub(tmp_path / "cache", env)(_echo(i))
    assert session.starts == 3 and len(fake.starts) == 3 and len(fake.execs) == 5


def test_gpu_flags_at_start(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    for kind, make in ENVIRONMENTS.items():
        env = make(warm=True)
        with env.warm([tmp_path], Device("cuda")) as handle:
            job = Job(
                task=shell.define("nvidia-smi")(),
                name="j",
                submitter=_sub(tmp_path / "cache", env),
            )
            cmd = handle._exec_command(job, {}, ["nvidia-smi"])
        if kind == "docker":
            start = fake.of("docker", "run", "-d")[-1]
            assert start[start.index("--gpus") + 1] == '"device=2,3"'
        else:
            assert "--nv" in fake.of("apptainer", "instance", "start")[-1]
            assert "CUDA_VISIBLE_DEVICES=2,3" in cmd
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with ENVIRONMENTS["docker"](warm=True).warm([tmp_path], Device("cuda", 1)):
        start = fake.of("docker", "run", "-d")[-1]
        assert start[start.index("--gpus") + 1] == '"device=1"'
    with ENVIRONMENTS["docker"](warm=True).warm([tmp_path], Device("cpu")):
        assert "--gpus" not in fake.of("docker", "run", "-d")[-1]


def test_read_only_docker(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    with docker.Docker(image=IMAGE, warm=True, read_only=True).warm([tmp_path]):
        start = fake.starts[-1]
    assert "--read-only" in start and "--tmpfs" not in start
    assert [v for v in start if v.endswith(":/tmp:rw")]


def test_setup_and_teardown_called(monkeypatch, tmp_path):
    FakeRuntime(monkeypatch, tmp_path)
    events = []

    @attrs.define
    class Counting(docker.Docker):
        _plugin_name: ty.ClassVar[str] = "docker"

        def setup(self):
            events.append("setup")

        def teardown(self):
            events.append("teardown")

    handle = Counting(image=IMAGE, warm=True).warm([tmp_path])
    assert events == ["setup"]
    handle.close()
    handle.close()
    assert events == ["setup", "teardown"]
    assert not handle.scratch.exists()


@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_failed_start_runs_cold_and_removes_by_name(kind, monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path, fail_start=True)
    env = ENVIRONMENTS[kind](warm=True)
    session = WarmSession(env)
    with serving(session):
        for i in range(3):
            result = _sub(tmp_path / "cache", env)(_echo(i))
            assert result.outputs.stdout.strip() == str(i)
    assert len(fake.starts) == 1  # tried once
    assert session.cold_runs == 3 and len(fake.cold) == 3
    # what the failed start may have left is removed by its name
    assert fake.stopped == list(fake.started)
    assert not list((tmp_path / "cache").glob(".pydra-warm-*"))
    assert base._OPEN == set()


def test_another_reference_runs_cold(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=IMAGE, warm=True)
    session = WarmSession(env)
    with serving(session):
        _sub(tmp_path / "a", env)(shell.define("echo")(append_args=["a"]))
        job = Job(
            task=shell.define("echo")(append_args=["b"]),
            name="j",
            submitter=_sub(tmp_path / "b", env),
        )
        job._resolved = (env, base.ResolvedRef("docker@1:other", "org/tool@other"))
        assert not session.handle.covers(job)
    assert len(fake.starts) == 1


def _dead_pid():
    proc = sp.Popen(["true"])
    proc.wait()
    return proc.pid


def test_orphaned_containers_reaped(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    host, pid = base.owner()
    uid, ns = os.getuid(), base.pid_namespace() or "-"
    dead = _dead_pid()
    fake.labels = {
        "orphan": f"{host}:{uid}:{ns}:{dead}",
        "elsewhere": f"otherhost:{uid}:{ns}:{dead}",
        "live": f"{host}:{uid}:{ns}:{pid}",
    }
    with docker.Docker(image=IMAGE, warm=True).warm([tmp_path]):
        pass
    assert "orphan" in fake.stopped
    assert "elsewhere" not in fake.stopped and "live" not in fake.stopped
    (listing,) = fake.of("docker", "ps", "-a")
    assert f"label=pydra.warm.uid={uid}" in listing


def test_reaping_spares_other_users_and_pid_namespaces(monkeypatch, tmp_path):
    """Another user's container, or one whose owner lives in another pid namespace
    (pydra in a container with the host's network), is never removed, nor one whose
    label does not say both"""
    fake = FakeRuntime(monkeypatch, tmp_path)
    host, _ = base.owner()
    uid, ns = os.getuid(), base.pid_namespace() or "-"
    dead = _dead_pid()
    fake.labels = {
        "other-user": f"{host}:{uid + 1}:{ns}:{dead}",
        "other-pidns": f"{host}:{uid}:4026599999:{dead}",
        "unlabelled-uid": f"{host}:{dead}",
    }
    with docker.Docker(image=IMAGE, warm=True).warm([tmp_path]):
        pass
    assert not set(fake.labels) & set(fake.stopped)


def test_failing_setup_leaves_no_scratch(monkeypatch, tmp_path):
    FakeRuntime(monkeypatch, tmp_path)

    @attrs.define
    class Failing(docker.Docker):
        _plugin_name: ty.ClassVar[str] = "docker"

        def setup(self):
            raise RuntimeError("setup failed")

    with pytest.raises(RuntimeError, match="setup failed"):
        Failing(image=IMAGE, warm=True).warm([tmp_path], scratch=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob(".pydra-warm-*"))
    assert base._OPEN == set()


def test_orphaned_instances_reaped(monkeypatch, tmp_path):
    fake = FakeRuntime(monkeypatch, tmp_path)
    host, pid = base.owner()
    dead = _dead_pid()
    orphan = f"pydra-warm-{dead}-aaaaaaaaaaaa-{host}"
    fake.foreign = [
        orphan,
        f"pydra-warm-{dead}-bbbbbbbbbbbb-otherhost",
        f"pydra-warm-{pid}-cccccccccccc-{host}",
        "someone-else",
    ]
    env = singularity.Singularity(image=f"docker://{IMAGE}", warm=True)
    with env.warm([tmp_path]):
        pass
    assert orphan in fake.stopped
    assert not set(fake.foreign[1:]) & set(fake.stopped)


@pytest.mark.parametrize("ending", ["sigterm", "exit"])
def test_open_environment_closed_on_sigterm_and_exit(ending, tmp_path):
    """A process ended by SIGTERM, or exiting without closing, removes its container"""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "docker.log"
    script = bindir / "docker"
    script.write_text(textwrap.dedent(f"""\
            #!/bin/sh
            echo "$*" >> "{log}"
            case "$1 $2" in
              "run -d") echo cid1 ;;
              "image inspect") echo null ;;
            esac
            exit 0
            """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    child = textwrap.dedent(f"""\
        import time
        from pydra.environments import docker
        handle = docker.Docker(image="{IMAGE}", warm=True).warm(
            [{str(tmp_path)!r}], scratch={str(tmp_path / "scratch")!r}
        )
        print("ready", flush=True)
        if {ending!r} == "sigterm":
            time.sleep(60)
        """)
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}")
    proc = sp.Popen([sys.executable, "-c", child], stdout=sp.PIPE, env=env, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        if ending == "sigterm":
            proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert rc == (-signal.SIGTERM if ending == "sigterm" else 0)
    assert "rm -f cid1" in log.read_text().splitlines()


def test_two_threads_share_one_session(monkeypatch, tmp_path):
    """A restart for one thread's new root never kills the other thread's exec"""
    fake = FakeRuntime(monkeypatch, tmp_path)
    fake.exec_delay = 0.05
    env = docker.Docker(image=IMAGE, warm=True)
    sub = _sub(tmp_path / "cache", env)
    jobs = {}
    for name in ("in1", "in2"):
        jobs[name] = []
        for i in range(4):
            src = _input(tmp_path / name / f"{i}.txt", f"{name}{i}\n")
            job = Job(task=Cp(in_file=src), name=f"{name}{i}", submitter=sub)
            job.resolved
            jobs[name].append(job)
    session = WarmSession(env)
    errors = []

    def run(name):
        for job in jobs[name]:
            try:
                session.execute(job)
            except Exception as e:  # collected and asserted below
                errors.append(e)

    threads = [threading.Thread(target=run, args=(n,)) for n in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    session.close()
    assert errors == []
    assert session.starts == len(fake.starts) == len(fake.stopped)
    assert len(fake.execs) == 8


# Real runtimes: Docker on the CI ubuntu runner, Apptainer and Lmod on an HPC cluster ------


def _docker_up():
    if shutil.which("docker") is None:
        return False
    return not sp.run(["docker", "info"], capture_output=True).returncode


@pytest.mark.docker
@pytest.mark.skipif(shutil.which("docker") is None, reason="no docker")
def test_warm_docker_real(tmp_path):
    if not _docker_up():
        pytest.skip("no docker daemon")
    src = _input(tmp_path / "inputs" / "in.txt")
    cold_env = docker.Docker(image="busybox")
    warm_env = docker.Docker(image="busybox", warm=True)
    task = Cp(in_file=src)
    cold = _sub(tmp_path / "cold", cold_env)(task)
    session = WarmSession(warm_env)
    with serving(session):
        warm = _sub(tmp_path / "warm", warm_env)(task)
        a = _sub(tmp_path / "warm", warm_env)(
            _sh('echo a > "$HOME/x"; echo b > /tmp/y')
        )
        b = _sub(tmp_path / "warm", warm_env)(
            _sh(
                'test -e "$HOME/x" && echo seen-x; test -e /tmp/y && echo seen-y;'
                " id -u"
            )
        )
        with pytest.raises(RuntimeError):
            _sub(tmp_path / "warm", warm_env)(
                Mutate(
                    script='echo mutated >> "$0"', in_file=Path(warm.outputs.out_file)
                )
            )
        assert session.starts == 1 and session.cold_runs == 0
    assert a.outputs.return_code == 0
    assert "seen" not in b.outputs.stdout
    assert b.outputs.stdout.split()[-1] == str(os.getuid())
    assert Path(warm.outputs.out_file).read_bytes() == src.read_bytes()
    assert Path(warm.outputs.out_file).stat().st_uid == os.getuid()
    assert cold.outputs.stdout == warm.outputs.stdout
    assert cold.cache_dir.name == warm.cache_dir.name

    # the run-against-exec figure (bench_warm_environment's measure, 50 commands),
    # reported as a warning so the CI log shows it for a passing test
    walls = {}
    for label, env in (("cold", cold_env), ("warm", warm_env)):
        sub = _sub(tmp_path / f"bench-{label}", env)
        jobs = [Job(task=_echo(i), name=f"e{i}", submitter=sub) for i in range(50)]
        session = WarmSession(env)
        run = (
            (lambda j: j.environment.execute(j)) if label == "cold" else session.execute
        )
        walls[label] = []
        try:
            for job in jobs:
                job.resolved
                start = time.perf_counter()
                run(job)
                walls[label].append(time.perf_counter() - start)
        finally:
            session.close()
    warnings.warn(
        "bench_warm_environment docker (50 commands): "
        f"docker run median {statistics.median(walls['cold']) * 1000:.0f} ms, "
        f"docker exec median {statistics.median(walls['warm'][1:]) * 1000:.0f} ms, "
        f"warm start {walls['warm'][0] * 1000:.0f} ms"
    )


@pytest.mark.singularity
@pytest.mark.skipif(
    not (shutil.which("apptainer") or shutil.which("singularity")),
    reason="no apptainer",
)
def test_warm_apptainer_real(tmp_path, monkeypatch):
    """One instance for many jobs, from the local SIF, equal to cold"""
    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "apptainer-cache"))
    monkeypatch.setenv("SINGULARITY_CACHEDIR", str(tmp_path / "apptainer-cache"))
    monkeypatch.setenv("PYDRA_IMAGE_CACHE", str(tmp_path / "images"))
    ref = singularity.Singularity(image="docker://alpine", tag="3.20").resolve()
    cold_env = singularity.Singularity(image=ref.run)
    warm_env = singularity.Singularity(image=ref.run, warm=True)
    src = _input(tmp_path / "inputs" / "in.txt")
    task = Cp(in_file=src)
    cold = _sub(tmp_path / "cold", cold_env)(task)
    # warm: the local SIF, never a registry
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    session = WarmSession(warm_env)
    with serving(session):
        warm = _sub(tmp_path / "warm", warm_env)(task)
        for i in range(3):
            _sub(tmp_path / "warm", warm_env)(
                _sh(f'echo {i} > "$HOME/x"; echo {i} > /tmp/y')
            )
        b = _sub(tmp_path / "warm", warm_env)(
            _sh(
                'test -e "$HOME/x" && echo seen-x; test -e /tmp/y && echo seen-y;'
                " true"
            )
        )
        with pytest.raises(RuntimeError):
            _sub(tmp_path / "warm", warm_env)(
                Mutate(
                    script='echo mutated >> "$0"', in_file=Path(warm.outputs.out_file)
                )
            )
        assert session.starts == 1 and session.cold_runs == 0
    assert "seen" not in b.outputs.stdout
    assert Path(warm.outputs.out_file).read_bytes() == src.read_bytes()
    assert cold.outputs.stdout == warm.outputs.stdout
    assert cold.cache_dir.name == warm.cache_dir.name


@pytest.mark.lmod
@pytest.mark.skipif("MODULEPATH" not in os.environ, reason="modules not available")
def test_warm_lmod_real(tmp_path, monkeypatch):
    """One resolution for many commands, equal to cold"""
    modules = tmp_path / "modules" / "tool"
    modules.mkdir(parents=True)
    (modules / "2.0.lua").write_text('setenv("TOOL_VERSION", "2.0")\n')
    monkeypatch.setenv("MODULEPATH", str(tmp_path / "modules"))
    env = lmod.Lmod(modules=["tool"], warm=True)
    calls = []
    real = lmod.Lmod._purge_and_load.__func__

    def counting(cls, names):
        calls.append(list(names))
        return real(cls, names)

    monkeypatch.setattr(lmod.Lmod, "_purge_and_load", classmethod(counting))
    session = WarmSession(env)
    with serving(session):
        for i in range(5):
            result = _sub(tmp_path / "warm", env)(_sh(f"echo $TOOL_VERSION {i}"))
            assert result.outputs.stdout.split() == ["2.0", str(i)]
    cold = _sub(tmp_path / "cold", lmod.Lmod(modules=["tool"]))(
        _sh("echo $TOOL_VERSION 0")
    )
    assert cold.outputs.stdout.split() == ["2.0", "0"]
    assert calls.count(["tool/2.0"]) == 2  # one warm start, one cold run


def test_sigterm_inside_the_registry_lock_does_not_deadlock():
    """The SIGTERM handler runs on the main thread, which may be inside the open-environments
    lock when the signal arrives: the lock is re-entrant, so closing them never waits on itself.
    """
    from pydra.environments import base

    with base._OPEN_LOCK:
        assert base._OPEN_LOCK.acquire(blocking=False)
        base._OPEN_LOCK.release()
        base.close_open_environments()
