"""Warm environments (RFC 0005 section 3.1, W3): one started container, Apptainer
instance or Lmod resolution serves many jobs. The runtimes are fakes that run each
command on the host, mapping the container paths back and refusing a path outside the
bound roots, so outputs and isolation are those of real commands."""

import os
import re
import shutil
import subprocess as sp
import sys
import tempfile
import typing as ty
from pathlib import Path

import attrs
import pytest

from pydra.compose import shell
from pydra.engine.job import Job
from pydra.engine.resident import Device, ResidentKey
from pydra.engine.submitter import Submitter
from pydra.environments import base, docker, images, lmod, native, singularity
from pydra.environments.base import WarmSession, serving

D1 = "sha256:" + "1" * 64
ROOT = "/mnt/pydra"

posix_only = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="the fakes run POSIX commands"
)
pytestmark = posix_only

Cp = shell.define("cp <in_file> <out|out_file>")


class FakeRuntime:
    """Stands for ``docker`` and ``apptainer``: a started container or instance is a
    set of bound host roots; a command in it runs on the host with the container
    paths mapped back, and fails as a missing file outside the bound roots."""

    def __init__(self, monkeypatch, tmp_path, fail_start=False):
        self.calls: list[list[str]] = []
        self.started: dict[str, dict] = {}
        self.stopped: list[str] = []
        self.sifs: list[str] = []
        self.fail_start = fail_start
        self.images = tmp_path / "images"
        self.cold_home = tmp_path / "image-home"
        self.cold_home.mkdir()
        monkeypatch.setattr(base, "execute", self.execute)
        monkeypatch.setattr(singularity, "runtime", lambda: "apptainer")
        monkeypatch.setattr(images, "local_sif", self.local_sif)

    def local_sif(self, uri, digest):
        self.sifs.append(uri)
        self.images.mkdir(exist_ok=True)
        path = self.images / f"{digest.replace(':', '-')}.sif"
        path.write_text(f"SIF of {uri}")
        return path

    def of(self, *prefix):
        return [c for c in self.calls if c[: len(prefix)] == list(prefix)]

    @property
    def starts(self):
        return self.of("docker", "run", "-d") + self.of(
            "apptainer", "instance", "start"
        )

    @property
    def execs(self):
        return self.of("docker", "exec") + [
            c
            for c in self.of("apptainer", "exec")
            if any(a.startswith("instance://") for a in c)
        ]

    @property
    def cold(self):
        return [
            c
            for c in self.of("docker", "run") + self.of("singularity", "exec")
            if "-d" not in c
        ]

    def execute(self, cmd, strip=False, cwd=None, env=None, **kwargs):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        tool, rest = cmd[0], cmd[1:]
        if tool == "docker" and rest[:2] == ["run", "-d"]:
            return self._start(cmd, "-v")
        if tool == "docker" and rest[0] == "run":
            return self._cold(rest[1:], "-v", "-w", {"--rm"})
        if tool == "docker" and rest[0] == "exec":
            return self._exec_docker(rest[1:])
        if tool == "docker" and rest[:2] == ["rm", "-f"]:
            self.stopped.append(rest[2])
            return 0, rest[2], ""
        if tool == "apptainer" and rest[:2] == ["instance", "start"]:
            return self._start(cmd, "-B")
        if tool == "apptainer" and rest[:2] == ["instance", "stop"]:
            self.stopped.append(rest[2])
            return 0, "", ""
        if tool == "apptainer" and rest[0] == "exec":
            return self._exec_apptainer(rest[1:])
        if tool == "singularity" and rest[0] == "exec":
            return self._cold(rest[1:], "-B", "--pwd", set())
        raise AssertionError(f"unexpected command {cmd}")

    def _start(self, cmd, bind_flag):
        if self.fail_start:
            return 125, "", "cannot start"
        valued = {bind_flag, "--name", "--user", "--gpus", "--tmpfs", "--entrypoint"}
        binds, args, i = [], [], 2
        while i < len(cmd):
            if cmd[i] in valued:
                if cmd[i] == bind_flag:
                    binds.append(Path(cmd[i + 1].split(":")[0]))
                i += 2
            else:
                args.append(cmd[i])
                i += 1
        if cmd[0] == "docker":
            name = f"id{len(self.started)}"
            image = [a for a in args if not a.startswith("-")][0]
        else:
            name = args[-1]
            image = args[-2]
        self.started[name] = {"binds": binds, "image": image, "cmd": cmd}
        return 0, name, ""

    def _host(self, binds, value):
        def repl(match):
            host = match.group(1)
            if not any(Path(host) == b or b in Path(host).parents for b in binds):
                raise FileNotFoundError(host)
            return host

        return re.sub(rf"{ROOT}(/[^\s:=]*)", repl, value)

    def _run(self, binds, workdir, env, args):
        try:
            workdir = self._host(binds, workdir)
            env = {k: self._host(binds, v) for k, v in env.items()}
            args = [self._host(binds, a) for a in args]
        except FileNotFoundError as e:
            return 1, "", f"No such file or directory: {e}"
        proc = sp.run(
            args,
            cwd=workdir,
            env={"PATH": os.environ["PATH"], **env},
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def _exec_docker(self, rest):
        workdir, env, i = None, {}, 0
        while rest[i].startswith("-"):
            if rest[i] == "-w":
                workdir = rest[i + 1]
            elif rest[i] == "-e":
                k, _, v = rest[i + 1].partition("=")
                env[k] = v
            i += 2
        name, args = rest[i], rest[i + 1 :]
        if name in self.stopped or name not in self.started:
            return 1, "", f"No such container: {name}"
        return self._run(self.started[name]["binds"], workdir, env, args)

    def _exec_apptainer(self, rest):
        workdir, i = None, 0
        while rest[i].startswith("-"):
            workdir = rest[i + 1] if rest[i] == "--pwd" else workdir
            i += 2
        name = rest[i].removeprefix("instance://")
        if name in self.stopped or name not in self.started:
            return 255, "", f"no instance found with name {name}"
        return self._run(self.started[name]["binds"], workdir, {}, rest[i + 1 :])

    def _cold(self, rest, bind_flag, workdir_flag, bare):
        binds, workdir, i = [], None, 0
        while rest[i].startswith("-"):
            if rest[i] in bare:
                i += 1
                continue
            if rest[i] == bind_flag:
                binds.append(Path(rest[i + 1].split(":")[0]))
            elif rest[i] == workdir_flag:
                workdir = rest[i + 1]
            i += 2
        args = rest[i + 1 :]
        return self._run(binds, workdir, {"HOME": str(self.cold_home)}, args)


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


ENVIRONMENTS = {
    "docker": lambda **kw: docker.Docker(image=f"org/tool@{D1}", **kw),
    "apptainer": lambda **kw: singularity.Singularity(
        image=f"docker://org/tool@{D1}", **kw
    ),
}


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Warm scratch directories under ``tmp_path``"""
    (tmp_path / "scratch").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))


@pytest.fixture
def roots_below(tmp_path, monkeypatch, scratch):
    """A binding root is a directory directly under ``tmp_path``, so a test can make
    jobs that need distinct roots"""
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
def test_one_run_n_exec(kind, monkeypatch, tmp_path, scratch):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = ENVIRONMENTS[kind](warm=True)
    session = WarmSession(env)
    with serving(session):
        for i in range(5):
            result = _sub(tmp_path / "cache", env)(
                shell.define("echo")(executable="echo", append_args=[str(i)])
            )
            assert result.outputs.stdout.strip() == str(i)
    assert len(fake.starts) == 1
    assert len(fake.execs) == 5
    assert fake.cold == []
    assert len(fake.stopped) == 1
    assert session.starts == 1 and session.cold_runs == 0
    if kind == "docker":
        start = fake.starts[0]
        assert start[:5] == ["docker", "run", "-d", "--rm", "--init"]
        assert start[-3:] == ["sleep", f"org/tool@{D1}", "2147483647"]
    else:
        # the instance starts from the local SIF, never from the docker:// URI
        start = fake.starts[0]
        assert start[:5] == ["apptainer", "instance", "start", "-c", "--no-home"]
        assert not any("docker://" in a for a in start)
        assert fake.started[start[-1]]["image"].endswith(".sif")
        assert fake.sifs == [f"docker://org/tool@{D1}"]


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
    resolutions = [n for n in loads if n == ["tool"]]
    worker_loads = [n for n in loads if n == ["tool/2.0"]]
    assert len(worker_loads) == 1  # one purge and load of the full names
    assert resolutions  # the submitting process resolved the requested names
    assert session.starts == 1 and session.cold_runs == 0
    # cold, each command loads
    loads.clear()
    for i in range(3):
        _sub(tmp_path / "cold", env)(_sh(f"echo $TOOL_VERSION {i}"))
    assert len([n for n in loads if n == ["tool/2.0"]]) == 3


@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_job_home_and_tmp_not_visible_to_next_job(kind, monkeypatch, tmp_path, scratch):
    FakeRuntime(monkeypatch, tmp_path)
    env = ENVIRONMENTS[kind](warm=True)
    session = WarmSession(env)
    with serving(session):
        a = _sub(tmp_path / "cache", env)(
            _sh('echo a > "$HOME/x"; echo b > "$TMPDIR/y"; echo "$HOME $TMPDIR"')
        )
        b = _sub(tmp_path / "cache", env)(
            _sh(
                'test -e "$HOME/x" && echo seen-x; test -e "$TMPDIR/y" && echo seen-y;'
                ' echo "$HOME $TMPDIR"'
            )
        )
        assert session.starts == 1
    home_a, tmp_a = a.outputs.stdout.split()
    lines = b.outputs.stdout.split()
    assert "seen-x" not in lines and "seen-y" not in lines
    home_b, tmp_b = lines[-2:]
    assert {home_a, tmp_a}.isdisjoint({home_b, tmp_b})
    assert not Path(home_a).exists() and not Path(tmp_a).exists()
    # neither is in the job's own directory, which stays as a cold run leaves it
    assert not (a.cache_dir / "home").exists()


@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_outputs_owned_by_submitting_user(kind, monkeypatch, tmp_path, scratch):
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


def test_unmounted_root_one_restart(monkeypatch, tmp_path, roots_below):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=f"org/tool@{D1}", warm=True)
    session = WarmSession(env)
    with serving(session):
        for name in ("in1", "in2", "in1", "in2"):
            src = _input(tmp_path / name / "in.txt", f"{name}\n")
            result = _sub(tmp_path / "cache", env)(Cp(in_file=src))
            assert Path(result.outputs.out_file).read_text() == f"{name}\n"
    assert session.starts == 2  # in2 was not mounted: one restart
    assert session.cold_runs == 0 and fake.cold == []
    assert len(fake.starts) == 2 and len(fake.execs) == 2  # the rest were cached
    first, second = (fake.started[n]["binds"] for n in ("id0", "id1"))
    assert tmp_path / "in2" not in first
    assert {tmp_path / "in1", tmp_path / "in2"} <= set(second)
    assert fake.stopped[0] == "id0"


def test_beyond_max_mount_roots_runs_cold(monkeypatch, tmp_path, roots_below):
    fake = FakeRuntime(monkeypatch, tmp_path)
    # cache, scratch and in1 are three roots; in2 would be a fourth
    env = docker.Docker(image=f"org/tool@{D1}", warm=True, max_mount_roots=3)
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


@pytest.mark.parametrize("kind", ENVIRONMENTS)
def test_warm_and_cold_outputs_equal(kind, monkeypatch, tmp_path, scratch):
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
    for result in (cold, warm):
        assert not result.errored
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


def test_native_warm_is_none(monkeypatch, tmp_path):
    assert native.Native().warm() is None
    assert native.Native(warm=True).warm([tmp_path], Device()) is None
    assert docker.Docker(image="busybox").warm([tmp_path]) is None  # not opted in
    env = native.Native(warm=True)
    session = WarmSession(env)
    with serving(session):
        result = _sub(tmp_path / "cache", env)(
            shell.define("echo")(executable="echo", append_args=["hi"])
        )
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


def test_resident_key_is_the_resolved_reference(monkeypatch, tmp_path, scratch):
    FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=f"org/tool@{D1}", warm=True)
    ref = env.resolve()
    with env.warm([tmp_path], ref=ref) as handle:
        assert handle.key == ResidentKey("environment", "docker", f"docker@1:{D1}")
        assert tmp_path in handle.roots


def test_max_jobs_restarts(monkeypatch, tmp_path, scratch):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=f"org/tool@{D1}", warm=True, max_jobs=2)
    session = WarmSession(env)
    with serving(session):
        for i in range(5):
            _sub(tmp_path / "cache", env)(
                shell.define("echo")(executable="echo", append_args=[str(i)])
            )
    assert session.starts == 3 and len(fake.starts) == 3 and len(fake.execs) == 5


def test_gpu_flags_at_start(monkeypatch, tmp_path, scratch):
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
        start = fake.calls[-2]  # the start; the stop follows
        if kind == "docker":
            assert start[start.index("--gpus") + 1] == '"device=2,3"'
        else:
            assert "--nv" in start
            assert "CUDA_VISIBLE_DEVICES=2,3" in cmd
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with ENVIRONMENTS["docker"](warm=True).warm([tmp_path], Device("cuda", 1)):
        start = fake.of("docker", "run", "-d")[-1]
        assert start[start.index("--gpus") + 1] == '"device=1"'
    with ENVIRONMENTS["docker"](warm=True).warm([tmp_path], Device("cpu")):
        assert "--gpus" not in fake.of("docker", "run", "-d")[-1]


def test_read_only_docker(monkeypatch, tmp_path, scratch):
    fake = FakeRuntime(monkeypatch, tmp_path)
    with docker.Docker(image=f"org/tool@{D1}", warm=True, read_only=True).warm(
        [tmp_path]
    ):
        start = fake.starts[-1]
    assert "--read-only" in start
    assert start[start.index("--tmpfs") + 1] == "/tmp"


def test_setup_and_teardown_called(monkeypatch, tmp_path, scratch):
    FakeRuntime(monkeypatch, tmp_path)
    events = []

    @attrs.define
    class Counting(docker.Docker):
        _plugin_name: ty.ClassVar[str] = "docker"

        def setup(self):
            events.append("setup")

        def teardown(self):
            events.append("teardown")

    handle = Counting(image=f"org/tool@{D1}", warm=True).warm([tmp_path])
    assert events == ["setup"]
    handle.close()
    handle.close()
    assert events == ["setup", "teardown"]
    assert not handle.scratch.exists()


def test_failed_start_runs_cold(monkeypatch, tmp_path, scratch):
    fake = FakeRuntime(monkeypatch, tmp_path, fail_start=True)
    env = docker.Docker(image=f"org/tool@{D1}", warm=True)
    session = WarmSession(env)
    with serving(session):
        for i in range(3):
            result = _sub(tmp_path / "cache", env)(
                shell.define("echo")(executable="echo", append_args=[str(i)])
            )
            assert result.outputs.stdout.strip() == str(i)
    assert len(fake.starts) == 1  # tried once
    assert session.cold_runs == 3 and len(fake.cold) == 3
    assert list((tmp_path / "scratch").iterdir()) == []


def test_another_reference_runs_cold(monkeypatch, tmp_path, scratch):
    fake = FakeRuntime(monkeypatch, tmp_path)
    env = docker.Docker(image=f"org/tool@{D1}", warm=True)
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


# Real runtimes: Docker on the CI ubuntu runner, Apptainer and Lmod on ORCD (O5) ------


def _docker_up():
    if shutil.which("docker") is None:
        return False
    return not sp.run(["docker", "info"], capture_output=True).returncode


@pytest.mark.docker
@pytest.mark.skipif(shutil.which("docker") is None, reason="no docker")
def test_warm_docker_real(tmp_path, scratch):
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
            _sh('echo a > "$HOME/x"; echo b > "$TMPDIR/y"')
        )
        b = _sub(tmp_path / "warm", warm_env)(
            _sh(
                'test -e "$HOME/x" && echo seen-x; test -e "$TMPDIR/y" && echo seen-y;'
                " id -u"
            )
        )
        assert session.starts == 1 and session.cold_runs == 0
    assert not a.errored
    assert "seen" not in b.outputs.stdout
    assert b.outputs.stdout.split()[-1] == str(os.getuid())
    assert Path(warm.outputs.out_file).read_bytes() == src.read_bytes()
    assert Path(warm.outputs.out_file).stat().st_uid == os.getuid()
    assert cold.outputs.stdout == warm.outputs.stdout
    assert cold.cache_dir.name == warm.cache_dir.name


@pytest.mark.singularity
@pytest.mark.skipif(
    not (shutil.which("apptainer") or shutil.which("singularity")),
    reason="no apptainer",
)
def test_warm_apptainer_real(tmp_path, monkeypatch, scratch):
    """O5: one instance for many jobs, from the local SIF, equal to cold"""
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
            _sub(tmp_path / "warm", warm_env)(_sh(f'echo {i} > "$HOME/x"; echo {i}'))
        b = _sub(tmp_path / "warm", warm_env)(
            _sh(
                'test -e "$HOME/x" && echo seen-x; test -e "$TMPDIR/y" && echo seen-y;'
                " true"
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
    """O5: one resolution for many commands, equal to cold"""
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
