import os
import sys
import textwrap
from pathlib import Path
import cloudpickle
import pytest
from pydra.compose import shell
from pydra.engine.job import Job
from pydra.engine.submitter import Submitter
from pydra.environments import base, images, lmod, singularity
from pydra.environments.base import ImageResolutionError
from pydra.engine.tests.utils import need_lmod, need_singularity, no_win

D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64

posix_only = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="fakes are POSIX shell scripts"
)


class FakeRegistry:
    """Stands for the registry ``HEAD`` (and the platform lookup behind it)"""

    def __init__(self, monkeypatch, answers=None):
        self.answers = answers or {}  # "repo:tag" -> (digest, platform digest)
        self.asked = []
        monkeypatch.setattr(images, "registry_manifest", self.ask)

    def ask(self, repository, tag):
        self.asked.append((repository, tag))
        return self.answers.get(f"{repository}:{tag}")


def _ran(monkeypatch):
    commands = []
    monkeypatch.setattr(
        base,
        "execute",
        lambda cmd, cwd=None, strip=False, **kw: commands.append(cmd) or (0, "", ""),
    )
    return commands


def _job(tmp_path, env, **submitter_kwargs):
    sub = Submitter(
        store="pickle",
        environment=env,
        cache_root=tmp_path / "cache",
        telemetry=False,
        **submitter_kwargs,
    )
    return Job(task=shell.define("whoami")(), name="j", submitter=sub)


@pytest.fixture(autouse=True)
def hash_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYDRA_HASH_CACHE", str(tmp_path / "hashes"))
    getattr(images, "_SIF_DIGESTS", {}).clear()


# Singularity / Apptainer ------------------------------------------------------------


def test_singularity_docker_uri_digest(monkeypatch, tmp_path):
    fake = FakeRegistry(monkeypatch, {"org/tool:1": (D1, D2)})
    env = singularity.Singularity(image="docker://org/tool", tag="1")
    job = _job(tmp_path, env)
    ref = job.resolved
    # the key is the Docker key of the same image
    assert ref.key == f"docker@1:{D1}"
    assert ref.run == f"docker://org/tool@{D1}"
    assert ref.carried["repository"] == "org/tool"
    # the per-platform manifest digest is recorded, not keyed
    assert ref.recorded["platform_digest"] == D2
    assert job.context["environment_facts"]["platform_digest"] == D2
    assert job.record["job"]["environment"] == f"docker@1:{D1}"
    # once per tag per run
    for name in "abc":
        Job(task=shell.define("whoami")(), name=name, submitter=job.submitter).resolved
    assert fake.asked == [("org/tool", "1")]
    # a digest given is keyed with no lookup, and the tag in the URI is honoured
    fake.asked.clear()
    given = singularity.Singularity(image=f"docker://org/tool@{D1}").resolve()
    assert (given.key, given.run) == (f"docker@1:{D1}", f"docker://org/tool@{D1}")
    assert fake.asked == []
    singularity.Singularity(image="docker://org/tool:2").resolve()
    assert fake.asked == [("org/tool", "2")]


def test_singularity_docker_uri_offline_unresolved(monkeypatch, tmp_path):
    FakeRegistry(monkeypatch)
    env = singularity.Singularity(image="docker://org/tool", tag="1")
    job = _job(tmp_path, env)
    assert job.resolved.unresolved
    assert job.resolved.run == "docker://org/tool:1"
    assert "unknown" in job.context
    with pytest.raises(ImageResolutionError, match="org/tool"):
        _job(tmp_path, env, resolve_images="require").checksum


def _sif(tmp_path, content=b"SIF image bytes"):
    path = tmp_path / "tool.sif"
    path.write_bytes(content)
    return path


def test_sif_content_key_and_worker_check(monkeypatch, tmp_path):
    import hashlib

    path = _sif(tmp_path)
    env = singularity.Singularity(image=str(path))
    job = _job(tmp_path, env)
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert job.resolved.key == f"sif@1:{digest}"
    assert job.record["job"]["environment"] == f"sif@1:{digest}"
    blob = cloudpickle.dumps(job)

    # the worker checks the file through its own stat cache and runs it
    getattr(images, "_SIF_DIGESTS", {}).clear()
    worker_job = cloudpickle.loads(blob)
    commands = _ran(monkeypatch)
    worker_job.environment.execute(worker_job)
    assert str(path) in commands[0]

    # the file changed under the key: the job fails and nothing runs
    path.write_bytes(b"another image")
    os.utime(path, ns=(1, 1))
    worker_job = cloudpickle.loads(blob)
    commands.clear()
    with pytest.raises(ImageResolutionError, match=digest):
        worker_job.environment.execute(worker_job)
    assert commands == []


def test_sif_digest_read_once(monkeypatch, tmp_path):
    path = _sif(tmp_path)
    reads = []
    real = images._sha256_file
    monkeypatch.setattr(images, "_sha256_file", lambda p: reads.append(p) or real(p))
    first = images.sif_digest(path)
    assert images.sif_digest(path) == first
    images._SIF_DIGESTS.clear()  # a new process: the persistent cache answers
    assert images.sif_digest(path) == first
    assert len(reads) == 1


def test_rebuilt_sif_gets_a_new_key(tmp_path):
    path = _sif(tmp_path)
    before = singularity.Singularity(image=str(path)).resolve().key
    path.write_bytes(b"rebuilt image")
    os.utime(path, ns=(2, 2))
    assert singularity.Singularity(image=str(path)).resolve().key != before


def test_sif_path_runs_as_given(monkeypatch, tmp_path):
    path = _sif(tmp_path)
    env = singularity.Singularity(image=str(path), tag="ignored")
    job = _job(tmp_path, env)
    assert job.resolved.run == str(path)
    commands = _ran(monkeypatch)
    env.execute(job)
    assert str(path) in commands[0]
    assert f"{path}:ignored" not in commands[0]
    assert not any(a.endswith(":latest") for a in commands[0])


def test_relative_sif_runs_by_its_absolute_path(monkeypatch, tmp_path):
    path = _sif(tmp_path)
    monkeypatch.chdir(tmp_path)
    env = singularity.Singularity(image="tool.sif")
    job = _job(tmp_path, env)
    assert job.resolved.run == str(path.resolve())
    blob = cloudpickle.dumps(job)
    monkeypatch.chdir(tmp_path / "cache")  # a worker elsewhere
    worker_job = cloudpickle.loads(blob)
    commands = _ran(monkeypatch)
    worker_job.environment.execute(worker_job)
    assert str(path.resolve()) in commands[0]
    assert "tool.sif" not in [a for a in commands[0] if a != str(path.resolve())]


def test_sif_missing_unresolved(tmp_path):
    env = singularity.Singularity(image=str(tmp_path / "absent.sif"))
    job = _job(tmp_path, env)
    assert job.resolved.unresolved
    assert "unknown" in job.context


def test_library_uri_unresolved(monkeypatch, tmp_path):
    fake = FakeRegistry(monkeypatch)
    for image in ("library://user/col/tool", "shub://user/tool"):
        env = singularity.Singularity(image=image, tag="1")
        job = _job(tmp_path, env)
        assert job.resolved.unresolved
        assert job.resolved.run == f"{image}:1"
        assert "unknown" in job.context
    assert fake.asked == []


def test_singularity_checksum_never_resolves(monkeypatch, tmp_path):
    fake = FakeRegistry(monkeypatch, {"org/tool:1": (D1, None)})
    env = singularity.Singularity(image="docker://org/tool", tag="1")
    assert env.checksum() == env.checksum()
    assert fake.asked == []


def _fake_singularity(tmp_path, monkeypatch, exit_code=0, pull_sleep=0):
    """A fake ``singularity``: ``pull <dest> <uri>`` writes ``dest`` (after
    ``pull_sleep`` seconds); ``exec`` writes INFO lines to stderr and exits
    ``exit_code``. Each call is logged, one line of arguments each."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "singularity.log"
    script = bindir / "singularity"
    script.write_text(textwrap.dedent(f"""\
            #!/bin/sh
            echo "$*" >> "{log}"
            if [ "$1" = pull ]; then
                sleep {pull_sleep}
                echo "SIF of $3" > "$2"
                exit 0
            fi
            echo "INFO:    Converting OCI blobs to SIF format" >&2
            echo "INFO:    Starting build..." >&2
            echo "ran"
            exit {exit_code}
            """))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYDRA_IMAGE_CACHE", str(tmp_path / "images"))
    return log


@posix_only
def test_docker_uri_pulled_once_to_a_local_sif(monkeypatch, tmp_path):
    log = _fake_singularity(tmp_path, monkeypatch)
    env = singularity.Singularity(image=f"docker://org/tool@{D1}")
    for name in ("a", "b"):
        job = _job(tmp_path / name, env)
        assert job.resolved.key == f"docker@1:{D1}"
        env.execute(job)
    sif = images.local_sif_path(D1)
    assert sif.parent == tmp_path / "images" / "sif"
    assert sif.read_text().strip() == f"SIF of docker://org/tool@{D1}"
    calls = log.read_text().splitlines()
    assert [c for c in calls if c.startswith("pull ")] == [
        f"pull {calls[0].split()[1]} docker://org/tool@{D1}"
    ]
    execs = [c for c in calls if c.startswith("exec ")]
    assert len(execs) == 2
    assert all(str(sif) in c.split() and "docker://" not in c for c in execs)


@posix_only
def test_concurrent_first_use_pulls_once(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    log = _fake_singularity(tmp_path, monkeypatch, pull_sleep=0.5)
    with ThreadPoolExecutor(4) as pool:
        paths = list(
            pool.map(
                lambda _: images.local_sif(f"docker://org/tool@{D1}", D1), range(4)
            )
        )
    assert len(set(paths)) == 1 and Path(paths[0]).is_file()
    assert sum(c.startswith("pull ") for c in log.read_text().splitlines()) == 1
    assert not [
        p
        for p in Path(paths[0]).parent.iterdir()
        if p.suffix != ".sif" and not p.name.endswith(".lock")
    ]


@posix_only
def test_sif_name_carries_the_platform(monkeypatch, tmp_path):
    log = _fake_singularity(tmp_path, monkeypatch)
    uri = f"docker://org/tool@{D1}"
    monkeypatch.setattr(images, "host_platform", lambda: ("linux", "amd64", ""))
    amd = images.local_sif(uri, D1)
    monkeypatch.setattr(images, "host_platform", lambda: ("linux", "arm64", "v8"))
    arm = images.local_sif(uri, D1)
    assert amd != arm and amd.is_file() and arm.is_file()
    assert "linux-amd64" in amd.name and "linux-arm64-v8" in arm.name
    assert sum(c.startswith("pull ") for c in log.read_text().splitlines()) == 2


@posix_only
def test_lock_unavailable_pulls_unlocked(monkeypatch, tmp_path):
    import errno
    import fcntl

    _fake_singularity(tmp_path, monkeypatch)

    def nolock(fd, op):
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(fcntl, "flock", nolock)
    sif = images.local_sif(f"docker://org/tool@{D1}", D1)
    assert sif.read_text().strip() == f"SIF of docker://org/tool@{D1}"


@posix_only
def test_read_only_lock_file_pulls(monkeypatch, tmp_path):
    _fake_singularity(tmp_path, monkeypatch)
    sif = images.local_sif_path(D1)
    sif.parent.mkdir(parents=True)
    lock = sif.parent / f"{sif.name}.lock"
    lock.write_text("")
    lock.chmod(0o444)  # created by another user of a group-shared cache
    try:
        assert images.local_sif(f"docker://org/tool@{D1}", D1).is_file()
    finally:
        lock.chmod(0o644)


def test_image_cache_prefers_apptainer_cachedir(monkeypatch, tmp_path):
    monkeypatch.delenv("PYDRA_IMAGE_CACHE", raising=False)
    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "apptainer"))
    assert images.image_cache() == tmp_path / "apptainer" / "pydra-sif"
    monkeypatch.setenv("PYDRA_IMAGE_CACHE", str(tmp_path / "mine"))
    assert images.image_cache() == tmp_path / "mine"


@posix_only
def test_failed_pull_leaves_no_sif(monkeypatch, tmp_path):
    _fake_singularity(tmp_path, monkeypatch)
    monkeypatch.setattr(
        base, "execute", lambda cmd, **kw: (255, "", "FATAL: registry unreachable")
    )
    with pytest.raises(RuntimeError, match="registry unreachable"):
        images.local_sif(f"docker://org/tool@{D1}", D1)
    assert not list((tmp_path / "images" / "sif").glob("*.sif"))


@posix_only
def test_singularity_info_on_stderr_with_exit_0_succeeds(monkeypatch, tmp_path):
    _fake_singularity(tmp_path, monkeypatch, 0)
    env = singularity.Singularity(image=f"docker://org/tool@{D1}")
    job = _job(tmp_path, env)
    output = env.execute(job)
    assert output["return_code"] == 0
    assert output["stdout"].strip() == "ran"
    assert "Converting OCI blobs" in output["stderr"]


@posix_only
def test_singularity_exit_1_fails_with_stderr(monkeypatch, tmp_path):
    _fake_singularity(tmp_path, monkeypatch, 1)
    env = singularity.Singularity(image=f"docker://org/tool@{D1}")
    job = _job(tmp_path, env)
    with pytest.raises(RuntimeError, match="Converting OCI blobs") as info:
        env.execute(job)
    assert "exit code 1" in str(info.value)


@no_win
@need_singularity
@pytest.mark.singularity
def test_apptainer_docker_digest_runs(tmp_path, monkeypatch):
    import time

    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "apptainer-cache"))
    monkeypatch.setenv("SINGULARITY_CACHEDIR", str(tmp_path / "apptainer-cache"))
    monkeypatch.setenv("PYDRA_IMAGE_CACHE", str(tmp_path / "images"))
    env = singularity.Singularity(image="docker://alpine", tag="3.20")
    ref = env.resolve()
    assert ref.key.startswith("docker@1:sha256:")
    assert ref.run.startswith("docker://alpine@sha256:")
    echo = shell.define("echo")(executable="echo", append_args=["hi"])
    with Submitter(
        store="pickle",
        environment=env,
        cache_root=tmp_path / "cold",
        telemetry=False,
        resolve_images="require",
    ) as sub:
        result = sub(echo)
    assert result.outputs.stdout.strip() == "hi"
    assert result.context["environment"] == ref.key
    assert images.local_sif_path(ref.key.removeprefix("docker@1:")).is_file()

    # warm: the same digest again neither converts nor reaches a registry
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    warm = singularity.Singularity(image=ref.run)
    start = time.perf_counter()
    with Submitter(
        store="pickle",
        environment=warm,
        cache_root=tmp_path / "warm",
        telemetry=False,
        resolve_images="require",
    ) as sub:
        result = sub(echo)
    print(f"warm apptainer exec: {(time.perf_counter() - start) * 1000:.0f} ms")
    assert result.outputs.stdout.strip() == "hi"
    assert "Converting" not in result.outputs.stderr
    assert result.context["environment"] == ref.key


@no_win
@need_singularity
@pytest.mark.singularity
def test_apptainer_relative_sif_runs(tmp_path, monkeypatch):
    import subprocess as sp

    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "apptainer-cache"))
    monkeypatch.setenv("SINGULARITY_CACHEDIR", str(tmp_path / "apptainer-cache"))
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    sp.run(
        ["singularity", "pull", "tool.sif", "docker://alpine:3.20"],
        cwd=images_dir,
        check=True,
    )
    monkeypatch.chdir(images_dir)
    env = singularity.Singularity(image="tool.sif")
    with Submitter(
        store="pickle",
        environment=env,
        cache_root=tmp_path / "cache",
        telemetry=False,
        resolve_images="require",
    ) as sub:
        result = sub(shell.define("echo")(executable="echo", append_args=["hi"]))
    assert result.outputs.stdout.strip() == "hi"
    assert result.context["environment"].startswith("sif@1:sha256:")


# Lmod -------------------------------------------------------------------------------


def _fake_lmod(tmp_path, monkeypatch, loads, shell="shellmod/1.0", path="/tree/x"):
    """A fake ``$MODULESHOME/libexec/lmod``. ``loads`` maps the requested names
    (space-joined; ``"<MODULEPATH>|<names>"`` to match one ``MODULEPATH`` only) to the
    full names loaded, dependencies included. ``purge`` unwinds ``MODULEPATH`` to
    ``/tree/core`` only when it sees loaded modules, as Lmod does. Each call is logged
    with the ``LOADEDMODULES`` and ``MODULEPATH`` it saw."""
    home = tmp_path / "lmod"
    (home / "libexec").mkdir(parents=True)
    log = tmp_path / "lmod.log"

    def pattern(key):
        return f'"{key}"' if "|" in key else f'*"|{key}"'

    cases = "\n".join(f"""  {pattern(requested)})
    echo "os.environ['LOADEDMODULES'] = '{':'.join(full)}';"
    echo "os.environ['_LMFILES_'] = '{':'.join(f'/tree/{m}.lua' for m in full)}';"
    echo "os.environ['PATH'] = '/tree/bin';"
    ;;""" for requested, full in loads.items())
    script = home / "libexec" / "lmod"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$* | ${{LOADEDMODULES-unset}} | $MODULEPATH" >> "{log}"\n'
        'if [ "$2" = purge ]; then\n'
        '  if [ -n "$LOADEDMODULES" ]; then\n'
        "    echo \"os.environ['MODULEPATH'] = '/tree/core';\"\n"
        "  fi\n"
        "  echo \"os.environ['LOADEDMODULES'] = '';\"\n"
        "  exit 0\n"
        "fi\n"
        "shift 2\n"
        'case "$MODULEPATH|$*" in\n'
        f"{cases}\n"
        "  *)\n"
        '    echo "_mlstatus = False"\n'
        "    ;;\n"
        "esac\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("MODULESHOME", str(home))
    monkeypatch.setenv("MODULEPATH", path)
    monkeypatch.setenv("LOADEDMODULES", shell)
    return log


FSL = {
    "fsl python/3.11": ["gcc/12.2", "fsl/6.0.7.9", "python/3.11.5"],
    "fsl/6.0.7.9 python/3.11.5": ["gcc/12.2", "fsl/6.0.7.9", "python/3.11.5"],
}


@posix_only
def test_lmod_full_names_and_mapping(monkeypatch, tmp_path):
    log = _fake_lmod(tmp_path, monkeypatch, FSL)
    env = lmod.Lmod(modules=["fsl", "python/3.11"])
    job = _job(tmp_path, env)
    ref = job.resolved
    # the key carries the full names of the modules asked for, in their order
    assert ref.key == "lmod@1:fsl/6.0.7.9 python/3.11.5"
    assert job.record["job"]["environment"] == ref.key
    # dependencies and the calling shell's modules are recorded, not keyed
    facts = job.context["environment_facts"]
    assert facts["modules"] == {"fsl": "fsl/6.0.7.9", "python/3.11": "python/3.11.5"}
    assert facts["dependencies"] == ["gcc/12.2"]
    assert facts["shell_modules"] == ["shellmod/1.0"]
    assert facts["modulepath"] == "/tree/core"
    assert facts["tree"] == os.environ["MODULESHOME"]
    assert "gcc" not in ref.key and "shellmod" not in ref.key
    # purge with the shell's modules in place, so Lmod unwinds them; then load
    calls = log.read_text().splitlines()
    assert calls[0] == "python purge | shellmod/1.0 | /tree/x"
    assert calls[1] == "python load fsl python/3.11 |  | /tree/core"
    # once per module list per run
    Job(task=shell.define("whoami")(), name="k", submitter=job.submitter).resolved
    assert len(log.read_text().splitlines()) == 2


@posix_only
def test_lmod_shell_compiler_does_not_leak(monkeypatch, tmp_path):
    hierarchy = "/tree/core:/tree/compiler/gcc12"
    _fake_lmod(
        tmp_path,
        monkeypatch,
        {
            f"{hierarchy}|openmpi": ["openmpi/4.1.5"],
            "gcc/12.2 openmpi": ["gcc/12.2", "openmpi/4.1.5"],
            "gcc/11.2 openmpi": ["gcc/11.2", "openmpi/4.1.5"],
        },
        shell="gcc/12.2",
        path=hierarchy,
    )
    # the shell's compiler does not select the build: it must be named, and keyed
    assert lmod.Lmod(modules=["openmpi"]).resolve().unresolved
    a = lmod.Lmod(modules=["gcc/12.2", "openmpi"]).resolve()
    b = lmod.Lmod(modules=["gcc/11.2", "openmpi"]).resolve()
    assert a.key == "lmod@1:gcc/12.2 openmpi/4.1.5"
    assert a.key != b.key


@posix_only
def test_lmod_worker_purges_then_loads_the_keyed_names(monkeypatch, tmp_path):
    log = _fake_lmod(tmp_path, monkeypatch, FSL)
    job = _job(tmp_path, lmod.Lmod(modules=["fsl", "python/3.11"]))
    job.resolved
    blob = cloudpickle.dumps(job)
    # the worker's shell has another compiler loaded
    monkeypatch.setenv("LOADEDMODULES", "gcc/11.2")
    monkeypatch.setenv("MODULEPATH", "/tree/core:/tree/compiler/gcc11")
    log.write_text("")
    worker_job = cloudpickle.loads(blob)
    envs = []
    monkeypatch.setattr(
        base,
        "execute",
        lambda cmd, cwd=None, env=None, **kw: envs.append(env) or (0, "", ""),
    )
    worker_job.environment.execute(worker_job)
    calls = log.read_text().splitlines()
    assert calls == [
        "python purge | gcc/11.2 | /tree/core:/tree/compiler/gcc11",
        "python load fsl/6.0.7.9 python/3.11.5 |  | /tree/core",
    ]
    assert envs[0]["LOADEDMODULES"] == "gcc/12.2:fsl/6.0.7.9:python/3.11.5"
    assert envs[0]["PATH"] == "/tree/bin"


@posix_only
def test_lmod_worker_warns_when_dependencies_differ(monkeypatch, tmp_path, caplog):
    _fake_lmod(tmp_path, monkeypatch, FSL)
    job = _job(tmp_path, lmod.Lmod(modules=["fsl", "python/3.11"]))
    job.resolved
    blob = cloudpickle.dumps(job)
    _fake_lmod(
        tmp_path / "worker",
        monkeypatch,
        {"fsl/6.0.7.9 python/3.11.5": ["gcc/11.2", "fsl/6.0.7.9", "python/3.11.5"]},
    )
    worker_job = cloudpickle.loads(blob)
    _ran(monkeypatch)
    with caplog.at_level("WARNING", logger="pydra"):
        worker_job.environment.execute(worker_job)
    assert "gcc/11.2" in caplog.text and "gcc/12.2" in caplog.text


@posix_only
def test_lmod_unknown_module_unresolved(monkeypatch, tmp_path):
    _fake_lmod(tmp_path, monkeypatch, {})
    job = _job(tmp_path, lmod.Lmod(modules=["nosuch"]))
    assert job.resolved.unresolved
    assert "unknown" in job.context


def test_lmod_name_mapping_needs_a_boundary():
    match = lmod.Lmod._match
    assert match("fsl", ["fslpy/1.0"]) is None
    assert match("python/3.1", ["python/3.10.4"]) is None
    assert match("python/3.10", ["python/3.10.4"]) == "python/3.10.4"
    assert match("fsl/6.0", ["fsl/6.0.7.9"]) == "fsl/6.0.7.9"


@posix_only
def test_lmod_names_mapping_to_one_module_key_it_once(monkeypatch, tmp_path):
    _fake_lmod(tmp_path, monkeypatch, {"fsl fsl/6": ["fsl/6.0.7.9"]})
    assert lmod.Lmod(modules=["fsl", "fsl/6"]).resolve().key == "lmod@1:fsl/6.0.7.9"


def test_lmod_absent_unresolved(monkeypatch, tmp_path):
    monkeypatch.delenv("MODULESHOME", raising=False)
    env = lmod.Lmod(modules=["fsl", "python"])
    job = _job(tmp_path, env)
    ref = job.resolved
    assert ref.unresolved
    assert "fsl python" in ref.key and not ref.key.startswith("lmod@1:")
    assert ref.run == "fsl python"
    assert job.record["job"]["environment"] == ref.key
    assert "unknown" in job.context
    with pytest.raises(ImageResolutionError, match="fsl"):
        _job(tmp_path, env, resolve_images="require").checksum


def test_lmod_execute_refuses_another_environments_job(monkeypatch, tmp_path):
    monkeypatch.delenv("MODULESHOME", raising=False)
    job = _job(tmp_path, None)
    with pytest.raises(ImageResolutionError, match="environment"):
        lmod.Lmod(modules=["fsl"]).execute(job)


def test_environment_facts_recorded_only(monkeypatch, tmp_path):
    from pydra.engine.portable.identity import PydraIdentityV2
    from pydra.engine.store import ConsistentReuse, context_key

    contexts = []
    for platform_digest in (D1, D2):
        FakeRegistry(monkeypatch, {"org/tool:1": (D1, platform_digest)})
        job = _job(
            tmp_path / platform_digest[-1],
            singularity.Singularity(image="docker://org/tool", tag="1"),
        )
        contexts.append(job.context)
    assert contexts[0]["environment_facts"] != contexts[1]["environment_facts"]
    ident = PydraIdentityV2()
    assert "environment_facts" in ident.recorded_only
    assert ident.fingerprint(contexts[0]) == ident.fingerprint(contexts[1])
    assert context_key(contexts[0]) == context_key(contexts[1])
    assert ConsistentReuse().accept(contexts[0], contexts[1], None)


@no_win
@need_lmod
@pytest.mark.lmod
def test_lmod_real_resolution(tmp_path, monkeypatch):
    import time

    modules = tmp_path / "modules" / "tool"
    modules.mkdir(parents=True)
    (modules / "1.0.lua").write_text('setenv("TOOL_VERSION", "1.0")\n')
    (modules / "2.0.lua").write_text('setenv("TOOL_VERSION", "2.0")\n')
    monkeypatch.setenv("MODULEPATH", str(tmp_path / "modules"))
    env = lmod.Lmod(modules=["tool"])
    start = time.perf_counter()
    ref = env.resolve()
    elapsed = time.perf_counter() - start
    print(f"lmod resolution: {elapsed * 1000:.1f} ms")  # measured on the cluster (§7)
    assert ref.key == "lmod@1:tool/2.0"
    assert ref.recorded["modules"] == {"tool": "tool/2.0"}
    with Submitter(
        store="pickle", environment=env, cache_root=tmp_path / "c", telemetry=False
    ) as sub:
        result = sub(shell.define("printenv TOOL_VERSION")())
    assert result.outputs.stdout.strip() == "2.0"
    assert result.context["environment"] == "lmod@1:tool/2.0"


def test_an_empty_image_cache_variable_counts_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("PYDRA_IMAGE_CACHE", "")
    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "apptainer"))
    assert images.image_cache() == tmp_path / "apptainer" / "pydra-sif"


@posix_only
def test_an_unreadable_cached_sif_is_refused(monkeypatch, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root reads any file")
    monkeypatch.setenv("PYDRA_IMAGE_CACHE", str(tmp_path / "cache"))
    sif = images.local_sif_path(D1)
    sif.parent.mkdir(parents=True)
    sif.write_text("SIF")
    sif.chmod(0)
    with pytest.raises(PermissionError, match="cannot be read"):
        images.local_sif(f"docker://org/tool@{D1}", D1)
