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
    commands = _ran(monkeypatch)
    env.execute(job)
    assert f"docker://org/tool@{D1}" in commands[0]
    assert not any(a.endswith(":1") or a.endswith(":latest") for a in commands[0])
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


def _fake_singularity(tmp_path, monkeypatch, exit_code):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "singularity"
    script.write_text(textwrap.dedent(f"""\
            #!/bin/sh
            echo "INFO:    Converting OCI blobs to SIF format" >&2
            echo "INFO:    Starting build..." >&2
            echo "ran"
            exit {exit_code}
            """))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


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
def test_apptainer_docker_digest_runs(tmp_path):
    env = singularity.Singularity(image="docker://alpine", tag="3.20")
    ref = env.resolve()
    assert ref.key.startswith("docker@1:sha256:")
    assert ref.run.startswith("docker://alpine@sha256:")
    assert ref.recorded.get("platform_digest", "").startswith("sha256:")
    with Submitter(
        environment=env,
        cache_root=tmp_path,
        telemetry=False,
        resolve_images="require",
    ) as sub:
        result = sub(shell.define("echo")(executable="echo", append_args=["hi"]))
    assert result.outputs.stdout.strip() == "hi"
    assert result.context["environment"] == ref.key


# Lmod -------------------------------------------------------------------------------


def _fake_lmod(tmp_path, monkeypatch, loads):
    """A fake ``$MODULESHOME/libexec/lmod``: ``loads`` maps the requested names
    (space-joined) to the full names loaded, dependencies included. Each call is logged
    with the ``LOADEDMODULES`` it saw."""
    home = tmp_path / "lmod"
    (home / "libexec").mkdir(parents=True)
    log = tmp_path / "lmod.log"
    cases = "\n".join(f"""  "{requested}")
    echo "os.environ['LOADEDMODULES'] = '{':'.join(full)}';"
    echo "os.environ['_LMFILES_'] = '{':'.join(f'/tree/{m}.lua' for m in full)}';"
    echo "os.environ['PATH'] = '/tree/bin';"
    ;;""" for requested, full in loads.items())
    script = home / "libexec" / "lmod"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$* | ${{LOADEDMODULES-unset}}" >> "{log}"\n'
        'if [ "$2" = purge ]; then\n'
        "  echo \"os.environ['LOADEDMODULES'] = '';\"\n"
        "  exit 0\n"
        "fi\n"
        "shift 2\n"
        'case "$*" in\n'
        f"{cases}\n"
        "  *)\n"
        '    echo "_mlstatus = False"\n'
        "    ;;\n"
        "esac\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("MODULESHOME", str(home))
    monkeypatch.setenv("MODULEPATH", "/tree/modules")
    monkeypatch.setenv("LOADEDMODULES", "shellmod/1.0")
    return log


@posix_only
def test_lmod_full_names_and_mapping(monkeypatch, tmp_path):
    log = _fake_lmod(
        tmp_path,
        monkeypatch,
        {"fsl python/3.11": ["gcc/12.2", "fsl/6.0.7.9", "python/3.11.5"]},
    )
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
    assert facts["modulepath"] == "/tree/modules"
    assert facts["tree"] == os.environ["MODULESHOME"]
    assert "gcc" not in ref.key and "shellmod" not in ref.key
    # resolved in a child whose LOADEDMODULES is cleared: purge, then load
    calls = log.read_text().splitlines()
    assert calls[0].startswith("python purge |") and "shellmod" not in calls[0]
    assert calls[1].startswith("python load fsl python/3.11 |")
    assert "shellmod" not in calls[1]
    # once per module list per run
    Job(task=shell.define("whoami")(), name="k", submitter=job.submitter).resolved
    assert len(log.read_text().splitlines()) == 2

    # the worker loads the full names
    monkeypatch.setenv("LOADEDMODULES", "shellmod/1.0")
    worker_job = cloudpickle.loads(cloudpickle.dumps(job))
    loaded = []
    monkeypatch.setattr(
        lmod.Lmod,
        "run_lmod_cmd",
        classmethod(lambda cls, *args, env=None: loaded.append(args) or ""),
    )
    commands = _ran(monkeypatch)
    worker_job.environment.execute(worker_job)
    assert loaded == [("python", "load", "fsl/6.0.7.9", "python/3.11.5")]
    assert commands


@posix_only
def test_lmod_unknown_module_unresolved(monkeypatch, tmp_path):
    _fake_lmod(tmp_path, monkeypatch, {})
    job = _job(tmp_path, lmod.Lmod(modules=["nosuch"]))
    assert job.resolved.unresolved
    assert "unknown" in job.context


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
    with Submitter(environment=env, cache_root=tmp_path / "c", telemetry=False) as sub:
        result = sub(shell.define("printenv TOOL_VERSION")())
    assert result.outputs.stdout.strip() == "2.0"
    assert result.context["environment"] == "lmod@1:tool/2.0"
