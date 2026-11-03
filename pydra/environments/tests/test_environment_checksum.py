import sys
import attrs
import cloudpickle
import pytest
from pydra.environments import native, docker, singularity, lmod

# Worker processes cannot import the test modules, so they receive their classes by value
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def test_checksum_is_stable_hex():
    assert native.Native().checksum() == native.Native().checksum()
    assert len(native.Native().checksum()) == 64


def test_checksum_distinguishes_environments():
    assert lmod.Lmod(modules=["a"]).checksum() != lmod.Lmod(modules=["b"]).checksum()
    assert native.Native().checksum() != lmod.Lmod(modules=["a"]).checksum()


@pytest.mark.parametrize("env", [docker.Docker, singularity.Singularity])
def test_identity_replaces_image_resolution(env, monkeypatch):
    def fail(self):
        raise AssertionError("must not resolve a digest when identity is given")

    monkeypatch.setattr(env, "_resolve_digest", fail)
    a = env(image="busybox", identity="sha256:1")
    assert a.checksum() == env(image="other", identity="sha256:1").checksum()
    assert a.checksum() != env(image="busybox", identity="sha256:2").checksum()


@pytest.mark.parametrize("env", [docker.Docker, singularity.Singularity])
def test_rebuilt_image_changes_checksum_via_digest(env, monkeypatch):
    digest = ["sha256:a"]
    monkeypatch.setattr(env, "_resolve_digest", lambda self: digest[0])
    before = env(image="busybox").checksum()
    digest[0] = "sha256:b"
    assert env(image="busybox").checksum() != before


@attrs.define
class Tagged(native.Native):
    """A runnable environment whose identity is set by ``tag``."""

    tag: str = ""


def _run(tmp_path, env, worker="debug"):
    from pydra.compose import shell
    from pydra.engine.submitter import Submitter
    from pydra.engine.job import Job

    job = Job(
        task=shell.define("whoami")(),
        name="j",
        submitter=Submitter(worker=worker, environment=env, cache_root=tmp_path),
    )
    job.run()
    return job


def test_environment_checksum_recorded_not_in_job_checksum(tmp_path):
    a = _run(tmp_path / "a", Tagged(tag="a"))
    b = _run(tmp_path / "b", Tagged(tag="b"))
    assert a.checksum == b.checksum
    assert a.result().environment_checksum == Tagged(tag="a").checksum()
    assert b.result().environment_checksum == Tagged(tag="b").checksum()
    assert a.result().environment_checksum != b.result().environment_checksum


def test_environment_checksum_survives_reload(tmp_path):
    job = _run(tmp_path, Tagged(tag="a"))
    stored = job.store.lookup(job.checksum, job.context, job.task)
    assert stored.environment_checksum == Tagged(tag="a").checksum()


def test_default_environment_records_native_checksum(tmp_path):
    assert (
        _run(tmp_path, None).result().environment_checksum == native.Native().checksum()
    )


def test_digest_resolved_once_per_process(monkeypatch):
    calls = []

    def resolve(self):
        calls.append(1)
        return "sha256:a"

    monkeypatch.setattr(docker.Docker, "_resolve_digest", resolve)
    env = docker.Docker(image="busybox")
    assert env.checksum() == env.checksum()
    assert len(calls) == 1


def test_unresolved_digest_is_not_cached(monkeypatch):
    digests = [None, "sha256:a"]
    monkeypatch.setattr(docker.Docker, "_resolve_digest", lambda self: digests.pop(0))
    env = docker.Docker(image="busybox")
    assert env.checksum() != env.checksum()


def test_docker_digest_never_pulls_and_tolerates_no_docker(monkeypatch):
    from pydra.environments import base

    commands = []

    def missing(cmd, strip=False):
        commands.append(cmd)
        return 1, "", "no such image"

    monkeypatch.setattr(base, "execute", missing)
    assert docker.Docker(image="busybox")._resolve_digest() is None
    assert [c[1] for c in commands] == ["image"]

    def no_docker(cmd, strip=False):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(base, "execute", no_docker)
    env = docker.Docker(image="busybox")
    assert env.checksum() == env.checksum()


def test_environment_checksum_recorded_under_each_worker(worker, tmp_path):
    """The environment identity is recorded whichever process runs the job"""
    from pydra.compose import shell
    from pydra.engine.submitter import Submitter

    with Submitter(
        worker=worker, environment=Tagged(tag="a"), cache_root=tmp_path
    ) as sub:
        result = sub(shell.define("whoami")())
    assert result.environment_checksum == Tagged(tag="a").checksum()
