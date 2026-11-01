import pytest
from pydra.environments import native, docker, singularity, lmod


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


def test_environment_checksum_not_in_job_checksum(tmp_path):
    from pydra.compose import shell
    from pydra.engine.submitter import Submitter
    from pydra.engine.job import Job

    Shelly = shell.define("whoami")
    sums = {
        Job(
            task=Shelly(),
            name="j",
            submitter=Submitter(environment=env, cache_root=tmp_path),
        ).checksum
        for env in (
            native.Native(),
            docker.Docker(image="busybox", identity="x"),
            docker.Docker(image="busybox", identity="y"),
        )
    }
    assert len(sums) == 1
