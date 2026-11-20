import json
import pytest
from pydra.compose import shell
from pydra.engine.job import Job
from pydra.engine.submitter import Submitter
from pydra.environments import base, docker
from pydra.environments.base import ImageResolutionError, ResolvedRef

D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
ID = "sha256:" + "a" * 64


class FakeDocker:
    """Stands for ``docker image inspect`` and the registry ``HEAD``"""

    def __init__(self, monkeypatch, local=None, registry=None, docker_missing=False):
        self.local = local or {}  # "repo:tag" -> (Id, [RepoDigests])
        self.registry = registry or {}  # "repo:tag" -> digest
        self.docker_missing = docker_missing
        self.inspected = []
        self.heads = []
        monkeypatch.setattr(base, "execute", self.execute)
        monkeypatch.setattr(docker, "_registry_digest", self.head)

    def execute(self, cmd, strip=False, **kwargs):
        if self.docker_missing:
            raise FileNotFoundError("docker")
        assert cmd[:3] == ["docker", "image", "inspect"]
        ref = cmd[-1]
        self.inspected.append(ref)
        if ref not in self.local:
            return 1, "", "No such image"
        image_id, repo_digests = self.local[ref]
        return 0, f"{image_id}\t{json.dumps(repo_digests)}", ""

    def head(self, repository, tag):
        self.heads.append((repository, tag))
        return self.registry.get(f"{repository}:{tag}")


def test_docker_given_digest_no_lookup(monkeypatch):
    fake = FakeDocker(monkeypatch)
    ref = docker.Docker(image=f"org/tool@{D1}").resolve()
    assert ref == ResolvedRef(
        key=f"docker@1:{D1}", run=f"org/tool@{D1}", carried={"repository": "org/tool"}
    )
    assert fake.inspected == [] and fake.heads == []
    assert (
        docker.Docker(image="org/tool", identity=D2).resolve().key == f"docker@1:{D2}"
    )
    assert fake.inspected == [] and fake.heads == []


def test_docker_repodigests_by_repository(monkeypatch):
    fake = FakeDocker(
        monkeypatch,
        local={"org/tool:1": (ID, [f"other/tool@{D2}", f"org/tool@{D1}"])},
    )
    ref = docker.Docker(image="org/tool", tag="1").resolve()
    assert ref.key == f"docker@1:{D1}"
    assert ref.run == f"org/tool@{D1}"
    assert ref.carried == {"repository": "org/tool"}
    assert fake.heads == []


def test_docker_local_build_keyed_by_id(monkeypatch):
    fake = FakeDocker(monkeypatch, local={"tool:latest": (ID, [])})
    ref = docker.Docker(image="tool").resolve()
    assert (ref.key, ref.run) == (f"docker-local@1:{ID}", ID)
    assert fake.heads == []


def test_docker_retagged_image_keyed_by_id(monkeypatch):
    fake = FakeDocker(
        monkeypatch, local={"tool:latest": (ID, [f"ghcr.io/org/tool@{D1}"])}
    )
    ref = docker.Docker(image="tool").resolve()
    assert (ref.key, ref.run) == (f"docker-local@1:{ID}", ID)
    assert fake.heads == []


def test_docker_several_entries_keyed_by_id(monkeypatch):
    FakeDocker(
        monkeypatch,
        local={"org/tool:latest": (ID, [f"org/tool@{D1}", f"org/tool@{D2}"])},
    )
    ref = docker.Docker(image="org/tool").resolve()
    assert (ref.key, ref.run) == (f"docker-local@1:{ID}", ID)


def test_docker_registry_head_once_per_tag(monkeypatch, tmp_path):
    fake = FakeDocker(monkeypatch, registry={"org/tool:1": D1})
    env = docker.Docker(image="org/tool", tag="1")
    sub = Submitter(environment=env, cache_root=tmp_path, telemetry=False)
    refs = []
    for name in "abc":
        job = Job(task=shell.define("whoami")(), name=name, submitter=sub)
        refs.append(job.resolved)
    assert fake.heads == [("org/tool", "1")]
    assert {r.key for r in refs} == {f"docker@1:{D1}"}
    assert refs[0].run == f"org/tool@{D1}"


def test_docker_offline_unresolved(monkeypatch, tmp_path):
    FakeDocker(monkeypatch, docker_missing=True)
    env = docker.Docker(image="org/tool", tag="1")
    ref = env.resolve()
    assert ref.unresolved
    assert ref.run == "org/tool:1"
    assert "docker@1:" not in ref.key and "org/tool:1" in ref.key
    sub = Submitter(environment=env, cache_root=tmp_path, telemetry=False)
    job = Job(task=shell.define("whoami")(), name="j", submitter=sub)
    assert job.record["job"]["environment"] == job.resolved.key
    # the result is used by this process only
    assert "unknown" in job.context


def test_resolve_images_require_raises(monkeypatch, tmp_path):
    FakeDocker(monkeypatch, docker_missing=True)
    sub = Submitter(
        environment=docker.Docker(image="org/tool"),
        cache_root=tmp_path,
        resolve_images="require",
        telemetry=False,
    )
    job = Job(task=shell.define("whoami")(), name="j", submitter=sub)
    with pytest.raises(ImageResolutionError, match="org/tool"):
        job.checksum


def test_resolve_images_value_checked(tmp_path):
    with pytest.raises(ValueError, match="resolve_images"):
        Submitter(cache_root=tmp_path, resolve_images="always", telemetry=False)


def test_mirror_shares_key(monkeypatch):
    FakeDocker(monkeypatch)
    a = docker.Docker(image=f"org/fsl@{D1}").resolve()
    b = docker.Docker(image=f"mirror.site/org/fsl@{D1}").resolve()
    assert a.key == b.key
    assert a.run != b.run
    assert b.carried == {"repository": "mirror.site/org/fsl"}


def test_resolved_job_record_and_context_use_key(monkeypatch, tmp_path):
    FakeDocker(monkeypatch)
    sub = Submitter(
        environment=docker.Docker(image=f"org/fsl@{D1}"),
        cache_root=tmp_path,
        telemetry=False,
    )
    job = Job(task=shell.define("whoami")(), name="j", submitter=sub)
    assert job.record["job"]["environment"] == f"docker@1:{D1}"
    assert job.context["environment"] == f"docker@1:{D1}"
    assert "unknown" not in job.context


def test_execute_no_tag_suffix_on_resolved(monkeypatch, tmp_path):
    fake = FakeDocker(monkeypatch)
    commands = []

    def run(cmd, cwd=None, strip=False):
        commands.append(cmd)
        return 0, "", ""

    sub = Submitter(
        environment=docker.Docker(image=f"org/tool@{D1}"),
        cache_root=tmp_path,
        telemetry=False,
    )
    job = Job(task=shell.define("whoami")(), name="j", submitter=sub)
    job.resolved  # the submitting process resolves
    monkeypatch.setattr(base, "execute", run)
    job.environment.execute(job)
    assert f"org/tool@{D1}" in commands[0]
    assert not any(a.endswith(":latest") for a in commands[0])
    assert fake.heads == []


def test_execute_runs_local_image_by_id(monkeypatch, tmp_path):
    FakeDocker(monkeypatch, local={"tool:latest": (ID, [])})
    commands = []
    sub = Submitter(
        environment=docker.Docker(image="tool"), cache_root=tmp_path, telemetry=False
    )
    job = Job(task=shell.define("whoami")(), name="j", submitter=sub)
    job.resolved
    monkeypatch.setattr(
        base,
        "execute",
        lambda cmd, cwd=None, strip=False: commands.append(cmd) or (0, "", ""),
    )
    job.environment.execute(job)
    assert ID in commands[0] and "tool:latest" not in commands[0]


@pytest.mark.docker
def test_docker_real_daemon(tmp_path):
    env = docker.Docker(image="busybox")
    ref = env.resolve()
    assert not ref.unresolved
    assert ref.key.startswith(("docker@1:sha256:", "docker-local@1:sha256:"))
    with Submitter(environment=env, cache_root=tmp_path, telemetry=False) as sub:
        result = sub(shell.define("echo")(executable="echo", append_args=["hi"]))
    assert result.outputs.stdout.strip() == "hi"
