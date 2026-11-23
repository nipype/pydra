import json
import cloudpickle
import pytest
from pydra.compose import shell
from pydra.engine.job import Job
from pydra.engine.submitter import Submitter
from pydra.environments import base, docker, images
from pydra.engine.tests.utils import need_docker, no_win
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
        monkeypatch.setattr(images, "registry_manifest", self.head)

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
        digest = self.registry.get(f"{repository}:{tag}")
        return None if digest is None else (digest, None)


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
    sub = Submitter(
        store="pickle", environment=env, cache_root=tmp_path, telemetry=False
    )
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
    sub = Submitter(
        store="pickle", environment=env, cache_root=tmp_path, telemetry=False
    )
    job = Job(task=shell.define("whoami")(), name="j", submitter=sub)
    assert job.record["job"]["environment"] == job.resolved.key
    # the result is used by this process only
    assert "unknown" in job.context


def test_resolve_images_require_raises(monkeypatch, tmp_path):
    FakeDocker(monkeypatch, docker_missing=True)
    sub = Submitter(
        store="pickle",
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
        Submitter(
            store="pickle",
            cache_root=tmp_path,
            resolve_images="always",
            telemetry=False,
        )


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
        store="pickle",
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
        store="pickle",
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
        store="pickle",
        environment=docker.Docker(image="tool"),
        cache_root=tmp_path,
        telemetry=False,
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


@no_win
@need_docker
@pytest.mark.docker
def test_docker_real_daemon(tmp_path):
    env = docker.Docker(image="busybox")
    ref = env.resolve()
    assert not ref.unresolved
    assert ref.key.startswith(("docker@1:sha256:", "docker-local@1:sha256:"))
    with Submitter(
        store="pickle", environment=env, cache_root=tmp_path, telemetry=False
    ) as sub:
        result = sub(shell.define("echo")(executable="echo", append_args=["hi"]))
    assert result.outputs.stdout.strip() == "hi"


def _job(tmp_path, env, supplied=False):
    task = shell.define("whoami")
    if supplied:
        task.__checksum__ = lambda self, inputs: "shell-fixed"
    sub = Submitter(
        store="pickle", environment=env, cache_root=tmp_path, telemetry=False
    )
    return Job(task=task(), name="j", submitter=sub)


def _ran(monkeypatch):
    commands = []
    monkeypatch.setattr(
        base,
        "execute",
        lambda cmd, cwd=None, strip=False: commands.append(cmd) or (0, "", ""),
    )
    return commands


def test_pickled_job_runs_the_original_reference(monkeypatch, tmp_path):
    fake = FakeDocker(monkeypatch, local={"tool:latest": (ID, [])})
    job = _job(tmp_path, docker.Docker(image="tool"), supplied=True)
    blob = cloudpickle.dumps(job)
    rebuilt = "sha256:" + "b" * 64
    fake.local = {"tool:latest": (rebuilt, [])}
    fake.inspected.clear()
    worker_job = cloudpickle.loads(blob)
    commands = _ran(monkeypatch)
    worker_job.environment.execute(worker_job)
    assert ID in commands[0] and rebuilt not in commands[0]
    assert fake.inspected == []


def test_worker_without_carried_reference_raises(monkeypatch, tmp_path):
    fake = FakeDocker(monkeypatch, local={"tool:latest": (ID, [])})
    job = _job(tmp_path, docker.Docker(image="tool"))
    state = job.__getstate__()
    state["_resolved"] = None
    worker_job = Job.__new__(Job)
    worker_job.__setstate__(state)
    fake.inspected.clear()
    commands = _ran(monkeypatch)
    with pytest.raises(ImageResolutionError, match="without the reference"):
        worker_job.environment.execute(worker_job)
    assert fake.inspected == [] and commands == []


def test_five_jobs_one_inspect(monkeypatch, tmp_path):
    fake = FakeDocker(monkeypatch, local={"tool:latest": (ID, [])})
    sub = Submitter(
        store="pickle",
        environment=docker.Docker(image="tool"),
        cache_root=tmp_path,
        telemetry=False,
    )
    for name in "abcde":
        Job(task=shell.define("whoami")(), name=name, submitter=sub).resolved
    assert fake.inspected == ["tool:latest"]


def test_checksum_never_reaches_docker_or_registry(monkeypatch):
    fake = FakeDocker(monkeypatch, registry={"org/tool:1": D1})
    env = docker.Docker(image="org/tool", tag="1")
    assert env.checksum() == env.checksum()
    assert fake.inspected == [] and fake.heads == []


def test_repository_is_recorded_only_in_the_context(monkeypatch, tmp_path):
    from pydra.engine.portable.identity import PydraIdentityV2
    from pydra.engine.store import ConsistentReuse, context_key

    FakeDocker(monkeypatch)
    contexts = []
    for image in (f"org/fsl@{D1}", f"mirror.site/org/fsl@{D1}"):
        job = _job(tmp_path, docker.Docker(image=image))
        contexts.append(job.context)
    assert contexts[0]["environment_repository"] == "org/fsl"
    assert contexts[1]["environment_repository"] == "mirror.site/org/fsl"
    assert "environment_repository" in PydraIdentityV2().recorded_only
    ident = PydraIdentityV2()
    assert ident.fingerprint(contexts[0]) == ident.fingerprint(contexts[1])
    assert context_key(contexts[0]) == context_key(contexts[1])
    assert ConsistentReuse().accept(contexts[0], contexts[1], None)


def test_hub_names_are_normalised(monkeypatch):
    FakeDocker(
        monkeypatch,
        local={"library/busybox:latest": (ID, [f"busybox@{D1}"])},
    )
    ref = docker.Docker(image="library/busybox").resolve()
    assert ref.key == f"docker@1:{D1}"
    FakeDocker(
        monkeypatch, local={"docker.io/org/tool:latest": (ID, [f"org/tool@{D2}"])}
    )
    assert docker.Docker(image="docker.io/org/tool").resolve().key == f"docker@1:{D2}"


def test_rebuilt_local_image_gets_a_new_key(monkeypatch):
    fake = FakeDocker(monkeypatch, local={"tool:latest": (ID, [])})
    before = docker.Docker(image="tool").resolve().key
    fake.local = {"tool:latest": ("sha256:" + "b" * 64, [])}
    assert docker.Docker(image="tool").resolve().key != before


def test_replacing_the_environment_resolves_again(monkeypatch, tmp_path):
    FakeDocker(
        monkeypatch,
        local={"a:latest": (ID, []), "b:latest": ("sha256:" + "c" * 64, [])},
    )
    job = _job(tmp_path, docker.Docker(image="a"))
    first = job.resolved.key
    job.environment = docker.Docker(image="b")
    assert job.resolved.key != first


def test_portable_identity_context_keys_the_resolved_reference(monkeypatch, tmp_path):
    from pydra.engine.portable.identity import PydraIdentityV2

    FakeDocker(monkeypatch)
    job = _job(tmp_path, docker.Docker(image=f"org/fsl@{D1}"))
    ctx = PydraIdentityV2().context(job, "lookup")
    assert ctx["environment"] == f"docker@1:{D1}"


def test_malformed_inspect_output_is_an_image_error(monkeypatch):
    monkeypatch.setattr(
        base, "execute", lambda cmd, strip=False, **kw: (0, "sha256:x\tnot-json", "")
    )
    with pytest.raises(ImageResolutionError):
        docker.Docker(image="tool").resolve()


def test_registry_errors_leave_the_image_unresolved(monkeypatch):
    import http.client
    import urllib.request

    def broken(*args, **kwargs):
        raise http.client.LineTooLong("header")

    monkeypatch.setattr(urllib.request, "urlopen", broken)
    assert images.registry_manifest("org/tool", "1") is None


def test_environments_differing_only_in_tag_resolve_apart(monkeypatch, tmp_path):
    fake = FakeDocker(
        monkeypatch,
        local={"tool:1": (ID, []), "tool:2": ("sha256:" + "b" * 64, [])},
    )
    sub = Submitter(store="pickle", cache_root=tmp_path, telemetry=False)
    refs = []
    for tag in ("1", "2"):
        job = Job(
            task=shell.define("whoami")(),
            name=tag,
            submitter=sub,
            environment=docker.Docker(image="tool", tag=tag),
        )
        refs.append(job.resolved)
    assert refs[0].key != refs[1].key and refs[0].run != refs[1].run
    assert sorted(fake.inspected) == ["tool:1", "tool:2"]


def test_environments_differing_only_in_identity_resolve_apart(monkeypatch, tmp_path):
    FakeDocker(monkeypatch)
    sub = Submitter(store="pickle", cache_root=tmp_path, telemetry=False)
    refs = []
    for identity in (D1, D2):
        job = Job(
            task=shell.define("whoami")(),
            name="j",
            submitter=sub,
            environment=docker.Docker(image="tool", identity=identity),
        )
        refs.append(job.resolved)
    assert [r.key for r in refs] == [f"docker@1:{D1}", f"docker@1:{D2}"]


def test_execute_on_a_job_of_another_environment_raises(monkeypatch, tmp_path):
    FakeDocker(monkeypatch)
    job = _job(tmp_path, None)
    commands = _ran(monkeypatch)
    with pytest.raises(ImageResolutionError, match="environment"):
        docker.Docker(image="tool").execute(job)
    assert commands == []


def test_python_task_resolves_no_image(monkeypatch, tmp_path):
    from pydra.compose import python

    FakeDocker(monkeypatch, docker_missing=True)
    sub = Submitter(
        store="pickle",
        environment=docker.Docker(image="tool"),
        cache_root=tmp_path,
        resolve_images="require",
        telemetry=False,
    )

    @python.define
    def Plus(a: int) -> int:
        return a + 1

    job = Job(
        task=Plus(a=1), name="j", submitter=sub, environment=docker.Docker(image="tool")
    )
    job.__getstate__()


def test_docker_records_local_id_and_platform_digest(monkeypatch, tmp_path):
    FakeDocker(monkeypatch, local={"org/tool:1": (ID, [f"org/tool@{D1}"])})
    job = _job(tmp_path, docker.Docker(image="org/tool", tag="1"))
    assert job.context["environment_facts"] == {"image_id": ID}
    assert job.record["job"]["environment"] == f"docker@1:{D1}"
    FakeDocker(monkeypatch)
    monkeypatch.setattr(images, "registry_manifest", lambda repo, tag: (D1, D2))
    ref = docker.Docker(image="org/tool", tag="2").resolve()
    assert ref.key == f"docker@1:{D1}" and ref.recorded == {"platform_digest": D2}


class _Response:
    def __init__(self, headers, body=b""):
        self.headers = headers
        self.body = body

    def read(self, *args):
        body, self.body = self.body, b""
        return body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_registry_manifest_is_one_head(monkeypatch):
    import urllib.request

    requests = []

    def head(media_type, digest):
        def urlopen(request, timeout=None):
            requests.append(request.get_method())
            return _Response(
                {"Docker-Content-Digest": digest, "Content-Type": media_type}
            )

        return urlopen

    # an index: its platform digest is not read, since a GET counts as a pull
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        head("application/vnd.oci.image.index.v1+json", ID),
    )
    assert images.registry_manifest("ghcr.io/org/tool", "1") == (ID, None)
    assert requests == ["HEAD"]
    # a single manifest is its own platform manifest
    requests.clear()
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        head("application/vnd.oci.image.manifest.v1+json", D1),
    )
    assert images.registry_manifest("ghcr.io/org/tool", "1") == (D1, D1)
    assert requests == ["HEAD"]
