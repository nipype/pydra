"""A workflow's identity: the run facts of its nodes and the deferred rule (RFC 0004
section 4.3, I5 part 2)."""

import sys
import warnings

import pytest

from pydra.compose import python, shell, workflow
from pydra.engine.job import Job
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter
from pydra.environments import base, docker, images
from pydra.engine.tests.test_portable_store import make
from pydra.engine.tests.test_portable_submitter import drive_with_run
from pydra.engine.tests.test_runner_flatten import count_expansions
from pydra.utils.identity import record_checksum

D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64


class Registry:
    """Stands for the registry ``HEAD`` of a tag, with no local Docker"""

    def __init__(self, monkeypatch, tags=None):
        self.tags = dict(tags or {})  # "repo:tag" -> digest

        def no_docker(cmd, strip=False, **kwargs):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(base, "execute", no_docker)
        monkeypatch.setattr(images, "registry_manifest", self.head)

    def head(self, repository, tag):
        digest = self.tags.get(f"{repository}:{tag}")
        return None if digest is None else (digest, None)


Tool = shell.define("tool <x:int>")
Versioned = shell.define(
    "tool <x:int>",
    version_command=[sys.executable, "-c", "print('tool 1.2.3')"],
    version_pattern=r"tool (\d+\.\d+\.\d+)",
)
NoMatch = shell.define(
    "tool <x:int>",
    version_command=[sys.executable, "-c", "print('no version here')"],
    version_pattern=r"tool (\d+\.\d+\.\d+)",
)


@python.define
def Add1(x: int) -> int:
    return x + 1


@workflow.define
def RunFacts(x: int) -> int:
    workflow.add(
        Tool(x=x), name="dock", environment=docker.Docker(image="org/tool", tag="1")
    )
    workflow.add(Versioned(x=x), name="native")
    add = workflow.add(Add1(x=x), name="add")
    return add.out


@workflow.define
def Inner(x: int) -> int:
    add = workflow.add(Add1(x=x), name="add")
    return add.out


@workflow.define
def LazyOuter(x: int) -> int:
    a = workflow.add(Add1(x=x), name="a")
    inner = workflow.add(Inner(x=a.out), name="inner")
    return inner.out


@workflow.define
def ConcreteOuter(x: int) -> int:
    inner = workflow.add(Inner(x=x), name="inner")
    return inner.out


@workflow.define
def SplitOuter(xs: list[int]) -> list[int]:
    inner = workflow.add(Inner().split(x=xs), name="inner")
    return inner.out


@workflow.define
def Mid(x: int) -> int:
    a = workflow.add(Add1(x=x), name="a")
    inner = workflow.add(Inner(x=a.out), name="inner")
    return inner.out


@workflow.define
def Deep(x: int) -> int:
    mid = workflow.add(Mid(x=x), name="mid")
    return mid.out


@workflow.define
def Bad(x: int) -> int:
    if x > 1:
        raise ValueError(f"cannot build Bad for {x}")
    add = workflow.add(Add1(x=x), name="add")
    return add.out


@workflow.define(outputs=["bad", "ok"])
def KeyingFails(x: int):
    a = workflow.add(Add1(x=x), name="a")
    bad = workflow.add(Bad(x=a.out), name="bad")
    ok = workflow.add(Add1(x=x), name="ok")
    return bad.out, ok.out


@workflow.define
def NoMatchInner(x: int) -> int:
    tool = workflow.add(NoMatch(x=x), name="tool")
    return tool.return_code


@workflow.define(outputs=["bad", "ok"])
def VersionFails(x: int):
    a = workflow.add(Add1(x=x), name="a")
    bad = workflow.add(NoMatchInner(x=a.out), name="bad")
    ok = workflow.add(Add1(x=x), name="ok")
    return bad.out, ok.out


@workflow.define(outputs=["bad", "ok"])
def Root(x: int):
    bad = workflow.add(Bad(x=x), name="bad")  # concrete at the parent's keying
    ok = workflow.add(Add1(x=x), name="ok")
    return bad.out, ok.out


@workflow.define(outputs=["mid", "ok2"])
def Top(x: int):
    a = workflow.add(Add1(x=x), name="a")
    mid = workflow.add(Root(x=a.out), name="mid")
    ok2 = workflow.add(Add1(x=x), name="ok2")
    return mid.ok, ok2.out


@workflow.define(outputs=["tool", "ok"])
def VersionRoot(x: int):
    tool = workflow.add(NoMatch(x=x), name="tool")  # concrete at the parent's keying
    ok = workflow.add(Add1(x=x), name="ok")
    return tool.return_code, ok.out


def _job(task, sub=None):
    return Job(name="main", task=task, submitter=sub or Submitter(store="pickle"))


def _owned_runner(sub, task):
    return GraphRunner(
        task,
        sub.store,
        on_error="raise_at_end",
        submitter=sub,
        job=Job(task, submitter=sub, name="main"),
    )


def test_workflow_nodes_carry_run_facts(monkeypatch):
    Registry(monkeypatch, {"org/tool:1": D1})
    job = _job(RunFacts(x=1))
    nodes = job.record["nodes"]
    # each shell node's job fields, resolved as a shell job resolves them; a python
    # node has none and is left out
    native = _job(Versioned(x=1)).record["job"]
    assert nodes == {
        "dock": {"environment": f"docker@1:{D1}", "tool_version": None},
        "native": native,
    }
    assert native["tool_version"] == "1.2.3"
    assert record_checksum(job.record) == job.checksum
    assert not job.deferred


def test_moved_tag_under_node_rekeys_workflow(monkeypatch):
    registry = Registry(monkeypatch, {"org/tool:1": D1})
    before = _job(RunFacts(x=1))
    before.checksum  # keyed before the move
    registry.tags["org/tool:1"] = D2  # the tag is re-pushed
    after = _job(RunFacts(x=1))  # a new submitter resolves it again
    assert before.task._checksum == after.task._checksum  # same code and inputs
    assert before.checksum != after.checksum
    assert after.record["nodes"]["dock"]["environment"] == f"docker@1:{D2}"


def test_unresolved_node_deferred(monkeypatch):
    Registry(monkeypatch)  # neither local nor in the registry
    job = _job(RunFacts(x=1))
    assert job.record["nodes"]["dock"] == "deferred"
    assert job.deferred


def test_nested_lazy_input_deferred_is_miss(tmp_path, monkeypatch):
    job = _job(LazyOuter(x=1))
    assert job.record["nodes"] == {"inner": "deferred"}
    assert job.deferred
    concrete = _job(ConcreteOuter(x=1))
    assert concrete.record["nodes"] == {"inner": {"nodes": {}}}
    assert not concrete.deferred
    # its entry is written, and a lookup of it is a miss: it is expanded every time
    with Submitter(store="pickle", cache_root=tmp_path) as sub:
        assert sub(LazyOuter(x=1)).outputs.out == 3
        stored = _job(LazyOuter(x=1), sub)
        assert stored.result() is not None
        expansions = count_expansions(monkeypatch)
        assert sub(LazyOuter(x=1)).outputs.out == 3
    assert "LazyOuter" in expansions


def test_split_nested_deferred():
    job = _job(SplitOuter(xs=[1, 2]))
    assert job.record["nodes"] == {"inner": "deferred"}
    assert job.deferred


def test_deferred_at_depth_defers_parent():
    job = _job(Deep(x=1))
    assert job.record["nodes"] == {"mid": {"nodes": {"inner": "deferred"}}}
    assert job.deferred


def test_deferred_workflow_is_miss_under_flattened_runner(tmp_path, monkeypatch):
    with Submitter(worker="debug", store="pickle", cache_root=tmp_path) as sub:
        assert sub(LazyOuter(x=1)).outputs.out == 3
        assert _job(LazyOuter(x=1), sub).result() is not None  # pre-stored
        expansions = count_expansions(monkeypatch)
        runner = _owned_runner(sub, LazyOuter(x=1))
        rounds = drive_with_run(runner)
        assert runner.result().outputs.out == 3
    settled = {(s.job.name, s.how) for r in rounds for s in r.settled}
    assert ("main", "workflow-stored") not in settled
    assert ("a", "stored") in settled  # expanded node by node
    assert expansions[0] == "LazyOuter"


def test_deferred_workflow_is_miss_under_portable_find(tmp_path, monkeypatch):
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        assert sub(LazyOuter(x=1)).outputs.out == 3
    asked = []
    real = store.find

    def find(job, *args, **kwargs):
        asked.append(job.name)
        return real(job, *args, **kwargs)

    monkeypatch.setattr(store, "find", find)
    expansions = count_expansions(monkeypatch)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = _owned_runner(sub, LazyOuter(x=1))
        rounds = drive_with_run(runner)
        assert runner.result().outputs.out == 3
    assert "main" not in asked  # find is not asked to serve the outer job whole
    hows = {s.job.name: s.how for r in rounds for s in r.settled}
    assert "main" not in hows
    assert hows["a"] == "stored"  # unchanged nodes are served from their own entries
    assert expansions[0] == "LazyOuter"


@pytest.mark.parametrize("task", [KeyingFails, VersionFails])
def test_constructor_error_at_keying_is_job_error(tmp_path, task):
    with Submitter(worker="debug", store="pickle", cache_root=tmp_path) as sub:
        runner = GraphRunner(
            task(x=1), sub.store, on_error="raise_at_end", submitter=sub
        )
        drive_with_run(runner)
        ok = runner.exec_graph["ok"].job(None)
        assert ok.result().outputs.out == 2  # the parent's loop continued
    assert runner.failed
    (entry,) = runner.entries
    # the node is deferred at the parent's keying; its own job fails when it is built
    assert (entry.job_name, entry.phase) == ("bad", "run")
    assert ("cannot build Bad" if task is KeyingFails else "no version") in (
        entry.message + entry.traceback
    )


@pytest.mark.parametrize(
    "task, done, message",
    [
        (Root(x=2), [Add1(x=2)], "cannot build Bad"),
        (Top(x=1), [Add1(x=1), Add1(x=2)], "cannot build Bad"),
        (VersionRoot(x=1), [Add1(x=1)], "no version here"),
    ],
    ids=["root", "nested", "version"],
)
def test_keying_error_in_a_concrete_node_stays_on_that_node(
    tmp_path, task, done, message
):
    with Submitter(store="pickle", cache_root=tmp_path) as sub:
        result = sub(task, raise_errors=False)  # the call does not raise
        assert result.errored
        for ok in done:  # the siblings ran
            assert _job(ok, sub).result().outputs.out == ok.x + 1
        if message == "cannot build Bad":  # the error is on Bad's own result
            bad = _job(Bad(x=2), sub).result()
            assert bad.errored
            assert message in "".join(bad.errors["error message"])
    assert message in str(result.errors)


def test_version_error_at_keying_is_the_shell_nodes(tmp_path):
    with Submitter(worker="debug", store="pickle", cache_root=tmp_path) as sub:
        runner = GraphRunner(
            VersionRoot(x=1), sub.store, on_error="raise_at_end", submitter=sub
        )
        drive_with_run(runner)
        assert runner.exec_graph["ok"].job(None).result().outputs.out == 2
    (entry,) = runner.entries
    assert (entry.job_name, entry.phase) == ("tool", "keying")
    assert "no version here" in entry.message


@pytest.mark.parametrize("kind", ["pickle", "portable"])
def test_deferred_nested_workflow_is_not_served_whole(tmp_path, monkeypatch, kind):
    store = "pickle" if kind == "pickle" else make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "c") as sub:
        assert sub(Deep(x=1)).outputs.out == 3
    asked = []
    if kind == "portable":
        real = store.find

        def find(job, *args, **kwargs):
            asked.append(job.name)
            return real(job, *args, **kwargs)

        monkeypatch.setattr(store, "find", find)
    expansions = count_expansions(monkeypatch)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "c") as sub:
        runner = _owned_runner(sub, Deep(x=1))
        rounds = drive_with_run(runner)
        assert runner.result().outputs.out == 3
    settled = {(s.job.name, s.how) for r in rounds for s in r.settled}
    assert ("mid", "workflow-stored") not in settled
    assert "mid" not in asked
    assert expansions[:2] == ["Deep", "Mid"]
    assert ("inner", "workflow-stored") in settled  # concrete at run: served whole
    assert ("a", "stored") in settled


def test_deferred_workflow_is_miss_under_job_run(tmp_path, monkeypatch):
    with Submitter(store="pickle", cache_root=tmp_path) as sub:
        Job(LazyOuter(x=1), submitter=sub, name="main").run()
        expansions = count_expansions(monkeypatch)
        result = Job(LazyOuter(x=1), submitter=sub, name="main").run()
    assert result.outputs.out == 3
    assert expansions == ["LazyOuter"]  # looked up under its lease: a miss


def _versioned_pipeline(edited: bool):
    """Two constructors of one name and version, whose code differs"""
    if edited:

        def Pipeline(x: int) -> int:
            workflow.add(
                Tool(x=x + 5),
                name="dock",
                environment=docker.Docker(image="org/tool", tag="1"),
            )
            return workflow.add(Add1(x=x), name="add").out

    else:

        def Pipeline(x: int) -> int:
            workflow.add(
                Tool(x=x),
                name="dock",
                environment=docker.Docker(image="org/tool", tag="1"),
            )
            return workflow.add(Add1(x=x), name="add").out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return workflow.define(Pipeline, version="1")


def test_workflow_version_replaces_constructor_code_not_nodes(monkeypatch):
    registry = Registry(monkeypatch, {"org/tool:1": D1})
    one = _job(_versioned_pipeline(edited=False)(x=1))
    edited = _job(_versioned_pipeline(edited=True)(x=1))
    one.checksum, edited.checksum
    # the declared version stands for the constructor's code
    assert one.record["task"]["code"] == edited.record["task"]["code"]
    assert one.checksum == edited.checksum
    # it does not stand for the run facts of the nodes
    registry.tags["org/tool:1"] = D2
    retagged = _job(_versioned_pipeline(edited=False)(x=1))
    assert one.record["task"] == retagged.record["task"]
    assert one.record["nodes"] != retagged.record["nodes"]
    assert one.checksum != retagged.checksum
    # without a version the constructor's code is keyed
    assert _job(Inner(x=1)).record["task"]["code"] != (
        _job(ConcreteOuter(x=1)).record["task"]["code"]
    )

    def Local(x: int) -> int:
        return workflow.add(Add1(x=x)).out

    with pytest.warns(UserWarning, match="version='1'"):
        workflow.define(Local, version="1")


def test_a_record_whose_construction_failed_is_deferred():
    from pydra.utils.identity import is_deferred

    assert is_deferred({"nodes": "deferred"})
    assert is_deferred({"nodes": {"inner": {"nodes": "deferred"}}})
    assert not is_deferred({"nodes": {"inner": {"nodes": {}}}})
