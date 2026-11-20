"""An external driver that completes jobs with opaque references in place of output values
(RFC 0002 section 3.5): the runner passes them to readers as they are, and reads values only
where it must, as a splitter's sequence"""

import typing as ty

import attrs
import pytest

from pydra.compose import python, workflow
from pydra.engine.result import Result
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter


@attrs.frozen
class Ref:
    """A reference to an output held by the driver"""

    key: str


@python.define
def Step(a: ty.Any, b: ty.Any) -> ty.Any:
    raise AssertionError("the driver runs the jobs")


@python.define
def Source(n: int) -> ty.Any:
    raise AssertionError("the driver runs the jobs")


@python.define
def Gather(items: ty.Any) -> ty.Any:
    raise AssertionError("the driver runs the jobs")


@workflow.define
def RefChain(x: int) -> ty.Any:
    a = workflow.add(Step(a=x, b=1), name="A")
    b = workflow.add(Step(a=a.out, b=2), name="B")
    return b.out


@workflow.define
def RefSplit(n: int) -> ty.Any:
    src = workflow.add(Source(n=n), name="src")
    each = workflow.add(Step(b=1).split(a=src.out), name="each")
    return each.out


@workflow.define
def RefGather(xs: list[int]) -> ty.Any:
    each = workflow.add(Step(b=1).split(a=xs).combine("a"), name="each")
    g = workflow.add(Gather(items=each.out), name="G")
    return g.out


def ref_of(job) -> Ref:
    index = "" if job.state_index is None else f"[{job.state_index}]"
    return Ref(f"{job.name}{index}")


def complete_with_refs(job, source_value=None) -> Result:
    value = source_value if job.name == "src" and source_value is not None else None
    return Result(
        cache_dir=job.cache_dir,
        outputs=job.task.Outputs(out=value if value is not None else ref_of(job)),
        task=job.task,
    )


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        yield sub


def drive_refs(runner, source_value=None, rounds=50):
    sent = []
    for _ in range(rounds):
        if runner.done:
            return sent
        rnd = runner.ready()
        for job in rnd.dispatch:
            sent.append((job, runner.upstream(job)))
            runner.complete(job, complete_with_refs(job, source_value))
    raise AssertionError(runner.stuck())


def runner_for(submitter, task):
    return GraphRunner(
        task,
        submitter.store,
        on_error="raise_at_end",
        submitter=submitter,
        lookup_before_dispatch=True,
    )


def test_reference_outputs_flow_to_readers(submitter):
    runner = runner_for(submitter, RefChain(x=1))
    sent = drive_refs(runner)
    (a, _), (b, upstream) = sent
    assert b.task.a == Ref("A")  # the reference, as the driver gave it
    assert upstream == {"a": [(a, None)]}
    assert runner.done and not runner.failed


def test_real_list_source_drives_downstream_split(submitter):
    runner = runner_for(submitter, RefSplit(n=3))
    refs = [Ref("r0"), Ref("r1"), Ref("r2")]
    sent = drive_refs(runner, source_value=refs)
    src = sent[0][0]
    each = [(job, up) for job, up in sent if job.name == "each"]
    assert [job.task.a for job, _ in each] == refs
    assert [up for _, up in each] == [{"a": [(src, None)]}] * 3
    assert runner.done and not runner.failed


def test_upstream_combined_read_index_none(submitter):
    runner = runner_for(submitter, RefGather(xs=[1, 2, 3]))
    sent = drive_refs(runner)
    each = [job for job, _ in sent if job.name == "each"]
    g, upstream = sent[-1]
    assert g.name == "G"
    assert upstream == {"items": [(job, None) for job in each]}
    assert g.task.items == [Ref("each[0]"), Ref("each[1]"), Ref("each[2]")]


def test_reference_in_splitter_position_is_clear_error(submitter):
    runner = runner_for(submitter, RefSplit(n=3))
    (src,) = runner.ready().dispatch
    runner.complete(src, complete_with_refs(src))  # a reference, not a sequence
    with pytest.raises(TypeError, match=r"'each' splits over 'a'.*real sequence"):
        runner.ready()
