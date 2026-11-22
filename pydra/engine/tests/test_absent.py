"""Typed absent values (RFC 0002 section 2.3(a), R10a): ``Absent``, the ``absent=``
field attribute, propagation at release and the absence suffix of a checksum that may not
see an accepted or dropped ``Absent``
"""

import pytest

from pydra.compose import python, shell, workflow
from pydra.engine.absent import Absent, absence_suffix, absence_triples
from pydra.engine.result import Result
from pydra.engine.runner import GraphRunner
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.utils.hash import hash_function

# -- tasks -------------------------------------------------------------------------------------------


@python.define
def Maybe(x: int) -> int:
    """``x``, or an ``Absent`` when it is negative"""
    if x < 0:
        return Absent(cause="negative", source="Maybe")
    return x


@python.define
def MaybeList(n: int) -> list[int]:
    if n < 0:
        return Absent(cause="no-list", source="MaybeList")
    return list(range(n))


@python.define
def Inc(a: int) -> int:
    return a + 1


@python.define(inputs={"a": python.arg(type=int, absent="accept")})
def Describe(a) -> str:
    return f"absent:{a.cause}" if isinstance(a, Absent) else f"value:{a}"


@python.define
def Gather(vals: list[int]) -> list[str]:
    return [f"absent:{v.cause}" if isinstance(v, Absent) else str(v) for v in vals]


@python.define(inputs={"vals": python.arg(type=list[int], absent="drop")})
def GatherDrop(vals) -> list[int]:
    return list(vals)


def by_length(task, inputs):
    """A supplied checksum built from how many values a gather reads, not from them"""
    return f"gather-n{len(inputs['vals'])}"


@python.define(checksum=by_length)
def KeyedGather(vals: list[int]) -> list[str]:
    return [f"absent:{v.cause}" if isinstance(v, Absent) else str(v) for v in vals]


@python.define(
    checksum=by_length, inputs={"vals": python.arg(type=list[int], absent="drop")}
)
def KeyedGatherDrop(vals) -> list[int]:
    return list(vals)


@python.define
def Fail(a: int) -> int:
    raise ValueError("deliberate")


# -- workflows ---------------------------------------------------------------------------------------


@workflow.define
def Chain(x: int) -> int:
    a = workflow.add(Maybe(x=x), name="A")
    b = workflow.add(Inc(a=a.out), name="B")
    return b.out


@workflow.define(outputs=["gathered", "xs"])
def Aligned(xs: list[int]):
    a = workflow.add(Maybe().split(x=xs).combine("x"), name="A")
    g = workflow.add(Gather(vals=a.out), name="G")
    return g.out, xs


@workflow.define
def Dropped(xs: list[int]) -> list[int]:
    a = workflow.add(Maybe().split(x=xs).combine("x"), name="A")
    g = workflow.add(GatherDrop(vals=a.out), name="G")
    return g.out


@workflow.define
def Keyed(xs: list[int]) -> list[str]:
    a = workflow.add(Maybe().split(x=xs).combine("x"), name="A")
    g = workflow.add(KeyedGather(vals=a.out), name="G")
    return g.out


@workflow.define
def KeyedDrop(xs: list[int]) -> list[int]:
    a = workflow.add(Maybe().split(x=xs).combine("x"), name="A")
    g = workflow.add(KeyedGatherDrop(vals=a.out), name="G")
    return g.out


@workflow.define
def Passthrough(x: int) -> int:
    a = workflow.add(Maybe(x=x), name="A")
    return a.out


@workflow.define
def SplitOverAbsent(n: int) -> list[int]:
    a = workflow.add(MaybeList(n=n), name="A")
    b = workflow.add(Inc().split(a=a.out), name="B")
    return b.out


@workflow.define(outputs=["absent", "failed"])
def AbsentAndFailing(x: int):
    a = workflow.add(Maybe(x=x), name="A")
    b = workflow.add(Inc(a=a.out), name="B")
    f = workflow.add(Fail(a=1), name="F")
    return b.out, f.out


# -- helpers -----------------------------------------------------------------------------------------


class CountingStore(PickleStore):
    """A PickleStore that records the checksum of every lookup, record read and commit"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[str, str]] = []

    def lookup(self, checksum, context, *args, **kwargs):
        self.calls.append(("lookup", checksum))
        return super().lookup(checksum, context, *args, **kwargs)

    def read_succeeded(self, checksum, *args, **kwargs):
        self.calls.append(("read_succeeded", checksum))
        return super().read_succeeded(checksum, *args, **kwargs)

    def commit(self, lease, result, job=None):
        self.calls.append(("commit", job.checksum if job is not None else None))
        return super().commit(lease, result, job=job)


def run_inline(job) -> Result:
    """The outcome of a python job, computed by the test without the store"""
    task = job.task
    inputs = {
        f.name: getattr(task, f.name)
        for f in type(task).__attrs_attrs__
        if f.name != "function" and not f.name.startswith("_")
    }
    out = task.function(**inputs)
    return Result(cache_dir=job.cache_dir, outputs=task.Outputs(out=out), task=task)


def drive(runner):
    """Drive a runner by hand to the end, returning every round"""
    rounds = []
    for _ in range(100):
        if runner.done:
            return rounds
        rnd = runner.ready()
        rounds.append(rnd)
        for job in rnd.dispatch:
            runner.complete(job, run_inline(job))
        if not (rnd.dispatch or rnd.settled or rnd.unrunnable) and not runner.done:
            raise AssertionError(runner.stuck())
    raise AssertionError("the run did not finish")


def handed_job(runner, name, index=None):
    return runner.exec_graph[name].job(index)


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        yield sub


# -- the value ---------------------------------------------------------------------------------------


def test_absent_output_committed_under_checksum(submitter):
    assert isinstance(submitter.store, PickleStore)
    task = Maybe(x=-1)
    result = submitter(task)
    assert result.outputs.out == Absent(cause="negative", source="Maybe")
    # a successful result, committed like any other under the task's checksum
    stored = submitter.store.lookup(task._checksum, result.context, task=task)
    assert stored is not None and not stored.errored
    assert stored.outputs.out == Absent(cause="negative", source="Maybe")


def test_absent_hash_ignores_source():
    one, other = Absent("negative", "A"), Absent("negative", "renamed")
    assert hash_function(one) == hash_function(other)
    assert Describe(a=one)._checksum == Describe(a=other)._checksum
    assert hash_function(one) != hash_function(Absent("negative", None).cause)


def test_two_causes_two_checksums():
    assert hash_function(Absent("negative", "A")) != hash_function(
        Absent("not-routed", "A")
    )
    assert Describe(a=Absent("negative"))._checksum != (
        Describe(a=Absent("not-routed"))._checksum
    )


# -- what a reader does with one ---------------------------------------------------------------------


def test_propagate_skips_lookup_and_dispatch(tmp_path):
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        runner = GraphRunner(
            Chain(x=-1),
            store,
            submitter=sub,
            on_error="raise_at_end",
            lookup_before_dispatch=True,
        )
        rounds = drive(runner)
    dispatched = [j.name for r in rounds for j in r.dispatch]
    assert dispatched == ["A"]
    settled = [s for r in rounds for s in r.settled]
    assert [(s.job.name, s.how) for s in settled] == [("B", "absent")]
    b = settled[0].job
    assert b.checksum not in {checksum for _, checksum in store.calls}
    assert not [c for c in store.calls if c[0] == "commit"]
    # every output is the Absent of the field, cause and source unchanged
    assert settled[0].result.outputs.out == Absent("negative", "Maybe")
    assert settled[0].result.outputs.out.source == "Maybe"
    assert not runner.failed and not runner.entries
    assert runner.exec_graph["B"].successful


def test_combined_accept_keeps_alignment(submitter):
    outputs = submitter(Aligned(xs=[1, -1, 3])).outputs
    assert outputs.gathered == ["1", "absent:negative", "3"]
    assert len(outputs.gathered) == len(outputs.xs)


def test_drop_removes_absent_elements(submitter):
    assert submitter(Dropped(xs=[1, -1, 3])).outputs.out == [1, 3]


# -- the absence suffix ------------------------------------------------------------------------------


def test_suffix_supplied_checksum_accepting_absent(submitter):
    runner = GraphRunner(
        Keyed(xs=[1, -1, 3]),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    drive(runner)
    job = handed_job(runner, "G")
    assert job.unsuffixed_checksum == "gather-n3"
    assert job.absent == [("vals", 1, "negative")]
    assert absence_triples(job) == job.absent
    assert job.checksum == "gather-n3" + absence_suffix(job.absent)
    assert job.checksum.startswith("gather-n3-absent-")
    assert "/" not in job.checksum
    assert absence_suffix([]) == ""


def test_suffix_unhashed_field(submitter):
    Echo = shell.define(
        "echo",
        inputs={
            "msg": shell.arg(
                type=int, argstr="", position=1, container_path=True, absent="accept"
            )
        },
    )

    @workflow.define
    def Unhashed(x: int) -> str:
        a = workflow.add(Maybe(x=x), name="A")
        e = workflow.add(Echo(msg=a.out), name="E")
        return e.stdout

    runner = GraphRunner(
        Unhashed(x=-1), submitter.store, submitter=submitter, on_error="raise_at_end"
    )
    first = runner.ready()
    (a,) = first.dispatch
    runner.complete(a, run_inline(a))
    (e,) = runner.ready().dispatch
    assert e.absent == [("msg", -1, "negative")]
    assert e.unsuffixed_checksum == e.task._checksum
    assert e.checksum == e.unsuffixed_checksum + absence_suffix(e.absent)
    runner.close()


def test_suffix_drop_indices_before_dropping(submitter):
    runner = GraphRunner(
        KeyedDrop(xs=[-1, 2, -3, 4]),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    drive(runner)
    job = handed_job(runner, "G")
    assert job.task.vals == [2, 4]
    assert job.unsuffixed_checksum == "gather-n2"
    assert job.absent == [("vals", 0, "negative"), ("vals", 2, "negative")]
    assert job.checksum == "gather-n2" + absence_suffix(job.absent)


def test_suffix_sorted_and_hashed_over_triples():
    assert absence_suffix([("b", 0, "x"), ("a", 1, "y")]) == absence_suffix(
        [("a", 1, "y"), ("b", 0, "x")]
    )
    assert absence_suffix([("a", 1, "y")]) != absence_suffix([("a", 2, "y")])
    assert absence_suffix([("a", 1, "y")]) != absence_suffix([("a", 1, "z")])


def test_two_runs_not_served_across_absence(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        full = sub(Keyed(xs=[1, 2, 3])).outputs.out
        routed = sub(Keyed(xs=[1, -2, 3])).outputs.out
    # the supplied key is the same in both runs; the suffix keeps them apart
    assert full == ["1", "2", "3"]
    assert routed == ["1", "absent:negative", "3"]


def test_no_absent_checksum_unchanged(submitter):
    runner = GraphRunner(
        Keyed(xs=[1, 2, 3]),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    drive(runner)
    job = handed_job(runner, "G")
    assert job.absent == []
    assert job.checksum == job.unsuffixed_checksum == "gather-n3"
    plain = handed_job(runner, "A", 0)
    assert plain.checksum == plain.task._checksum == plain.unsuffixed_checksum


def test_hash_change_check_uses_unsuffixed(submitter):
    runner = GraphRunner(
        Keyed(xs=[1, -1, 3]),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    drive(runner)
    job = handed_job(runner, "G")
    assert job.checksum != job.task._checksum
    job._check_for_hash_changes()  # compares the unsuffixed checksum: no raise
    # and a supplied-checksum job that runs with an absence does not fail the check
    assert submitter(Keyed(xs=[1, -1, 3])).outputs.out == ["1", "absent:negative", "3"]


# -- splitters, workflow outputs, failures -----------------------------------------------------------


def test_absent_splitter_whole_node(submitter):
    out = submitter(SplitOverAbsent(n=-1)).outputs.out
    # absent as a whole: the reader receives the Absent, not an empty list
    assert out == Absent("no-list", "MaybeList")
    assert submitter(SplitOverAbsent(n=2)).outputs.out == [1, 2]


def test_absent_in_workflow_outputs(tmp_path):
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        task = Passthrough(x=-1)
        assert sub(task).outputs.out == Absent("negative", "Maybe")
        # the workflow's result is committed, and served on a second run
        assert ("commit", task._checksum) in store.calls
        assert sub(task).outputs.out == Absent("negative", "Maybe")


def test_absent_never_in_error_summary(submitter):
    runner = GraphRunner(
        AbsentAndFailing(x=-1),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    for _ in range(50):
        if runner.done:
            break
        rnd = runner.ready()
        for job in rnd.dispatch:
            try:
                outcome = run_inline(job)
            except Exception as exc:
                from pydra.engine.runner import Failed

                outcome = Failed(exc)
            runner.complete(job, outcome)
    assert runner.failed
    assert [e.job_name for e in runner.entries] == ["F"]
    assert not any("Absent(" in e or "Job 'B'" in e for e in runner.errors)
    assert not any("Absent(" in e.message for e in runner.entries)
    b = runner.exec_graph["B"]
    assert b.successful and not b.errored and not b.unrunnable


def test_absent_field_attribute_validated():
    with pytest.raises((ValueError, TypeError)):
        python.arg(type=int, absent="sometimes")
    assert python.arg(type=int).absent is None


def test_coercion_passes_absent_for_typed_fields():
    assert Describe(a=Absent("x")).a == Absent("x")
    assert Gather(vals=[1, Absent("x"), 3]).vals == [1, Absent("x"), 3]
    with pytest.raises(TypeError):
        Gather(vals=[1, "not-an-int-or-absent", 3])
