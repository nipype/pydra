"""Typed absent values (RFC 0002 section 2.3(a), R10a): ``Absent``, the ``absent=``
field attribute, propagation at release and the absence suffix of a checksum that may not
see an accepted or dropped ``Absent``
"""

import cloudpickle as cp
import pytest

from pydra.compose import python, shell, workflow
from pydra.engine.absent import (
    Absent,
    AbsentNotAccepted,
    absence_suffix,
    absence_triples,
)
from pydra.engine.result import Result
from pydra.engine.runner import GraphRunner
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.utils.hash import hash_function
from pydra.utils.identity import record_checksum

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


@python.define(inputs={"vals": python.arg(type=list[int], absent="accept")})
def Gather(vals) -> list[str]:
    return [f"absent:{v.cause}" if isinstance(v, Absent) else str(v) for v in vals]


@python.define(inputs={"vals": python.arg(type=list[int], absent="drop")})
def GatherDrop(vals) -> list[int]:
    return list(vals)


def by_length(task, inputs):
    """A supplied checksum built from how many values a gather reads, not from them"""
    return f"gather-n{len(inputs['vals'])}"


@python.define(
    checksum=by_length, inputs={"vals": python.arg(type=list[int], absent="accept")}
)
def KeyedGather(vals) -> list[str]:
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
    # under whichever store the submitter has (PortableStore on this branch)
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
    assert e.unsuffixed_checksum == record_checksum(e.record)
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
        # the workflow's result is committed under the key it was looked up by, and
        # served on a second run
        (looked_up,) = [
            c for op, c in store.calls if op == "lookup" and c.startswith("workflow-")
        ]
        assert ("commit", looked_up) in store.calls
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


def test_coercion_passes_absent_only_for_fields_that_accept_it():
    assert Describe(a=Absent("x")).a == Absent("x")
    assert GatherDrop(vals=[1, Absent("x"), 3]).vals == [1, Absent("x"), 3]
    assert Maybe(x=1).Outputs(out=Absent("x")).out == Absent("x")  # an output may be
    with pytest.raises(TypeError):
        Inc(a=Absent("x"))  # propagates by default: it is not accepted as a value
    with pytest.raises(TypeError):
        Count(vals=[1, Absent("x"), 3])
    with pytest.raises(TypeError):
        Gather(vals=[1, "not-an-int-or-absent", 3])


# -- downstream of a wholly absent node --------------------------------------------


@workflow.define
def ReadersOfAbsent(n: int) -> list[int]:
    a = workflow.add(MaybeList(n=n), name="A")
    b = workflow.add(Inc().split(a=a.out), name="B")
    c = workflow.add(Inc(a=b.out), name="C")
    d = workflow.add(Inc(a=c.out), name="D")
    return d.out


@python.define
def Count(vals: list[int]) -> int:
    return len(vals)


@python.define(inputs={"vals": python.arg(type=list[int], absent="accept")})
def Summarize(vals) -> str:
    """A combined read that accepts an ``Absent``"""
    if isinstance(vals, Absent):
        return f"absent:{vals.cause}"
    return ",".join(str(v) for v in vals)


@python.define(inputs={"vals": python.arg(type=list[int], absent="propagate")})
def GatherPropagate(vals) -> list[int]:
    return list(vals)


@workflow.define(outputs=["summary", "dropped", "propagated", "count"])
def CombiningReadersOfAbsent(n: int):
    a = workflow.add(MaybeList(n=n), name="A")
    b = workflow.add(Inc().split(a=a.out).combine("a"), name="B")
    s = workflow.add(Summarize(vals=b.out), name="S")
    d = workflow.add(GatherDrop(vals=b.out), name="D")
    p = workflow.add(GatherPropagate(vals=b.out), name="P")
    c = workflow.add(Count(vals=p.out), name="C")
    return s.out, d.out, p.out, c.out


def test_non_combining_readers_of_absent_node_are_absent(submitter):
    runner = GraphRunner(
        ReadersOfAbsent(n=-1),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    rounds = drive(runner)
    assert [j.name for r in rounds for j in r.dispatch] == ["A"]
    assert not runner.failed and not runner.entries
    cause = Absent("no-list", "MaybeList")
    for name in ("B", "C", "D"):
        node = runner.exec_graph[name]
        assert node.absent == cause and node._tasks == {}
    assert submitter(ReadersOfAbsent(n=-1)).outputs.out == cause
    assert submitter(ReadersOfAbsent(n=2)).outputs.out == [3, 4]


def test_combining_readers_of_absent_node_follow_their_mode(submitter):
    cause = Absent("no-list", "MaybeList")
    runner = GraphRunner(
        CombiningReadersOfAbsent(n=-1),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    rounds = drive(runner)
    assert sorted(j.name for r in rounds for j in r.dispatch) == ["A", "D", "S"]
    # accept: the value is the node's Absent, cause kept
    assert handed_job(runner, "S").task.vals == cause
    # drop: an empty list
    assert handed_job(runner, "D").task.vals == []
    # propagate: the reader is absent, and so is its own reader
    settled = {s.job.name: s for r in rounds for s in r.settled}
    assert settled["P"].how == "absent" and settled["C"].how == "absent"
    assert settled["C"].result.outputs.out == cause
    outputs = submitter(CombiningReadersOfAbsent(n=-1)).outputs
    assert outputs.summary == "absent:no-list"
    assert outputs.dropped == [] and outputs.propagated == cause
    assert outputs.count == cause
    outputs = submitter(CombiningReadersOfAbsent(n=2)).outputs
    assert outputs.summary == "1,2" and outputs.dropped == [1, 2]
    assert outputs.count == 2


def test_round_reports_wholly_absent_nodes(submitter):
    runner = GraphRunner(
        ReadersOfAbsent(n=-1),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    rounds = drive(runner)
    reported = [e for r in rounds for e in r.absent_nodes]
    cause = Absent("no-list", "MaybeList")
    assert [(e.node, e.absent, e.inherits) for e in reported] == [
        ("B", cause, None),
        ("C", cause, "B"),
        ("D", cause, "C"),
    ]
    # every node is visible from the rounds alone
    seen = {j.name for r in rounds for j in r.dispatch}
    seen |= {s.job.name for r in rounds for s in r.settled}
    seen |= {e.node for e in reported}
    assert seen == {"A", "B", "C", "D"}


def test_plan_lists_a_node_that_will_be_wholly_absent(submitter):
    runner = GraphRunner(
        ReadersOfAbsent(n=-1),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    (a,) = runner.ready().dispatch
    runner.complete(a, run_inline(a))
    nodes = {n.path: n for n in runner.plan().nodes}
    assert nodes["B"].state == "absent" and nodes["B"].elements == 0
    assert nodes["B"].expected_hits == 0
    runner.close()


# -- a propagated job has no task checksum -------------------------------------

SEEN: list = []


def strict_key(task, inputs):
    SEEN.append(inputs["a"])
    return "strict-" + str(int(inputs["a"]))


@python.define(checksum=strict_key)
def Strict(a: int) -> int:
    return a * 2


@workflow.define
def StrictChain(x: int) -> int:
    a = workflow.add(Maybe(x=x), name="A")
    s = workflow.add(Strict(a=a.out), name="S")
    return s.out


def test_propagated_job_never_calls_checksum_hook(tmp_path):
    SEEN.clear()
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        assert sub(StrictChain(x=-1)).outputs.out == Absent("negative", "Maybe")
        assert sub(StrictChain(x=3)).outputs.out == 6
    assert not any(isinstance(v, Absent) for v in SEEN)


def test_propagated_identity_is_the_absence_record(submitter):
    SEEN.clear()
    runner = GraphRunner(
        StrictChain(x=-1), submitter.store, submitter=submitter, on_error="raise_at_end"
    )
    drive(runner)
    a, s = handed_job(runner, "A"), handed_job(runner, "S")
    assert s._propagated_absent == Absent("negative", "Maybe")
    assert s.checksum == s.unsuffixed_checksum
    assert s.checksum.startswith("python-absent-") and "/" not in s.checksum
    assert s.absent == []
    # a function of the producer's checksum and the (field, index, cause) triple
    other = GraphRunner(
        StrictChain(x=-2), submitter.store, submitter=submitter, on_error="raise_at_end"
    )
    drive(other)
    assert handed_job(other, "A").checksum != a.checksum
    assert handed_job(other, "S").checksum != s.checksum
    cp.loads(cp.dumps(s))  # pickling does not compute a task checksum either
    assert not any(isinstance(v, Absent) for v in SEEN)


@python.define(outputs=["p", "q"])
def Pair(a: int):
    return a, a


@python.define
def Add(a: int, b: int) -> int:
    return a + b


@workflow.define(outputs=["i", "p", "s"])
def ManyReaders(x: int):
    a = workflow.add(Maybe(x=x), name="A")
    i = workflow.add(Inc(a=a.out), name="I")
    p = workflow.add(Pair(a=a.out), name="P")
    s = workflow.add(Add(a=a.out).split(b=[1, 2]), name="S")
    return i.out, p.p, s.out


def test_absence_identity_includes_the_reader_task(submitter):
    def identities():
        runner = GraphRunner(
            ManyReaders(x=-1),
            submitter.store,
            submitter=submitter,
            on_error="raise_at_end",
        )
        drive(runner)
        return {
            name: handed_job(runner, *key).checksum
            for name, key in {
                "I": ("I",),
                "P": ("P",),
                "S0": ("S", 0),
                "S1": ("S", 1),
            }.items()
        }

    first = identities()
    # different task types reading the same absent producer differ
    assert len({first["I"], first["P"], first["S0"]}) == 3
    # the same task type with the same absence is deterministic: across its elements
    # (equal absent results) and across runs
    assert first["S0"] == first["S1"]
    assert identities() == first


def test_plan_counts_propagated_elements_without_store_reads(tmp_path):
    SEEN.clear()
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        runner = GraphRunner(
            StrictChain(x=-1),
            store,
            submitter=sub,
            on_error="raise_at_end",
            lookup_before_dispatch=True,
        )
        (a,) = runner.ready().dispatch
        runner.complete(a, run_inline(a))
        store.calls.clear()
        nodes = {n.path: n for n in runner.plan().nodes}
    assert nodes["S"].absent == 1 and nodes["S"].expected_hits == 0
    assert store.calls == []
    assert not any(isinstance(v, Absent) for v in SEEN)


# -- an Absent held as a constant ---------------------------------------------------------


def count_key(task, inputs):
    """A supplied checksum built from how many values a task reads, not from them"""
    return f"keyedshow-n{len(inputs['vals'])}"


@python.define(
    checksum=count_key, inputs={"vals": python.arg(type=list[int], absent="accept")}
)
def KeyedShow(vals) -> list[str]:
    return [f"absent:{v.cause}" if isinstance(v, Absent) else str(v) for v in vals]


@workflow.define(inputs={"vals": workflow.arg(type=list[int], absent="accept")})
def KeyedShowWf(vals) -> list[str]:
    k = workflow.add(KeyedShow(vals=vals), name="K")
    return k.out


def test_constant_absent_enters_the_suffix(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        # a nested workflow's node receives the list as a constant
        nested = sub(KeyedShowWf(vals=[1, Absent("negative"), 3])).outputs.out
        assert nested == ["1", "absent:negative", "3"]
        # a run without the absence is not served the inner node's entry
        assert sub(KeyedShow(vals=[1, 2, 3])).outputs.out == ["1", "2", "3"]
        # a root job holding a constant Absent has its own suffix and cause
        root = sub(KeyedShow(vals=[1, Absent("x"), 3])).outputs.out
        assert root == ["1", "absent:x", "3"]
    job_task = KeyedShow(vals=[1, 2, 3])
    assert job_task._checksum == "keyedshow-n3"  # no Absent: no suffix


# -- with partial expansion, placement and the store -------------------


def test_partial_readers_classified_before_their_upstream_is_absent(submitter):
    runner = GraphRunner(
        ReadersOfAbsent(n=-1),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    (a,) = runner.ready().dispatch
    runner.ready()  # A is out: B, C and D start as partial nodes with no block yet
    assert [runner.exec_graph[n].mode for n in "BCD"] == ["partial"] * 3
    runner.complete(a, run_inline(a))
    rounds = drive(runner)
    cause = Absent("no-list", "MaybeList")
    assert [(e.node, e.absent, e.inherits) for r in rounds for e in r.absent_nodes] == [
        ("B", cause, None),
        ("C", cause, "B"),
        ("D", cause, "C"),
    ]
    assert not [j for r in rounds for j in r.dispatch]
    assert all(not runner.exec_graph[n].is_partial for n in "BCD")
    runner.close()


@workflow.define
def PerElement(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Inc().split("a", a=a.out), name="B")
    return b.out


def test_absent_block_of_a_partial_node_has_no_element(submitter):
    # an upstream element's Absent gives its sub-state no elements (test_absent_split)
    result = submitter(PerElement(ns=[2, -1]))
    assert result.outputs.out == [1, 2]


@workflow.define
def SplitMaybe(xs: list[int]) -> list[int]:
    a = workflow.add(Maybe().split(x=xs), name="A")
    s = workflow.add(Inc(a=a.out), name="S")
    return s.out


class Placer:
    """An executor that records what ``plan()`` shows its ``place``"""

    def __init__(self):
        self.shown = []

    def place(self, jobs):
        self.shown.append(list(jobs))
        return "placed"


class AnyWorker(dict):
    def __init__(self, executor):
        super().__init__()
        self.executor = executor

    def get(self, key, default=None):
        return self.executor


def test_plan_places_and_looks_up_only_elements_that_run(tmp_path):
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        runner = GraphRunner(
            SplitMaybe(xs=[1, -1]),
            store,
            submitter=sub,
            on_error="raise_at_end",
            lookup_before_dispatch=True,
        )
        for job in runner.ready().dispatch:
            runner.complete(job, run_inline(job))
        store.calls.clear()
        placer = Placer()
        (node,) = runner.plan(AnyWorker(placer)).nodes
        runner.close()
    assert node.path == "S" and node.absent == 1 and node.elements == 2
    ((shown,),) = placer.shown
    assert shown.task.a == 1 and shown._propagated_absent is None
    assert [checksum for _, checksum in store.calls] == [shown.checksum]


# -- absent modes: propagate by default, accept, fill ------------------------------------------------


@python.define(inputs={"vals": python.arg(type=list[int], absent="fill", fill=0)})
def GatherFill(vals) -> list[int]:
    return list(vals)


@python.define(inputs={"a": python.arg(type=int, absent="fill", fill=0)})
def IncFill(a) -> int:
    return a + 1


@workflow.define(outputs=["counted", "filled", "single"])
def Modes(xs: list[int], x: int):
    a = workflow.add(Maybe().split(x=xs).combine("x"), name="A")
    c = workflow.add(Count(vals=a.out), name="C")
    f = workflow.add(GatherFill(vals=a.out), name="F")
    m = workflow.add(Maybe(x=x), name="M")
    s = workflow.add(IncFill(a=m.out), name="S")
    return c.out, f.out, s.out


def test_a_combined_read_propagates_by_default(submitter):
    outputs = submitter(Modes(xs=[1, -1, 3], x=1)).outputs
    assert isinstance(outputs.counted, Absent)
    assert outputs.counted.cause == "negative"


def test_fill_replaces_each_absent_element_and_value(submitter):
    outputs = submitter(Modes(xs=[1, -1, 3], x=-1)).outputs
    assert outputs.filled == [1, 0, 3]
    assert outputs.single == 1


def test_a_filled_value_is_keyed_apart_from_the_same_real_value(submitter):
    runner = GraphRunner(
        Modes(xs=[1, -1, 3], x=-1),
        submitter.store,
        submitter=submitter,
        on_error="raise_at_end",
    )
    drive(runner)
    filled = handed_job(runner, "S")
    assert filled.task.a == 0 and filled.absent == [("a", -1, "negative")]
    assert filled.checksum.endswith(absence_suffix(filled.absent))
    assert filled.checksum != filled._unsuffixed_checksum


@python.define
def Nested(n: int) -> list:
    return [n, Absent("inner")]


@workflow.define
def NestedRead(n: int) -> int:
    a = workflow.add(Nested(n=n), name="A")
    c = workflow.add(Count(vals=a.out), name="C")
    return c.out


def test_an_absent_inside_a_value_is_refused_unless_accepted(submitter):
    runner = GraphRunner(
        NestedRead(n=1), submitter.store, submitter=submitter, on_error="raise_at_end"
    )
    drive(runner)
    (error,) = runner.exec_graph["C"].errored.values()
    assert isinstance(error, AbsentNotAccepted) and "C.vals" in str(error)
    assert runner.failed


@pytest.mark.parametrize("kwargs", [dict(absent="fill"), dict(fill=0)])
def test_fill_and_its_mode_go_together(kwargs):
    with pytest.raises(ValueError, match="fill"):
        python.arg(type=int, **kwargs)


def test_an_absent_default_on_a_propagating_field_is_refused():
    """A field that does not accept an ``Absent`` refuses one as its default"""
    with pytest.raises(TypeError):

        @python.define(inputs={"x": python.arg(type=int, default=Absent("unset"))})
        def Takes(x: int) -> int:
            return x


def test_a_parser_built_outside_a_field_refuses_an_absent():
    """Only a parser told so lets an ``Absent`` through"""
    from pydra.utils.typing import TypeParser

    with pytest.raises(TypeError):
        TypeParser(int)(Absent("unset"))
    assert isinstance(TypeParser(int, accepts_absent=True)(Absent("unset")), Absent)
