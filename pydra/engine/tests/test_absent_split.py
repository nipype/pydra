"""An ``Absent`` splitter input per element (RFC 0002 section 2.3(a) "Splitter inputs",
a partial node): an upstream element whose output is an ``Absent`` gives that element's sub-state
no elements, and a reader of the absent sub-state receives the ``Absent``
"""

import attrs

from pydra.compose import python, workflow
from pydra.engine.absent import Absent, absence_suffix, resolving
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter
from pydra.engine.tests.test_absent import (
    Count,
    CountingStore,
    Gather,
    GatherDrop,
    GatherPropagate,
    Inc,
    KeyedGather,
    MaybeList,
    Summarize,
    drive,
    handed_job,
    submitter,  # noqa: F401 (a fixture)
)
from pydra.engine.tests.test_loop_runner import HalveLoop
from pydra.engine.tests.test_loop_runner import drive as drive_loops

NO_LIST = Absent("no-list", "MaybeList")

# -- workflows ---------------------------------------------------------------------------------------


@workflow.define
def PerElement(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Inc().split("a", a=a.out), name="B")
    return b.out


@workflow.define(outputs=["summary", "dropped", "propagated", "count", "groups"])
def PerGroup(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Inc().split("a", a=a.out).combine("a"), name="B")
    s = workflow.add(Summarize(vals=b.out), name="S")
    d = workflow.add(GatherDrop(vals=b.out), name="D")
    p = workflow.add(GatherPropagate(vals=b.out), name="P")
    c = workflow.add(Count(vals=p.out), name="C")
    return s.out, d.out, p.out, c.out, b.out


def show_key(task, inputs):
    """A supplied checksum that does not look at the values it is given"""
    return "show"


@python.define(
    checksum=show_key, inputs={"vals": python.arg(type=list[int], absent="accept")}
)
def KeyedSummarize(vals) -> str:
    if isinstance(vals, Absent):
        return f"absent:{vals.cause}"
    return ",".join(str(v) for v in vals)


@workflow.define
def KeyedPerGroup(ns: list[int]) -> list[str]:
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Inc().split("a", a=a.out).combine("a"), name="B")
    k = workflow.add(KeyedSummarize(vals=b.out), name="K")
    return k.out


def drop_key(task, inputs):
    return f"drop-n{len(inputs['vals'])}"


@python.define(
    checksum=drop_key, inputs={"vals": python.arg(type=list[int], absent="drop")}
)
def DropKeyed(vals) -> list[int]:
    return list(vals)


@workflow.define(outputs=["gathered", "keyed", "keyed_drop"])
def Readers(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Inc().split("a", a=a.out).combine(["A.n", "a"]), name="B")
    g = workflow.add(Gather(vals=b.out), name="G")
    k = workflow.add(KeyedGather(vals=b.out), name="K")
    d = workflow.add(DropKeyed(vals=b.out), name="D")
    return g.out, k.out, d.out


@python.define
def Flag(on: bool) -> bool:
    return on


@workflow.define
def GatedPerGroup(ns: list[int], on: bool):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    f = workflow.add(Flag(on=on), name="F")
    b = workflow.add(Inc().split("a", a=a.out).combine("a"), name="B", when=f.out)
    return b.out


@python.define
def MaybeUpto(n: int) -> list[int]:
    if n < 0:
        return Absent(cause="no-list", source="MaybeUpto")
    return [4 * 2**k for k in range(n)]


@workflow.define(outputs=["x", "rounds"])
def LoopOverAbsent(ns: list[int]):
    up = workflow.add(MaybeUpto().split("n", n=ns), name="up")
    loop = workflow.add(HalveLoop(target=1).split("x", x=up.out), name="halve")
    return loop.x, loop.rounds


def runner_for(sub, task, **kwargs):
    return GraphRunner(
        task, sub.store, submitter=sub, on_error="raise_at_end", **kwargs
    )


def settled(rounds, name):
    return [s for r in rounds for s in r.settled if s.job.name == name]


# -- the brief's two tests ---------------------------------------------------------------------------


def test_absent_upstream_element_gives_empty_substate(submitter):
    runner = runner_for(submitter, PerElement(ns=[2, -1]))
    rounds = drive(runner)
    b = runner.exec_graph["B"]
    # the absent element of A opens no element of B: nothing is settled absent
    assert len(b._tasks) == 2 and not settled(rounds, "B")
    assert sorted(j.task.a for r in rounds for j in r.dispatch if j.name == "B") == [
        0,
        1,
    ]
    assert b.absent is None and not runner.failed
    # the element vanishes from a reader of the whole node
    assert submitter(PerElement(ns=[2, -1])).outputs.out == [1, 2]


def test_reader_of_absent_substate_gets_absent_not_empty_list(submitter):
    runner = runner_for(submitter, PerGroup(ns=[2, -1]))
    drive(runner)
    s = runner.exec_graph["S"]
    assert [s.job(key).task.vals for key in sorted(s._tasks)] == [[1, 2], NO_LIST]
    # the whole read of the groups holds the Absent in place of the absent group
    outputs = submitter(PerGroup(ns=[2, -1])).outputs
    assert outputs.groups == [[1, 2], NO_LIST]
    assert outputs.summary == ["1,2", "absent:no-list"]


# -- a combined reader of an absent sub-state follows its mode ---------------------------------------


def test_combined_reader_of_absent_substate_follows_its_mode(submitter):
    runner = runner_for(submitter, PerGroup(ns=[-1]))
    rounds = drive(runner)
    # accept: the value is the sub-state's Absent, cause kept
    assert handed_job(runner, "S", (0,)).task.vals == NO_LIST
    # drop: an empty list
    assert handed_job(runner, "D", (0,)).task.vals == []
    # propagate: the reader completes absent, and so does its own reader
    (p,) = settled(rounds, "P")
    (c,) = settled(rounds, "C")
    assert p.how == c.how == "absent"
    assert c.result.outputs.out == NO_LIST
    assert sorted(j.name for r in rounds for j in r.dispatch) == ["A", "D", "S"]
    outputs = submitter(PerGroup(ns=[2, -1])).outputs
    assert outputs.summary == ["1,2", "absent:no-list"]
    assert outputs.dropped == [[1, 2], []]
    assert outputs.propagated == [[1, 2], NO_LIST]
    assert outputs.count == [2, NO_LIST]


# -- identity --------------------------------------------------------------------------------------


def test_propagated_reader_of_absent_substate_names_its_producer(submitter):
    def identity(ns):
        runner = runner_for(submitter, PerGroup(ns=ns))
        drive(runner)
        return (
            handed_job(runner, "A", 0).unsuffixed_checksum,
            handed_job(runner, "P", (0,)).checksum,
        )

    (a1, p1), (a2, p2) = identity([-1]), identity([-2])
    # the same cause from two upstream jobs: two identities, as the RFC's record says
    assert a1 != a2 and p1 != p2
    assert identity([-1]) == (a1, p1)


def test_reader_of_absent_substate_has_the_absence_suffix(tmp_path):
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        runner = runner_for(sub, KeyedPerGroup(ns=[-1]))
        drive(runner)
        absent_key = handed_job(runner, "K", (0,)).checksum
        assert absent_key == "show" + absence_suffix([("vals", -1, "no-list")])
        # the run with the element present is not served the absent run's entry
        assert sub(KeyedPerGroup(ns=[-1])).outputs.out == ["absent:no-list"]
        assert sub(KeyedPerGroup(ns=[2])).outputs.out == ["1,2"]
        assert sub(KeyedPerGroup(ns=[-1])).outputs.out == ["absent:no-list"]
    looked_up = {key for call, key in store.calls if call == "lookup"}
    assert {"show", absent_key} <= looked_up


def old_keys():
    """The keys the readers of ``Readers(ns=[2, -1])`` had while the absent element of A
    opened one element of B, which propagated: they read ``[1, 2, Absent]``"""
    with resolving():  # as the runner resolved G's element
        task = attrs.evolve(Gather(vals=[]), vals=[1, 2, NO_LIST])
    gathered = task._checksum
    triple = [("vals", 2, "no-list")]
    return {
        "G": gathered,
        "K": "gather-n3" + absence_suffix(triple),
        "D": "drop-n2" + absence_suffix(triple),
    }


def test_keys_written_before_the_change_are_never_looked_up(tmp_path):
    old = old_keys()
    # each carries the Absent: hashed by cause, or in the suffix
    assert old["G"] != Gather(vals=[1, 2])._checksum
    assert old["K"].startswith("gather-n3-absent-")
    assert old["D"].startswith("drop-n2-absent-")
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        out = sub(Readers(ns=[2, -1])).outputs
        assert (out.gathered, out.keyed, out.keyed_drop) == (
            ["1", "2"],
            ["1", "2"],
            [1, 2],
        )
        looked_up = {key for call, key in store.calls if call == "lookup"}
        assert not looked_up & set(old.values())
        # the readers' keys are those of a run in which the element is not there at all
        runner = runner_for(sub, Readers(ns=[2, -1]))
        drive(runner)
        now = {n: handed_job(runner, n).checksum for n in "GKD"}
        runner = runner_for(sub, Readers(ns=[2]))
        drive(runner)
        assert {n: handed_job(runner, n).checksum for n in "GKD"} == now
    assert (now["K"], now["D"]) == ("gather-n2", "drop-n2")


# -- a gated element is an element; an absent element is not -----------------------------------------


def test_gated_elements_are_elements_absent_substate_is_not(submitter):
    runner = runner_for(submitter, GatedPerGroup(ns=[2, -1], on=False))
    rounds = drive(runner)
    gated = settled(rounds, "B")
    # the two elements of the present sub-state exist and complete gated
    assert [s.how for s in gated] == ["gated", "gated"]
    assert len(runner.exec_graph["B"]._tasks) == 2
    not_routed = Absent("not-routed", "B")
    out = submitter(GatedPerGroup(ns=[2, -1], on=False)).outputs.out
    assert out == [[not_routed, not_routed], NO_LIST]
    assert submitter(GatedPerGroup(ns=[2, -1], on=True)).outputs.out == [
        [1, 2],
        NO_LIST,
    ]


# -- a loop under a split ----------------------------------------------------------------------------


def test_loop_under_split_with_an_absent_element(submitter):
    runner = runner_for(submitter, LoopOverAbsent(ns=[2, -1, 1]))
    rounds = drive_loops(runner)
    # no element of the absent sub-state: nothing is settled absent
    assert "absent" not in {s.how for s in settled(rounds, "halve")}
    # one loop per element that exists: 4 and 8 of n=2, 4 of n=1
    assert sorted(runner.exec_graph["halve"]._tasks) == [(0, 0), (0, 1), (2, 0)]
    out = submitter(LoopOverAbsent(ns=[2, -1, 1])).outputs
    assert (out.x, out.rounds) == ([1, 1, 1], [2, 3, 2])


# -- barrier nodes and barrier readers: an absent sub-state is an empty one -------------------------


@python.define
def AddLength(a: int, b: list[int]) -> int:
    return a + len(b)


@python.define
def Ident(x: int) -> int:
    return x


@workflow.define
def BarrierOverAbsent(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    # a non-split input read from the inner source makes B a barrier node
    b = workflow.add(AddLength(b=a.out).split("a", a=a.out), name="B")
    return b.out


@workflow.define
def CombinedBarrierOverAbsent(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(AddLength(b=a.out).split("a", a=a.out).combine("a"), name="B")
    return b.out


@workflow.define
def BarrierReader(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Inc().split("a", a=a.out), name="B")
    # combines the block axis of partial node B but not its inner axis: a barrier reader
    c = workflow.add(Ident(x=b.out).combine("A.n"), name="C")
    return c.out


def failed_with(sub, task, node):
    runner = runner_for(sub, task)
    rounds = drive(runner)
    assert not settled(rounds, node)
    (entry,) = runner.entries
    return runner, entry.message


def test_barrier_node_gives_an_absent_substate_no_elements(submitter):
    runner = runner_for(submitter, BarrierOverAbsent(ns=[-1]))
    rounds = drive(runner)
    b = runner.exec_graph["B"]
    assert b.mode == "barrier" and b._tasks == {} and not runner.entries
    assert not settled(rounds, "B")
    assert submitter(BarrierOverAbsent(ns=[-1])).outputs.out == []
    assert submitter(BarrierOverAbsent(ns=[2, 2])).outputs.out == [2, 3, 2, 3]
    # beside a present element, the empty sub-state is ragged, which today's State
    # cannot index
    _, message = failed_with(submitter, BarrierOverAbsent(ns=[2, -1]), "B")
    assert "unequal lengths [0, 2]" in message and "Absent" in message


def test_combining_barrier_node_refuses_an_absent_upstream_element(submitter):
    # the reader of the absent group would need its Absent, which today's State has not
    runner, message = failed_with(submitter, CombinedBarrierOverAbsent(ns=[-1]), "B")
    assert runner.exec_graph["B"].mode == "barrier"
    assert "combines" in message and "Absent" in message and "'A'" in message


def test_barrier_reader_of_an_absent_substate(submitter):
    runner = runner_for(submitter, BarrierReader(ns=[-1]))
    rounds = drive(runner)
    assert runner.exec_graph["C"].mode == "barrier" and not runner.entries
    assert not settled(rounds, "C")
    assert submitter(BarrierReader(ns=[-1])).outputs.out == []
    _, message = failed_with(submitter, BarrierReader(ns=[2, -1]), "C")
    assert "'C'" in message and "unequal lengths [0, 2]" in message


# -- a combine downstream of the node whose sub-state is absent ----------------------------------------


@workflow.define(outputs=["groups", "summary"])
def PartialThenCombine(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Ident(x=b.out).combine("B.x"), name="C")
    s = workflow.add(Summarize(vals=c.out), name="S")
    return c.out, s.out


@workflow.define(outputs=["groups", "summary", "dropped", "count"])
def PartialChainThenCombine(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Inc(a=b.out), name="C")
    d = workflow.add(Ident(x=c.out).combine("B.x"), name="D")
    s = workflow.add(Summarize(vals=d.out), name="S")
    g = workflow.add(GatherDrop(vals=d.out), name="G")
    p = workflow.add(GatherPropagate(vals=d.out), name="P")
    k = workflow.add(Count(vals=p.out), name="K")
    return d.out, s.out, g.out, k.out


def test_downstream_partial_combine_forms_the_absent_group(submitter):
    out = submitter(PartialThenCombine(ns=[-1])).outputs
    assert (out.groups, out.summary) == ([NO_LIST], ["absent:no-list"])
    out = submitter(PartialThenCombine(ns=[2, -1])).outputs
    assert (out.groups, out.summary) == ([[0, 1], NO_LIST], ["0,1", "absent:no-list"])
    # through an uncombined partial node in between, and under each mode
    out = submitter(PartialChainThenCombine(ns=[2, -1])).outputs
    assert out.groups == [[1, 2], NO_LIST]
    assert out.summary == ["1,2", "absent:no-list"]
    assert out.dropped == [[1, 2], []]
    assert out.count == [2, NO_LIST]


@workflow.define(outputs=["groups", "summary"])
def BarrierThenCombine(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(AddLength(b=a.out).split("a", a=a.out), name="B")  # a barrier
    c = workflow.add(Ident(x=b.out).combine("B.a"), name="C")
    s = workflow.add(Summarize(vals=c.out), name="S")
    return c.out, s.out


@workflow.define(outputs=["groups", "summary"])
def BarrierChainThenCombine(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(AddLength(b=a.out).split("a", a=a.out), name="B")  # a barrier
    c = workflow.add(Inc(a=b.out), name="C")
    d = workflow.add(Ident(x=c.out).combine("B.a"), name="D")
    s = workflow.add(Summarize(vals=d.out), name="S")
    return d.out, s.out


def test_combine_over_a_barrier_nodes_absent_substate_is_refused(submitter):
    for task, node in (
        (BarrierThenCombine(ns=[-1]), "C"),
        (BarrierThenCombine(ns=[-1, -2]), "C"),
        (BarrierChainThenCombine(ns=[-1]), "D"),
    ):
        runner = runner_for(submitter, task)
        rounds = drive(runner)
        assert not settled(rounds, node) and not settled(rounds, "S")
        assert len(runner.entries) == 1, "the combine was not refused"
        (entry,) = runner.entries
        assert entry.job_name == node
        assert entry.exception_type.endswith("AbsentGroupError")
        assert "'B.a'" in entry.message and "'A'" in entry.message
    # a combine that does not keep the absent element's group is not refused
    assert submitter(BarrierReader(ns=[-1])).outputs.out == []


# -- a downstream combine that keeps an axis of its own ----------------------------------------


@python.define
def Id2(x: int, y: int) -> int:
    return x + 100 * y


@workflow.define(outputs=["groups", "summary"])
def OwnAxisCombine(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Id2(x=b.out).split("y", y=[1, 2]).combine("B.x"), name="C")
    s = workflow.add(Summarize(vals=c.out), name="S")
    return c.out, s.out


@python.define
def Ys() -> list[int]:
    return [1, 2]


@workflow.define(outputs=["groups", "summary"])
def OtherSourceAxisCombine(ns: list[int]):
    a = workflow.add(MaybeList().split("n", n=ns), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    e = workflow.add(Ys(), name="E")
    c = workflow.add(Id2(x=b.out).split("y", y=e.out).combine("B.x"), name="C")
    s = workflow.add(Summarize(vals=c.out), name="S")
    return c.out, s.out


def test_downstream_combine_keeping_its_own_axis_forms_each_absent_group(submitter):
    out = submitter(OwnAxisCombine(ns=[2, -1])).outputs
    assert out.groups == [[100, 101], [200, 201], NO_LIST, NO_LIST]
    assert out.summary == ["100,101", "200,201", "absent:no-list", "absent:no-list"]
    out = submitter(OwnAxisCombine(ns=[-1])).outputs
    assert (out.groups, out.summary) == ([NO_LIST] * 2, ["absent:no-list"] * 2)


def test_absent_groups_that_cannot_be_enumerated_are_refused(submitter):
    runner = runner_for(submitter, OtherSourceAxisCombine(ns=[-1]))
    rounds = drive(runner)
    assert not settled(rounds, "S")
    errors = [
        e for e in runner.entries if e.exception_type.endswith("AbsentGroupError")
    ]
    assert [e.job_name for e in errors] == ["C"], "the combine was not refused"
    assert "'B'" in errors[0].message
    # with nothing absent the same workflow runs
    out = submitter(OtherSourceAxisCombine(ns=[2])).outputs
    assert out.summary == ["100,101", "200,201"]
