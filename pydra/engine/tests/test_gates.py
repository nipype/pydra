"""Gates and ``select(n)`` (RFC 0002 section 2.3(b)(c), R10b): ``Workflow.add(...,
when=, unless=)``, the precedence gate -> absence -> store -> sharing -> dispatch, and the
merge of routes keyed on the gates
"""

import typing as ty

import attrs
import pytest

from pydra.compose import python, workflow
from pydra.compose import select
from pydra.engine.absent import Absent
from pydra.engine.result import Result
from pydra.engine.runner import GraphRunner
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.engine.workflow import Workflow

# -- tasks -------------------------------------------------------------------------------------------


@python.define
def Check(x: int) -> bool:
    return x > 0


@python.define
def IsNeg(x: int) -> bool:
    return x < 0


@python.define
def MaybeCheck(x: int) -> bool:
    """Whether ``x`` is positive, or an ``Absent`` when it is zero"""
    if x == 0:
        return Absent(cause="unknown", source="MaybeCheck")
    return x > 0


@python.define
def Count(x: int) -> int:
    return x


@python.define
def Download(url: str) -> str:
    return url.upper()


@python.define
def Maybe(x: int) -> int:
    if x < 0:
        return Absent(cause="negative", source="Maybe")
    return x


@python.define
def Inc(a: int) -> int:
    return a + 1


@python.define
def Bad(x: int) -> ty.Any:
    """A payload of the wrong type, as a probe step gives when it fails"""
    return "not-an-int"


@python.define
def Dec(a: int) -> int:
    return a - 1


# -- workflows ---------------------------------------------------------------------------------------


@workflow.define
def Gated(x: int, url: str) -> str:
    c = workflow.add(Check(x=x), name="C")
    d = workflow.add(Download(url=url), name="D", when=c.out)
    return d.out


@workflow.define
def GatedUnless(x: int, url: str) -> str:
    c = workflow.add(Check(x=x), name="C")
    d = workflow.add(Download(url=url), name="D", unless=c.out)
    return d.out


@workflow.define
def Fanout(xs: list[int]) -> list[str]:
    c = workflow.add(Check().split(x=xs), name="C")
    d = workflow.add(Download(url="x"), name="D", when=c.out)
    return d.out


@workflow.define
def AbsentGate(x: int) -> str:
    c = workflow.add(MaybeCheck(x=x), name="C")
    d = workflow.add(Download(url="u"), name="D", when=c.out)
    return d.out


@workflow.define
def GatedOverAbsentData(x: int) -> int:
    c = workflow.add(Check(x=x), name="C")
    m = workflow.add(Maybe(x=x), name="M")
    d = workflow.add(Inc(a=m.out), name="D", when=c.out)
    return d.out


@workflow.define
def Route(x: int) -> ty.Any:
    pos = workflow.add(Check(x=x), name="pos")
    neg = workflow.add(IsNeg(x=x), name="neg")
    up = workflow.add(Inc(a=x), name="up", when=pos.out)
    down = workflow.add(Dec(a=x), name="down", when=neg.out)
    merged = workflow.add(
        select(2)(when_0=pos.out, value_0=up.out, when_1=neg.out, value_1=down.out),
        name="merge",
    )
    return merged.out


@workflow.define
def RouteContent(x: int) -> ty.Any:
    on = workflow.add(Check(x=1), name="on")
    branch = workflow.add(Maybe(x=x), name="branch", when=on.out)
    merged = workflow.add(
        select(1)(when_0=on.out, value_0=branch.out),
        name="merge",
    )
    return merged.out


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


def drive(runner, complete=run_inline):
    """Drive a runner by hand to the end, returning every round"""
    rounds = []
    for _ in range(100):
        if runner.done:
            return rounds
        rnd = runner.ready()
        rounds.append(rnd)
        for job in rnd.dispatch:
            runner.complete(job, complete(job))
        if not (rnd.dispatch or rnd.settled or rnd.unrunnable or rnd.absent_nodes):
            raise AssertionError(runner.stuck())
    raise AssertionError("the run did not finish")


def runner_for(sub, task, store=None):
    return GraphRunner(
        task,
        store if store is not None else sub.store,
        submitter=sub,
        on_error="raise_at_end",
        lookup_before_dispatch=True,
    )


def settled(rounds, name):
    return [s for r in rounds for s in r.settled if s.job.name == name]


def dispatched(rounds, name=None):
    return [j for r in rounds for j in r.dispatch if name is None or j.name == name]


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        yield sub


# -- the gate ----------------------------------------------------------------------------------------


def test_gated_off_completes_absent_not_routed(submitter):
    runner = runner_for(submitter, Gated(x=-1, url="u"))
    rounds = drive(runner)
    assert [j.name for j in dispatched(rounds)] == ["C"]
    (d,) = settled(rounds, "D")
    assert d.how == "gated"
    assert d.result.outputs.out == Absent(cause="not-routed", source="D")
    # through the submitter, both senses of the gate
    assert submitter(Gated(x=1, url="u")).outputs.out == "U"
    assert submitter(Gated(x=-1, url="u")).outputs.out.cause == "not-routed"
    assert submitter(GatedUnless(x=-1, url="u")).outputs.out == "U"
    assert submitter(GatedUnless(x=1, url="u")).outputs.out.cause == "not-routed"


def test_gate_precedes_absence(submitter):
    """Step 1 before step 2: a gated-off element whose data input is ``Absent`` is gated"""
    rounds = drive(runner_for(submitter, GatedOverAbsentData(x=-1)))
    (d,) = settled(rounds, "D")
    assert d.how == "gated"
    assert d.result.outputs.out.cause == "not-routed"


@workflow.define
def GatedOverMistypedData(x: int) -> int:
    c = workflow.add(Check(x=x), name="C")
    b = workflow.add(Bad(x=x), name="B")
    d = workflow.add(Inc(a=b.out), name="D", when=c.out)
    return d.out


def test_gated_off_data_never_coerced(submitter):
    """A gated-off element waits for its data upstreams but never resolves their values"""
    runner = runner_for(submitter, GatedOverMistypedData(x=-1))
    rnd = runner.ready()
    for job in rnd.dispatch:  # C and B
        runner.complete(job, run_inline(job))
    rnd = runner.ready()
    assert not runner.errors  # D's data is not resolved, so it cannot fail coercion
    assert [(s.job.name, s.how) for s in rnd.settled] == [("D", "gated")]
    assert rnd.settled[0].result.outputs.out.cause == "not-routed"
    result = submitter(GatedOverMistypedData(x=-1))
    assert result.outputs.out.cause == "not-routed"


@python.define
def Names(n: int) -> list[str]:
    return [f"u{i}" for i in range(n)]


@python.define
def Tagged(url: str, tag: str) -> str:
    return url.upper() + tag


@workflow.define
def GatedUpstreamSplit(xs: list[int], n: int) -> list[str]:
    c = workflow.add(Check().split(x=xs), name="C")
    names = workflow.add(Names(n=n), name="N")
    d = workflow.add(
        Tagged(tag="!").split(["_C", "url"], url=names.out), name="D", when=c.out
    )
    return d.out


@workflow.define
def GatedStaticSplit(xs: list[int]) -> list[str]:
    c = workflow.add(Check().split(x=xs), name="C")
    d = workflow.add(
        Tagged(tag="!").split(["_C", "url"], url=["u0", "u1"]), name="D", when=c.out
    )
    return d.out


def test_gated_off_split_elements_keep_their_split_values(submitter):
    """A gated-off element of a split node gets its own split value: a static one as it
    is, one split from an upstream output as the Absent it completes with"""
    out = submitter(GatedUpstreamSplit(xs=[1, -1], n=3)).outputs.out
    assert out[:3] == ["U0!", "U1!", "U2!"]
    assert [o.cause for o in out[3:]] == ["not-routed"] * 3
    rounds = drive(runner_for(submitter, GatedUpstreamSplit(xs=[1, -1], n=3)))
    gated = [s for s in settled(rounds, "D") if s.how == "gated"]
    assert len(gated) == 3
    assert all(s.job.task.url.cause == "not-routed" for s in gated)
    assert all(s.job.task.tag == "!" for s in gated)

    out = submitter(GatedStaticSplit(xs=[1, -1])).outputs.out
    assert out[:2] == ["U0!", "U1!"]
    assert [o.cause for o in out[2:]] == ["not-routed"] * 2
    rounds = drive(runner_for(submitter, GatedStaticSplit(xs=[1, -1])))
    gated = [s for s in settled(rounds, "D") if s.how == "gated"]
    assert sorted(s.job.task.url for s in gated) == ["u0", "u1"]


def test_gated_off_never_looked_up(tmp_path):
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        runner = runner_for(sub, Gated(x=-1, url="u"), store=store)
        rounds = drive(runner)
    (d,) = settled(rounds, "D")
    twin = Download(url="u")._checksum
    touched = {checksum for _, checksum in store.calls}
    assert d.job.checksum not in touched
    assert twin not in touched
    assert not dispatched(rounds, "D")
    # its absence is never committed: a submitted run commits C and the workflow only,
    # and an ungated twin afterwards runs rather than being served the absence
    with Submitter(worker="debug", store=store) as sub:
        store.calls.clear()
        assert sub(Gated(x=-1, url="u")).outputs.out.cause == "not-routed"
        commits = [checksum for kind, checksum in store.calls if kind == "commit"]
        assert len(commits) == 2 and twin not in commits
        assert sub(Download(url="u")).outputs.out == "U"


def test_gated_off_never_served_stored_twin(tmp_path):
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        stored = sub(Download(url="u"))  # the gated-on result is committed
        assert stored.outputs.out == "U"
        store.calls.clear()
        runner = runner_for(sub, Gated(x=-1, url="u"), store=store)
        rounds = drive(runner)
    (d,) = settled(rounds, "D")
    assert d.how == "gated"
    assert d.result.outputs.out.cause == "not-routed"
    twin = Download(url="u")._checksum
    assert twin not in {checksum for _, checksum in store.calls}


def test_gated_on_twin_runs_once(submitter):
    runner = runner_for(submitter, Fanout(xs=[1, -1, 2]))
    rounds = drive(runner)
    ds = dispatched(rounds, "D")
    assert len(ds) == 1  # three elements, one checksum among the gated-on ones
    hows = sorted(s.how for s in settled(rounds, "D"))
    assert hows == ["gated", "shared"]
    gated = [s for s in settled(rounds, "D") if s.how == "gated"]
    assert gated[0].job.state_index == 1
    assert gated[0].job.checksum != ds[0].checksum  # never in the sharing table
    out = submitter(Fanout(xs=[1, -1, 2])).outputs.out
    assert out[0] == "X" and out[2] == "X"
    assert out[1].cause == "not-routed"


@workflow.define
def InputGate(flag: bool, x: int) -> int:
    d = workflow.add(Inc(a=x), name="D", when=flag)
    return d.out


def test_workflow_input_gate(submitter):
    """A workflow input is a bool when the workflow is constructed: a constant gate"""
    assert submitter(InputGate(flag=True, x=1)).outputs.out == 2
    assert submitter(InputGate(flag=False, x=1)).outputs.out.cause == "not-routed"


def test_gate_absent_propagates(submitter):
    rounds = drive(runner_for(submitter, AbsentGate(x=0)))
    (d,) = settled(rounds, "D")
    assert d.how == "absent"
    assert d.result.outputs.out.cause == "unknown"
    assert not dispatched(rounds, "D")


def test_gate_wrong_type_is_wiring_error():
    @workflow.define
    def IntGate(x: int) -> int:
        n = workflow.add(Count(x=x), name="N")
        d = workflow.add(Inc(a=x), name="D", when=n.out)
        return d.out

    with pytest.raises(TypeError, match=r"gate 'when' of node 'D'.*bool"):
        Workflow.construct(IntGate(x=1))

    @workflow.define
    def CombinedGate(xs: list[int]) -> int:
        c = workflow.add(Check().split(x=xs).combine("x"), name="C")
        d = workflow.add(Inc(a=1), name="D", unless=c.out)
        return d.out

    with pytest.raises(TypeError, match=r"gate 'unless' of node 'D'.*bool"):
        Workflow.construct(CombinedGate(xs=[1, 2]))

    @workflow.define
    def StaticGate(x: int) -> int:
        d = workflow.add(Inc(a=x), name="D", when="yes")
        return d.out

    with pytest.raises(TypeError, match=r"gate 'when' of node 'D'.*bool"):
        Workflow.construct(StaticGate(x=1))


# -- select(n) ---------------------------------------------------------------------------------------


def test_select_first_true_and_none(submitter):
    sel = select(2)
    assert select(2) is sel  # one class per arity
    assert (
        submitter(sel(when_0=False, value_0=1, when_1=True, value_1=2)).outputs.out == 2
    )
    assert (
        submitter(sel(when_0=True, value_0=1, when_1=True, value_1=2)).outputs.out == 1
    )
    none = submitter(sel(when_0=False, value_0=1, when_1=False, value_1=2)).outputs.out
    assert isinstance(none, Absent) and none.cause == "not-routed"
    # a route: gated branches merged on their gates
    assert submitter(Route(x=3)).outputs.out == 4
    assert submitter(Route(x=-3)).outputs.out == -4
    assert submitter(Route(x=0)).outputs.out.cause == "not-routed"


def test_select_pickles_and_reloads():
    """The callable module pickles by reference and survives a reload"""
    import importlib

    import cloudpickle as cp

    import pydra.compose.select as module

    assert cp.loads(cp.dumps(select)) is module
    assert cp.loads(cp.dumps(select(2))) is select(2)
    reloaded = importlib.reload(module)
    assert (
        cp.loads(cp.dumps(reloaded)) is reloaded
    )  # still callable, still by reference
    fields = set(attrs.fields_dict(reloaded(2)))
    assert {"when_0", "when_1", "value_0", "value_1"} <= fields


def test_select_passes_content_absent(submitter):
    sel = select(2)
    content = Absent(cause="empty", source="branch")
    out = submitter(
        sel(when_0=True, value_0=content, when_1=True, value_1=2)
    ).outputs.out
    assert out == content
    # a routed branch that returns a content Absent reaches the reader with its cause
    assert submitter(RouteContent(x=-1)).outputs.out.cause == "negative"
    assert submitter(RouteContent(x=5)).outputs.out == 5


# -- plan() ------------------------------------------------------------------------------------------


def test_plan_counts_gated_from_predicate(tmp_path):
    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        runner = runner_for(sub, Fanout(xs=[1, -1, -2]), store=store)
        rnd = runner.ready()
        assert [j.name for j in rnd.dispatch] == ["C", "C", "C"]
        for job in rnd.dispatch:
            runner.complete(job, run_inline(job))
        store.calls.clear()
        (d,) = [n for n in runner.plan().nodes if n.path == "D"]
        assert d.elements == 3
        assert d.gated == 2  # read from the predicate's results, not from D's slots
        assert d.absent == 0
        # only the gated-on element is looked up
        assert len([c for c in store.calls if c[0] == "lookup"]) == 1
        drive(runner)


# -- an external driver passing references --------------------------------------------


@attrs.frozen
class Ref:
    """A reference to an output held by the driver"""

    key: str


@python.define
def Pred(x: int) -> bool:
    raise AssertionError("the driver runs the jobs")


@python.define
def PredAny(x: int) -> ty.Any:
    raise AssertionError("the driver runs the jobs")


@python.define
def Step(a: ty.Any, b: ty.Any) -> ty.Any:
    raise AssertionError("the driver runs the jobs")


@workflow.define
def RefGated(x: int) -> ty.Any:
    p = workflow.add(Pred(x=x), name="P")
    a = workflow.add(Step(a=x, b=0), name="A")
    d = workflow.add(Step(a=a.out, b=1), name="D", when=p.out)
    e = workflow.add(Step(a=d.out, b=2), name="E")
    return e.out


@workflow.define
def RefInPredicate(x: int) -> ty.Any:
    p = workflow.add(PredAny(x=x), name="P")
    d = workflow.add(Step(a=x, b=1), name="D", when=p.out)
    return d.out


def complete_with_refs(job) -> Result:
    """Every output an opaque reference, except a predicate's real ``bool``"""
    if job.name == "P" and isinstance(job.task, Pred):
        out = job.task.x > 0
    else:
        out = Ref(job.name)
    return Result(
        cache_dir=job.cache_dir, outputs=job.task.Outputs(out=out), task=job.task
    )


def test_external_driver_real_bool_predicate_gates(submitter):
    on = drive(runner_for(submitter, RefGated(x=1)), complete=complete_with_refs)
    assert [j.name for j in dispatched(on)] == ["A", "P", "D", "E"]
    (d,) = dispatched(on, "D")
    assert d.task.a == Ref("A")  # the reference, as the driver gave it

    off_runner = runner_for(submitter, RefGated(x=-1))
    off = drive(off_runner, complete=complete_with_refs)
    assert [j.name for j in dispatched(off)] == ["A", "P"]
    assert [s.how for s in settled(off, "D")] == ["gated"]
    (e,) = settled(off, "E")
    assert e.how == "absent"
    assert e.result.outputs.out.cause == "not-routed"


def test_reference_in_predicate_position_is_wiring_error(submitter):
    runner = runner_for(submitter, RefInPredicate(x=1))
    (p,) = runner.ready().dispatch
    runner.complete(p, complete_with_refs(p))  # a reference, not a bool
    with pytest.raises(TypeError, match=r"gate 'when' of node 'D'.*Ref.*bool"):
        runner.ready()


def test_plan_places_only_gated_on_elements(tmp_path):
    from pydra.engine.tests.test_absent import AnyWorker, Placer

    store = CountingStore(tmp_path / "cache")
    with Submitter(worker="debug", store=store) as sub:
        runner = runner_for(sub, Fanout(xs=[1, -1, -2]), store=store)
        for job in runner.ready().dispatch:
            runner.complete(job, run_inline(job))
        placer = Placer()
        (d,) = [n for n in runner.plan(AnyWorker(placer)).nodes if n.path == "D"]
        drive(runner)
    assert d.gated == 2 and d.absent == 0
    ((shown,),) = placer.shown
    assert not shown._gated and shown._propagated_absent is None
