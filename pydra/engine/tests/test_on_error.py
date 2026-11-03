"""Error propagation: an errored element marks only the downstream elements that read
it unrunnable, its siblings and their descendants complete, and the Submitter's
``on_error`` mode decides whether the run stops at the first error or raises them all
at the end"""

import pickle
import typing as ty
import pytest
from pydra.compose import python, workflow
from pydra.engine.submitter import Submitter
from pydra.engine.workflow import Workflow
from pydra.engine.tests.utils import FunAddVar, Identity, RaiseXeq1
from pydra.workers.debug import DebugWorker


class SyncWorker(DebugWorker):
    """A synchronous worker that is not *debug*"""

    _plugin_name = "sync"


@python.define
def Ratio(a: int, b: int) -> float:
    return a / (b - a)


@workflow.define(outputs=["c", "t"])
def Propagate(x, y):
    a = workflow.add(RaiseXeq1().split("x", x=x), name="A")
    b = workflow.add(FunAddVar(a=a.out).split("b", b=y).combine("b"), name="B")
    c = workflow.add(Identity(x=b.out), name="C")
    s = workflow.add(FunAddVar(b=0).split("a", a=y), name="S")
    t = workflow.add(Identity(x=s.out), name="T")
    return c.out, t.out


@workflow.define
def PartialGroup(x, y):
    a = workflow.add(Ratio().split(["a", "b"], a=x, b=y).combine("a"), name="A")
    b = workflow.add(Identity(x=a.out), name="B")
    return b.out


# Each case: the workflow, the error it raises, and for each node the successful
# elements, the errored elements and the unrunnable elements with the upstream nodes
# that made them so (worked out by hand from the splitters).
CASES = {
    # x=1 raises. B[2i+j] reads A[i] and C[i] reads the group B{2i, 2i+1}, so A[1] marks
    # B[2], B[3] and C[1]; S and T do not read A
    "chain": (
        Propagate(x=[2, 1, 3], y=[10, 20]),
        "i'm raising an exception!",
        {
            "A": ([0, 2], [1], {}),
            "B": ([0, 1, 4, 5], [], {2: ["A"], 3: ["A"]}),
            "C": ([0, 2], [], {1: ["B"]}),
            "S": ([0, 1], [], {}),
            "T": ([0, 1], [], {}),
        },
    ),
    # A[2a+b] for a in [1, 2], b in [2, 3]; only (2, 2) divides by zero. Combining over
    # "a" groups A{0, 2} (b=2), which partly failed, and A{1, 3} (b=3)
    "partial-group": (
        PartialGroup(x=[1, 2], y=[2, 3]),
        "division by zero",
        {
            "A": ([0, 1, 3], [2], {}),
            "B": ([1], [], {0: ["A"]}),
        },
    ),
}


def outcome(graph):
    return {
        node.name: (
            sorted(node.successful),
            sorted(node.errored),
            {i: [p.name for p in ps] for i, ps in node.unrunnable.items()},
        )
        for node in graph.nodes
    }


def test_on_error_marks_descendants(tmp_path):
    """An errored element marks the downstream elements that read it, and their own
    descendants, unrunnable as soon as it errors, while the elements that read its
    siblings are released as those complete"""
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        wf = Workflow.construct(Propagate(x=[2, 1, 3], y=[10, 20]))
        graph = wf.execution_graph(submitter=sub)
        a, b, c = graph["A"], graph["B"], graph["C"]
        jobs = a.get_runnable_tasks(graph)
        with pytest.raises(Exception, match="i'm raising an exception!"):
            jobs[1].run()
        a.update_status()
        assert b.get_runnable_tasks(graph) == []
        assert b.unrunnable == {2: [a], 3: [a]}
        assert sorted(b.blocked) == [0, 1, 4, 5]

        jobs[0].run()
        a.update_status()
        released = b.get_runnable_tasks(graph)
        assert [j.state_index for j in released] == [0, 1]
        for job in released:
            job.run()
        b.update_status()
        released = c.get_runnable_tasks(graph)
        assert [j.state_index for j in released] == [0]
        assert c.unrunnable == {1: [b]}
        assert sorted(c.blocked) == [2]
        assert released[0].run().outputs.out == [12, 22]

        jobs[2].run()
        a.update_status()
        for job in b.get_runnable_tasks(graph):
            job.run()
        b.update_status()
        released = c.get_runnable_tasks(graph)
        assert [j.state_index for j in released] == [0, 2]
        assert released[1].run().outputs.out == [13, 23]
        assert a.done and b.done and c.done
        assert c.unrunnable == {1: [b]}


@pytest.mark.timeout(120)
@pytest.mark.parametrize("case", CASES)
def test_on_error_raise_at_end_debug_parity(case, worker, tmp_path):
    """With ``on_error="raise_at_end"`` every element that does not read an errored
    one runs, and the errors are raised together at the end, under debug as under
    each worker"""
    task, error, expected = CASES[case]
    outcomes = []
    for name, cache in (("debug", "reference"), (worker, "worker")):
        with Submitter(
            worker=name, on_error="raise_at_end", cache_root=tmp_path / cache
        ) as sub:
            result = sub(task, raise_errors=False)
        assert result.errored
        assert error in "".join(sub.errors)
        message = "".join(result.errors["error message"])
        assert "failed with errors" in message
        assert error in message
        outcomes.append(outcome(result.job.return_values["exec_graph"]))
    reference, outputs = outcomes
    assert reference == expected
    assert outputs == reference


@pytest.mark.timeout(120)
def test_on_error_fail_fast(tmp_path):
    """With ``on_error="fail_fast"`` the cf worker raises the first error without
    waiting for the elements still running: A[2] here only finishes once a release
    marker is written, which happens after the Submitter has returned"""

    @python.define
    def Hold(x: int, release: str) -> int:
        import time
        from pathlib import Path

        if x == 1:
            raise Exception("x is 1, so i'm raising an exception!")
        if x == 2:
            deadline = time.monotonic() + 60
            while not Path(release).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{release} was not written")
                time.sleep(0.1)
        return x

    @workflow.define
    def FailFast(xs, release):
        a = workflow.add(Hold(release=release).split(x=xs), name="A")
        b = workflow.add(Identity(x=a.out), name="B")
        return b.out

    release = tmp_path / "release"
    task = FailFast(xs=[0, 1, 2], release=str(release))
    sub = Submitter(
        worker="cf", n_procs=3, on_error="fail_fast", cache_root=tmp_path / "cache"
    )
    try:
        with pytest.raises(RuntimeError, match="i'm raising an exception!") as excinfo:
            sub(task, raise_errors=True)
        assert "was not written" not in str(excinfo.value)
        assert not release.exists()
    finally:
        release.touch()
        sub.close()


@pytest.mark.timeout(120)
def test_on_error_raise_at_end_with_job_in_flight_cf(tmp_path):
    """With ``on_error="raise_at_end"`` under two cf processes, an element errors while
    a sibling is still running: the sibling only returns once the failure has been
    recorded, so the two overlap, and it still completes before the error is raised"""

    @python.define
    def Overlap(x: int, sync: str) -> int:
        import time
        from pathlib import Path

        sync = Path(sync)
        (sync / f"started_{x}").touch()

        def wait_for(name):
            deadline = time.monotonic() + 60
            while not (sync / name).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} never appeared")
                time.sleep(0.05)

        if x == 1:
            wait_for("started_0")
            (sync / "failing").touch()
            raise Exception("x is 1, so i'm raising an exception!")
        wait_for("failing")
        (sync / "finished_0").touch()
        return x

    @workflow.define
    def InFlight(xs, sync):
        a = workflow.add(Overlap(sync=sync).split(x=xs), name="A")
        b = workflow.add(Identity(x=a.out), name="B")
        return b.out

    from pydra.engine import WorkflowError

    sync = tmp_path / "sync"
    sync.mkdir()
    with Submitter(
        worker="cf", n_procs=2, on_error="raise_at_end", cache_root=tmp_path / "cache"
    ) as sub:
        with pytest.raises(WorkflowError, match="failed with errors") as excinfo:
            sub(InFlight(xs=[0, 1], sync=str(sync)), raise_errors=True)
    assert len(excinfo.value.errors) == 1
    assert "i'm raising an exception!" in excinfo.value.errors[0]
    assert (sync / "finished_0").exists()


@pytest.mark.parametrize(
    "plugin,on_error",
    [("debug", "fail_fast"), ("cf", "raise_at_end"), (SyncWorker, "fail_fast")],
)
def test_on_error_default(plugin, on_error, tmp_path):
    """The default keeps each worker's existing error behaviour: synchronous workers
    stop at the first error, asynchronous ones raise at the end"""
    with Submitter(worker=plugin, cache_root=tmp_path) as sub:
        assert sub.on_error == on_error


def test_on_error_invalid(tmp_path):
    with pytest.raises(ValueError, match="on_error"):
        Submitter(on_error="ignore", cache_root=tmp_path)


def test_on_error_fail_fast_debug_raises_original(tmp_path):
    """Under debug, ``fail_fast`` raises the errored element's own exception"""
    task = Propagate(x=[2, 1, 3], y=[10, 20])
    with pytest.raises(Exception, match="i'm raising an exception!") as excinfo:
        task(worker="debug", on_error="fail_fast", cache_root=tmp_path)
    assert "failed with errors" not in str(excinfo.value)


@python.define
def Stringify(x: int) -> ty.Any:
    return "one" if x == 1 else x


@python.define
def Mark(a: int, marks: str) -> int:
    from pathlib import Path

    (Path(marks) / str(a)).touch()
    return a


@workflow.define
def BadInput(xs, marks):
    a = workflow.add(Stringify().split("x", x=xs), name="A")
    b = workflow.add(Mark(marks=marks).split("a", a=a.out), name="B")
    return b.out


def test_workflow_error_is_typed():
    from pydra.engine import WorkflowError

    error = pickle.loads(pickle.dumps(WorkflowError("failed", ["one", "two"])))
    assert isinstance(error, RuntimeError)
    assert (str(error), error.errors) == ("failed", ["one", "two"])


@pytest.mark.timeout(120)
def test_on_error_raises_workflow_error_cf(tmp_path):
    """Under several cf processes the workflow raises a WorkflowError holding one entry
    per errored job, which the Submitter keeps for the last call only"""
    from pydra.engine import WorkflowError

    task = Propagate(x=[2, 1, 3, 1], y=[10, 20])
    with Submitter(worker="cf", n_procs=4, cache_root=tmp_path) as sub:
        with pytest.raises(WorkflowError) as excinfo:
            sub(task, raise_errors=True)
        assert len(excinfo.value.errors) == 1
        assert sub.errors == excinfo.value.errors
        assert all("i'm raising an exception!" in e for e in sub.errors)
        sub(Identity(x=1), raise_errors=True)
        assert sub.errors == []


@pytest.mark.timeout(120)
@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_on_error_unresolvable_input_raise_at_end(worker, tmp_path):
    """An element whose inputs cannot be resolved is an error of the run, raised at the
    end after its siblings have run"""
    from pydra.engine import WorkflowError

    marks = tmp_path / "marks"
    marks.mkdir()
    task = BadInput(xs=[0, 1, 2, 3], marks=str(marks))
    with Submitter(
        worker=worker, on_error="raise_at_end", cache_root=tmp_path / "cache"
    ) as sub:
        with pytest.raises(WorkflowError) as excinfo:
            sub(task, raise_errors=True)
    assert len(excinfo.value.errors) == 1
    assert "B(1)" in excinfo.value.errors[0]
    assert sorted(p.name for p in marks.iterdir()) == ["0", "2", "3"]


@pytest.mark.timeout(120)
def test_on_error_unresolvable_input_fail_fast(worker, tmp_path):
    from pydra.engine import WorkflowError

    marks = tmp_path / "marks"
    marks.mkdir()
    task = BadInput(xs=[0, 1, 2], marks=str(marks))
    with Submitter(worker=worker, on_error="fail_fast", cache_root=tmp_path) as sub:
        with pytest.raises(WorkflowError, match=r"B\(1\)"):
            sub(task, raise_errors=True)


def test_on_error_debug_loop_wraps_submit(tmp_path, monkeypatch):
    """The debug loop catches errors from ``Submitter.submit``, where jobs are placed,
    so that ``raise_at_end`` collects them wherever the job was routed"""
    from pydra.engine import WorkflowError

    submit = Submitter.submit

    def failing(self, job, *args, **kwargs):
        if job.name == "S":
            raise ValueError("could not be placed")
        return submit(self, job, *args, **kwargs)

    monkeypatch.setattr(Submitter, "submit", failing)
    task = Propagate(x=[2, 3], y=[10, 20])
    with Submitter(worker="debug", on_error="raise_at_end", cache_root=tmp_path) as sub:
        with pytest.raises(WorkflowError) as excinfo:
            sub(task, raise_errors=True)
    assert len(excinfo.value.errors) == 2
    assert all("could not be placed" in e for e in excinfo.value.errors)


@pytest.mark.parametrize("worker,is_async", [("debug", True), ("cf", False)])
def test_on_error_default_follows_the_loop(worker, is_async, tmp_path):
    """The default is that of the loop expanding the workflow, which is what a router
    choosing a different worker per workflow needs, not the default worker's"""
    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        assert sub._loop_on_error(is_async) == (
            "raise_at_end" if is_async else "fail_fast"
        )
    with Submitter(worker=worker, on_error="fail_fast", cache_root=tmp_path) as sub:
        assert sub._loop_on_error(is_async) == "fail_fast"
