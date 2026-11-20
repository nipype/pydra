"""``GraphRunner.plan()``: what a run would execute, reuse or wait for, read from the
runner's graph and the store without running a job, taking a lease or writing a result.
"""

import copy
import typing as ty

import attrs

from pydra.utils.general import is_workflow

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.runner import GraphRunner, NodeExecution, _Frame
    from pydra.utils.resources import ComputeResources


@attrs.frozen
class PlanNode:
    """One node of the graph, as the runner would run it now

    Parameters
    ----------
    path : str
        The node's name, with the nested workflows that hold it (``"outer/inner/node"``);
        a workflow job whose lease is not yet held is named by its own path
    elements : int, optional
        How many elements the node releases, whatever a concurrency cap would hand out at
        once; None if that depends on an upstream output that has not been read
    resources : ComputeResources, optional
        What the node asks of its worker
    worker : str, optional
        The name of the worker the submitter's router sends the node's first element to.
        An element whose job cannot be formed yet is routed as a stand-in (name, task with
        lazy inputs, resources), and the answer is ``provisional``; a stand-in the router
        fails on gives None. A real job the router fails on raises, as the run would
    expected_hits : int
        How many of the node's unfinished elements have a stored result to reuse. The store
        may answer with an ``expected_hits(jobs)`` method; otherwise the records are
        counted, and the results ``lookup`` returns when the runner has
        ``lookup_before_dispatch``. A store with neither (an adapter) reports 0. Only
        elements whose inputs are resolvable now have a checksum and are counted.
    state : "pending" | "waiting" | "deferred"
        ``pending``: its elements are released or can be. ``waiting``: a workflow job
        whose lease another process holds. ``deferred``: what it holds is not known yet (a
        workflow job not yet expanded, a split over an upstream output)
    gated : int
        Elements a gate switches off; always 0 until gates exist
    placement : object, optional
        What the worker's executor's ``place(jobs)`` returned for the node's elements,
        when it has one (RFC 0005 section 7.4). Elements whose jobs cannot be formed yet are
        shown to ``place`` as the same stand-ins the router sees, so the answer is then
        ``provisional``
    provisional : bool
        Whether ``worker`` or ``placement`` was derived from a stand-in rather than from
        the jobs the run will dispatch, so a later plan may differ
    """

    path: str
    elements: int | None
    resources: "ComputeResources | None"
    worker: str | None
    expected_hits: int
    state: ty.Literal["pending", "waiting", "deferred"]
    gated: int = 0
    placement: object | None = None
    provisional: bool = False


@attrs.frozen
class Plan:
    """The nodes a run still has to do, in the order the runner releases them"""

    nodes: tuple[PlanNode, ...]


class _Stand:
    """What a router or ``place`` is shown for an element whose job cannot be formed yet:
    the node's name, its task (inputs still lazy), resources and no state index"""

    def __init__(self, node: "NodeExecution"):
        self.name = node.name
        self.task = node.node._task
        self.resources = node.node._resources or node.node._task._resources
        self.state_index = None


def _view(node: "NodeExecution", graph) -> "NodeExecution | None":
    """The node itself if it has started, else a copy that can say what its elements are
    (None if that waits on upstream elements). The copy has its own ``State``, so the live
    node, its ``State`` and its tables are never written."""
    if node._tasks is not None:
        return node
    preds = graph.predecessors[node.name]
    # the runner starts a node once its predecessors have started, or are done if its
    # splitter splits over an upstream output (``get_runnable_tasks``)
    if node._splits_upstream_output:
        if not all(p.done and not (p.errored or p.unrunnable) for p in preds):
            return None
    elif not all(p.started for p in preds):
        return None
    from pydra.engine.runner import NodeExecution

    twin = copy.copy(node.node)
    twin._state = copy.deepcopy(node.node._state)
    view = NodeExecution(twin, node.submitter, node.workflow, node.path)
    view.graph = node.graph
    try:
        view.start()
    except Exception:  # not resolvable yet: the element count is unknown
        return None
    return view


def _elements(view: "NodeExecution") -> "tuple[list[Job], int]":
    """The jobs of the unfinished elements of a node that exist or can be formed from
    upstream elements the runner has accepted, and the number of unfinished elements
    that cannot be formed yet. A job is formed only when every upstream element it reads
    is in the upstream node's ``successful``: a job handed out and not yet reported is
    never read, and nothing is kept on a live node."""
    from pydra.engine.runner import _has_succeeded

    finished = set(view.successful) | set(view.errored) | set(view.unrunnable)
    jobs: list = []
    unformed = 0
    for index, job in view._tasks.items():
        if index in finished:
            continue
        if job is None:
            upstream = (view.blocked or {}).get(index)
            if upstream is not None and all(_has_succeeded(p, i) for p, i in upstream):
                try:
                    job = view._new_job(index)
                except Exception:  # its inputs cannot be resolved: the run will say so
                    job = None
        if job is None:
            unformed += 1
        else:
            jobs.append(job)
    return jobs, unformed


def _hits(runner: "GraphRunner", jobs: "list[Job]") -> int:
    if runner._inner_rerun or not jobs:  # ``ready`` skips the store on a rerun
        return 0
    store = runner.store
    hook = getattr(store, "expected_hits", None)
    if hook is not None:
        return int(hook(jobs))
    record = getattr(store, "success_record", None)
    count = 0
    for job in jobs:
        if record is not None and _record_exists(record(job.checksum, job.context)):
            count += 1
        elif runner.lookup_before_dispatch and runner._lookup(job) is not None:
            count += 1
    return count


def _record_exists(path) -> bool:
    from pydra.engine import store

    return store._record_exists(path)


def build_plan(
    runner: "GraphRunner",
    executors: "ty.Mapping[str, ty.Any] | None" = None,
) -> Plan:
    """The plan of a runner; see :meth:`GraphRunner.plan`"""
    if executors is None:
        executors = runner.submitter.workers
    nodes: list[PlanNode] = []
    for frame in runner._frames:
        if frame.state == "leasing":
            nodes.append(_leasing(frame))
        elif frame.state == "expanding" and frame.graph is not None:
            for node in frame.graph.sorted_nodes:
                entry = _node(runner, frame, node, executors)
                if entry is not None:
                    nodes.append(entry)
    return Plan(tuple(nodes))


def _leasing(frame: "_Frame") -> PlanNode:
    path = "/".join(frame.path) or frame.job.name
    return PlanNode(
        path=path,
        elements=1,
        resources=frame.job.resources,
        worker=None,
        expected_hits=0,
        state="waiting" if frame.retry_at is not None else "deferred",
        provisional=True,
    )


def _node(
    runner: "GraphRunner",
    frame: "_Frame",
    node: "NodeExecution",
    executors: "ty.Mapping[str, ty.Any]",
) -> PlanNode | None:
    if node.done:
        return None
    view = _view(node, frame.graph)
    jobs, unformed = _elements(view) if view is not None else ([], 0)
    workflow_node = is_workflow(node.node._task)
    if workflow_node and view is not None:
        if any(
            j in runner._child or j in runner._handed for j in view._tasks.values() if j
        ):
            return None  # entered: its own nodes are planned, or it is settled
    elements = len(view._tasks) if view is not None else (None if node.state else 1)
    resources = node.node._resources or node.node._task._resources
    unhanded = [j for j in jobs if j not in runner._handed]
    stand = _Stand(node)
    worker = None
    provisional = False
    if not workflow_node:
        real = jobs[0] if jobs else None
        provisional = real is None
        try:
            worker = runner.submitter.route_name(real or stand)
        except Exception:
            if real is not None:  # a misrouted job is the run's failure too
                raise
    placement = None
    place = getattr(
        executors.get(worker) if worker is not None else None, "place", None
    )
    if place is not None:
        shown: list = unhanded + [stand] * (unformed or (0 if view is not None else 1))
        if shown:
            placement = place(shown)
            provisional = provisional or len(shown) > len(unhanded)
    deferred = elements is None or workflow_node
    return PlanNode(
        path="/".join((*node.path, node.name)),
        elements=elements,
        resources=resources,
        worker=worker,
        expected_hits=_hits(runner, unhanded),
        state="deferred" if deferred else "pending",
        placement=placement,
        provisional=provisional,
    )
