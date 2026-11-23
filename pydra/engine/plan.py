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
        The node's name with the nested workflows that hold it (``"outer/inner/node"``)
    elements : int, optional
        How many elements the node releases; None while that depends on an upstream output
        (see ``known_elements``)
    resources : ComputeResources, optional
        What the node asks of its worker
    worker : str, optional
        The worker the router sends the node's first element to; None if the router fails
        on a stand-in for an element whose job cannot be formed yet
    expected_hits : int
        How many unfinished elements with a checksum have a stored result to reuse, as the
        store answers (``expected_hits`` or ``recorded``; 0 for a store with neither)
    state : "pending" | "waiting" | "deferred" | "absent" | "looping"
        Released or releasable; a workflow job whose lease another process holds; not
        known yet (a workflow job not expanded, a split over an upstream output); or absent
        as a whole (RFC 0002 section 2.3(a)), with no elements
        ``looping``: an entered loop (RFC 0002 section 2.3(d)) and its live rounds, whose
        total is never predicted (``elements`` is None)
    gated : int
        Elements a gate switches off (RFC 0002 section 2.3(b)), counted from the
        predicate's result; they are not looked up in the store, routed or placed
    absent : int
        Elements that complete absent because a propagating input, or a gate, holds an
        ``Absent`` (RFC 0002 section 2.3(a)); they are not looked up in the store, routed
        or placed
    placement : object, optional
        What the worker's executor's ``place(jobs)`` returned (RFC 0005 section 7.4)
    provisional : bool
        Whether ``worker`` or ``placement`` came from a stand-in, so a later plan may differ
    known_elements : int, optional
        How many elements are known now: ``elements`` when known; on a partial node, those
        of the blocks expanded so far; None when nothing can be read yet
    rounds : int, optional
        For a ``looping`` entry, the rounds of the loop that are done
    max_rounds : int, optional
        For a ``looping`` entry, the loop's ``max_rounds``, None if it has none
    """

    path: str
    elements: int | None
    resources: "ComputeResources | None"
    worker: str | None
    expected_hits: int
    state: ty.Literal["pending", "waiting", "deferred", "absent", "looping"]
    gated: int = 0
    absent: int = 0
    placement: object | None = None
    provisional: bool = False
    known_elements: int | None = None
    rounds: int | None = None
    max_rounds: int | None = None


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
    """The node itself if it has started and is not partial, else a copy that can say
    what its elements are (None if that waits on upstream elements). The copy has its own
    ``State`` and, for a partial node, its own blocks: it expands every block whose upstream
    elements are in the upstream nodes' ``successful``, as the next ``ready`` would; its
    element keys are the live node's.

    No job table of a live node is written and no in-flight result is read. Reading a live
    partial node may advance its derived bookkeeping (blocks, groups, its seal), exactly as
    the next ``ready()`` would."""
    if node._tasks is not None and not node.is_partial:
        return node
    preds = graph.predecessors[node.name]
    # the runner classifies a node once its predecessors have started; a barrier node
    # starts once they are done (``get_runnable_tasks``)
    if not all(p.started for p in preds) or any(
        p._node_error is not None or (p._tasks is None and p.unrunnable) for p in preds
    ):
        return None
    from pydra.engine.runner import NodeExecution

    twin = copy.copy(node.node)
    twin._state = copy.deepcopy(node.node._state)
    view = NodeExecution(twin, node.submitter, node.workflow, node.path)
    view.graph = node.graph
    if view._inherit_absent():  # absent as a whole, as the runner would make it
        return view
    try:
        mode = view._classify()
        if mode == "error":
            return None
        if mode == "barrier":
            if not all(p.done and not (p.errored or p.unrunnable) for p in preds):
                return None
            view.start()
        elif mode == "partial":
            view._partial.start()
            view._partial.expand_ready()
        else:
            view.start()
    except Exception:  # not resolvable yet: the element count is unknown
        return None
    return view


def _elements(view: "NodeExecution", live: "NodeExecution") -> "tuple[list[Job], int]":
    """The jobs of the unfinished elements of a node that exist or can be formed from
    upstream elements the runner has accepted, and the number of unfinished elements
    that cannot be formed yet. A job is formed only when every upstream element it reads
    is in the upstream node's ``successful``: a job handed out and not yet reported is
    never read, and nothing is kept on a live node. ``live`` is the node itself, whose
    tables say which elements are finished and which jobs exist when ``view`` is a copy
    of a started partial node."""
    from pydra.engine.runner import _has_succeeded

    finished = set(live.successful) | set(live.errored) | set(live.unrunnable)
    existing = live._tasks or {}
    jobs: list = []
    unformed = 0
    for index, job in view._tasks.items():
        if index in finished:
            continue
        job = existing.get(index, job)
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


def _hits(runner: "GraphRunner", jobs: "list[Job]", listed: dict) -> int:
    """How many of the jobs ``ready`` would settle from the store: those with a success
    record and, under ``lookup_before_dispatch``, those ``lookup`` returns a result for.
    A store's own ``expected_hits(jobs)`` answers for all of them, unless it has the finer
    ``recorded`` (``PickleStore``), which names the recorded jobs so the rest can be looked
    up."""
    if runner._inner_rerun or not jobs:  # ``ready`` skips the store on a rerun
        return 0
    store = runner.store
    have = store.recorded(jobs, listed)
    if have is not None:
        rest = (
            [j for j in jobs if j not in have] if runner.lookup_before_dispatch else []
        )
        return len(have) + sum(runner._lookup(j) is not None for j in rest)
    hits = store.expected_hits(jobs)
    if hits is not None:
        return hits
    if not runner.lookup_before_dispatch:
        return 0
    return sum(runner._lookup(job) is not None for job in jobs)


def build_plan(
    runner: "GraphRunner",
    executors: "ty.Mapping[str, ty.Any] | None" = None,
) -> Plan:
    """The plan of a runner; see :meth:`GraphRunner.plan`"""
    if executors is None:
        executors = runner.submitter.workers
    nodes: list[PlanNode] = []
    listed: dict = {}  # record directories listed so far, once per plan
    for frame in runner._frames:
        if frame.state == "leasing":
            nodes.append(_leasing(frame))
        elif frame.state == "expanding" and frame.graph is not None:
            if frame.loop is not None:
                nodes.append(_looping(frame))
            for node in frame.graph.sorted_nodes:
                entry = _node(runner, frame, node, executors, listed)
                if entry is not None:
                    nodes.append(entry)
    return Plan(tuple(nodes))


def _looping(frame: "_Frame") -> PlanNode:
    """The entry of an entered loop: the rounds done so far, never a predicted total;
    no round is instantiated"""
    from pydra.engine.loop import loop_label

    loop = frame.loop
    return PlanNode(
        path=loop_label(frame),
        elements=None,
        resources=frame.job.resources if frame.job is not None else None,
        worker=None,
        expected_hits=0,
        state="looping",
        rounds=loop.rounds - (0 if loop.last.done else 1),
        max_rounds=loop.spec.max_rounds,
    )


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
    listed: dict,
) -> PlanNode | None:
    if node.done:
        return None
    view = _view(node, frame.graph)
    if view is not None and view.absent is not None:
        return PlanNode(
            path="/".join((*node.path, node.name)),
            elements=0,
            resources=node.node._resources or node.node._task._resources,
            worker=None,
            expected_hits=0,
            state="absent",
            known_elements=0,
        )
    jobs, unformed = _elements(view, node) if view is not None else ([], 0)
    # elements completing absent or gated off are counted, and never routed, placed or
    # looked up; a gate is read from the predicate's result, as the run reads it, never
    # from the gated job's store slot
    gated = sum(j._gated for j in jobs)
    absent = sum(j._propagated_absent is not None and not j._gated for j in jobs)
    jobs = [j for j in jobs if j._propagated_absent is None]
    workflow_node = is_workflow(node.node._task)
    if workflow_node and view is not None:
        if any(
            j in runner._child or j in runner._handed
            for j in (node._tasks or {}).values()
            if j
        ):
            return None  # entered: its own nodes are planned, or it is settled
    if view is None:
        elements = known = None if node.state else 1
    else:
        known = len(view._tasks)
        elements = None if view.is_partial and not view.settled else known
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
    target = executors.get(worker) if worker is not None else None
    if target is not None:
        shown: list = unhanded + [stand] * (unformed or (0 if view is not None else 1))
        if shown:
            placement = target.place(shown)
            if placement is not None:
                provisional = provisional or len(shown) > len(unhanded)
    deferred = workflow_node or (elements is None and not known)
    return PlanNode(
        path="/".join((*node.path, node.name)),
        elements=elements,
        resources=resources,
        worker=worker,
        expected_hits=_hits(runner, unhanded, listed),
        state="deferred" if deferred else "pending",
        gated=gated,
        absent=absent,
        placement=placement,
        provisional=provisional,
        known_elements=known,
    )
