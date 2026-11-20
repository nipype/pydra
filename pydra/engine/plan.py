"""``GraphRunner.plan()``: what a run would execute, reuse or wait for, read from the
runner's graph and the store without running a job, taking a lease or writing a result.
"""

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
        The name of the worker the submitter's router sends the node's first element to
        (None if it cannot be told yet)
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
        when it has one (RFC 0005 section 7.4)
    """

    path: str
    elements: int | None
    resources: "ComputeResources | None"
    worker: str | None
    expected_hits: int
    state: ty.Literal["pending", "waiting", "deferred"]
    gated: int = 0
    placement: object | None = None


@attrs.frozen
class Plan:
    """The nodes a run still has to do, in the order the runner releases them"""

    nodes: tuple[PlanNode, ...]


class _Stand:
    """What a router is shown for a node whose jobs cannot be formed yet"""

    def __init__(self, node: "NodeExecution"):
        self.name = node.name
        self.task = node.node._task
        self.resources = node.node._resources or node.node._task._resources
        self.state_index = None


def _probe(node: "NodeExecution", graph) -> "NodeExecution | None":
    """The node itself if it has started, else a copy whose elements can be formed now (no
    upstream output unread), else None. A copy leaves the runner's graph as it was."""
    if node._tasks is not None:
        return node
    preds = graph.predecessors[node.name]
    # the runner starts a node once its predecessors have started, or are done if its
    # splitter splits over an upstream output (``get_runnable_tasks``)
    if not all(p.done if node._splits_upstream_output else p.started for p in preds):
        return None
    from pydra.engine.runner import NodeExecution

    copy = NodeExecution(node.node, node.submitter, node.workflow, node.path)
    copy.graph = node.graph
    try:
        copy.start()
    except Exception:  # not resolvable yet: the element count is unknown
        return None
    return copy


def _jobs(probe: "NodeExecution", runner: "GraphRunner") -> "list[Job]":
    """The jobs of the elements of a node that are unfinished and not yet handed out, as
    far as their inputs can be resolved"""
    done = set(probe.successful) | set(probe.errored) | set(probe.unrunnable)
    jobs = []
    for index in probe._tasks:
        if index in done:
            continue
        try:
            job = probe.job(index)
        except Exception:  # an upstream element it reads has not completed
            continue
        if job not in runner._handed:
            jobs.append(job)
    return jobs


def _hits(runner: "GraphRunner", jobs: "list[Job]") -> int:
    store = runner.store
    hook = getattr(store, "expected_hits", None)
    if hook is not None:
        return int(hook(jobs))
    record = getattr(store, "success_record", None)
    count = 0
    for job in jobs:
        if record is not None:
            from pydra.engine import store as store_module

            try:
                if store_module._record_exists(record(job.checksum, job.context)):
                    count += 1
                    continue
            except Exception:
                pass
        if runner.lookup_before_dispatch and runner._lookup(job) is not None:
            count += 1
    return count


def _route(runner: "GraphRunner", rep) -> str | None:
    try:
        return runner.submitter.route_name(rep)
    except Exception:
        return None


def build_plan(
    runner: "GraphRunner",
    executors: "ty.Mapping[str, ty.Any] | None" = None,
) -> Plan:
    """The plan of a runner; see :meth:`GraphRunner.plan`"""
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
    )


def _node(
    runner: "GraphRunner",
    frame: "_Frame",
    node: "NodeExecution",
    executors: "ty.Mapping[str, ty.Any] | None",
) -> PlanNode | None:
    if node.done:
        return None
    probe = _probe(node, frame.graph)
    jobs = _jobs(probe, runner) if probe is not None else []
    workflow_node = is_workflow(node.node._task)
    if workflow_node and any(
        j in runner._child or j in runner._handed
        for j in (probe._tasks.values() if probe else ())
        if j is not None
    ):
        return None  # entered: its own nodes are planned, or it is settled
    if probe is not None:
        elements: int | None = len(probe._tasks)
    else:
        elements = None if node.state else 1
    resources = node.node._resources or node.node._task._resources
    rep = (
        jobs[0]
        if jobs
        else (next((j for j in (probe._tasks.values() if probe else ()) if j), None))
    )
    worker = None if workflow_node else _route(runner, rep or _Stand(node))
    placement = None
    executor = (executors or {}).get(worker) if worker is not None else None
    place = getattr(executor, "place", None)
    if place is not None and jobs:
        placement = place(jobs)
    deferred = elements is None or workflow_node
    return PlanNode(
        path="/".join((*node.path, node.name)),
        elements=elements,
        resources=resources,
        worker=worker,
        expected_hits=_hits(runner, jobs) if jobs else 0,
        state="deferred" if deferred else "pending",
        placement=placement,
    )
