"""Runtime while-loops (RFC 0002 section 2.3(d)): the parameters of a loop and the
constructor of its round ``r0``.

A loop is a workflow job whose graph the runner grows one round at a time (see
:mod:`pydra.engine.runner`): round ``r<k+1>`` is added only once round ``r<k>`` is
done and its ``until`` output is false.
"""

import typing as ty

import attrs

from pydra.utils.general import get_fields
from pydra.utils.hash import hash_single
from pydra.engine.progress import UNSET, changed

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.hooks import TaskHooks
    from pydra.engine.runner import GraphRunner, NodeExecution, _Frame
    from pydra.environments.base import Environment
    from pydra.utils.resources import ComputeResources

#: The outputs a loop adds to those of its body
LOOP_OUTPUTS = ("rounds", "converged", "stopped")


@attrs.frozen
class LoopSpec:
    """The parameters of a loop

    Attributes
    ----------
    body : type[Task]
        The task class of one round
    until : str
        The ``bool`` output of ``body`` that ends the loop when true
    carry : tuple[tuple[str, str], ...]
        ``(input, output)`` pairs, sorted: each input of round ``k+1`` reads that output
        of round ``k``
    index : str | None
        The input of ``body`` given the round number, if any
    max_rounds : int | None
        The most rounds instantiated, None for no bound
    environment, hooks, resources
        Given to each round as to a node added to a workflow
    """

    body: "type[Task]"
    until: str
    carry: tuple[tuple[str, str], ...]
    index: str | None = None
    max_rounds: int | None = None
    environment: "Environment | None" = attrs.field(default=None, eq=False)
    hooks: "TaskHooks | None" = attrs.field(default=None, eq=False)
    resources: "ComputeResources | None" = attrs.field(default=None, eq=False)

    @property
    def body_inputs(self) -> list[str]:
        """The inputs of ``body``, without the executor"""
        executor = self.body._executor_name
        return [f.name for f in get_fields(self.body) if f.name != executor]

    @property
    def body_outputs(self) -> list[str]:
        return [f.name for f in get_fields(self.body.Outputs)]

    def hash_key(self) -> tuple:
        """What of the loop enters the loop job's checksum: the body's definition and
        every parameter that changes what the loop computes (hooks and resources do
        not)"""
        return (
            self.body,
            self.until,
            self.carry,
            self.index,
            self.max_rounds,
            self.environment,
        )


class LoopConstructor:
    """The constructor of a loop's workflow: adds round ``r0``, given the loop's inputs.
    It is hashed as the loop's :class:`LoopSpec`, so the loop job's checksum covers the
    body's definition and the loop's parameters."""

    def __init__(self, spec: LoopSpec):
        self.spec = spec

    def __repr__(self) -> str:
        return f"LoopConstructor({self.spec.body.__name__}, until={self.spec.until!r})"

    def __bytes_repr__(self, cache) -> ty.Iterator[bytes]:
        yield b"pydra.loop("
        yield bytes(hash_single(self.spec.hash_key(), cache))
        yield b")"

    def __call__(self, **inputs: ty.Any) -> tuple:
        from pydra.compose.workflow import this

        spec = self.spec
        workflow = this()
        given = {
            n: inputs[n]
            for n in spec.body_inputs
            if n in inputs and inputs[n] is not attrs.NOTHING
        }
        if spec.index is not None:
            given[spec.index] = 0
        first = workflow.add(
            spec.body(**given),
            name="r0",
            environment=spec.environment,
            hooks=spec.hooks,
            resources=spec.resources,
        )
        # placeholders: the loop's outputs are read from its last round when it stops
        return tuple(getattr(first, n) for n in spec.body_outputs) + (0, False, "")


class LoopState:
    """The runner's state of one loop instance, kept on the frame of its job

    Attributes
    ----------
    spec : LoopSpec
        The loop's parameters
    rounds : int
        The rounds instantiated so far
    last : NodeExecution
        The newest round
    prev : NodeExecution | None
        The round before it
    seen : set[str]
        The checksums of the completed rounds that did not end the loop, kept only when
        ``spec.index`` is None
    until : bool | Absent | None
        The ``until`` value last read, None before the first round is done
    stopped : str | None
        Why the loop ended: ``"until"``, ``"max-rounds"``, ``"cycle"``, ``"absent"`` or
        ``"error"``; None while it is open
    advanced : int
        The rounds added during the current call of ``GraphRunner.ready``
    """

    def __setattr__(self, name, value):
        if name in ("rounds", "stopped") and self.__dict__.get(name, UNSET) != value:
            changed()  # progress for `GraphRunner.ready`
        super().__setattr__(name, value)

    def __init__(self, spec: LoopSpec, first: "NodeExecution"):
        self.spec = spec
        self.rounds = 1
        self.last = first
        self.prev: "NodeExecution | None" = None
        self.seen: set[str] = set()
        self.until: ty.Any = None
        self.stopped: str | None = None
        self.advanced = 0

    @property
    def ran(self) -> int:
        """The rounds instantiated, less a last round that completed absent by
        propagation (it did not run)"""
        job = (self.last._tasks or {}).get(None)
        absent = job is not None and job._propagated_absent is not None
        return self.rounds - int(absent)


def loop_label(frame: "_Frame") -> str:
    """The path of a loop's job, its element index or key included"""
    if frame.path:
        return "/".join(frame.path)
    if frame.job is not None:
        return frame.job.name
    return type(frame.workflow.inputs).__name__


def read_until(frame: "_Frame") -> ty.Any:
    """The ``until`` output of a loop's newest round, read as a gate is read: a
    ``bool`` or an ``Absent``

    Raises
    ------
    GateTypeError
        It resolved to anything else, such as a reference an external driver completed
        the round with
    """
    from pydra.engine.absent import Absent
    from pydra.engine.runner import GateTypeError

    loop = frame.loop
    field = getattr(loop.last.node.lzout, loop.spec.until)
    value = field._get_value(workflow=frame.workflow, graph=frame.graph)
    if not isinstance(value, (bool, Absent)):
        raise GateTypeError(
            f"The until output {loop.spec.until!r} of loop {loop_label(frame)!r} "
            f"resolved to a {type(value).__name__} ({value!r}) in round "
            f"{loop.last.name!r}: it must be a bool or an Absent, not a reference to one"
        )
    return value


def decide(loop: LoopState, until: ty.Any, checksum: str) -> str | None:
    """Why the loop ends after its newest round, None if another round follows. The
    tests are taken in order: an ``Absent`` until, a true until, a round whose checksum
    an earlier round of the loop had (under ``index=None`` the loop's state repeats from
    there, so until can never become true; a fixed point is the cycle of length one),
    and ``max_rounds``"""
    from pydra.engine.absent import Absent

    if isinstance(until, Absent):
        return "absent"
    if until:
        return "until"
    if loop.spec.index is None and checksum in loop.seen:
        return "cycle"
    if loop.spec.max_rounds is not None and loop.rounds >= loop.spec.max_rounds:
        return "max-rounds"
    return None


def instantiate(runner: "GraphRunner", frame: "_Frame") -> "NodeExecution":
    """Add the next round to a loop's graph, wired lazily to the newest round's outputs,
    and return it"""
    from pydra.engine.runner import NodeExecution
    from pydra.utils.general import attrs_values

    loop = frame.loop
    spec = loop.spec
    workflow = frame.workflow
    last = loop.last
    k = loop.rounds
    given = attrs_values(workflow.inputs)
    inputs = {
        n: given[n]
        for n in spec.body_inputs
        if n in given and given[n] is not attrs.NOTHING
    }
    for inp, outp in spec.carry:
        inputs[inp] = getattr(last.node.lzout, outp)
    if spec.index is not None:
        inputs[spec.index] = k
    outputs = workflow.add(
        spec.body(**inputs),
        name=f"r{k}",
        environment=spec.environment,
        hooks=spec.hooks,
        resources=spec.resources,
    )
    new = NodeExecution(
        workflow[f"r{k}"], runner.submitter, workflow=workflow, path=frame.path
    )
    new.graph = frame.graph
    frame.graph.add_nodes(new)
    frame.graph.add_edges((last, new))
    # the workflow's outputs follow the newest round, so they hold no older round
    for name in spec.body_outputs:
        setattr(workflow.outputs, name, getattr(outputs, name))
    if loop.prev is not None:  # only the two newest rounds are held
        runner._to_release.append((frame, loop.prev))
    loop.prev, loop.last = last, new
    loop.rounds += 1
    loop.advanced += 1
    return new


def advance(runner: "GraphRunner", frame: "_Frame") -> None:
    """Close the loop of an expanding frame, or add its next round, once its newest
    round is done; called by the runner on every pass of ``ready``"""
    loop = frame.loop
    if loop.stopped is not None or not loop.last.done:
        return
    last = loop.last
    if last.errored or last.unrunnable or last._node_error is not None:
        # the round's error is the frame's, which fails the loop job when it finishes
        runner._close_loop(frame, "error")
        return
    try:
        until = read_until(frame)
    except Exception as exc:
        runner._fail_loop(frame, exc)
        return
    loop.until = until
    checksum = last.job(None).checksum
    stopped = decide(loop, until, checksum)
    if stopped is not None:
        runner._close_loop(frame, stopped)
        return
    if loop.advanced >= runner.max_advances:
        return  # the driver calls ``ready`` again, as this call settled rounds
    if loop.spec.index is None:
        loop.seen.add(checksum)
    instantiate(runner, frame)


MAX_ADVANCES = 64
"""The default of ``GraphRunner(max_advances=)``: the most rounds each loop adds during one
call of ``GraphRunner.ready``, so a loop replayed from the store hands control back to its
driver. The cap is per loop: a split of n replayed loops may settle n times as many rounds
in one call. Derivation in RFC 0002 section 2.3(d)."""
