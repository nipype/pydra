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

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.hooks import TaskHooks
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
        executor = getattr(self.body, "_executor_name", None)
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
