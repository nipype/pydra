import typing as ty
from copy import copy
from operator import itemgetter
from typing_extensions import Self
import attrs
from pydra.engine.helpers import list_fields
from pydra.engine.specs import TaskSpec, OutputsSpec
from .lazy import LazyInField
from pydra.utils.hash import hash_function
from pydra.utils.typing import TypeParser, StateArray
from .node import Node


OutputType = ty.TypeVar("OutputType", bound=OutputsSpec)


@attrs.define(auto_attribs=False)
class Workflow(ty.Generic[OutputType]):
    """A workflow, constructed from a workflow specification

    Parameters
    ----------
    name : str
        The name of the workflow
    inputs : TaskSpec
        The input specification of the workflow
    outputs : TaskSpec
        The output specification of the workflow
    """

    name: str = attrs.field()
    inputs: TaskSpec[OutputType] = attrs.field()
    outputs: OutputType = attrs.field()
    _nodes: dict[str, Node] = attrs.field(factory=dict)

    @classmethod
    def construct(
        cls,
        spec: TaskSpec[OutputType],
    ) -> Self:
        """Construct a workflow from a specification, caching the constructed worklow"""

        lazy_inputs = [f for f in list_fields(type(spec)) if f.lazy]

        # Create a cache key by hashing all the non-lazy input values in the spec
        # and use this to store the constructed workflow in case it is reused or nested
        # and split over within another workflow
        lazy_input_names = {f.name for f in lazy_inputs}
        non_lazy_vals = tuple(
            sorted(
                (
                    i
                    for i in attrs.asdict(spec, recurse=False).items()
                    if i[0] not in lazy_input_names
                ),
                key=itemgetter(0),
            )
        )
        hash_key = hash_function(non_lazy_vals)
        if hash_key in cls._constructed:
            return cls._constructed[hash_key]

        # Initialise the outputs of the workflow
        outputs = spec.Outputs(
            **{f.name: attrs.NOTHING for f in attrs.fields(spec.Outputs)}
        )

        # Initialise the lzin fields
        lazy_spec = copy(spec)
        wf = cls.under_construction = Workflow(
            name=type(spec).__name__,
            inputs=lazy_spec,
            outputs=outputs,
        )
        for lzy_inpt in lazy_inputs:
            setattr(
                lazy_spec,
                lzy_inpt.name,
                LazyInField(
                    node=wf,
                    field=lzy_inpt.name,
                    type=lzy_inpt.type,
                ),
            )

        input_values = attrs.asdict(lazy_spec, recurse=False)
        constructor = input_values.pop("constructor")
        cls._under_construction = wf
        try:
            # Call the user defined constructor to set the outputs
            output_lazy_fields = constructor(**input_values)
            # Check to see whether any mandatory inputs are not set
            for node in wf.nodes:
                node.inputs._check_for_unset_values()
            # Check that the outputs are set correctly, either directly by the constructor
            # or via returned values that can be zipped with the output names
            if output_lazy_fields:
                if not isinstance(output_lazy_fields, (list, tuple)):
                    output_lazy_fields = [output_lazy_fields]
                output_fields = list_fields(spec.Outputs)
                if len(output_lazy_fields) != len(output_fields):
                    raise ValueError(
                        f"Expected {len(output_fields)} outputs, got "
                        f"{len(output_lazy_fields)} ({output_lazy_fields})"
                    )
                for outpt, outpt_lf in zip(output_fields, output_lazy_fields):
                    if TypeParser.get_origin(outpt_lf.type) is StateArray:
                        # Automatically combine any uncombined state arrays into lists
                        tp, _ = TypeParser.strip_splits(outpt_lf.type)
                        outpt_lf.type = list[tp]
                        outpt_lf.splits = frozenset()
                    setattr(outputs, outpt.name, outpt_lf)
            else:
                if unset_outputs := [
                    a
                    for a, v in attrs.asdict(outputs, recurse=False).items()
                    if v is attrs.NOTHING
                ]:
                    raise ValueError(
                        f"Expected outputs {unset_outputs} to be set by the "
                        f"constructor of {wf!r}"
                    )
        finally:
            cls._under_construction = None

        cls._constructed[hash_key] = wf

        return wf

    def add(self, task_spec: TaskSpec[OutputType], name=None) -> OutputType:
        """Add a node to the workflow

        Parameters
        ----------
        task_spec : TaskSpec
            The specification of the task to add to the workflow as a node
        name : str, optional
            The name of the node, by default it will be the name of the task specification
            class

        Returns
        -------
        OutputType
            The outputs specification of the node
        """
        if name is None:
            name = type(task_spec).__name__
        if name in self._nodes:
            raise ValueError(f"Node with name {name!r} already exists in the workflow")
        node = Node[OutputType](name=name, spec=task_spec, workflow=self)
        self._nodes[name] = node
        return node.lzout

    def __getitem__(self, key: str) -> Node:
        return self._nodes[key]

    @property
    def nodes(self) -> ty.Iterable[Node]:
        return self._nodes.values()

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes)

    @property
    @classmethod
    def under_construction(cls) -> "Workflow[ty.Any]":
        if cls._under_construction is None:
            raise ValueError(
                "pydra.design.workflow.this() can only be called from within a workflow "
                "constructor function (see 'pydra.design.workflow.define')"
            )
        return cls._under_construction

    # Used to store the workflow that is currently being constructed
    _under_construction: "Workflow[ty.Any]" = None
    # Used to cache the constructed workflows by their hashed input values
    _constructed: dict[int, "Workflow[ty.Any]"] = {}
