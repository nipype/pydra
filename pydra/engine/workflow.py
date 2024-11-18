import typing as ty
from copy import copy
from operator import itemgetter
from typing_extensions import Self
import attrs
from pydra.design.base import list_fields, TaskSpec
from pydra.engine.specs import LazyField, StateArray
from pydra.utils.hash import hash_function


OutputType = ty.TypeVar("OutputType")


@attrs.define
class Node(ty.Generic[OutputType]):
    """A node in a workflow

    Parameters
    ----------
    name : str
        The name of the node
    inputs : TaskSpec
        The specification of the node
    """

    name: str
    inputs: TaskSpec[OutputType]
    _splitter: str | list[str] | tuple[str] | None = None
    _combiner: list | str = None
    _workflow: "Workflow" = None

    @property
    def lzout(self) -> OutputType:
        """The output spec of the node populated with lazy fields"""
        return self.inputs.Outputs(
            **{
                f.name: LazyField(name=self.name, field=f.name, type=f.type)
                for f in list_fields(self.inputs.Outputs)
            }
        )

    def split(self, splitter=None, /, **inputs) -> None:
        """Split the node over the specified inputs

        Parameters
        ----------
        splitter : str | list[str] | tuple[str], optional
            The input field(s) to split over. If a list then an "outer" product
            split is performed over all the fields (all combinations). If a tuple then a
            the input values must be the same length and "inner" product split is
            performed over the fields (pairs of combinations). If a splitter is not provided
            then all the inputs are taken to be an outer product split.
        **inputs
            The input values to split over
        """
        if self._splitter is not None:
            raise ValueError(f"Splitter already set to {self._splitter!r}")
        self._splitter = splitter or list(inputs)
        for name, value in inputs.items():
            setattr(self.inputs, name, StateArray(value))

    def combine(self, combiner: list | str) -> None:
        """Combine the node over the specified inputs

        Parameters
        ----------
        combiner : list | str
            Either a single field or a list of fields to combine in the node
        """
        self._combiner = combiner


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
                (i for i in attrs.asdict(spec).items() if i[0] not in lazy_input_names),
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
                lzy_inpt,
                LazyField(
                    wf.name,  # This shouldn't be the name of the workflow, but this is currently
                    lzy_inpt.name,
                    lzy_inpt.type,
                ),
            )

        input_values = attrs.asdict(lazy_spec)
        constructor = input_values.pop("constructor")
        cls._under_construction = wf
        try:
            # Call the user defined constructor to set the outputs
            output_values = constructor(**input_values)

            # Check that the outputs are set correctly, either directly by the constructor
            # returned values that can be zipped with the output names
            if output_values:
                if not isinstance(output_values, (list, tuple)):
                    output_values = [output_values]
                output_fields = list_fields(spec.Outputs)
                if len(output_values) != len(output_fields):
                    raise ValueError(
                        f"Expected {len(output_fields)} outputs, got "
                        f"{len(output_values)} ({output_values})"
                    )
                for outpt, oupt_val in zip(output_fields, output_values):
                    setattr(outputs, outpt.name, oupt_val)
            else:
                if unset_outputs := {
                    a: v for a, v in attrs.asdict(outputs).items() if v is attrs.NOTHING
                }:
                    raise ValueError(
                        f"Expected outputs {list(unset_outputs)} to be set by the "
                        f"constructor of {wf!r}"
                    )
        finally:
            cls._under_construction = None

        cls._constructed[hash_key] = wf

        return wf

    def add(self, task_spec: TaskSpec[OutputType], name=None) -> OutputType:
        if name is None:
            name = type(task_spec).__name__
        if name in self._nodes:
            raise ValueError(f"Node with name {name!r} already exists in the workflow")
        node = Node[OutputType](name=name, inputs=task_spec, workflow=self)
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
                "constructor function"
            )
        return cls._under_construction

    _under_construction: "Workflow[ty.Any]" = None
    _constructed: dict[int, "Workflow[ty.Any]"] = {}
