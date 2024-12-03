import typing as ty
from copy import copy, deepcopy
from operator import itemgetter
from typing_extensions import Self
import attrs
from pydra.design.base import list_fields, TaskSpec, OutputsSpec
from pydra.engine.specs import LazyField, LazyInField, LazyOutField, StateArray
from pydra.utils.hash import hash_function
from pydra.utils.typing import TypeParser
from . import helpers_state as hlpst
from .helpers import ensure_list
from . import state


OutputType = ty.TypeVar("OutputType", bound=OutputsSpec)


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
    _workflow: "Workflow" = None
    _lzout: OutputType | None = None
    _state: state.State | None = None
    _cont_dim: dict[str, int] | None = (
        None  # QUESTION: should this be included in the state?
    )

    @property
    def lzout(self) -> OutputType:
        """The output spec of the node populated with lazy fields"""
        if self._lzout is not None:
            return self._lzout
        combined_splitter = set()
        for inpt_name, inpt_val in attrs.asdict(self.inputs, recurse=False).items():
            if isinstance(inpt_val, LazyField):
                combined_splitter.update(inpt_val.splits)
        lazy_fields = {}
        for field in list_fields(self.inputs.Outputs):
            type_ = field.type
            # Wrap types of lazy outputs in StateArray types if the input fields are split
            # over state values
            for _ in range(len(combined_splitter)):
                type_ = StateArray[type_]
            lazy_fields[field.name] = LazyOutField(
                name=self.name,
                field=field.name,
                type=type_,
                splits=frozenset(iter(combined_splitter)),
            )
        outputs = self.inputs.Outputs(**lazy_fields)
        # Flag the output lazy fields as being not typed checked (i.e. assigned to another
        # node's inputs) yet
        for outpt in attrs.asdict(outputs, recurse=False).values():
            outpt.type_checked = False
        outputs._node = self
        self._lzout = outputs
        return outputs

    def split(
        self,
        splitter: ty.Union[str, ty.List[str], ty.Tuple[str, ...], None] = None,
        /,
        overwrite: bool = False,
        cont_dim: ty.Optional[dict] = None,
        **inputs,
    ):
        """
        Run this task parametrically over lists of split inputs.

        Parameters
        ----------
        splitter : str or list[str] or tuple[str] or None
            the fields which to split over. If splitting over multiple fields, lists of
            fields are interpreted as outer-products and tuples inner-products. If None,
            then the fields to split are taken from the keyword-arg names.
        overwrite : bool, optional
            whether to overwrite an existing split on the node, by default False
        cont_dim : dict, optional
            Container dimensions for specific inputs, used in the splitter.
            If input name is not in cont_dim, it is assumed that the input values has
            a container dimension of 1, so only the most outer dim will be used for splitting.
        **inputs
            fields to split over, will automatically be wrapped in a StateArray object
            and passed to the node inputs

        Returns
        -------
        self : TaskSpec
            a reference to the task
        """
        self._check_if_outputs_have_been_used()
        if splitter is None and inputs:
            splitter = list(inputs)
        elif splitter:
            missing = set(hlpst.unwrap_splitter(splitter)) - set(inputs)
            missing = [m for m in missing if not m.startswith("_")]
            if missing:
                raise ValueError(
                    f"Split is missing values for the following fields {list(missing)}"
                )
        splitter = hlpst.add_name_splitter(splitter, self.name)
        # if user want to update the splitter, overwrite has to be True
        if self._state and not overwrite and self._state.splitter != splitter:
            raise Exception(
                "splitter has been already set, "
                "if you want to overwrite it - use overwrite=True"
            )
        if cont_dim:
            for key, vel in cont_dim.items():
                self._cont_dim[f"{self.name}.{key}"] = vel
        if inputs:
            new_inputs = {}
            split_inputs = set(
                f"{self.name}.{n}" if "." not in n else n
                for n in hlpst.unwrap_splitter(splitter)
                if not n.startswith("_")
            )
            for inpt_name, inpt_val in inputs.items():
                new_val: ty.Any
                if f"{self.name}.{inpt_name}" in split_inputs:  # type: ignore
                    if isinstance(inpt_val, LazyField):
                        new_val = inpt_val.split(splitter)
                    elif isinstance(inpt_val, ty.Iterable) and not isinstance(
                        inpt_val, (ty.Mapping, str)
                    ):
                        new_val = StateArray(inpt_val)
                    else:
                        raise TypeError(
                            f"Could not split {inpt_val} as it is not a sequence type"
                        )
                else:
                    new_val = inpt_val
                new_inputs[inpt_name] = new_val
            # Update the inputs with the new split values
            self.inputs = attrs.evolve(self.inputs, **new_inputs)
        if not self._state or splitter != self._state.splitter:
            self._set_state(splitter)
        # Wrap types of lazy outputs in StateArray types
        split_depth = len(LazyField.normalize_splitter(splitter))
        outpt_lf: LazyOutField
        for outpt_lf in attrs.asdict(self.lzout, recurse=False).values():
            assert not outpt_lf.type_checked
            outpt_type = outpt_lf.type
            for d in range(split_depth):
                outpt_type = StateArray[outpt_type]
            outpt_lf.type = outpt_type
            outpt_lf.splits = frozenset(iter(self._state.splitter))
        return self

    def combine(
        self,
        combiner: ty.Union[ty.List[str], str],
        overwrite: bool = False,  # **kwargs
    ):
        """
        Combine inputs parameterized by one or more previous tasks.

        Parameters
        ----------
        combiner : list[str] or str
            the field or list of inputs to be combined (i.e. not left split) after the
            task has been run
        overwrite : bool
            whether to overwrite an existing combiner on the node
        **kwargs : dict[str, Any]
            values for the task that will be "combined" before they are provided to the
            node

        Returns
        -------
        self : TaskSpec
            a reference to the task
        """
        if not isinstance(combiner, (str, list)):
            raise Exception("combiner has to be a string or a list")
        combiner = hlpst.add_name_combiner(ensure_list(combiner), self.name)
        if not_split := [
            c for c in combiner if not any(c in s for s in self._state.splitter)
        ]:
            raise ValueError(
                f"Combiner fields {not_split} for Node {self.name!r} are not in the "
                f"splitter fields {self._state.splitter}"
            )
        if (
            self._state
            and self._state.combiner
            and combiner != self._state.combiner
            and not overwrite
        ):
            raise Exception(
                "combiner has been already set, "
                "if you want to overwrite it - use overwrite=True"
            )
        if not self._state:
            self.split(splitter=None)
            # a task can have a combiner without a splitter
            # if is connected to one with a splitter;
            # self.fut_combiner will be used later as a combiner
            self._state.fut_combiner = combiner
        else:  # self.state and not self.state.combiner
            self._set_state(splitter=self._state.splitter, combiner=combiner)
        # Wrap types of lazy outputs in StateArray types
        norm_splitter = LazyField.normalize_splitter(self._state.splitter)
        remaining_splits = [
            s for s in norm_splitter if not any(c in s for c in combiner)
        ]
        combine_depth = len(norm_splitter) - len(remaining_splits)
        outpt_lf: LazyOutField
        for outpt_lf in attrs.asdict(self.lzout, recurse=False).values():
            assert not outpt_lf.type_checked
            outpt_type, split_depth = TypeParser.strip_splits(outpt_lf.type)
            assert split_depth >= combine_depth, (
                f"Attempting to combine a field that has not been split enough times: "
                f"{outpt_lf.name} ({outpt_lf.type}), {self._state.splitter} -> {combiner}"
            )
            outpt_lf.type = list[outpt_type]
            for _ in range(split_depth - combine_depth):
                outpt_lf.type = StateArray[outpt_lf.type]
            outpt_lf.splits = frozenset(iter(remaining_splits))
        return self

    def _set_state(self, splitter, combiner=None):
        """
        Set a particular state on this task.

        Parameters
        ----------
        splitter : str | list[str] | tuple[str]
            the fields which to split over. If splitting over multiple fields, lists of
            fields are interpreted as outer-products and tuples inner-products. If None,
            then the fields to split are taken from the keyword-arg names.
        combiner : list[str] | str, optional
            the field or list of inputs to be combined (i.e. not left split) after the
            task has been run
        """
        if splitter is not None:
            self._state = state.State(
                name=self.name, splitter=splitter, combiner=combiner
            )
        else:
            self._state = None
        return self._state

    @property
    def cont_dim(self):
        # adding inner_cont_dim to the general container_dimension provided by the users
        cont_dim_all = deepcopy(self._cont_dim)
        for k, v in self._inner_cont_dim.items():
            cont_dim_all[k] = cont_dim_all.get(k, 1) + v
        return cont_dim_all

    @cont_dim.setter
    def cont_dim(self, cont_dim):
        if cont_dim is None:
            self._cont_dim = {}
        else:
            self._cont_dim = cont_dim

    @property
    def splitter(self):
        if not self._state:
            return None
        return self._state.splitter

    @property
    def combiner(self):
        if not self._state:
            return None
        return self._state.combiner

    def _check_if_outputs_have_been_used(self):
        used = []
        if self._lzout:
            for outpt_name, outpt_val in attrs.asdict(
                self._lzout, recurse=False
            ).items():
                if outpt_val.type_checked:
                    used.append(outpt_name)
        if used:
            raise RuntimeError(
                f"Outputs {used} of {self} have already been accessed and therefore cannot "
                "be split or combined"
            )


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
