import typing as ty
import enum
from copy import copy, deepcopy
from operator import itemgetter
from typing_extensions import Self
import attrs
from pydra.design.base import list_fields, TaskSpec
from pydra.engine.specs import LazyField, StateArray
from pydra.utils.hash import hash_function
from . import helpers_state as hlpst
from .helpers import ensure_list
from . import state


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
    _lzout: OutputType | None = None
    _state: state.State | None = None
    _cont_dim: dict[str, int] | None = None

    @property
    def lzout(self) -> OutputType:
        """The output spec of the node populated with lazy fields"""
        if self._lzout is not None:
            return self._lzout
        state_array_depth = 0
        for inpt_name, inpt_val in attrs.asdict(self.inputs).items():
            if isinstance(inpt_val, LazyField):
                tp = inpt_val.type
                depth = 0
                while isinstance(tp, StateArray):
                    tp = ty.get_args(tp)[0]
                    depth += 1
                # FIXME: This won't be able to differentiate between state arrays
                # from different splits and those from the same split, we might need to
                # keep track of that in the LazyField... (am I right about this??)
                state_array_depth = max(state_array_depth, depth)
        lazy_fields = {}
        for field in list_fields(self.inputs.Outputs):
            # TODO: need to reimplement the full spliter/combiner logic here
            if self._splitter and field.name in self._splitter:
                if field.name in self._combiner:
                    type_ = list[field.type]
                else:
                    type_ = StateArray(field.type)
            else:
                type_ = field.type
            for _ in range(state_array_depth):
                type_ = StateArray[type_]
            lazy_fields[field.name] = LazyField(
                name=self.name,
                field=field.name,
                type=type_,
            )
        outputs = self.inputs.Outputs(**lazy_fields)
        # Flag the output lazy fields as being not typed checked (i.e. assigned to another
        # node's inputs) yet
        for outpt in attrs.asdict(outputs, recurse=False).values():
            outpt.type_checked = False
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
        self : TaskBase
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
        self : TaskBase
            a reference to the task
        """
        if not isinstance(combiner, (str, list)):
            raise Exception("combiner has to be a string or a list")
        combiner = hlpst.add_name_combiner(ensure_list(combiner), self.name)
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
                lzy_inpt,
                LazyField(
                    WORKFLOW_LZIN,
                    lzy_inpt.name,
                    lzy_inpt.type,
                ),
            )

        input_values = attrs.asdict(lazy_spec, recurse=False)
        constructor = input_values.pop("constructor")
        cls._under_construction = wf
        try:
            # Call the user defined constructor to set the outputs
            output_values = constructor(**input_values)
            # Check to see whether any mandatory inputs are not set
            for node in wf.nodes:
                node.inputs._check_for_unset_values()
            # Check that the outputs are set correctly, either directly by the constructor
            # or via returned values that can be zipped with the output names
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


class _WorkflowLzin(enum.Enum):

    WORKFLOW_LZIN = enum.auto()

    def __repr__(self):
        return "WORKFLOW_LZIN"


WORKFLOW_LZIN = _WorkflowLzin.WORKFLOW_LZIN
