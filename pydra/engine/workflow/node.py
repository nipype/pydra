import typing as ty
from copy import deepcopy
from enum import Enum
import attrs
from pydra.utils.typing import TypeParser, StateArray
from . import lazy
from ..specs import TaskSpec, Outputs
from ..helpers import ensure_list, attrs_values, is_lazy
from .. import helpers_state as hlpst
from ..state import State

if ty.TYPE_CHECKING:
    from .base import Workflow


OutputType = ty.TypeVar("OutputType", bound=Outputs)
Splitter = ty.Union[str, ty.Tuple[str, ...]]

_not_set = Enum("_not_set", "NOT_SET")

NOT_SET = _not_set.NOT_SET


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
    _spec: TaskSpec[OutputType]
    _workflow: "Workflow" = attrs.field(default=None, eq=False, hash=False)
    _lzout: OutputType | None = attrs.field(
        init=False, default=None, eq=False, hash=False
    )
    _state: State | None = attrs.field(init=False, default=NOT_SET)
    _cont_dim: dict[str, int] | None = attrs.field(
        init=False, default=None
    )  # QUESTION: should this be included in the state?
    _inner_cont_dim: dict[str, int] = attrs.field(
        init=False, factory=dict
    )  # QUESTION: should this be included in the state?

    class Inputs:
        """A class to wrap the inputs of a node and control access to them so lazy fields
        that will change the downstream state (i.e. with new splits) aren't set after
        the node has been split, combined or its outputs accessed.
        """

        _node: "Node"

        def __init__(self, node: "Node") -> None:
            super().__setattr__("_node", node)

        def __getattr__(self, name: str) -> ty.Any:
            return getattr(self._node._spec, name)

        def __setattr__(self, name: str, value: ty.Any) -> None:
            setattr(self._node._spec, name, value)
            if is_lazy(value):
                upstream_states = self._node._get_upstream_states()
                if (
                    not self._node._state
                    or self._node._state.other_states != upstream_states
                ):
                    self._node._check_if_outputs_have_been_used(
                        f"cannot set {name!r} input to {value} because it changes the "
                        f"state"
                    )

    @property
    def inputs(self) -> Inputs:
        return self.Inputs(self)

    @property
    def state(self):
        """Initialise the state of the node just after it has been created (i.e. before
        it has been split or combined) based on the upstream connections
        """
        if self._state is not NOT_SET:
            return self._state
        self._set_state(other_states=self._get_upstream_states())
        return self._state

    @property
    def input_values(self) -> tuple[tuple[str, ty.Any]]:
        return tuple(attrs_values(self._spec).items())

    @property
    def lzout(self) -> OutputType:
        from pydra.engine.helpers import list_fields

        """The output spec of the node populated with lazy fields"""
        if self._lzout is not None:
            return self._lzout
        lazy_fields = {}
        for field in list_fields(self.inputs.Outputs):
            lazy_fields[field.name] = lazy.LazyOutField(
                node=self,
                field=field.name,
                type=field.type,
            )
        outputs = self.inputs.Outputs(**lazy_fields)
        # Flag the output lazy fields as being not typed checked (i.e. assigned to another
        # node's inputs) yet
        for outpt in attrs_values(outputs).values():
            outpt.type_checked = False
        outputs._node = self
        self._lzout = outputs
        self._wrap_lzout_types_in_state_arrays()
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
        self._check_if_outputs_have_been_used("the node cannot be split or combined")
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
                    if isinstance(inpt_val, lazy.LazyField):
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
            self._spec = attrs.evolve(self._spec, **new_inputs)
        self._set_state(splitter=splitter)
        # Wrap types of lazy outputs in StateArray types
        self._wrap_lzout_types_in_state_arrays()
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
            c for c in combiner if not any(c in s for s in self.state.splitter_rpn)
        ]:
            raise ValueError(
                f"Combiner fields {not_split} for Node {self.name!r} are not in the "
                f"splitter fields {self.splitter}"
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
        self._set_state(combiner=combiner)
        self._wrap_lzout_types_in_state_arrays()
        return self

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
            return ()
        return self._state.splitter

    @property
    def combiner(self):
        if not self._state:
            return ()
        return self._state.combiner

    def _check_if_outputs_have_been_used(self, msg):
        used = []
        if self._lzout:
            for outpt_name, outpt_val in attrs.asdict(
                self._lzout, recurse=False
            ).items():
                if outpt_val.type_checked:
                    used.append(outpt_name)
        if used:
            raise RuntimeError(
                f"Outputs {used} of {self} have already been accessed and therefore "
                + msg
            )

    def _wrap_lzout_types_in_state_arrays(self) -> None:
        """Wraps a types of the lazy out fields in a number of nested StateArray types
        based on the number of states the node is split over"""
        # Unwrap StateArray types from the output types
        if not self.state:
            return
        outpt_lf: lazy.LazyOutField
        for outpt_lf in attrs_values(self.lzout).values():
            assert not outpt_lf.type_checked
            type_, _ = TypeParser.strip_splits(outpt_lf.type)
            for _ in range(self._state.depth):
                type_ = StateArray[type_]
            outpt_lf.type = type_

    def _set_state(
        self,
        splitter: list[str] | tuple[str, ...] | None = None,
        combiner: list[str] | None = None,
        other_states: dict[str, tuple["State", list[str]]] | None = None,
    ) -> None:
        if self._state not in (NOT_SET, None):
            if splitter is None:
                splitter = self._state.current_splitter
            if combiner is None:
                combiner = self._state.current_combiner
            if other_states is None:
                other_states = self._state.other_states
        if not (splitter or combiner or other_states):
            self._state = None
        else:
            self._state = State(
                self.name,
                splitter=splitter,
                other_states=other_states,
                combiner=combiner,
            )

    def _get_upstream_states(self) -> dict[str, tuple["State", list[str]]]:
        """Get the states of the upstream nodes that are connected to this node"""
        upstream_states = {}
        for inpt_name, val in self.input_values:
            if isinstance(val, lazy.LazyOutField) and val.node.state:
                node: Node = val.node
                # variables that are part of inner splitters should be treated as a containers
                if node.state and f"{node.name}.{inpt_name}" in node.state.splitter:
                    node._inner_cont_dim[f"{node.name}.{inpt_name}"] = 1
                # adding task_name: (task.state, [a field from the connection]
                if node.name not in upstream_states:
                    upstream_states[node.name] = (node.state, [inpt_name])
                else:
                    # if the task already exist in other_state,
                    # additional field name should be added to the list of fields
                    upstream_states[node.name][1].append(inpt_name)
        return upstream_states
