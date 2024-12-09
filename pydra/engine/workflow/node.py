import typing as ty
from copy import deepcopy
from enum import Enum
import attrs
from pydra.utils.typing import TypeParser, StateArray
from . import lazy
from ..specs import TaskSpec, Outputs
from ..helpers import ensure_list, attrs_values
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
            if isinstance(value, lazy.LazyField):
                # Save the current state for comparison later
                prev_state = self._node.state
                if value.node.state:
                    # Reset the state to allow the lazy field to be set
                    self._node._state = NOT_SET
                setattr(self._node._spec, name, value)
                if value.node.state and self._node.state != prev_state:
                    self._node._check_if_outputs_have_been_used(
                        f"cannot set {name!r} input to {value} because it changes the "
                        f"state of the node from {prev_state} to {value.node.state}"
                    )

    @property
    def inputs(self) -> Inputs:
        return self.Inputs(self)

    @property
    def state(self):
        if self._state is not NOT_SET:
            return self._state
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
        if upstream_states:
            state = State(
                node.name,
                splitter=None,
                other_states=upstream_states,
                combiner=None,
            )
        else:
            state = None
        self._state = state
        return state

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
        if not self._state or splitter != self._state.splitter:
            self._set_state(splitter)
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
            c for c in combiner if not any(c in s for s in self.state.splitter)
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
        if not self._state:
            self.split(splitter=None)
            # a task can have a combiner without a splitter
            # if is connected to one with a splitter;
            # self.fut_combiner will be used later as a combiner
            self._state.fut_combiner = (
                combiner  # QUESTION: why separate combiner and fut_combiner?
            )
        else:  # self.state and not self.state.combiner
            self._set_state(splitter=self._state.splitter, combiner=combiner)
        self._wrap_lzout_types_in_state_arrays()
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
            self._state = State(name=self.name, splitter=splitter, combiner=combiner)
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
        remaining_splits = []
        for split in self.state.splitter:
            if isinstance(split, str):
                if split not in self.state.combiner:
                    remaining_splits.append(split)
            elif all(s not in self.state.combiner for s in split):
                remaining_splits.append(split)
        state_depth = len(remaining_splits)
        for outpt_lf in attrs_values(self.lzout).values():
            assert not outpt_lf.type_checked
            type_, _ = TypeParser.strip_splits(outpt_lf.type)
            for _ in range(state_depth):
                type_ = StateArray[type_]
            outpt_lf.type = type_

    # @classmethod
    # def _normalize_splitter(
    #     cls, splitter: Splitter, strip_previous: bool = True
    # ) -> ty.Tuple[ty.Tuple[str, ...], ...]:
    #     """Converts the splitter spec into a consistent tuple[tuple[str, ...], ...] form
    #     used in LazyFields"""
    #     if isinstance(splitter, str):
    #         splitter = (splitter,)
    #     if isinstance(splitter, tuple):
    #         splitter = (splitter,)  # type: ignore
    #     else:
    #         assert isinstance(splitter, list)
    #         # convert to frozenset to differentiate from tuple, yet still be hashable
    #         # (NB: order of fields in list splitters aren't relevant)
    #         splitter = tuple((s,) if isinstance(s, str) else s for s in splitter)
    #     # Strip out fields starting with "_" designating splits in upstream nodes
    #     if strip_previous:
    #         stripped = tuple(
    #             tuple(f for f in i if not f.startswith("_")) for i in splitter
    #         )
    #         splitter = tuple(s for s in stripped if s)  # type: ignore
    #     return splitter  # type: ignore
