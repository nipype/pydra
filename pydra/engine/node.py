import typing as ty
from copy import deepcopy
from enum import Enum
import attrs
from . import lazy
from pydra.engine.helpers import (
    attrs_values,
    is_lazy,
)
from pydra.engine import helpers_state as hlpst
from pydra.engine.state import State

if ty.TYPE_CHECKING:
    from .core import Workflow
    from .environments import Environment
    from pydra.engine.specs import TaskDef, TaskOutputs, TaskHooks


OutputType = ty.TypeVar("OutputType", bound="TaskOutputs")
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
    inputs : TaskDef
        The definition of the node
    """

    name: str
    _definition: "TaskDef[OutputType]"
    _environment: "Environment | None" = None
    _hooks: "TaskHooks | None" = None
    _workflow: "Workflow" = attrs.field(default=None, eq=False, hash=False, repr=False)
    _lzout: OutputType | None = attrs.field(
        init=False, default=None, eq=False, hash=False, repr=False
    )
    _state: State | None = attrs.field(init=False, default=NOT_SET)

    def __attrs_post_init__(self):
        self._set_state()

    class Inputs:
        """A class to wrap the inputs of a node and control access to them so lazy fields
        that will change the downstream state (i.e. with new splits) aren't set after
        the node has been split, combined or its outputs accessed.
        """

        _node: "Node"

        def __init__(self, node: "Node") -> None:
            super().__setattr__("_node", node)

        def __getattr__(self, name: str) -> ty.Any:
            return getattr(self._node._definition, name)

        def __getstate__(self) -> ty.Dict[str, ty.Any]:
            return {"_node": self._node}

        def __setstate__(self, state: ty.Dict[str, ty.Any]) -> None:
            super().__setattr__("_node", state["_node"])

        def __setattr__(self, name: str, value: ty.Any) -> None:
            setattr(self._node._definition, name, value)
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
                    self._set_state()

    @property
    def inputs(self) -> Inputs:
        return self.Inputs(self)

    @property
    def input_names(self) -> list[str]:
        return list(attrs_values(self._definition).keys())

    @property
    def state(self):
        """Initialise the state of the node just after it has been created (i.e. before
        it has been split or combined) based on the upstream connections
        """
        return self._state

    @property
    def input_values(self) -> tuple[tuple[str, ty.Any]]:
        return tuple(attrs_values(self._definition).items())

    @property
    def state_values(self) -> dict[str, ty.Any]:
        """Get the values of the task definition, scoped by the name of the node to be
        used in the state

        Returns
        -------
        dict[str, Any]
            The values of the task definition
        """
        return {
            f"{self.name}.{n}": v for n, v in attrs_values(self._definition).items()
        }

    @property
    def lzout(self) -> OutputType:
        from pydra.engine.helpers import list_fields

        """The output definition of the node populated with lazy fields"""
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

        outpt: lazy.LazyOutField
        for outpt in attrs_values(outputs).values():
            # Assign the current node to the lazy fields so they can access the state
            outpt._node = self
            # If the node has a non-empty state, wrap the type of the lazy field in
            # a combination of an optional list and a number of nested StateArrays
            # types based on the number of states the node is split over and whether
            # it has a combiner
            if self._state:
                outpt._type = self._state.nest_output_type(outpt._type)
            # Flag the output lazy fields as being not typed checked (i.e. assigned to
            # another node's inputs) yet. This is used to prevent the user from changing
            # the type of the output after it has been accessed by connecting it to an
            # output of an upstream node with additional state variables.
            outpt._type_checked = False
        self._lzout = outputs
        return outputs

    @property
    def cont_dim(self):
        # adding inner_cont_dim to the general container_dimension provided by the users
        cont_dim_all = deepcopy(self._cont_dim)
        for k, v in self.state._inner_cont_dim.items():
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

    def _set_state(self) -> None:
        # Add node name to state's splitter, combiner and cont_dim loaded from the def
        splitter = deepcopy(
            self._definition._splitter
        )  # these can be modified in state
        combiner = deepcopy(
            self._definition._combiner
        )  # these can be modified in state
        cont_dim = {}
        if splitter:
            splitter = hlpst.add_name_splitter(splitter, self.name)
        if combiner:
            combiner = hlpst.add_name_combiner(combiner, self.name)
        if self._definition._cont_dim:
            for key, val in self._definition._cont_dim.items():
                cont_dim[f"{self.name}.{key}"] = val
        other_states = self._get_upstream_states()
        if splitter or combiner or other_states:
            self._state = State(
                self.name,
                splitter=splitter,
                other_states=other_states,
                combiner=combiner,
                cont_dim=cont_dim,
            )
            if combiner:
                if not_split := [
                    c
                    for c in combiner
                    if not any(c in s for s in self.state.splitter_rpn)
                ]:
                    raise ValueError(
                        f"Combiner fields {not_split} for Node {self.name!r} are not in the "
                        f"splitter fields {self.state.splitter_rpn}"
                    )
        else:
            self._state = None

    def _get_upstream_states(self) -> dict[str, tuple["State", list[str]]]:
        """Get the states of the upstream nodes that are connected to this node"""
        upstream_states = {}
        for inpt_name, val in self.input_values:
            if (
                isinstance(val, lazy.LazyOutField)
                and val._node.state
                and val._node.state.depth()
            ):
                node: Node = val._node
                # variables that are part of inner splitters should be treated as a containers
                if node.state and f"{node.name}.{inpt_name}" in node.state.splitter:
                    node.state._inner_cont_dim[f"{node.name}.{inpt_name}"] = 1
                # adding task_name: (task.state, [a field from the connection]
                if node.name not in upstream_states:
                    upstream_states[node.name] = (node.state, [inpt_name])
                else:
                    # if the task already exist in other_state,
                    # additional field name should be added to the list of fields
                    upstream_states[node.name][1].append(inpt_name)
        return upstream_states

        # else:
        #     # todo it never gets here
        #     breakpoint()
        #     inputs_dict = {inp: getattr(self.inputs, inp) for inp in self.input_names}
        #     return None, inputs_dict
