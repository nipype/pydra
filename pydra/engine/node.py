import typing as ty
from copy import deepcopy, copy
from enum import Enum
import attrs
from pydra.utils.typing import TypeParser, StateArray
from . import lazy
from pydra.engine.helpers import (
    ensure_list,
    attrs_values,
    is_lazy,
    create_checksum,
)
from pydra.utils.hash import hash_function
from pydra.engine import helpers_state as hlpst
from pydra.engine.state import State, StateIndex

if ty.TYPE_CHECKING:
    from .core import Workflow
    from pydra.engine.specs import TaskDef, TaskOutputs


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

    def __attrs_post_init__(self):
        # Add node name to state's splitter, combiner and cont_dim loaded from the def
        splitter = self._definition._splitter
        combiner = self._definition._combiner
        if splitter:
            splitter = hlpst.add_name_splitter(splitter, self.name)
        if combiner:
            combiner = hlpst.add_name_combiner(combiner, self.name)
        if self._definition._cont_dim:
            self._cont_dim = {}
            for key, val in self._definition._cont_dim.items():
                self._cont_dim[f"{self.name}.{key}"] = val
        self._set_state(splitter=splitter, combiner=combiner)
        if combiner:
            if not_split := [
                c for c in combiner if not any(c in s for s in self.state.splitter_rpn)
            ]:
                raise ValueError(
                    f"Combiner fields {not_split} for Node {self.name!r} are not in the "
                    f"splitter fields {self.state.splitter_rpn}"
                )

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
        if self._state is not NOT_SET:
            return self._state
        self._set_state(other_states=self._get_upstream_states())
        return self._state

    @property
    def input_values(self) -> tuple[tuple[str, ty.Any]]:
        return tuple(attrs_values(self._definition).items())

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
        # Flag the output lazy fields as being not typed checked (i.e. assigned to another
        # node's inputs) yet
        for outpt in attrs_values(outputs).values():
            outpt.type_checked = False
        outputs._node = self
        self._lzout = outputs
        self._wrap_lzout_types_in_state_arrays()
        return outputs

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

    def _checksum_states(self, state_index=None):
        """
        Calculate a checksum for the specific state or all of the states of the task.
        Replaces state-arrays in the inputs fields with a specific values for states.
        Used to recreate names of the task directories,

        Parameters
        ----------
        state_index :
            TODO

        """
        # if is_workflow(self) and self._definition._graph_checksums is attr.NOTHING:
        #     self._definition._graph_checksums = {
        #         nd.name: nd.checksum for nd in self.graph_sorted
        #     }
        from pydra.engine.specs import WorkflowDef

        if state_index is not None:
            inputs_copy = copy(self._definition)
            for key, ind in self.state.inputs_ind[state_index].items():
                val = self._extract_input_el(
                    inputs=self._definition, inp_nm=key.split(".")[1], ind=ind
                )
                setattr(inputs_copy, key.split(".")[1], val)
            # setting files_hash again in case it was cleaned by setting specific element
            # that might be important for outer splitter of input variable with big files
            # the file can be changed with every single index even if there are only two files
            input_hash = inputs_copy.hash
            if isinstance(self._definition, WorkflowDef):
                con_hash = hash_function(self._connections)
                # TODO: hash list is not used
                hash_list = [input_hash, con_hash]  # noqa: F841
                checksum_ind = create_checksum(
                    self.__class__.__name__, self._checksum_wf(input_hash)
                )
            else:
                checksum_ind = create_checksum(self.__class__.__name__, input_hash)
            return checksum_ind
        else:
            checksum_list = []
            if not hasattr(self.state, "inputs_ind"):
                self.state.prepare_states(self._definition, cont_dim=self.cont_dim)
                self.state.prepare_inputs()
            for ind in range(len(self.state.inputs_ind)):
                checksum_list.append(self._checksum_states(state_index=ind))
            return checksum_list

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

    def _extract_input_el(self, inputs, inp_nm, ind):
        """
        Extracting element of the inputs taking into account
        container dimension of the specific element that can be set in self.cont_dim.
        If input name is not in cont_dim, it is assumed that the input values has
        a container dimension of 1, so only the most outer dim will be used for splitting.
        If
        """
        if f"{self.name}.{inp_nm}" in self.cont_dim:
            return list(
                hlpst.flatten(
                    ensure_list(getattr(inputs, inp_nm)),
                    max_depth=self.cont_dim[f"{self.name}.{inp_nm}"],
                )
            )[ind]
        else:
            return getattr(inputs, inp_nm)[ind]

    def _split_definition(self) -> dict[StateIndex, "TaskDef[OutputType]"]:
        """Split the definition into the different states it will be run over"""
        # TODO: doesn't work properly for more cmplicated wf (check if still an issue)
        if not self.state:
            return {None: self._definition}
        split_defs = {}
        for input_ind in self.state.inputs_ind:
            inputs_dict = {}
            for inp in set(self.input_names):
                if f"{self.name}.{inp}" in input_ind:
                    inputs_dict[inp] = self._extract_input_el(
                        inputs=self._definition,
                        inp_nm=inp,
                        ind=input_ind[f"{self.name}.{inp}"],
                    )
            split_defs[StateIndex(input_ind)] = attrs.evolve(
                self._definition, inputs_dict
            )
        return split_defs

        # else:
        #     # todo it never gets here
        #     breakpoint()
        #     inputs_dict = {inp: getattr(self.inputs, inp) for inp in self.input_names}
        #     return None, inputs_dict
