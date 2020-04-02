"""Keeping track of mapping and reduce operations over tasks."""
from copy import deepcopy
from . import helpers_state as hlpst
from .helpers import ensure_list
from .specs import BaseSpec


class State:
    """
    A class that specifies a State of all tasks.

      * It's only used when a task have a splitter.
      * It contains all information about splitter, combiner, final splitter,
        and input values for specific task states
        (specified by the splitter and the input).
      * It also contains information about the final groups and the final splitter
        if combiner is available..

    Attributes
    ----------
    name : :obj:`str`
        name of the state that is the same as name of the task
    splitter : :obj:`str`, :obj:`tuple`, :obj:`list`
        can be a str (name of a single input),
        tuple for scalar splitter, or list for outer splitter
    splitter_rpn_compact : :obj:`list`
        splitter in :abbr:`RPN (reverse Polish notation)`, using a compact
        notation for splitter from previous states, e.g. _NA
    splitter_rpn : :obj:`list`
        splitter represented in RPN,
        unwrapping splitters from previous states
    combiner : :obj:`list`
        list of fields that should be combined
        (order is not important)
    splitter_final :
        final splitter that includes the combining process
    other_states : :obj:`dict`
        used to create connections with previous states::

            {
              name of a previous state:
                (previous state, input from current state needed the connection)
            }

    inner_inputs : :obj:`dict`
        used to create connections with previous states
        ``{"{self.name}.input name for current inp": previous state}``
    states_ind : :obj:`list` of :obj:`dict`
        dictionary for every state that contains
        indices for all state inputs (i.e. inputs that are part of the splitter)
    states_val : :obj:`list` of :obj:`dict`
        dictionary for every state that contains
        values for all state inputs (i.e. inputs that are part of the splitter)
    inputs_ind : :obj:`list` of :obj:`dict`
        dictionary for every state that contains
        indices for all task inputs (i.e. inputs that are relevant
        for current task, can be outputs from previous nodes)
    group_for_inputs : :obj:`dict`
        specifying groups (axes) for each input field
        (depends on the splitter)
    group_for_inputs_final : :obj:`dict`
        specifying final groups (axes)
        for each input field (depends on the splitter and combiner)
    groups_stack_final : :obj:`list`
        specify stack of groups/axes (used to
        determine which field could be combined)
    final_combined_ind_mapping : :obj:`dict`
        mapping between final indices
        after combining and partial indices of the results

    """

    def __init__(self, name, splitter=None, combiner=None, other_states=None):
        """
        Initialize state.

        Parameters
        ----------
        name : :obj:`str`
            name (should be the same as task name)
        splitter : :obj:`str`, or :obj:`tuple`, or :obj:`list`
            splitter of a task
        combiner : :obj:`str`, or :obj:`list`)
            field/fields used to combine results
        other_states :obj:`dict`:
            ``{name of a previous state: (prefious state,
            input from current state needed the connection)}``

        """
        self.name = name
        self.other_states = other_states
        self.splitter = splitter
        # temporary combiner
        self.combiner = combiner
        # if other_states, the connections have to be updated
        if self.other_states:
            self.update_connections()

    def __str__(self):
        """Generate a string representation of the object."""
        return (
            f"State for {self.name} with a splitter: {self.splitter} "
            f"and combiner: {self.combiner}"
        )

    @property
    def splitter(self):
        """Get the splitter of the state."""
        return self._splitter

    @splitter.setter
    def splitter(self, splitter):
        if splitter and not isinstance(splitter, (str, tuple, list)):
            raise hlpst.PydraStateError(
                "splitter has to be a string, a tuple or a list"
            )
        if splitter:
            self._splitter = hlpst.add_name_splitter(splitter, self.name)
        else:
            self._splitter = None

    @property
    def splitter_rpn_final(self):
        if self.combiner:
            _splitter_rpn_final = hlpst.remove_inp_from_splitter_rpn(
                deepcopy(self.splitter_rpn),
                self.right_combiner_all + self.left_combiner_all,
            )
            return _splitter_rpn_final
        else:
            return self.splitter_rpn

    @property
    def splitter_final(self):
        """ final splitter, after removing the combined fields"""
        return hlpst.rpn2splitter(self.splitter_rpn_final)

    @property
    def splitter_rpn(self):
        _splitter_rpn = hlpst.splitter2rpn(
            self.splitter, other_states=self.other_states
        )
        return _splitter_rpn

    @property
    def splitter_rpn_compact(self):
        if self.other_states:
            _splitter_rpn_compact = hlpst.splitter2rpn(
                self.splitter, other_states=self.other_states, state_fields=False
            )
            return _splitter_rpn_compact
        else:
            return self.splitter_rpn

    @property
    def right_splitter(self):
        """ current state splitter (i.e. the Right part)"""
        lr_flag = self._left_right_check(self.splitter)
        if lr_flag == "Left":
            return None
        elif lr_flag == "Right":
            return self.splitter
        elif lr_flag == "[Left, Right]":
            return self.splitter[1]

    @property
    def right_splitter_rpn(self):
        if self.right_splitter:
            right_splitter_rpn = hlpst.splitter2rpn(
                self.right_splitter, other_states=self.other_states
            )
            return right_splitter_rpn
        else:
            return []

    @property
    def left_splitter(self):
        """ splitters from the previous stated (i.e. the Light part)"""
        if hasattr(self, "_left_splitter"):
            return self._left_splitter
        else:
            return None

    @property
    def left_splitter_rpn(self):
        if self.left_splitter:
            left_splitter_rpn = hlpst.splitter2rpn(
                self.left_splitter, other_states=self.other_states
            )
            return left_splitter_rpn
        else:
            return []

    @property
    def left_splitter_rpn_compact(self):
        # left rpn part, but keeping the names of the nodes, e.g. [_NA, _NB, *]
        if self.left_splitter:
            left_splitter_rpn_compact = hlpst.splitter2rpn(
                self.left_splitter, other_states=self.other_states, state_fields=False
            )
            return left_splitter_rpn_compact
        else:
            return []

    @property
    def combiner(self):
        """Get the combiner associated to the state."""
        return self._combiner

    @combiner.setter
    def combiner(self, combiner):
        if combiner:
            if not isinstance(combiner, (str, list)):
                raise hlpst.PydraStateError("combiner has to be a string or a list")
            self._combiner = hlpst.add_name_combiner(ensure_list(combiner), self.name)
        else:
            self._combiner = []

    @property
    def right_combiner(self):
        return [comb for comb in self.combiner if self.name in comb]

    @property
    def left_combiner(self):
        if hasattr(self, "_left_combiner"):
            return self._left_combiner
        else:
            return list(set(self.combiner) - set(self.right_combiner))

    @property
    def right_combiner_all(self):
        if hasattr(self, "_right_combiner_all"):
            return self._right_combiner_all
        else:
            return self.right_combiner

    @property
    def left_combiner_all(self):
        if hasattr(self, "_left_combiner_all"):
            return self._left_combiner_all
        else:
            return self.left_combiner

    @left_combiner_all.setter
    def left_combiner_all(self, left_combiner_all):
        self._left_combiner_all = list(set(left_combiner_all))

    @property
    def other_states(self):
        return self._other_states

    @other_states.setter
    def other_states(self, other_states):
        if other_states:
            if not isinstance(other_states, dict):
                raise hlpst.PydraStateError("other states has to be a dictionary")
            else:
                for key, val in other_states.items():
                    if not val:
                        raise hlpst.PydraStateError(
                            f"connection from node {key} is empty"
                        )
            self._other_states = other_states
        else:
            self._other_states = {}

    @property
    def inner_inputs(self):
        """input fields from previous nodes"""
        if self.other_states:
            _inner_inputs = {}
            for name, (st, inp) in self.other_states.items():
                if f"_{st.name}" in self.splitter_rpn_compact:
                    _inner_inputs[f"{self.name}.{inp}"] = st
            return _inner_inputs
        else:
            return {}

    def update_connections(self, new_other_states=None, new_combiner=None):
        """ updating states connections and input groups"""
        if new_other_states:
            self.other_states = new_other_states
        self._connect_splitters()
        if new_combiner:
            self.combiner = new_combiner

    def _connect_splitters(self):
        """
        Connect splitters from previous nodes.
        Evaluates Left (the part from previous states) and Right (current state) parts.
        If left splitter is not provided the splitter has to be completed.

        """
        # TODO: should this be in the left_Splitter property?
        if self.splitter:
            # if splitter is string, have to check if this is Left or Right part (Left is required)
            if isinstance(self.splitter, str):
                # so this is the Left part
                if self.splitter.startswith("_"):
                    self._left_splitter = self._complete_left(left=self.splitter)
                else:  # this is Right part
                    self._left_splitter = self._complete_left()
            elif isinstance(self.splitter, (tuple, list)):
                lr_flag = self._left_right_check(self.splitter)
                if lr_flag == "Left":
                    self._left_splitter = self._complete_left(left=self.splitter)
                elif lr_flag == "Right":
                    self._left_splitter = self._complete_left()
                elif lr_flag == "[Left, Right]":
                    self._left_splitter = self._complete_left(left=self.splitter[0])
        else:
            # if there is no splitter, I create the Left part
            self._left_splitter = self._complete_left()

        if self.right_splitter:
            self.splitter = [
                deepcopy(self._left_splitter),
                deepcopy(self.right_splitter),
            ]
        else:
            self.splitter = deepcopy(self._left_splitter)

    def _complete_left(self, left=None):
        """Add all splitters from previous nodes (completing the Left part)."""
        if left:
            rpn_left = hlpst.splitter2rpn(
                left, other_states=self.other_states, state_fields=False
            )
            for name, (st, inp) in list(self.other_states.items())[::-1]:
                if "_{}".format(name) not in rpn_left and st.splitter_final:
                    left = ["_{}".format(name), left]
        else:
            left = ["_{}".format(name) for name in self.other_states]
            if len(left) == 1:
                left = left[0]
        return left

    def _left_right_check(self, splitter_part, check_nested=True):
        """
        Check if splitter_part is purely Left, Right
        or [Left, Right] if the splitter_part is a list (outer splitter)

        String is returned.

        If the splitter_part is mixed exception is raised.

        """
        rpn_part = hlpst.splitter2rpn(
            splitter_part, other_states=self.other_states, state_fields=False
        )
        inputs_in_splitter = [i for i in rpn_part if i not in ["*", "."]]
        others_in_splitter = [
            True if el.startswith("_") else False for el in inputs_in_splitter
        ]
        if all(others_in_splitter):
            return "Left"
        elif (not all(others_in_splitter)) and (not any(others_in_splitter)):
            return "Right"
        elif (
            isinstance(self.splitter, list)
            and check_nested
            and self._left_right_check(self.splitter[0], check_nested=False) == "Left"
            and self._left_right_check(self.splitter[1], check_nested=False) == "Right"
        ):
            return "[Left, Right]"  # Left and Right parts separated in outer scalar
        else:
            raise hlpst.PydraStateError(
                "Left and Right splitters are mixed - splitter invalid"
            )

    def set_input_groups(self, state_fields=True):
        """Evaluate groups, especially the final groups that address the combiner."""
        right_splitter_rpn = hlpst.splitter2rpn(
            self.right_splitter,
            other_states=self.other_states,
            state_fields=state_fields,
        )
        # merging groups from previous nodes if any input come from previous the nodes
        if self.inner_inputs:
            self._merge_previous_groups()
        keys_f, group_for_inputs_f, groups_stack_f, combiner_all = hlpst.splits_groups(
            right_splitter_rpn,
            combiner=self.right_combiner,
            inner_inputs=self.inner_inputs,
        )
        self._right_combiner_all = combiner_all
        if self.left_splitter and state_fields:  # if splitter has also the left part
            self._right_keys_final = keys_f
            self._right_group_for_inputs_final = group_for_inputs_f
            self._right_groups_stack_final = groups_stack_f
            if self.right_splitter:  # if Right part, adding groups from current st
                self._add_current_groups()

        else:
            self.group_for_inputs_final = group_for_inputs_f
            self.groups_stack_final = groups_stack_f
            self.keys_final = keys_f

    def _merge_previous_groups(self):
        """Merge groups from  all previous nodes."""
        last_gr = 0
        self.groups_stack_final = []
        self.group_for_inputs_final = {}
        self.keys_final = []
        if self.left_combiner:
            _, _, _, self.left_combiner_all = hlpst.splits_groups(
                self.left_splitter_rpn, combiner=self.left_combiner
            )
        for i, left_nm in enumerate(self.left_splitter_rpn_compact):
            if left_nm in ["*", "."]:
                continue
            if (
                i + 1 < len(self.left_splitter_rpn_compact)
                and self.left_splitter_rpn_compact[i + 1] == "."
            ):
                last_gr = last_gr - 1
            if left_nm[1:] not in self.other_states:
                raise hlpst.PydraStateError(
                    f"can't ask for splitter from {left_nm[1:]}, other nodes that are connected: {self.other_states}"
                )
            st = self.other_states[left_nm[1:]][0]
            # checking if left combiner contains any element from the st splitter
            st_combiner = [
                comb for comb in self.left_combiner_all if comb in st.splitter_rpn_final
            ]
            if not hasattr(st, "keys_final"):
                st.set_input_groups()
            if st_combiner:
                # keys and groups from previous states
                # after taking into account combiner from current state
                (
                    keys_f_st,
                    group_for_inputs_f_st,
                    groups_stack_f_st,
                    combiner_all_st,
                ) = hlpst.splits_groups(
                    st.splitter_rpn_final,
                    combiner=st_combiner,
                    inner_inputs=st.inner_inputs,
                )
                self.keys_final += keys_f_st  # st.keys_final
                if not hasattr(st, "group_for_inputs_final"):
                    raise hlpst.PydraStateError("previous state has to run first")
                group_for_inputs = group_for_inputs_f_st
                groups_stack = groups_stack_f_st
                self.left_combiner_all += combiner_all_st
            else:
                # if no element from st.splitter is in the current combiner,
                # using st attributes without changes
                if st.keys_final:
                    self.keys_final += st.keys_final
                group_for_inputs = st.group_for_inputs_final
                groups_stack = st.groups_stack_final

            group_for_inputs = {k: v + last_gr for k, v in group_for_inputs.items()}
            self.group_for_inputs_final.update(group_for_inputs)

            nmb_gr = 0
            for i, groups in enumerate(groups_stack):
                if i < len(self.groups_stack_final):
                    for gr in groups:
                        nmb_gr += 1
                        if gr + last_gr not in self.groups_stack_final[i]:
                            self.groups_stack_final[i].append(gr + last_gr)
                else:
                    self.groups_stack_final.append([gr + last_gr for gr in groups])
                    nmb_gr += len(groups)
            last_gr += nmb_gr

    def _add_current_groups(self):
        """Add additional groups from the current state."""
        self.keys_final += self._right_keys_final
        nr_gr_f = max(self.group_for_inputs_final.values()) + 1
        for inp, grs in self._right_group_for_inputs_final.items():
            if isinstance(grs, int):
                grs = grs + nr_gr_f
            else:  # a list
                grs = [gr + nr_gr_f for gr in grs]
            self.group_for_inputs_final[inp] = grs
        for i, stack in enumerate(self._right_groups_stack_final):
            if i == 0:
                stack = [gr + nr_gr_f for gr in stack]
                self.groups_stack_final[-1] += stack
            else:
                stack = [gr + nr_gr_f for gr in stack]
                self.groups_stack_final.append(stack)

    def splitter_validation(self):
        """ validating if the splitter is correct (after all states are connected)"""
        for spl in self.splitter_rpn_compact:
            if not (
                spl in [".", "*"]
                or spl.startswith("_")
                or spl.split(".")[0] == self.name
            ):
                raise hlpst.PydraStateError(
                    "can't include {} in the splitter, consider using _{}".format(
                        spl, spl.split(".")[0]
                    )
                )

    def combiner_validation(self):
        """ validating if the combiner is correct (after all states are connected)"""
        if self.combiner:
            if not self.splitter:
                raise hlpst.PydraStateError(
                    "splitter has to be set before setting combiner"
                )
            if set(self._combiner) - set(self.splitter_rpn):
                raise hlpst.PydraStateError("all combiners have to be in the splitter")

    def prepare_states(self, inputs, cont_dim=None):
        """
        Prepare a full list of state indices and state values.

        State Indices
            number of elements depends on the splitter

        State Values
            specific elements from inputs that can be used running interfaces

        """
        # checking if splitter and combiner have valid forms
        self.splitter_validation()
        self.combiner_validation()
        self.set_input_groups()
        # container dimension for each input, specifies how nested the input is
        if cont_dim is None:
            self.cont_dim = {}
        else:
            self.cont_dim = cont_dim
        if isinstance(inputs, BaseSpec):
            self.inputs = hlpst.inputs_types_to_dict(self.name, inputs)
        else:
            self.inputs = inputs
        if self.other_states:
            for nm, (st, _) in self.other_states.items():
                # I think now this if is never used
                if not hasattr(st, "states_ind"):
                    st.prepare_states(self.inputs, cont_dim=cont_dim)
                self.inputs.update(st.inputs)
        self.prepare_states_ind()
        self.prepare_states_val()

    def prepare_states_ind(self):
        """
        Calculate a list of dictionaries with state indices.

        Uses hlpst.splits.

        """
        # removing elements that are connected to inner splitter
        # (they will be taken into account in hlpst.splits anyway)
        # _comb part will be used in prepare_states_combined_ind
        # TODO: need tests in test_Workflow.py
        elements_to_remove = []
        elements_to_remove_comb = []
        for name, (st, inp) in self.other_states.items():
            if (
                "{}.{}".format(self.name, inp) in self.splitter_rpn
                and "_{}".format(name) in self.splitter_rpn_compact
            ):
                elements_to_remove.append("_{}".format(name))
                if "{}.{}".format(self.name, inp) not in self.combiner:
                    elements_to_remove_comb.append("_{}".format(name))

        partial_rpn = hlpst.remove_inp_from_splitter_rpn(
            deepcopy(self.splitter_rpn_compact), elements_to_remove
        )
        values_out_pr, keys_out_pr = hlpst.splits(
            partial_rpn,
            self.inputs,
            inner_inputs=self.inner_inputs,
            cont_dim=self.cont_dim,
        )
        values_pr = list(values_out_pr)

        self.ind_l = values_pr
        self.keys = keys_out_pr
        self.states_ind = list(hlpst.iter_splits(values_pr, self.keys))
        self.keys_final = self.keys
        if self.combiner:
            self.prepare_states_combined_ind(elements_to_remove_comb)
        else:
            self.ind_l_final = self.ind_l
            self.keys_final = self.keys
            self.final_combined_ind_mapping = {
                i: [i] for i in range(len(self.states_ind))
            }
            self.states_ind_final = self.states_ind
        return self.states_ind

    def prepare_states_combined_ind(self, elements_to_remove_comb):
        """Prepare the final list of dictionaries with indices after combiner."""
        partial_rpn_compact = hlpst.remove_inp_from_splitter_rpn(
            deepcopy(self.splitter_rpn_compact), elements_to_remove_comb
        )
        # combiner can have parts from the left splitter, so have to have rpn with states
        partial_rpn = hlpst.splitter2rpn(
            hlpst.rpn2splitter(partial_rpn_compact), other_states=self.other_states
        )
        combined_rpn = hlpst.remove_inp_from_splitter_rpn(
            deepcopy(partial_rpn), self.right_combiner_all + self.left_combiner_all
        )
        if combined_rpn:
            val_r, key_r = hlpst.splits(
                combined_rpn,
                self.inputs,
                inner_inputs=self.inner_inputs,
                cont_dim=self.cont_dim,
            )
            values = list(val_r)
        else:
            values = []
            key_r = []

        keys_out = key_r
        if values:
            self.ind_l_final = values
            self.keys_final = keys_out
            # groups after combiner
            ind_map = {
                tuple(hlpst.flatten(tup, max_depth=10)): ind
                for ind, tup in enumerate(self.ind_l_final)
            }
            self.final_combined_ind_mapping = {
                i: [] for i in range(len(self.ind_l_final))
            }
            for ii, st in enumerate(self.states_ind):
                ind_f = tuple([st[k] for k in self.keys_final])
                self.final_combined_ind_mapping[ind_map[ind_f]].append(ii)
        else:
            self.ind_l_final = values
            self.keys_final = keys_out
            # should be 0 or None?
            self.final_combined_ind_mapping = {0: list(range(len(self.states_ind)))}
        self.states_ind_final = list(
            hlpst.iter_splits(self.ind_l_final, self.keys_final)
        )

    def prepare_states_val(self):
        """Evaluate states values having states indices."""
        self.states_val = list(
            hlpst.map_splits(self.states_ind, self.inputs, cont_dim=self.cont_dim)
        )
        return self.states_val

    def prepare_inputs(self):
        """
        Preparing inputs indices, merges input from previous states.

        Includes indices for fields from inner splitters
        (removes elements connected to the inner splitters fields).

        """
        if not self.other_states:
            self.inputs_ind = self.states_ind
        else:
            # elements from the current node (the Right part)
            if self.right_splitter_rpn:
                values_inp, keys_inp = hlpst.splits(
                    self.right_splitter_rpn,
                    self.inputs,
                    inner_inputs=self.inner_inputs,
                    cont_dim=self.cont_dim,
                )
                inputs_ind = values_inp
            else:
                keys_inp = []
                inputs_ind = []

            # merging elements that come from previous nodes outputs
            # states that are connected to inner splitters are treated differently
            # (already included in inputs_ind)
            keys_inp_prev = []
            inputs_ind_prev = []
            connected_to_inner = []
            for ii, el in enumerate(self.left_splitter_rpn_compact):
                if el in ["*", "."]:
                    continue
                st, inp = self.other_states[el[1:]]
                if (
                    "{}.{}".format(self.name, inp) in self.splitter_rpn
                ):  # inner splitter
                    connected_to_inner += [
                        el for el in st.splitter_rpn_final if el not in [".", "*"]
                    ]
                else:  # previous states that are not connected to inner splitter
                    st_ind = range(len(st.states_ind_final))
                    if inputs_ind_prev:
                        # in case the Left part has scalar parts (not very well tested)
                        if self.left_splitter_rpn_compact[ii + 1] == ".":
                            inputs_ind_prev = hlpst.op["."](inputs_ind_prev, st_ind)
                        else:
                            inputs_ind_prev = hlpst.op["*"](inputs_ind_prev, st_ind)
                    else:
                        inputs_ind_prev = hlpst.op["*"](st_ind)
                    keys_inp_prev += ["{}.{}".format(self.name, inp)]
            keys_inp = keys_inp_prev + keys_inp

            if inputs_ind and inputs_ind_prev:
                inputs_ind = hlpst.op["*"](inputs_ind_prev, inputs_ind)
            elif inputs_ind:
                inputs_ind = hlpst.op["*"](inputs_ind)
            elif inputs_ind_prev:
                inputs_ind = hlpst.op["*"](inputs_ind_prev)
            else:
                inputs_ind = []

            # iter_splits using inputs from current state/node
            self.inputs_ind = list(hlpst.iter_splits(inputs_ind, keys_inp))
            # removing elements that are connected to inner splitter
            # TODO - add tests to test_workflow.py (not sure if we want to remove it)
            for el in connected_to_inner:
                [dict.pop(el) for dict in self.inputs_ind]
