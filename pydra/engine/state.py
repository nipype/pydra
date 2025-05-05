"""Keeping track of mapping and reduce operations over tasks."""

from copy import deepcopy
import itertools
from math import prod
import logging
import typing as ty
from pydra.utils.typing import StateArray, TypeParser
from pydra.utils.general import ensure_list, attrs_values


logger = logging.getLogger("pydra")


OutputsType = ty.TypeVar("OutputsType")


class State:
    """
    A class that specifies a State of all tasks.

      * It's only used when a job have a splitter.
      * It contains all information about splitter, combiner, final splitter,
        and input values for specific job states
        (specified by the splitter and the input).
      * It also contains information about the final groups and the final splitter
        if combiner is available.

    Attributes
    ----------
    name : :obj:`str`
        name of the state that is the same as a name of the job
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
        indices for all job inputs (i.e. inputs that are relevant
        for current job, can be outputs from previous nodes)
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

    def __init__(
        self,
        name,
        splitter=None,
        combiner=None,
        container_ndim=None,
        other_states=None,
    ):
        """
        Initialize a state.

        Parameters
        ----------
        name : :obj:`str`
            name (should be the same as the job's name)
        splitter : :obj:`str`, or :obj:`tuple`, or :obj:`list`
            splitter of a job
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
        self.container_ndim = container_ndim or {}
        self._inner_container_ndim = {}
        self._inputs_ind = None
        # if other_states, the connections have to be updated
        if self.other_states:
            self.update_connections()

    def __str__(self):
        """Generate a string representation of the object."""
        return (
            f"State for {self.name} with a splitter: {self.splitter} "
            f"and combiner: {self.combiner}"
        )

    def depth(self, before_combine: bool = False) -> int:
        """Return the number of splits of the state, i.e. the number nested
        state arrays to wrap around the type of lazy out fields

        Parameters
        ----------
        before_combine : :obj:`bool`
            if True, the depth is after combining the fields, otherwise it is before
            any combinations

        Returns
        -------
        int
            number of splits in the state (i.e. linked splits only add 1)
        """

        # replace field names with 1 or 0 (1 if the field is included in the state)
        include_rpn = [
            (
                s
                if s in [".", "*"]
                else (1 if before_combine else int(s not in self.combiner))
            )
            for s in self.splitter_rpn
        ]

        stack = []
        for opr in include_rpn:
            if opr == ".":
                assert len(stack) >= 2
                opr1 = stack.pop()
                opr2 = stack.pop()
                stack.append(opr1 and opr2)
            elif opr == "*":
                assert len(stack) >= 2
                stack.append(stack.pop() + stack.pop())
            else:
                stack.append(opr)
        assert len(stack) == 1
        return stack[0]

    def nest_output_type(self, type_: type) -> type:
        """Nests a type of an output field in a combination of lists and state-arrays
        based on the state's splitter and combiner

        Parameters
        ----------
        type_ : type
            the type of the output field

        Returns
        -------
        type
            the nested type of the output field
        """

        state_array_depth = self.depth()

        # If there is a combination, it will get flattened into a single list
        if self.depth(before_combine=True) > state_array_depth:
            type_ = list[type_]

        # Nest the uncombined state arrays around the type
        for _ in range(state_array_depth):
            type_ = StateArray[type_]
        return type_

    @classmethod
    def combine_state_arrays(cls, type_: type) -> type:
        """Collapses (potentially nested) state array(s) into a single list"""
        if TypeParser.get_origin(type_) is StateArray:
            # Implicitly combine any remaining uncombined states into a single
            # list
            type_ = list[TypeParser.strip_splits(type_)[0]]
        return type_

    @property
    def splitter(self):
        """Get the splitter of the state."""
        return self._splitter

    @splitter.setter
    def splitter(self, splitter):
        if splitter and not isinstance(splitter, (str, tuple, list)):
            raise PydraStateError("splitter has to be a string, a tuple or a list")
        if splitter:
            self._splitter = add_name_splitter(splitter, self.name)
        else:
            self._splitter = None
        # updating splitter_rpn
        self._splitter_rpn_updates()

    def _splitter_rpn_updates(self):
        """updating splitter_rpn and splitter_rpn_compact"""
        try:
            self._splitter_rpn = splitter2rpn(
                self.splitter, other_states=self.other_states
            )
        # other_state might not be ready yet
        except PydraStateError:
            self._splitter_rpn = None

        if self.other_states or self._splitter_rpn is None:
            self._splitter_rpn_compact = splitter2rpn(
                self.splitter, other_states=self.other_states, state_fields=False
            )
        else:
            self._splitter_rpn_compact = self._splitter_rpn

    @property
    def splitter_rpn(self):
        """splitter in :abbr:`RPN (Reverse Polish Notation)`"""
        # if splitter_rpn was not calculated within splitter.setter
        if self._splitter_rpn is None:
            self._splitter_rpn = splitter2rpn(
                self.splitter, other_states=self.other_states
            )
        return self._splitter_rpn

    @property
    def splitter_rpn_compact(self):
        r"""splitter in :abbr:`RPN (Reverse Polish Notation)`
        with a compact representation of the prev-state part (i.e. without unwrapping
        the part that comes from the previous states), e.g., [_NA, _NB, \*]
        """
        return self._splitter_rpn_compact

    @property
    def splitter_final(self):
        """the final splitter, after removing the combined fields"""
        return rpn2splitter(self.splitter_rpn_final)

    @property
    def splitter_rpn_final(self):
        if self.combiner:
            _splitter_rpn_final = remove_inp_from_splitter_rpn(
                deepcopy(self.splitter_rpn),
                self.current_combiner_all + self.prev_state_combiner_all,
            )
            return _splitter_rpn_final
        else:
            return self.splitter_rpn

    @property
    def current_splitter(self):
        """the current part of the splitter,
        i.e. the part that is related to the current job's state only
        (doesn't include fields propagated from the previous tasks)
        """
        if hasattr(self, "_current_splitter"):
            return self._current_splitter
        else:
            return self.splitter

    @property
    def inputs_ind(self):
        """dictionary for every state that contains indices for all job inputs
        (i.e. inputs that are relevant for current job, can be outputs from previous nodes)
        """
        if self._inputs_ind is None:
            raise RuntimeError(
                "inputs_ind is not set, please run prepare_states() on the state first"
            )
        return self._inputs_ind

    @current_splitter.setter
    def current_splitter(self, current_splitter):
        self._current_splitter = current_splitter
        # updating rpn
        self._current_splitter_rpn_updates()

    def _current_splitter_rpn_updates(self):
        """updating current_splitter_rpn"""
        if self._current_splitter:
            self._current_splitter_rpn = splitter2rpn(
                self.current_splitter, other_states=self.other_states
            )
        else:
            self._current_splitter_rpn = []

    @property
    def current_splitter_rpn(self):
        """the current part of the splitter using RPN"""
        if hasattr(self, "_current_splitter_rpn"):
            return self._current_splitter_rpn
        else:
            return self.splitter_rpn

    @property
    def prev_state_splitter(self):
        """the prev-state part of the splitter,
        i.e. the part that comes from the previous tasks' states
        """
        if hasattr(self, "_prev_state_splitter"):
            return self._prev_state_splitter
        else:
            return None

    @prev_state_splitter.setter
    def prev_state_splitter(self, prev_state_splitter):
        self._prev_state_splitter = prev_state_splitter
        # updating rpn splitters
        self._prev_state_splitter_rpn_updates()

    def _prev_state_splitter_rpn_updates(self):
        """updating prev_state_splitter_rpn/_rpn_compact"""
        if self._prev_state_splitter:
            self._prev_state_splitter_rpn = splitter2rpn(
                self.prev_state_splitter, other_states=self.other_states
            )
        else:
            self._prev_state_splitter_rpn = []

        if self.other_states:
            self._prev_state_splitter_rpn_compact = splitter2rpn(
                self.prev_state_splitter,
                other_states=self.other_states,
                state_fields=False,
            )
        else:
            self._prev_state_splitter_rpn_compact = self._prev_state_splitter_rpn

    @property
    def prev_state_splitter_rpn(self):
        """the prev-state art of the splitter using RPN"""
        return self._prev_state_splitter_rpn

    @property
    def prev_state_splitter_rpn_compact(self):
        r"""the prev-state part of the splitter using RPN in a compact form,
        (without unwrapping the states from previous nodes), e.g. [_NA, _NB, \*]
        """
        return self._prev_state_splitter_rpn_compact

    @property
    def container_ndim_all(self):
        # adding inner_container_ndim to the general container_dimension provided by the users
        container_ndim_all = deepcopy(self.container_ndim)
        for k, v in self._inner_container_ndim.items():
            container_ndim_all[k] = container_ndim_all.get(k, 1) + v
        return container_ndim_all

    @property
    def combiner(self):
        """the combiner associated to the state."""
        return self._combiner

    @combiner.setter
    def combiner(self, combiner):
        if combiner:
            if not isinstance(combiner, (str, list)):
                raise PydraStateError("combiner has to be a string or a list")
            self._combiner = add_name_combiner(ensure_list(combiner), self.name)
        else:
            self._combiner = []

    @property
    def current_combiner(self):
        """the current part of the combiner,
        i.e. the part that is related to the current job's state only
        (doesn't include fields propagated from the previous tasks)
        """
        return [comb for comb in self.combiner if self.name in comb]

    @property
    def current_combiner_all(self):
        """the current part of the combiner including all the fields
        that should be combined (i.e. not only the fields that are explicitly
        set, but also the fields that re in the same group/axis and had to be combined
        together, e.g., if splitter is (a, b) a and b has to be combined together)
        """
        if hasattr(self, "_current_combiner_all"):
            return self._current_combiner_all
        else:
            return self.current_combiner

    @property
    def prev_state_combiner(self):
        """the prev-state part of the combiner,
        i.e. the part that comes from the previous tasks' states
        """
        if hasattr(self, "_prev_state_combiner"):
            return self._prev_state_combiner
        else:
            return list(set(self.combiner) - set(self.current_combiner))

    @property
    def prev_state_combiner_all(self):
        """the prev-state part of the combiner including all the fields
        that should be combined (i.e. not only the fields that are explicitly
        set, but also the fields that re in the same group/axis and had to be combined
        together, e.g., if splitter is (a, b) a and b has to be combined together)
        """
        if hasattr(self, "_prev_state_combiner_all"):
            return list(set(self._prev_state_combiner_all))
        else:
            return self.prev_state_combiner

    @property
    def other_states(self):
        """specifies the connections with previous states, uses dictionary:
        {name of a previous state: (previous state, input field from current state)}
        """
        return self._other_states

    @other_states.setter
    def other_states(self, other_states):
        if other_states:
            if not isinstance(other_states, dict):
                raise PydraStateError("other states has to be a dictionary")
            else:
                for key, val in other_states.items():
                    if not val:
                        raise PydraStateError(f"connection from node {key} is empty")
            # ensuring that the connected fields are set as a list
            self._other_states = {
                nm: (st, ensure_list(flds)) for nm, (st, flds) in other_states.items()
            }
        else:
            self._other_states = {}

    @property
    def inner_inputs(self):
        """specifies connections between fields from the current state
        with the specific state from the previous states, uses dictionary
        ``{input name for current state: the previous state}``
        """
        if self.other_states:
            _inner_inputs = {}
            for name, (st, inp_l) in self.other_states.items():
                if f"_{st.name}" in self.splitter_rpn_compact:
                    for inp in inp_l:
                        _inner_inputs[f"{self.name}.{inp}"] = st
            return _inner_inputs
        else:
            return {}

    def update_connections(self, new_other_states=None, new_combiner=None):
        """updating connections, can use a new other_states and combiner

        Parameters
        ----------
        new_other_states : :obj:`dict`, optional
            dictionary with new other_states, will be set before updating connections
        new_combiner : :obj:`str`, or :obj:`list`, optional
            new combiner
        """
        if new_other_states:
            self.other_states = new_other_states
        self._connect_splitters()
        if new_combiner:
            self.combiner = new_combiner

    def _connect_splitters(self):
        """
        Connect splitters from the previous nodes.
        Evaluates the prev-state part of the splitter (i.e. the part from the previous states)
        and the current part of the splitter(i.e., the current state).
        If the prev-state splitter is not provided the splitter has to be completed.
        """
        # TODO: should this be in the prev-state_Splitter property?
        if self.splitter:
            # if splitter is string, have to check if this is prev-state
            # or current part (prev-state is required)
            if isinstance(self.splitter, str):
                # so this is the prev-state part
                if self.splitter.startswith("_"):
                    self.prev_state_splitter = self._complete_prev_state(
                        prev_state=self.splitter
                    )
                    self.current_splitter = None
                else:  # this is the current part
                    self.prev_state_splitter = self._complete_prev_state()
                    self.current_splitter = self.splitter
            elif isinstance(self.splitter, (tuple, list)):
                lr_flag = self._prevst_current_check(self.splitter)
                if lr_flag == "prev-state":
                    self.prev_state_splitter = self._complete_prev_state(
                        prev_state=self.splitter
                    )
                    self.current_splitter = None
                elif lr_flag == "current":
                    self.prev_state_splitter = self._complete_prev_state()
                    self.current_splitter = self.splitter
                elif lr_flag == "[prev-state, current]":
                    self.prev_state_splitter = self._complete_prev_state(
                        prev_state=self.splitter[0]
                    )
                    self.current_splitter = self.splitter[1]
        else:
            # if there is no splitter, I create the prev-state part
            self.prev_state_splitter = self._complete_prev_state()
            self.current_splitter = None

        if self.current_splitter:
            self.splitter = [
                deepcopy(self.prev_state_splitter),
                deepcopy(self.current_splitter),
            ]
        else:
            self.splitter = deepcopy(self.prev_state_splitter)

    def _complete_prev_state(self, prev_state=None):
        """Add all splitters from the previous nodes (completing the prev-state part).

        Parameters
        ----------
        prev_state :  :obj:`str`, or :obj:`list`, or :obj:`tuple`, optional
            the prev-state part of the splitter, that has to be completed
        """
        if prev_state:
            rpn_prev_state = splitter2rpn(
                prev_state, other_states=self.other_states, state_fields=False
            )
            for name, (st, inp) in list(self.other_states.items())[::-1]:
                if f"_{name}" not in rpn_prev_state and st.splitter_final:
                    prev_state = [f"_{name}", prev_state]
        else:
            prev_state = [f"_{name}" for name in self.other_states]
        # TODO: what if prev state is a tuple
        if isinstance(prev_state, list):
            prev_state = self._remove_repeated(prev_state)
            prev_state = self._add_state_history(ensure_list(prev_state))
        if len(prev_state) == 1:
            prev_state = prev_state[0]
        return prev_state

    def _remove_repeated(self, previous_splitters):
        """removing states from previous tasks that are repeated"""
        for el in previous_splitters:
            if el[1:] not in self.other_states:
                raise PydraStateError(
                    f"can't ask for splitter from {el[1:]}, other nodes that are connected: "
                    f"{self.other_states}"
                )

        repeated = {
            (el, previous_splitters.count(el))
            for el in previous_splitters
            if previous_splitters.count(el) > 1
        }
        if repeated:
            # assuming that I want to remove from right
            previous_splitters.reverse()
            for el, cnt in repeated:
                for ii in range(cnt):
                    previous_splitters.remove(el)
            previous_splitters.reverse()
        return previous_splitters

    def _add_state_history(self, previous_splitters):
        """analysing history of each state from previous states
        expanding previous_splitters list and oter_states
        """
        othst_w_currst = []  # el from other_states that have only current states
        othst_w_prevst = (
            []
        )  # el from other_states that have only prev st (nm, st.prev_st_spl)
        othst_w_currst_prevst = (
            []
        )  # el from other_states that have both (nm, st.prev_st_spl)
        for el in previous_splitters:
            nm = el[1:]
            st = self.other_states[nm][0]
            if not st.other_states:
                # states that has no other connections
                othst_w_currst.append(el)
            else:  # element has previous_connection
                if st.current_splitter:  # final?
                    # states that has previous connections and it's own splitter
                    othst_w_currst_prevst.append((el, st.prev_state_splitter))
                else:
                    # states with previous connections but no additional splitter
                    othst_w_prevst.append((el, st.prev_state_splitter))

        for el in othst_w_prevst:
            el_nm, el_prevst = el[0][1:], el[1]
            # checking if the element's previous state is not already included
            repeated_prev = set(ensure_list(el_prevst)).intersection(othst_w_currst)
            if repeated_prev:
                for r_el in repeated_prev:
                    r_nm = r_el[1:]
                    # updating self.other_states
                    self._other_states[r_nm] = (
                        self.other_states[r_nm][0],
                        self.other_states[r_nm][1] + self.other_states[el_nm][1],
                    )
                new_st = set(ensure_list(el_prevst)) - set(othst_w_currst)
                if not new_st:
                    # removing element from the previous_splitter if no new_st
                    previous_splitters.remove(el[0])
                else:
                    # adding elements to self.other_states
                    for new_el in new_st:
                        new_nm = new_el[1:]
                        self._other_states[new_nm] = (
                            self.other_states[el_nm][0].other_states[new_nm][0],
                            self.other_states[el_nm][1],
                        )
                    # removing el of the splitter and adding new_st instead
                    ind = previous_splitters.index(el[0])
                    if ind == len(previous_splitters) - 1:
                        previous_splitters = previous_splitters[:-1] + list(new_st)
                    else:
                        previous_splitters = (
                            previous_splitters[:ind]
                            + list(new_st)
                            + previous_splitters[ind + 1 :]
                        )
        # TODO: this part is not tested, needs more work
        for el in othst_w_currst_prevst:
            repeated_prev = set(ensure_list(el[1])).intersection(othst_w_currst)
            # removing el if it's repeated: in el.other_states and othst_w_currst
            if repeated_prev:
                for r_el in repeated_prev:
                    previous_splitters.remove(r_el)
        return previous_splitters

    def _prevst_current_check(self, splitter_part, check_nested=True):
        """
        Check if splitter_part is purely prev-state part, the current part,
        or mixed ([prev-state, current]) if the splitter_part is a list (outer splitter)

        Parameters
        ----------
        splitter_part : :obj:`str`, or :obj:`list`, or :obj:`tuple`
            Part of the splitter that is being check
        check_nested : :obj:`bool`, optional
            If True, the nested parts are checked.

        Returns
        -------
        str
           describes the type - "prev-state" or "current"

        If the splitter_part is mixed exception is raised.

        """
        rpn_part = splitter2rpn(
            splitter_part, other_states=self.other_states, state_fields=False
        )
        inputs_in_splitter = [i for i in rpn_part if i not in ["*", "."]]
        others_in_splitter = [
            True if el.startswith("_") else False for el in inputs_in_splitter
        ]
        if all(others_in_splitter):
            return "prev-state"
        elif (not all(others_in_splitter)) and (not any(others_in_splitter)):
            return "current"
        elif (
            isinstance(self.splitter, list)
            and check_nested
            and self._prevst_current_check(self.splitter[0], check_nested=False)
            == "prev-state"
            and self._prevst_current_check(self.splitter[1], check_nested=False)
            == "current"
        ):
            # the prev-state and the current parts separated in outer scalar
            return "[prev-state, current]"
        else:
            raise PydraStateError(
                "prev-state and current splitters are mixed - splitter invalid"
            )

    def set_input_groups(self, state_fields=True):
        """Evaluates groups, especially the final groups that address the combiner.

        Parameters
        ----------
        state_fields : :obj:`bool`
            if False the splitter from the previous states are unwrapped
        """
        current_splitter_rpn = splitter2rpn(
            self.current_splitter,
            other_states=self.other_states,
            state_fields=state_fields,
        )
        # merging groups from previous nodes if any input come from previous states
        if self.other_states:
            self._merge_previous_groups()
        keys_f, group_for_inputs_f, groups_stack_f, combiner_all = splits_groups(
            current_splitter_rpn,
            combiner=self.current_combiner,
            inner_inputs=self.inner_inputs,
        )
        self._current_combiner_all = combiner_all
        if (
            self.prev_state_splitter and state_fields
        ):  # if splitter has also the prev-state part
            self._current_keys_final = keys_f
            self._current_group_for_inputs_final = group_for_inputs_f
            self._current_groups_stack_final = groups_stack_f
            if (
                self.current_splitter
            ):  # if the current part, adding groups from current st
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
        if self.prev_state_combiner:
            _, _, _, self._prev_state_combiner_all = splits_groups(
                self.prev_state_splitter_rpn, combiner=self.prev_state_combiner
            )
        for i, prev_nm in enumerate(self.prev_state_splitter_rpn_compact):
            if prev_nm in ["*", "."]:
                continue
            if (
                i + 1 < len(self.prev_state_splitter_rpn_compact)
                and self.prev_state_splitter_rpn_compact[i + 1] == "."
            ):
                last_gr = last_gr - 1
            if prev_nm[1:] not in self.other_states:
                raise PydraStateError(
                    f"can't ask for splitter from {prev_nm[1:]}, "
                    f"other nodes that are connected: {self.other_states}"
                )
            st = self.other_states[prev_nm[1:]][0]
            # checking if prev-state combiner contains any element from the st splitter
            st_combiner = [
                comb
                for comb in self.prev_state_combiner_all
                if comb in st.splitter_rpn_final
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
                ) = splits_groups(
                    st.splitter_rpn_final,
                    combiner=st_combiner,
                    inner_inputs=st.inner_inputs,
                )
                self.keys_final += keys_f_st  # st.keys_final
                if not hasattr(st, "group_for_inputs_final"):
                    raise PydraStateError("previous state has to run first")
                group_for_inputs = group_for_inputs_f_st
                groups_stack = groups_stack_f_st
                self._prev_state_combiner_all += combiner_all_st
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
        self.keys_final += self._current_keys_final
        nr_gr_f = max(self.group_for_inputs_final.values()) + 1
        for inp, grs in self._current_group_for_inputs_final.items():
            if isinstance(grs, int):
                grs = grs + nr_gr_f
            else:  # a list
                grs = [gr + nr_gr_f for gr in grs]
            self.group_for_inputs_final[inp] = grs
        for i, stack in enumerate(self._current_groups_stack_final):
            if i == 0:
                stack = [gr + nr_gr_f for gr in stack]
                self.groups_stack_final[-1] += stack
            else:
                stack = [gr + nr_gr_f for gr in stack]
                self.groups_stack_final.append(stack)

    def splitter_validation(self):
        """validating if the splitter is correct (after all states are connected)"""
        for spl in self.splitter_rpn_compact:
            if not (
                spl in [".", "*"]
                or (spl.startswith("_") and spl[1:] in self.other_states)
                or spl.split(".")[0] == self.name
            ):
                raise PydraStateError(
                    "can't include {} in the splitter, consider using _{}".format(
                        spl, spl.split(".")[0]
                    )
                )

    def combiner_validation(self):
        """validating if the combiner is correct (after all states are connected)"""
        if local_names := set(
            c for c in self.combiner if c.startswith(self.name + ".")
        ):
            if not self.splitter:
                raise PydraStateError(
                    "splitter has to be set before setting combiner with field names "
                    f"in the current node {list(local_names)}"
                )
            if missing := local_names - set(self.splitter_rpn):
                raise PydraStateError(
                    "The following field names from the current node referenced in the "
                    f"combiner, {list(missing)} are not in the splitter"
                )

    def prepare_states(
        self,
        inputs: dict[str, ty.Any],
        container_ndim: dict[str, int] | None = None,
    ):
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
        self.inputs = inputs
        if container_ndim is not None:
            self.container_ndim = container_ndim
        if self.other_states:
            st: State
            for nm, (st, _) in self.other_states.items():
                self.inputs.update(st.inputs)
                self.container_ndim.update(st.container_ndim_all)

        self.prepare_states_ind()
        self.prepare_states_val()

    def prepare_states_ind(self):
        """
        Calculate a list of dictionaries with state indices.

        Uses splits.

        """
        # removing elements that are connected to inner splitter
        # (they will be taken into account in splits anyway)
        # _comb part will be used in prepare_states_combined_ind
        # TODO: need tests in test_Workflow.py
        elements_to_remove = []
        elements_to_remove_comb = []
        for name, (st, inp_l) in self.other_states.items():
            for inp in inp_l:
                if (
                    f"{self.name}.{inp}" in self.splitter_rpn
                    and f"_{name}" in self.splitter_rpn_compact
                ):
                    elements_to_remove.append(f"_{name}")
                    if f"{self.name}.{inp}" not in self.combiner:
                        elements_to_remove_comb.append(f"_{name}")

        partial_rpn = remove_inp_from_splitter_rpn(
            deepcopy(self.splitter_rpn_compact), elements_to_remove
        )
        values_out_pr, keys_out_pr = self.splits(
            partial_rpn,
        )
        values_pr = list(values_out_pr)

        self.ind_l = values_pr
        self.keys = keys_out_pr
        self.states_ind = list(iter_splits(values_pr, self.keys))
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
        """Prepare the final list of dictionaries with indices after combiner.

        Parameters
        ----------
        elements_to_remove_comb : :obj:`list`
            elements of the splitter that should be removed due to the combining
        """
        partial_rpn_compact = remove_inp_from_splitter_rpn(
            deepcopy(self.splitter_rpn_compact), elements_to_remove_comb
        )
        # combiner can have parts from the prev-state splitter, so have to have rpn with states
        partial_rpn = splitter2rpn(
            rpn2splitter(partial_rpn_compact), other_states=self.other_states
        )
        combined_rpn = remove_inp_from_splitter_rpn(
            deepcopy(partial_rpn),
            self.current_combiner_all + self.prev_state_combiner_all,
        )
        if combined_rpn:
            val_r, key_r = self.splits(
                combined_rpn,
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
                tuple(flatten(tup, max_depth=10)): ind
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
        self.states_ind_final = list(iter_splits(self.ind_l_final, self.keys_final))

    def prepare_states_val(self):
        """Evaluate states values having states indices."""
        self.states_val = list(
            map_splits(
                self.states_ind, self.inputs, container_ndim=self.container_ndim_all
            )
        )
        return self.states_val

    def prepare_inputs(self):
        """
        Preparing inputs indices, merges input from previous states.

        Includes indices for fields from inner splitters
        (removes elements connected to the inner splitters fields).

        """
        if not self.other_states:
            self._inputs_ind = self.states_ind
        else:
            # elements from the current node (the current part of the splitter)
            if self.current_splitter_rpn:
                values_inp, keys_inp = self.splits(
                    self.current_splitter_rpn,
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
            for ii, el in enumerate(self.prev_state_splitter_rpn_compact):
                if el in ["*", "."]:
                    continue
                st, inp_l = self.other_states[el[1:]]
                inp_l = [f"{self.name}.{inp}" for inp in inp_l]
                if set(inp_l).intersection(self.splitter_rpn):  # inner splitter
                    connected_to_inner += [
                        el for el in st.splitter_rpn_final if el not in [".", "*"]
                    ]
                else:  # previous states that are not connected to inner splitter
                    st_ind = range(len(st.states_ind_final))
                    if inputs_ind_prev:
                        # in case the prev-state part has scalar parts (not very well tested)
                        if self.prev_state_splitter_rpn_compact[ii + 1] == ".":
                            inputs_ind_prev = op["."](inputs_ind_prev, st_ind)
                        else:
                            inputs_ind_prev = op["*"](inputs_ind_prev, st_ind)
                    else:
                        # TODO: more tests needed
                        inputs_ind_prev = op["."](*[st_ind] * len(inp_l))
                    keys_inp_prev += inp_l
            keys_inp = keys_inp_prev + keys_inp

            if inputs_ind and inputs_ind_prev:
                inputs_ind = op["*"](inputs_ind_prev, inputs_ind)
            elif inputs_ind:
                inputs_ind = op["*"](inputs_ind)
            elif inputs_ind_prev:
                inputs_ind = op["*"](inputs_ind_prev)
            else:
                inputs_ind = []

            # iter_splits using inputs from current state/node
            self._inputs_ind = list(iter_splits(inputs_ind, keys_inp))
            # removing elements that are connected to inner splitter
            # TODO - add tests to test_workflow.py (not sure if we want to remove it)
            for el in connected_to_inner:
                [dict.pop(el) for dict in self._inputs_ind]

    def splits(self, splitter_rpn):
        """
        Splits input variable as specified by splitter

        Parameters
        ----------
        splitter_rpn : list
            splitter in RPN notation
        Returns
        -------
        splitter : list
            each element contains indices for input variables
        keys: list
            names of input variables

        """
        # analysing states from connected tasks if inner_inputs
        previous_states_ind = {
            f"_{v.name}": (v.ind_l_final, v.keys_final)
            for v in self.inner_inputs.values()
        }

        # when splitter is a single element (no operators)
        if len(splitter_rpn) == 1:
            op_single = splitter_rpn[0]
            # splitter comes from the previous state
            if op_single.startswith("_"):
                return previous_states_ind[op_single]
            return self._single_op_splits(op_single)

        stack = []
        keys = []

        # iterating splitter_rpn
        for token in splitter_rpn:
            if token not in [".", "*"]:  # token is one of the input var
                # adding variable to the stack
                stack.append(token)
            else:
                # removing Right and Left var from the stack
                term_R = stack.pop()
                term_L = stack.pop()

                # analysing and processing Left and Right terms
                # both terms (Left and Right) are strings, so they were not processed yet by the function
                if isinstance(term_L, str) and isinstance(term_R, str):
                    shape_L, var_ind_L, new_keys_L = self._processing_terms(
                        term_L, previous_states_ind
                    )
                    shape_R, var_ind_R, new_keys_R = self._processing_terms(
                        term_R, previous_states_ind
                    )
                    keys = keys + new_keys_L + new_keys_R
                elif isinstance(term_L, str):
                    shape_L, var_ind_L, new_keys_L = self._processing_terms(
                        term_L, previous_states_ind
                    )
                    var_ind_R, shape_R = term_R
                    keys = new_keys_L + keys
                elif isinstance(term_R, str):
                    shape_R, var_ind_R, new_keys_R = self._processing_terms(
                        term_R, previous_states_ind
                    )
                    var_ind_L, shape_L = term_L
                    keys = keys + new_keys_R
                else:
                    var_ind_L, shape_L = term_L
                    var_ind_R, shape_R = term_R

                # checking shapes and creating newshape
                if token == ".":
                    if shape_L != shape_R:
                        raise ValueError(
                            f"Operands {term_R} and {term_L} do not have same shape"
                        )
                    newshape = shape_R
                elif token == "*":
                    newshape = tuple(list(shape_L) + list(shape_R))

                # creating a new iterator with all indices for the current operation
                # and pushing it to the stack
                pushval = (op[token](var_ind_L, var_ind_R), newshape)
                stack.append(pushval)

        # when everything is processed it should be one element in the stack
        # that contains iterator with variable indices after splitting for all keys
        var_ind = stack.pop()
        if isinstance(var_ind, tuple):
            var_ind = var_ind[0]

        return var_ind, keys

    def _processing_terms(self, term, previous_states_ind):
        """processing a specific term to get new keys from the term,
        an iterator with variable indices and matching keys
        """
        if term.startswith("_"):
            var_ind, new_keys = previous_states_ind[term]
            shape = (len(var_ind),)
        else:
            container_ndim = self.container_ndim_all.get(term, 1)
            shape = input_shape(self.inputs[term], container_ndim=container_ndim)
            var_ind = range(prod(shape))
            new_keys = [term]
            # checking if the term is in inner_inputs
            if term in self.inner_inputs:
                # TODO: have to be changed if differ length
                inner_len = [shape[-1]] * prod(shape[:-1])
                # this come from the previous node
                outer_ind = self.inner_inputs[term].ind_l
                var_ind_out = itertools.chain.from_iterable(
                    itertools.repeat(x, n) for x, n in zip(outer_ind, inner_len)
                )
                var_ind = op["."](var_ind_out, var_ind)
                new_keys = self.inner_inputs[term].keys_final + new_keys

        return shape, var_ind, new_keys

    def _single_op_splits(self, op_single):
        """splits function if splitter is a singleton"""
        shape = input_shape(
            self.inputs[op_single],
            container_ndim=self.container_ndim_all.get(op_single, 1),
        )
        val_ind = range(prod(shape))
        if op_single in self.inner_inputs:
            # TODO: have to be changed if differ length
            inner_len = [shape[-1]] * prod(shape[:-1])

            # this come from the previous node
            outer_ind = self.inner_inputs[op_single].ind_l
            op_out = itertools.chain.from_iterable(
                itertools.repeat(x, n) for x, n in zip(outer_ind, inner_len)
            )
            res = op["."](op_out, val_ind)
            val = res
            keys = self.inner_inputs[op_single].keys_final + [op_single]
            return val, keys
        else:
            val = op["*"](val_ind)
            keys = [op_single]
            return val, keys

    def _get_element(self, value: ty.Any, field_name: str, ind: int) -> ty.Any:
        """
        Extracting element of the inputs taking into account
        container dimension of the specific element that can be set in self.state.container_ndim.
        If input name is not in container_ndim, it is assumed that the input values has
        a container dimension of 1, so only the most outer dim will be used for splitting.

        Parameters
        ----------
        value : Any
            inputs of the job
        field_name : str
            name of the input field
        ind : int
            index of the element

        Returns
        -------
        Any
            specific element of the input field
        """
        if f"{self.name}.{field_name}" in self.container_ndim_all:
            return list(
                flatten(
                    ensure_list(value),
                    max_depth=self.container_ndim_all[f"{self.name}.{field_name}"],
                )
            )[ind]
        else:
            return value[ind]


def splitter2rpn(splitter, other_states=None, state_fields=True):
    """
    Translate user-provided splitter into *reverse polish notation*.

    The reverse polish notation is imposed by :class:`~pydra.engine.state.State`.

    Parameters
    ----------
    splitter :
        splitter (standard form)
    other_states :
        other states that are connected to the state
    state_fields : :obj:`bool`
        if False the splitter from the previous states are unwrapped

    """
    if not splitter:
        return []
    output_splitter = []
    _ordering(
        deepcopy(splitter),
        i=0,
        output_splitter=output_splitter,
        other_states=deepcopy(other_states),
        state_fields=state_fields,
    )
    return output_splitter


def _ordering(
    el, i, output_splitter, current_sign=None, other_states=None, state_fields=True
):
    """Get a proper order of fields and signs (used by splitter2rpn)."""
    if type(el) is tuple:
        # checking if the splitter dont contain splitter from previous nodes
        # i.e. has str "_NA", etc.
        if len(el) == 1:
            # treats .split(("x",)) like .split("x")
            el = el[0]
            _ordering(el, i, output_splitter, current_sign, other_states, state_fields)
        else:
            if type(el[0]) is str and el[0].startswith("_"):
                node_nm = el[0][1:]
                if node_nm not in other_states and state_fields:
                    raise PydraStateError(
                        "can't ask for splitter from {}, other nodes that are connected: {}".format(
                            node_nm, other_states.keys()
                        )
                    )
                elif state_fields:
                    splitter_mod = add_name_splitter(
                        splitter=other_states[node_nm][0].splitter_final, name=node_nm
                    )
                    el = (splitter_mod, el[1])
                    if other_states[node_nm][0].other_states:
                        other_states.update(other_states[node_nm][0].other_states)
            if type(el[1]) is str and el[1].startswith("_"):
                node_nm = el[1][1:]
                if node_nm not in other_states and state_fields:
                    raise PydraStateError(
                        "can't ask for splitter from {}, other nodes that are connected: {}".format(
                            node_nm, other_states.keys()
                        )
                    )
                elif state_fields:
                    splitter_mod = add_name_splitter(
                        splitter=other_states[node_nm][0].splitter_final, name=node_nm
                    )
                    el = (el[0], splitter_mod)
                    if other_states[node_nm][0].other_states:
                        other_states.update(other_states[node_nm][0].other_states)
            _iterate_list(
                el,
                ".",
                other_states,
                output_splitter=output_splitter,
                state_fields=state_fields,
            )
    elif type(el) is list:
        if len(el) == 1:
            # treats .split(["x"]) like .split("x")
            el = el[0]
            _ordering(el, i, output_splitter, current_sign, other_states, state_fields)
        else:
            if type(el[0]) is str and el[0].startswith("_"):
                node_nm = el[0][1:]
                if node_nm not in other_states and state_fields:
                    raise PydraStateError(
                        "can't ask for splitter from {}, other nodes that are connected: {}".format(
                            node_nm, other_states.keys()
                        )
                    )
                elif state_fields:
                    splitter_mod = add_name_splitter(
                        splitter=other_states[node_nm][0].splitter_final, name=node_nm
                    )
                    el[0] = splitter_mod
                    if other_states[node_nm][0].other_states:
                        other_states.update(other_states[node_nm][0].other_states)
            if type(el[1]) is str and el[1].startswith("_"):
                node_nm = el[1][1:]
                if node_nm not in other_states and state_fields:
                    raise PydraStateError(
                        "can't ask for splitter from {}, other nodes that are connected: {}".format(
                            node_nm, other_states.keys()
                        )
                    )
                elif state_fields:
                    splitter_mod = add_name_splitter(
                        splitter=other_states[node_nm][0].splitter_final, name=node_nm
                    )
                    el[1] = splitter_mod
                    if other_states[node_nm][0].other_states:
                        other_states.update(other_states[node_nm][0].other_states)
            _iterate_list(
                el,
                "*",
                other_states,
                output_splitter=output_splitter,
                state_fields=state_fields,
            )
    elif type(el) is str:
        if el.startswith("_"):
            node_nm = el[1:]
            if node_nm not in other_states and state_fields:
                raise PydraStateError(
                    "can't ask for splitter from {}, other nodes that are connected: {}".format(
                        node_nm, other_states.keys()
                    )
                )
            elif state_fields:
                splitter_mod = add_name_splitter(
                    splitter=other_states[node_nm][0].splitter_final, name=node_nm
                )
                el = splitter_mod
                if other_states[node_nm][0].other_states:
                    other_states.update(other_states[node_nm][0].other_states)
        if type(el) is str:
            output_splitter.append(el)
        elif type(el) is tuple:
            _iterate_list(
                el,
                ".",
                other_states,
                output_splitter=output_splitter,
                state_fields=state_fields,
            )
        elif type(el) is list:
            _iterate_list(
                el,
                "*",
                other_states,
                output_splitter=output_splitter,
                state_fields=state_fields,
            )
    else:
        raise PydraStateError("splitter has to be a string, a tuple or a list")
    if i > 0:
        output_splitter.append(current_sign)


def _iterate_list(element, sign, other_states, output_splitter, state_fields=True):
    """Iterate over list (used in the splitter2rpn to get recursion)."""
    for i, el in enumerate(element):
        _ordering(
            deepcopy(el),
            i,
            current_sign=sign,
            other_states=other_states,
            output_splitter=output_splitter,
            state_fields=state_fields,
        )


def converter_groups_to_input(group_for_inputs):
    """
    Return fields for each axis and number of all groups.

    Requires having axes for all the input fields.

    Parameters
    ----------
    group_for_inputs :
        specified axes (groups) for each input

    """
    input_for_axis = {}
    ngr = 0
    for inp, grs in group_for_inputs.items():
        for gr in ensure_list(grs):
            if gr in input_for_axis.keys():
                input_for_axis[gr].append(inp)
            else:
                ngr += 1
                input_for_axis[gr] = [inp]
    return input_for_axis, ngr


def remove_inp_from_splitter_rpn(splitter_rpn, inputs_to_remove):
    """
    Remove inputs due to combining.

    Mutates a splitter.

    Parameters
    ----------
    splitter_rpn :
        The splitter in reverse polish notation
    inputs_to_remove :
        input names that should be removed from the splitter

    """
    splitter_rpn_copy = splitter_rpn.copy()
    # reverting order
    splitter_rpn_copy.reverse()
    stack_inp = []
    stack_sgn = []
    from_last_sign = []
    for ii, el in enumerate(splitter_rpn_copy):
        # element is a sign
        if el == "." or el == "*":
            stack_sgn.append((ii, el))
            from_last_sign.append(0)
        # it's an input but not to remove
        elif el not in inputs_to_remove:
            if from_last_sign:
                from_last_sign[-1] += 1
            stack_inp.append((ii, el))
        # it'a an input that should be removed
        else:
            if not from_last_sign:
                pass
            elif from_last_sign[-1] <= 1:
                stack_sgn.pop()
                from_last_sign.pop()
            else:
                stack_sgn.pop(-1 * from_last_sign.pop())

    # creating the final splitter_rpn after combining
    remaining_elements = stack_sgn + stack_inp
    remaining_elements.sort(reverse=True)
    splitter_rpn_combined = [el for (i, el) in remaining_elements]
    return splitter_rpn_combined


def rpn2splitter(splitter_rpn):
    """
    Convert from splitter_rpn to splitter.

    Recurrent algorithm to perform the conversion.
    Every time combines pairs of input in one input,
    ends when the length is one.

    Parameters
    ----------
    splitter_rpn :
        splitter in reverse polish notation

    Returns
    -------
    splitter :
        splitter in the standard/original form

    """
    if splitter_rpn == []:
        return None
    if len(splitter_rpn) == 1:
        return splitter_rpn[0]

    splitter_rpn_copy = splitter_rpn.copy()
    signs = [".", "*"]
    splitter_modified = []

    while splitter_rpn_copy:
        el = splitter_rpn_copy.pop()
        # element is a sign
        if el in signs:
            if (
                splitter_rpn_copy[-1] not in signs
                and splitter_rpn_copy[-2] not in signs
            ):
                right, left = splitter_rpn_copy.pop(), splitter_rpn_copy.pop()
                if el == ".":
                    splitter_modified.append((left, right))
                elif el == "*":
                    splitter_modified.append([left, right])
            else:
                splitter_modified.append(el)
        else:
            splitter_modified.append(el)

    # reversing the list and combining more
    splitter_modified.reverse()
    return rpn2splitter(splitter_modified)


def add_name_combiner(combiner, name):
    """adding a node's name to each field from the combiner"""
    combiner_changed = []
    for comb in combiner:
        if "." not in comb:
            combiner_changed.append(f"{name}.{comb}")
        else:
            combiner_changed.append(comb)
    return combiner_changed


def add_name_splitter(
    splitter: ty.Union[str, ty.List[str], ty.Tuple[str, ...], None], name: str
) -> ty.Optional[ty.List[str]]:
    """adding a node's name to each field from the splitter"""
    if isinstance(splitter, str):
        return _add_name([splitter], name)[0]
    elif isinstance(splitter, list):
        return _add_name(list(splitter), name)
    elif isinstance(splitter, tuple):
        return tuple(_add_name(list(splitter), name))
    else:
        return None


def _add_name(mlist, name):
    """adding anem to each element from the list"""
    for i, elem in enumerate(mlist):
        if isinstance(elem, str):
            if "." in elem or elem.startswith("_"):
                pass
            else:
                mlist[i] = f"{name}.{mlist[i]}"
        elif isinstance(elem, list):
            mlist[i] = _add_name(elem, name)
        elif isinstance(elem, tuple):
            mlist[i] = list(elem)
            mlist[i] = _add_name(mlist[i], name)
            mlist[i] = tuple(mlist[i])
    return mlist


def flatten(vals, cur_depth=0, max_depth=None):
    """Flatten a list of values."""
    if max_depth is None:
        max_depth = len(list(input_shape(vals)))
    values = []
    if cur_depth >= max_depth:
        values.append([vals])
    else:
        for val in vals:
            if isinstance(val, (list, tuple)):
                values.append(flatten(val, cur_depth + 1, max_depth))
            else:
                values.append([val])
    return itertools.chain.from_iterable(values)


def iter_splits(iterable, keys):
    """Generate splits."""
    for iter in list(iterable):
        yield dict(zip(keys, list(flatten(iter, max_depth=1000))))


def input_shape(inp, container_ndim=1):
    """Get input shape, depends on the container dimension, if not specify it is assumed to be 1"""
    # TODO: have to be changed for inner splitter (sometimes different length)
    container_ndim -= 1
    shape = [len(inp)]
    last_shape = None
    for value in inp:
        if isinstance(value, list) and container_ndim > 0:
            cur_shape = input_shape(value, container_ndim)
            if last_shape is None:
                last_shape = cur_shape
            elif last_shape != cur_shape:
                last_shape = None
                break
        else:
            last_shape = None
            break
    if last_shape is not None:
        shape.extend(last_shape)
    return tuple(shape)


def splits_groups(splitter_rpn, combiner=None, inner_inputs=None):
    """splits inputs to groups (axes) and creates stacks for these groups
    This is used to specify which input can be combined.
    """
    if not splitter_rpn:
        return [], {}, [], []
    stack = []
    keys = []
    groups = {}
    group_count = None
    if not combiner:
        combiner = []
    if inner_inputs:
        previous_states_ind = {
            f"_{v.name}": v.keys_final for v in inner_inputs.values()
        }
        inner_inputs = {k: v for k, v in inner_inputs.items() if k in splitter_rpn}
    else:
        previous_states_ind = {}
        inner_inputs = {}

    # when splitter is a single element (no operators)
    if len(splitter_rpn) == 1:
        op_single = splitter_rpn[0]
        return _single_op_splits_groups(op_single, combiner, inner_inputs, groups)

    # len(splitter_rpn) > 1
    # iterating splitter_rpn
    for token in splitter_rpn:
        if token in [".", "*"]:
            terms = {}
            terms["R"] = stack.pop()
            terms["L"] = stack.pop()

            # checking if opL/R are strings
            trm_str = {"L": False, "R": False}
            oldgroups = {}

            for lr in ["L", "R"]:
                if isinstance(terms[lr], str):
                    trm_str[lr] = True
                else:
                    oldgroups[lr] = terms[lr]

            if token == ".":
                if all(trm_str.values()):
                    if group_count is None:
                        group_count = 0
                    else:
                        group_count += 1
                    oldgroup = groups[terms["L"]] = groups[terms["R"]] = group_count
                elif trm_str["R"]:
                    groups[terms["R"]] = oldgroups["L"]
                    oldgroup = oldgroups["L"]
                elif trm_str["L"]:
                    groups[terms["L"]] = oldgroups["R"]
                    oldgroup = oldgroups["R"]
                else:
                    if len(ensure_list(oldgroups["L"])) != len(
                        ensure_list(oldgroups["R"])
                    ):
                        raise ValueError(
                            "Operands do not have same shape "
                            "(left one is {}d and right one is {}d.".format(
                                len(ensure_list(oldgroups["L"])),
                                len(ensure_list(oldgroups["R"])),
                            )
                        )
                    oldgroup = oldgroups["L"]
                    # dj: changing axes for Right part of the scalar op.
                    for k, v in groups.items():
                        if v in ensure_list(oldgroups["R"]):
                            groups[k] = ensure_list(oldgroups["L"])[
                                ensure_list(oldgroups["R"]).index(v)
                            ]
            else:  # if token == "*":
                if all(trm_str.values()):
                    if group_count is None:
                        group_count = 0
                    else:
                        group_count += 1
                    groups[terms["L"]] = group_count
                    group_count += 1
                    groups[terms["R"]] = group_count
                    oldgroup = [groups[terms["L"]], groups[terms["R"]]]
                elif trm_str["R"]:
                    group_count += 1
                    groups[terms["R"]] = group_count
                    oldgroup = ensure_list(oldgroups["L"]) + [groups[terms["R"]]]
                elif trm_str["L"]:
                    group_count += 1
                    groups[terms["L"]] = group_count
                    oldgroup = [groups[terms["L"]]] + ensure_list(oldgroups["R"])
                else:
                    oldgroup = ensure_list(oldgroups["L"]) + ensure_list(oldgroups["R"])

            # creating list of keys
            if trm_str["L"]:
                if terms["L"].startswith("_"):
                    keys = previous_states_ind[terms["L"]] + keys
                else:
                    keys.insert(0, terms["L"])
            if trm_str["R"]:
                if terms["R"].startswith("_"):
                    keys += previous_states_ind[terms["R"]]
                else:
                    keys.append(terms["R"])

            pushgroup = oldgroup
            stack.append(pushgroup)

        else:  # name of one of the inputs
            stack.append(token)

    groups_stack = stack.pop()
    if isinstance(groups_stack, int):
        groups_stack = [groups_stack]
    if inner_inputs:
        groups_stack = [[], groups_stack]
    else:
        groups_stack = [groups_stack]

    if combiner:
        (
            keys_final,
            groups_final,
            groups_stack_final,
            combiner_all,
        ) = combine_final_groups(combiner, groups, groups_stack, keys)
        return keys_final, groups_final, groups_stack_final, combiner_all
    else:
        return keys, groups, groups_stack, []


def _single_op_splits_groups(op_single, combiner, inner_inputs, groups):
    """splits_groups function if splitter is a singleton"""
    if op_single in inner_inputs:
        # TODO: have to be changed if differ length
        # TODO: i think I don't want to add here from left part
        # keys = inner_inputs[op_single].keys_final + [op_single]
        keys = [op_single]
        groups[op_single], groups_stack = 0, [[], [0]]
    else:
        keys = [op_single]
        groups[op_single], groups_stack = 0, [[0]]
    if combiner:
        if combiner == [op_single]:
            return [], {}, [], combiner
        else:
            # TODO: probably not needed, should be already check by st.combiner_validation
            raise PydraStateError(
                f"all fields from the combiner have to be in splitter_rpn: {[op_single]}, "
                f"but combiner: {combiner} is set"
            )
    else:
        return keys, groups, groups_stack, []


def combine_final_groups(combiner, groups, groups_stack, keys):
    """Combine the final groups."""
    input_for_groups, _ = converter_groups_to_input(groups)
    combiner_all = []
    for comb in combiner:
        for gr in ensure_list(groups[comb]):
            combiner_all += input_for_groups[gr]
    combiner_all = list(set(combiner_all))
    combiner_all.sort()

    # groups that were removed (so not trying to remove twice)
    grs_removed = []
    groups_stack_final = deepcopy(groups_stack)
    for comb in combiner:
        grs = groups[comb]
        for gr in ensure_list(grs):
            if gr in groups_stack_final[-1]:
                grs_removed.append(gr)
                groups_stack_final[-1].remove(gr)
            elif gr in grs_removed:
                pass
            else:
                raise PydraStateError(
                    "input {} not ready to combine, you have to combine {} "
                    "first".format(comb, groups_stack[-1])
                )
    groups_final = {inp: gr for (inp, gr) in groups.items() if inp not in combiner_all}
    gr_final = set()
    for el in groups_final.values():
        gr_final.update(ensure_list(el))
    gr_final = list(gr_final)
    map_gr_nr = {nr: i for (i, nr) in enumerate(sorted(gr_final))}
    groups_final_map = {}
    for inp, gr in groups_final.items():
        if isinstance(gr, int):
            groups_final_map[inp] = map_gr_nr[gr]
        elif isinstance(gr, list):
            groups_final_map[inp] = [map_gr_nr[el] for el in gr]
        else:
            raise Exception("gr should be an int or a list, something wrong")
    for i, groups_l in enumerate(groups_stack_final):
        groups_stack_final[i] = [map_gr_nr[gr] for gr in groups_l]

    keys_final = [key for key in keys if key not in combiner_all]
    # TODO: not sure if I have to calculate and return keys, groups, groups_stack
    return keys_final, groups_final_map, groups_stack_final, combiner_all


def map_splits(split_iter, inputs, container_ndim=None):
    """generate a dictionary of inputs prescribed by the splitter."""
    if container_ndim is None:
        container_ndim = {}
    for split in split_iter:
        yield {
            k: list(
                flatten(ensure_list(inputs[k]), max_depth=container_ndim.get(k, None))
            )[v]
            for k, v in split.items()
        }


def inputs_types_to_dict(name, inputs):
    """Convert type.Inputs to dictionary."""
    # dj: any better option?
    input_names = [field for field in attrs_values(inputs) if field != "_func"]
    inputs_dict = {}
    for field in input_names:
        inputs_dict[f"{name}.{field}"] = getattr(inputs, field)
    return inputs_dict


def unwrap_splitter(
    splitter: ty.Union[str, ty.List[str], ty.Tuple[str, ...]],
) -> ty.Iterable[str]:
    """Unwraps a splitter into a flat list of fields that are split over, i.e.
    [("a", "b"), "c"] -> ["a", "b", "c"]

    Parameters
    ----------
    splitter: str or list[str] or tuple[str, ...]
        the splitter task to unwrap

    Returns
    -------
    unwrapped : ty.Iterable[str]
        the field names listed in the splitter
    """
    if isinstance(splitter, str):
        return [splitter]
    else:
        return itertools.chain(*(unwrap_splitter(s) for s in splitter))


class PydraStateError(Exception):
    """Custom error for Pydra State"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


op = {".": zip, "*": itertools.product}
