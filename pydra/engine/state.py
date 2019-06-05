from copy import deepcopy

from . import auxiliary as aux
from .specs import BaseSpec


class State:
    def __init__(self, name, splitter=None, combiner=None, other_states=None):
        self.name = name
        self.other_states = other_states
        self.splitter = splitter
        self.connect_splitters()
        self.combiner = combiner
        if not self.other_states:
            self.other_states = {}
        self.inner_inputs = {
            "{}.{}".format(self.name, inp): st
            for name, (st, inp) in self.other_states.items()
        }
        self.set_input_groups()
        self.set_splitter_final()
        self.states_val = []
        self.final_groups_mapping = {}

    @property
    def splitter(self):
        return self._splitter

    @splitter.setter
    def splitter(self, splitter):
        if splitter:
            self._splitter = aux.change_splitter(splitter, self.name)
            self.splitter_rpn = aux.splitter2rpn(
                deepcopy(self._splitter), other_states=self.other_states
            )
            self.splitter_rpn_nost = aux.splitter2rpn(
                deepcopy(self._splitter),
                other_states=self.other_states,
                state_fields=False,
            )
            for spl in self.splitter_rpn_nost:
                if (
                    spl in [".", "*"]
                    or spl.startswith("_")
                    or spl.split(".")[0] == self.name
                ):
                    pass
                else:
                    raise Exception(
                        "can't include {} in the splitter, consider using _{}".format(
                            spl, spl.split(".")[0]
                        )
                    )
            self.splitter_final = self._splitter
        else:
            self._splitter = None
            self.splitter_final = None
            self.splitter_rpn = []
            self.splitter_rpn_nost = []

    @property
    def combiner(self):
        return self._combiner

    @combiner.setter
    def combiner(self, combiner):
        if combiner:
            if not self.splitter:
                raise Exception("splitter has to be set before setting combiner")
            if type(combiner) is str:
                combiner = [combiner]
            elif type(combiner) is not list:
                raise Exception("combiner should be a string or a list")
            self._combiner = aux.change_combiner(combiner, self.name)
            if set(self._combiner) - set(self.splitter_rpn):
                raise Exception("all combiners should be in the splitter")
            self._right_combiner = [
                comb for comb in self._combiner if self.name in comb
            ]
            self._left_combiner = list(set(self._combiner) - set(self._right_combiner))
        else:
            self._combiner = []
            self._left_combiner = []
            self._right_combiner = []

    def connect_splitters(self):
        """
        connect splitters from previous nodes,
        evaluate Left (previous nodes) and Right (current node) parts
        """
        if self.other_states:
            self.splitter, self._left_splitter, self._right_splitter = aux.connect_splitters(
                splitter=self.splitter, other_states=self.other_states
            )
            # left rpn part, but keeping the names of the nodes, e.g. [_NA, _NB, *]
            self._left_splitter_rpn_nost = aux.splitter2rpn(
                deepcopy(self._left_splitter),
                other_states=self.other_states,
                state_fields=False,
            )
            self._left_splitter_rpn = aux.splitter2rpn(
                deepcopy(self._left_splitter), other_states=self.other_states
            )
        else:  # if other_states is empty there is only Right part
            self._left_splitter = None
            self._left_splitter_rpn_nost = []
            self._left_splitter_rpn = []
            self._right_splitter = self.splitter
        self._right_splitter_rpn = aux.splitter2rpn(
            deepcopy(self._right_splitter), other_states=self.other_states
        )

    def set_splitter_final(self):
        """evaluate a final splitter after combining"""
        _splitter_rpn_final = aux.remove_inp_from_splitter_rpn(
            deepcopy(self.splitter_rpn),
            self.right_combiner_all + self.left_combiner_all,
        )
        self.splitter_final = aux.rpn2splitter(_splitter_rpn_final)
        self.splitter_rpn_final = aux.splitter2rpn(
            self.splitter_final, other_states=self.other_states
        )

    def set_input_groups(self):
        """evaluate groups, especially the final groups that address the combiner"""
        keys_f, group_for_inputs_f, groups_stack_f, combiner_all = aux._splits_groups(
            self._right_splitter_rpn,
            combiner=self._right_combiner,
            inner_inputs=self.inner_inputs,
        )
        self.right_combiner_all = combiner_all
        if self._left_splitter:  # if splitter has also the left part
            self._right_keys_final = keys_f
            self._right_group_for_inputs_final = group_for_inputs_f
            self._right_groups_stack_final = groups_stack_f
            self.connect_groups()
        else:
            self.group_for_inputs_final = group_for_inputs_f
            self.groups_stack_final = groups_stack_f
            self.keys_final = keys_f
            self.left_combiner_all = []

    def connect_groups(self):
        """"connect previous states and evaluate the final groups"""
        self.merge_previous_states()
        if self._right_splitter:  # if Right part, adding groups from current st
            self.push_new_states()

    def merge_previous_states(self):
        """ merging groups from  all previous nodes"""
        last_gr = 0
        self.groups_stack_final = []
        self.group_for_inputs_final = {}
        self.keys_final = []
        self.left_combiner_all = []
        if self._left_combiner:
            _, _, _, self._left_combiner = aux._splits_groups(
                self._left_splitter_rpn, combiner=self._left_combiner
            )

        for i, left_nm in enumerate(self._left_splitter_rpn_nost):
            if left_nm in ["*", "."]:
                continue
            if (
                i + 1 < len(self._left_splitter_rpn_nost)
                and self._left_splitter_rpn_nost[i + 1] == "."
            ):
                last_gr = last_gr - 1
            st = self.other_states[left_nm[1:]][0]
            # checking if left combiner contains any element from the st splitter
            st_combiner = [
                comb for comb in self._left_combiner if comb in st.splitter_rpn_final
            ]
            if st_combiner:
                # keys and groups from previous states
                # after taking into account combiner from current state
                keys_f_st, group_for_inputs_f_st, groups_stack_f_st, combiner_all_st = aux._splits_groups(
                    st.splitter_rpn_final,
                    combiner=st_combiner,
                    inner_inputs=st.inner_inputs,
                )
                self.keys_final += keys_f_st  # st.keys_final
                if not hasattr(st, "group_for_inputs_final"):
                    raise Exception("previous state has to run first")
                group_for_inputs = group_for_inputs_f_st
                groups_stack = groups_stack_f_st
                self.left_combiner_all += combiner_all_st
            else:
                # if no element from st.splitter is in the current combiner,
                # using st attributes without changes
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

    def push_new_states(self):
        """adding additional groups from the current state"""
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

    def prepare_states(self, inputs):
        """
        preparing a full list of state indices (number of elements depends on the splitter)
        and state values (specific elements from inputs that can be used running interfaces)
        """
        if isinstance(inputs, BaseSpec):
            self.inputs = aux.inputs_types_to_dict(self.name, inputs)
        else:
            self.inputs = inputs
        if self.other_states:
            for nm, (st, _) in self.other_states.items():
                # I think now this if is never used
                if not hasattr(st, "states_ind"):
                    # dj: should i provide different inputs?
                    st.prepare_states(self.inputs)
                self.inputs.update(st.inputs)
        self.prepare_states_ind()
        self.prepare_states_val()

    def prepare_states_ind(self):
        """using aux._splits to calculate a list of dictionaries with state indices"""

        # removing elements that are connected to inner splitter
        # (they will be taken into account in aux._splits anyway)
        # _comb part will be used in prepare_states_combined_ind
        elements_to_remove = []
        elements_to_remove_comb = []
        for name, (st, inp) in self.other_states.items():
            if (
                "{}.{}".format(self.name, inp) in self.splitter_rpn
                and "_{}".format(name) in self.splitter_rpn_nost
            ):
                elements_to_remove.append("_{}".format(name))
                if "{}.{}".format(self.name, inp) not in self.combiner:
                    elements_to_remove_comb.append("_{}".format(name))

        partial_rpn = aux.remove_inp_from_splitter_rpn(
            deepcopy(self.splitter_rpn_nost), elements_to_remove
        )
        values_out_pr, keys_out_pr, _, kL = aux._splits(
            partial_rpn, self.inputs, inner_inputs=self.inner_inputs
        )
        values_pr = list(values_out_pr)

        self.ind_l = values_pr
        self.keys = keys_out_pr
        self.states_ind = list(aux.iter_splits(values_pr, self.keys))
        self.keys_final = self.keys
        if self.combiner:
            self.prepare_states_combined_ind(elements_to_remove_comb)
        else:
            self.ind_l_final = self.ind_l
            self.keys_final = self.keys
            self.final_groups_mapping = {i: [i] for i in range(len(self.states_ind))}
            self.states_ind_final = self.states_ind
        return self.states_ind

    def prepare_states_combined_ind(self, elements_to_remove_comb):
        """preparing the final list of dictionaries with indices after combiner"""
        partial_rpn_nost = aux.remove_inp_from_splitter_rpn(
            deepcopy(self.splitter_rpn_nost), elements_to_remove_comb
        )
        # combiner can have parts from the left splitter, so have to have rpn with states
        partial_rpn = aux.splitter2rpn(
            aux.rpn2splitter(partial_rpn_nost), other_states=self.other_states
        )
        combined_rpn = aux.remove_inp_from_splitter_rpn(
            deepcopy(partial_rpn), self.right_combiner_all + self.left_combiner_all
        )
        # TODO: create a function for this!!
        if combined_rpn:
            val_r, key_r, _, _ = aux._splits(
                combined_rpn, self.inputs, inner_inputs=self.inner_inputs
            )
            values = list(val_r)
        else:
            values = []
            key_r = []

        keys_out = key_r
        if values:
            # NOW TODO: move to init?
            self.ind_l_final = values
            self.keys_final = keys_out
            # groups after combiner
            ind_map = {
                tuple(aux.flatten(tup, max_depth=10)): ind
                for ind, tup in enumerate(self.ind_l_final)
            }
            self.final_groups_mapping = {i: [] for i in range(len(self.ind_l_final))}
            for ii, st in enumerate(self.states_ind):
                ind_f = tuple([st[k] for k in self.keys_final])
                self.final_groups_mapping[ind_map[ind_f]].append(ii)
        else:
            self.ind_l_final = values
            self.keys_final = keys_out
            # should be 0 or None?
            self.final_groups_mapping = {0: list(range(len(self.states_ind)))}
        self.states_ind_final = list(aux.iter_splits(self.ind_l_final, self.keys_final))

    def prepare_states_val(self):
        """evaluate states values having states indices"""
        self.states_val = list(aux.map_splits(self.states_ind, self.inputs))
        return self.states_val

    def prepare_inputs(self):
        if not self.other_states:
            self.inputs_ind = self.states_ind
        else:
            # removing elements that come from connected states
            elements_to_remove = [
                spl
                for spl in self.splitter_rpn_nost
                if spl[1:] in self.other_states.keys()
            ]
            partial_rpn = aux.remove_inp_from_splitter_rpn(
                deepcopy(self.splitter_rpn_nost), elements_to_remove
            )
            if partial_rpn:
                values_inp, keys_inp, _, _ = aux._splits(
                    partial_rpn, self.inputs, inner_inputs=self.inner_inputs
                )
                inputs_ind = values_inp
            else:
                keys_inp = []
                inputs_ind = []

            # merging elements that comes from previous nodes outputs
            # states that are connected to inner splitters are treated differently
            # (already included in inputs_ind)
            keys_inp_prev = []
            inputs_ind_prev = []
            connected_to_inner = []
            for ii, el in enumerate(self._left_splitter_rpn_nost):
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
                        if self._left_splitter_rpn_nost[ii + 1] == ".":
                            inputs_ind_prev = aux.op["."](inputs_ind_prev, st_ind)
                        else:
                            inputs_ind_prev = aux.op["*"](inputs_ind_prev, st_ind)
                    else:
                        inputs_ind_prev = aux.op["*"](st_ind)
                    keys_inp_prev += ["{}.{}".format(self.name, inp)]
            keys_inp = keys_inp_prev + keys_inp

            if inputs_ind and inputs_ind_prev:
                inputs_ind = aux.op["*"](inputs_ind_prev, inputs_ind)
            elif inputs_ind:
                inputs_ind = aux.op["*"](inputs_ind)
            elif inputs_ind_prev:
                inputs_ind = aux.op["*"](inputs_ind_prev)
            else:
                inputs_ind = []

            # iter_splits using inputs from current state/node
            self.inputs_ind = list(aux.iter_splits(inputs_ind, keys_inp))
            # removing elements that are connected to inner splitter
            for el in connected_to_inner:
                [dict.pop(el) for dict in self.inputs_ind]
