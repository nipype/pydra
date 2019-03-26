from copy import deepcopy
import pdb

from functools import lru_cache

from . import auxiliary as aux
from .specs import BaseSpec


class State:
    def __init__(self, name, splitter=None, combiner=None, other_states=None):
        self.name = name
        self.other_states = other_states
        self.splitter = splitter
        self.combiner = combiner
        if not self.other_states:
            self.other_states = {}
        self.inner_inputs = {
            "{}.{}".format(self.name, inp): st
            for name, (st, inp) in self.other_states.items()
        }
        self.connect_splitters()
        self.set_input_groups()
        self.set_splitter_final()

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
            self._combiner = aux.change_splitter(combiner, self.name)
            if set(self._combiner) - set(self.splitter_rpn):
                raise Exception("all combiners should be in the splitter")
        else:
            self._combiner = []

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
        else:  # if other_states is empty there is only Right part
            self._left_splitter = None
            self._left_splitter_rpn_nost = []
            self._right_splitter = self.splitter
        self._right_splitter_rpn = aux.splitter2rpn(
            deepcopy(self._right_splitter), other_states=self.other_states
        )

    def set_splitter_final(self):
        """evaluate a final splitter after combining"""
        _splitter_rpn_final = aux.remove_inp_from_splitter_rpn(
            deepcopy(self.splitter_rpn_nost), self.combiner_all
        )
        self.splitter_final = aux.rpn2splitter(_splitter_rpn_final)
        self.splitter_rpn_final = aux.splitter2rpn(
            self.splitter_final, other_states=self.other_states
        )

    def set_input_groups(self):
        """evaluate groups, especially the final groups that address cthe combine"""
        _, _, _, keys_f, group_for_inputs_f, groups_stack_f, combiner_all = aux._splits_groups(
            self._right_splitter_rpn,
            combiner=self.combiner,
            inner_inputs=self.inner_inputs,
        )
        self._right_keys_final = keys_f
        self.combiner_all = combiner_all
        self._right_group_for_inputs_final = group_for_inputs_f
        self._right_groups_stack_final = groups_stack_f
        self.connect_groups()

    def connect_groups(self):
        """"connect previous states and evaluate the final groups"""
        if self._left_splitter:  # if Left part, merging all previous nodes
            self.merge_previous_states()
            if self._right_splitter:  # if Right part, adding groups from current st
                self.push_new_states()
        else:  # if no Left part, than there is only Right part from thecurrent node
            self.group_for_inputs_final = self._right_group_for_inputs_final
            self.groups_stack_final = self._right_groups_stack_final
            self.keys_final = self._right_keys_final

    def merge_previous_states(self):
        """ merging groups from  all previous nodes"""
        last_gr = 0
        self.groups_stack_final = []
        self.group_for_inputs_final = {}
        self.keys_final = []
        for nm, (st, inp) in self.other_states.items():
            if "_{}".format(nm) not in self._left_splitter_rpn_nost:
                continue
            self.keys_final += st.keys_final
            if not hasattr(st, "group_for_inputs_final"):
                raise Exception("previous state has to run first")
            group_for_inputs = st.group_for_inputs_final
            groups_stack = st.groups_stack_final
            group_for_inputs = {k: v + last_gr for k, v in group_for_inputs.items()}
            self.group_for_inputs_final.update(group_for_inputs)

            nmb_gr = 0
            for i, groups in enumerate(groups_stack):
                if i < len(self.groups_stack_final):
                    for gr in groups:
                        nmb_gr += 1
                        self.groups_stack_final[i].append(gr + last_gr)
                else:
                    self.groups_stack_final.append([gr + last_gr for gr in groups])
                    nmb_gr += len(groups)
            last_gr += nmb_gr
        # TODO: left part also could be partially combined

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
        if self.other_states:
            for nm, (st, _) in self.other_states.items():
                if not hasattr(st, "states_ind"):
                    # dj: should i provide different inputs?
                    st.prepare_states(inputs)
        self.prepare_states_ind(inputs)
        self.prepare_states_val(inputs)


    def prepare_states_ind(self, inputs):
        """using aux._splits to calculate a list of dictionaries with state indices"""
        if isinstance(inputs, BaseSpec):
            inputs = aux.inputs_types_to_dict(self.name, inputs)
        # if there are Left and Right parts, evaluate keys/indices
        # from Left and RIght separately and merge them
        if self._right_splitter and self._left_splitter:
            val_r, key_r, _, keys_fromLeftSpl = aux._splits(
                self._right_splitter_rpn, inputs, inner_inputs=self.inner_inputs
            )
            val_r = list(val_r)
            updated_left_rpn = deepcopy(self._left_splitter_rpn_nost)
            updated_left_rpn = aux.remove_inp_from_splitter_rpn(
                updated_left_rpn, keys_fromLeftSpl
            )

            if updated_left_rpn:
                val_l, key_l, _, _ = aux._splits(
                    updated_left_rpn, inputs, inner_inputs=self.inner_inputs
                )
                val_l = list(val_l)
            else:
                val_l = []
                key_l = []

            if val_l and val_r:
                values = list(aux.op["*"](val_l, val_r))
            elif val_l:
                values = val_l
            elif val_r:
                values = val_r
            keys_out = key_l + key_r
            self.val_l = val_l
            self.key_l = key_l
        else:
            values_out, keys_out, _, _ = aux._splits(
                self.splitter_rpn, inputs, inner_inputs=self.inner_inputs
            )
            values = list(values_out)
            # dj: not sure if this shouldn't be already in the init
            self.key_l = []
            self.val_l = []
        self.ind_l = values
        self.keys = keys_out
        self.states_ind = list(aux.iter_splits(values, self.keys))
        self.keys_final = self.keys
        if self.combiner:
            self.prepare_states_combined_ind(inputs=inputs)
        else:
            self.ind_l_final = self.ind_l
            self.keys_final = self.keys
            self.final_groups_mapping = {i: [i] for i in range(len(self.states_ind))}
        return self.states_ind

    def prepare_states_combined_ind(self, inputs):
        """preparing the final list of dictionaries with indices after combiner"""
        # assuming for now that the combiner is only in the right part TODO
        if self._right_splitter and self._left_splitter:
            combined_right_rpn = aux.remove_inp_from_splitter_rpn(
                deepcopy(self._right_splitter_rpn), self.combiner_all
            )
        else:
            combined_right_rpn = aux.remove_inp_from_splitter_rpn(
                deepcopy(self.splitter_rpn), self.combiner_all
            )

        # TODO: create a function for this!!
        if combined_right_rpn:
            val_r, key_r, _, _ = aux._splits(
                combined_right_rpn, inputs, inner_inputs=self.inner_inputs
            )
            val_r = list(val_r)
        else:
            val_r = []
            key_r = []

        if self.val_l and val_r:
            values = list(aux.op["*"](self.val_l, val_r))
        elif self.val_l:
            values = self.val_l
        elif val_r:
            values = val_r
        else:
            values = []
        keys_out = self.key_l + key_r
        if values:
            # NOW TODO: move to init?
            self.ind_l_final = values
            self.keys_final = keys_out
            # groups after combiner
            ind_map = {tuple(aux.flatten(tup, max_depth=10)): ind for ind,tup in enumerate(self.ind_l_final)}
            self.final_groups_mapping = {i: [] for i in range(len(self.ind_l_final))}
            for ii, st in enumerate(self.states_ind):
                ind_f = tuple([st[k] for k in self.keys_final])
                self.final_groups_mapping[ind_map[ind_f]].append(ii)
        else:
            self.ind_l_final = values
            self.keys_final = keys_out
            # should be 0 or None?
            self.final_groups_mapping = {0: list(range(len(self.states_ind)))}


    def prepare_states_val(self, inputs):
        """evaluate states values having states indices"""
        if isinstance(inputs, BaseSpec):
            inputs = aux.inputs_types_to_dict(self.name, inputs)
        self.states_val = list(aux.map_splits(self.states_ind, inputs))
        return self.states_val


'''    
    def cross_combine(self, other):
        self.ndim += other.ndim


class State(object):
    def __init__(self, node):
        self.node = node
        self._splitter = node.splitter
        self._other_splitters = node._other_splitters
        self.node_name = node.name
        self._inner_splitter = []

        self._inner_combiner = []
        self.comb_inp_to_remove = []

        self.state_inputs = node.state_inputs
        # checking if self.node is actually a node (not a wf)
        if hasattr(self.node, "interface"):
            # inputs that are taken from other nodes
            self._inner_inputs_names = ["{}.{}".format(self.node_name, inp) for inp in self.node.inner_inputs_names]
            # adding inner splitters from other nodes
            self._inner_inputs_names = self._inner_inputs_names + self.node.wf_inner_splitters
            if self._splitter and self._inner_inputs_names:
                self._inner_splitter_separation(combiner=node.combiner)
        # not sure if we should allow for wf (wouldn't work for now anyway)
        if not self._inner_splitter:
            self._splitter_wo_inner = self._splitter
            self._combiner_wo_inner = node.combiner

        # changing splitter (as in rpn), so I can read from left to right
        # e.g. if splitter=('d', ['e', 'r']), _splitter_rpn=['d', 'e', 'r', '*', '.']
        self._splitter_rpn = aux.splitter2rpn(self._splitter, other_splitters=self._other_splitters)
        self._input_names_splitter = [i for i in self._splitter_rpn if i not in ["*", "."]]

        self._splitter_rpn_wo_inner = aux.splitter2rpn(self._splitter_wo_inner, other_splitters=self._other_splitters)

        # TODO: should try to change the order and move it higher
        if node.combiner:
            self.combiner = node.combiner
        else:
            self._combiner = node.combiner

        # adding inner splitters to the combined inner splitters from wf
        for spl in self._inner_splitter:
            if spl not in self.node.wf_inner_splitters:
                self.node.wf_inner_splitters.append(spl)
        # inner splitters that will stay after combining
        self._inner_splitter_comb = list(set(self._inner_splitter) - set(self._inner_combiner))
        if self._inner_splitter_comb and self._combiner_wo_inner:
            raise Exception("You can't combine {} before combining all inner splitters: {}".format(
                self._combiner_wo_inner, self._inner_splitter_comb
            ))


    # do I use it?
    def __getitem__(self, ind):
        if type(ind) is int:
            ind = (ind,)
        return self.state_values(ind)

    @property
    def combiner(self):
        return self._combiner

    @combiner.setter
    def combiner(self, combiner):
        # if self._combiner:
        #     raise Exception("combiner is already set")
        # else:
        self._combiner = combiner
        for el in self._combiner:
            if el not in self._splitter_rpn:
                raise Exception("element {} of combiner is not found in the splitter {}".format(
                    el, self._splitter))
        self._prepare_combine()

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape


    def _prepare_combine(self):
        # input and axes that will be removed when combiner applied
        self.comb_axes_to_remove = []
        # temporary axis_for_input (without knowing specific input)
        # need to know which inputs are bound together
        axis_for_input_tmp, ndim_tmp = aux.matching_input_from_splitter(self._splitter_rpn)
        input_for_axis_tmp = aux.converting_axis2input(axis_for_input=axis_for_input_tmp,
                                                       ndim=ndim_tmp)
        for comb_el in self._combiner_wo_inner:
            self.comb_inp_to_remove.append(comb_el)
            axes = axis_for_input_tmp[comb_el]
            for ax in axes:
                if ax in self.comb_axes_to_remove:
                    raise Exception("can't combine by {}, axis {} already removed".format(
                        comb_el, ax
                    ))
            self.comb_axes_to_remove += axes

        # sorting and removing repeated values
        self.comb_axes_to_remove = list(set(self.comb_axes_to_remove))
        self.comb_axes_to_remove.sort()

        # (original axis, new axis) for axes that remain after combining
        self.partial_comb_input_rem_axes = {}
        # (original axis of the input, axis from the combiner space) for axes that are combined
        self.partial_comb_input_comb_axes = {}
        for ax in self.comb_axes_to_remove:
            inputs = deepcopy(input_for_axis_tmp[ax])
            for other_inp in set(inputs) - set(self._combiner_wo_inner):
                self.comb_inp_to_remove.append(other_inp)
                # saving remaining axes and position in a final splitter
                remaining_axes = set(axis_for_input_tmp[other_inp])\
                                 - set(self.comb_axes_to_remove)
                if remaining_axes:
                    self.partial_comb_input_rem_axes[other_inp] = []
                    for rem_ax in remaining_axes:
                        axes_shift = (rem_ax, rem_ax - len([i for i in self.comb_axes_to_remove
                                                            if i < rem_ax]))
                        self.partial_comb_input_rem_axes[other_inp].append(axes_shift)

                combined_axes = list(set(axis_for_input_tmp[other_inp]) - set(remaining_axes))
                combined_axes.sort()
                if combined_axes:
                    self.partial_comb_input_comb_axes[other_inp] = []
                    for ind_ax, comb_ax in enumerate(combined_axes):
                        axes_shift = (ind_ax, self.comb_axes_to_remove.index(comb_ax))
                        self.partial_comb_input_comb_axes[other_inp].append(axes_shift)

        self._prepare_splitter_combine()


    # TODO: should I call it from splitter?
    def _prepare_splitter_combine(self):
        self._splitter_rpn_comb = aux.remove_inp_from_splitter_rpn(self._splitter_rpn_wo_inner,
                                                                   self.comb_inp_to_remove)
        self.splitter_comb = aux.rpn2splitter(self._splitter_rpn_comb)


    def _prepare_axis_inputs_combine(self):
        # todo: do i need it?
        self._state_inputs_comb = self.state_inputs.copy()
        for inp in self.comb_inp_to_remove:
            self._state_inputs_comb.pop(inp)
        self._axis_for_input_comb, self._ndim_comb = aux.splitting_axis(self._state_inputs_comb,
                                                                         self._splitter_rpn_comb)
        self._input_for_axis_comb, self._shape_comb = aux.converting_axis2input(
            state_inputs=self._state_inputs_comb, axis_for_input=self._axis_for_input_comb,
            ndim=self._ndim_comb)


    # TODO: this hast be review (the idea itself)
    def _inner_splitter_separation(self, combiner):
        """
        checking if splitter is ok, allowing for inner splitter,
        assuming that inner splitter can be only in the most outer layer,
        and that it has to be also in the combiner
        """
        if type(self._splitter) is str:
            if self._splitter in self._inner_inputs_names:
                self._inner_splitter.append(self._splitter)
                self._splitter_wo_inner = None
        elif type(self._splitter) is tuple:
            if all([x in self._inner_inputs_names for x in self._splitter]):
                # TODO: still they might not gave the same shapes...
                self._inner_splitter += list(self._splitter)
                self._splitter_wo_inner = None
            elif any([x in self._inner_inputs_names for x in self._splitter]):
                    raise Exception("the scalar splitter {} is not correct, either both or neither "
                                    "od the elements should be inner inputs".format(self._splitter))
        elif type(self._splitter) is list:
            if all([x in self._inner_inputs_names for x in self._splitter]):
                # TODO: should i allow it?
                raise Exception("the outer splitter {} is not correct, both elements "
                                "can't be from inner inputs".format(self._splitter))
            # checking if one of the element is an inner input
            else:
                for (i, spl) in enumerate(self._splitter):
                    if spl in self._inner_inputs_names:
                        self._inner_splitter.append(spl)
                        self._splitter_wo_inner = self._splitter[(i+1)%2]


        self._inner_combiner = [comb for comb in combiner if comb in self._inner_splitter]
        self._combiner_wo_inner = list(set(combiner) - set(self._inner_combiner))


    def prepare_state_input(self):
        """prepare all inputs, should be called once all input is available"""

        # not all input field have to be use in the splitter, can be an extra scalar
        self._input_names = list(self.state_inputs.keys())

        # dictionary[key=input names] = list of axes related to
        # e.g. {'r': [1], 'e': [0], 'd': [0, 1]}
        # ndim - int, number of dimension for the "final array" (that is not created)
        self._axis_for_input, self._ndim = aux.splitting_axis(self.state_inputs, self._splitter_rpn_wo_inner)

        # list of inputs variable for each axis
        # e.g. [['e', 'd'], ['r', 'd']]
        # shape - list, e.g. [2,3]
        # TODO: do I need it?
        self._input_for_axis, self._shape = aux.converting_axis2input(
            state_inputs=self.state_inputs, axis_for_input=self._axis_for_input,
            ndim=self._ndim)

        # list of all possible indexes in each dim, will be use to iterate
        # e.g. [[0, 1], [0, 1, 2]]
        self.all_elements = [range(i) for i in self._shape]
        self.index_generator = itertools.product(*self.all_elements)

        if self.combiner:
            self._prepare_axis_inputs_combine()


    def state_values(self, ind, value=True):
        """returns state input as a dictionary (input name, value)"""
        if len(ind) > self._ndim:
            raise IndexError("too many indices")

        for ii, index in enumerate(ind):
            if index > self._shape[ii] - 1:
                raise IndexError("index {} is out of bounds for axis {} with size {}".format(
                    index, ii, self._shape[ii]))

        state_dict = {}
        for input, ax in self._axis_for_input.items():
            # checking which axes are important for the input
            sl_ax = slice(ax[0], ax[-1] + 1)
            # taking the indexes for the axes
            ind_inp = tuple(ind[sl_ax])  #used to be list
            if value:
                state_dict[input] = self.state_inputs[input][ind_inp]
            else: # using index instead of value
                ind_inp_str = "x".join([str(el) for el in ind_inp])
                state_dict[input] = ind_inp_str
        # adding values from input that are not used in the splitter
        for input in set(self._input_names) - set(self._input_names_splitter):
            if value:
                state_dict[input] = self.state_inputs[input]
            else:
                state_dict[input] = None
        # in py3.7 we can skip OrderedDict
        # returning a named tuple?
        return OrderedDict(sorted(state_dict.items(), key=lambda t: t[0]))


    def state_ind(self, ind):
        """state_values, but returning indices instead of values"""
        return self.state_values(ind, value=False)
'''
