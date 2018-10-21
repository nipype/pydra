from collections import OrderedDict
import itertools
from copy import deepcopy
import pdb

from . import auxiliary as aux


class State(object):
    def __init__(self, node_name, mapper=None, other_mappers=None, combiner=None):
        self._mapper = mapper
        self._combiner = combiner
        self.node_name = node_name
        if self._mapper:
            # changing mapper (as in rpn), so I can read from left to right
            # e.g. if mapper=('d', ['e', 'r']), _mapper_rpn=['d', 'e', 'r', '*', '.']
            self._mapper_rpn = aux.mapper2rpn(self._mapper, other_mappers=other_mappers)
            self._input_names_mapper = [i for i in self._mapper_rpn if i not in ["*", "."]]
        else:
            self._mapper_rpn = []
            self._input_names_mapper = []

    def prepare_state_input(self, state_inputs):
        """prepare all inputs, should be called once all input is available"""

        # dj TOTHINK: I actually stopped using state_inputs for now, since people wanted to have mapper not only
        # for state inputs. Might have to come back....
        self.state_inputs = state_inputs

        # not all input field have to be use in the mapper, can be an extra scalar
        self._input_names = list(self.state_inputs.keys())

        # dictionary[key=input names] = list of axes related to
        # e.g. {'r': [1], 'e': [0], 'd': [0, 1]}
        # ndim - int, number of dimension for the "final array" (that is not created)
        self._axis_for_input, self._ndim = aux.mapping_axis(self.state_inputs, self._mapper_rpn)

        # list of inputs variable for each axis
        # e.g. [['e', 'd'], ['r', 'd']]
        # shape - list, e.g. [2,3]
        # TODO: do I need it?
        self._input_for_axis, self._shape = aux.converting_axis2input(
            self.state_inputs, self._axis_for_input, self._ndim)

        if self._combiner:
            self._prepare_combining()

        # list of all possible indexes in each dim, will be use to iterate
        # e.g. [[0, 1], [0, 1, 2]]
        self.all_elements = [range(i) for i in self._shape]
        self.index_generator = itertools.product(*self.all_elements)

    def __getitem__(self, ind):
        if type(ind) is int:
            ind = (ind, )
        return self.state_values(ind)

    # not used?
    #@property
    #def mapper(self):
    #    return self._mapper

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape


    def _prepare_combining(self):
        self.inp_to_remove = []
        axes_to_remove = []
        for comb_el in self._combiner:
            self.inp_to_remove.append(comb_el)
            axes = self._axis_for_input[comb_el]
            for ax in axes:
                if ax in axes_to_remove:
                    raise Exception("can't combine by {}, axis {} already removed".format(
                        comb_el, ax
                    ))
                inputs = deepcopy(self._input_for_axis[ax])
                inputs.remove(comb_el)
                if inputs:
                    for other_inp in inputs:
                        self.inp_to_remove.append(other_inp)
            axes_to_remove += axes
        self._prepare_final_mapper(axes_to_remove)


    def _prepare_final_mapper(self, axes_to_remove):
        # TODO not sure if I have to change these atr, or only mappers
        # TODO: this actually should be simply done using method from aux after the mapper is changed
        self._input_for_axis_comb = self._input_for_axis.copy()
        axes_to_remove.sort(reverse=True)
        for ax in axes_to_remove:
            self._input_for_axis_comb.pop(ax)
        self._axis_for_input_comb = self._axis_for_input.copy()
        for inp in self.inp_to_remove:
            self._axis_for_input_comb.pop(inp)
        # TODO
        # change mapper_rpn - > different mapper


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
        # adding values from input that are not used in the mapper
        for input in set(self._input_names) - set(self._input_names_mapper):
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