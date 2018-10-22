import pdb
import inspect, os
import logging
from nipype import Node as Nipype1Node
logger = logging.getLogger('nipype.workflow')

# dj: might create a new class or move to State


# Function to change user provided mapper to "reverse polish notation" used in State
def mapper2rpn(mapper, other_mappers=None):
    """ Functions that translate mapper to "reverse polish notation."""
    output_mapper = []
    _ordering(mapper, i=0, output_mapper=output_mapper, other_mappers=other_mappers)
    return output_mapper


def _ordering(el, i, output_mapper, current_sign=None, other_mappers=None):
    """ Used in the mapper2rpn to get a proper order of fields and signs. """
    if type(el) is tuple:
        # checking if the mapper dont contain mapper from previous nodes, i.e. has str "_NA", etc.
        if type(el[0]) is str and el[0].startswith("_"):
            node_nm = el[0][1:]
            if node_nm not in other_mappers:
                raise Exception("can't ask for mapper from {}".format(node_nm))
            mapper_mod = change_mapper(mapper=other_mappers[node_nm], name=node_nm)
            el = (mapper_mod, el[1])
        if type(el[1]) is str and el[1].startswith("_"):
            node_nm = el[1][1:]
            if node_nm not in other_mappers:
                raise Exception("can't ask for mapper from {}".format(node_nm))
            mapper_mod = change_mapper(mapper=other_mappers[node_nm], name=node_nm)
            el = (el[0], mapper_mod)
        _iterate_list(el, ".", other_mappers, output_mapper=output_mapper)
    elif type(el) is list:
        if type(el[0]) is str and el[0].startswith("_"):
            node_nm = el[0][1:]
            if node_nm not in other_mappers:
                raise Exception("can't ask for mapper from {}".format(node_nm))
            mapper_mod = change_mapper(mapper=other_mappers[node_nm], name=node_nm)
            el[0] = mapper_mod
        if type(el[1]) is str and el[1].startswith("_"):
            node_nm = el[1][1:]
            if node_nm not in other_mappers:
                raise Exception("can't ask for mapper from {}".format(node_nm))
            mapper_mod = change_mapper(mapper=other_mappers[node_nm], name=node_nm)
            el[1] = mapper_mod
        _iterate_list(el, "*", other_mappers, output_mapper=output_mapper)
    elif type(el) is str:
        output_mapper.append(el)
    else:
        raise Exception("mapper has to be a string, a tuple or a list")

    if i > 0:
        output_mapper.append(current_sign)


def _iterate_list(element, sign, other_mappers, output_mapper):
    """ Used in the mapper2rpn to get recursion. """
    for i, el in enumerate(element):
        _ordering(
            el, i, current_sign=sign, other_mappers=other_mappers, output_mapper=output_mapper)


# functions used in State to know which element should be used for a specific axis
# TODO: should I moved it to State?

def mapping_axis(state_inputs, mapper_rpn):
    """Having inputs and mapper (in rpn notation), functions returns the axes of output for every input."""
    axis_for_input = {}
    stack = []
    # to remember current axis
    current_axis = None
    # to remember shapes and axes for partial results
    out_shape = {}
    out_axes = {}
    # to remember imput names for partial results
    out_inputname = {}
    for el in mapper_rpn:
        # scalar mapper
        if el == ".":
            right = stack.pop()
            left = stack.pop()
            # when both, left and right, are already products of partial mapper
            if left.startswith("OUT") and right.startswith("OUT"):
                if out_shape[left] != out_shape[right]:
                    raise Exception("arrays for scalar operations should have the same size")
                current_inputs = out_inputname[left] + out_inputname[right]
            # when left is already product of partial mapper
            elif left.startswith("OUT"):
                if state_inputs[right].shape == out_shape[left]:  #todo:should we allow for one-element array?
                    axis_for_input[right] = out_axes[left]
                else:
                    raise Exception("arrays for scalar operations should have the same size")
                current_inputs = out_inputname[left] + [right]
            # when right is already product of partial mapper
            elif right.startswith("OUT"):
                if state_inputs[left].shape == out_shape[right]:
                    axis_for_input[left] = out_axes[right]
                else:
                    raise Exception("arrays for scalar operations should have the same size")
                current_inputs = out_inputname[right] + [left]

            else:
                if state_inputs[right].shape == state_inputs[left].shape:
                    current_axis = list(range(state_inputs[right].ndim))
                    current_shape = state_inputs[left].shape
                    axis_for_input[left] = current_axis
                    axis_for_input[right] = current_axis
                    current_inputs = [left, right]
                else:
                    raise Exception("arrays for scalar operations should have the same size")
            # adding partial output to the stack
            stack.append("OUT_{}".format(len(out_shape)))
            out_inputname["OUT_{}".format(len(out_shape))] = current_inputs
            out_axes["OUT_{}".format(len(out_shape))] = current_axis
            out_shape["OUT_{}".format(len(out_shape))] = current_shape

        # outer mapper
        elif el == "*":
            right = stack.pop()
            left = stack.pop()
            # when both, left and right, are already products of partial mapper
            if left.startswith("OUT") and right.startswith("OUT"):
                # changing all axis_for_input for inputs from right
                for key in out_inputname[right]:
                    axis_for_input[key] = [
                        i + len(out_axes[left]) for i in axis_for_input[key]
                    ]
                current_axis = out_axes[left] +\
                               [i + (out_axes[left][-1] + 1) for i in out_axes[right]]
                current_shape = tuple([i for i in out_shape[left] + out_shape[right]])
                current_inputs = out_inputname[left] + out_inputname[right]
            # when left is already product of partial mapper
            elif left.startswith("OUT"):
                axis_for_input[right] = [
                    i + (out_axes[left][-1] + 1) for i in range(state_inputs[right].ndim)
                ]
                current_axis = out_axes[left] + axis_for_input[right]
                current_shape = tuple([i for i in out_shape[left] + state_inputs[right].shape])
                current_inputs = out_inputname[left] + [right]
            # when right is already product of partial mapper
            elif right.startswith("OUT"):
                # changing all axis_for_input for inputs from right
                for key in out_inputname[right]:
                    axis_for_input[key] = [
                        i + state_inputs[left].ndim for i in axis_for_input[key]
                    ]
                axis_for_input[left] = [
                    i - len(out_shape[right]) + (out_axes[right][-1] + 1)
                    for i in range(state_inputs[left].ndim)
                ]
                current_axis = out_axes[right] + [
                    i + (out_axes[right][-1] + 1) for i in range(state_inputs[left].ndim)
                ]
                current_shape = tuple([i for i in state_inputs[left].shape + out_shape[right]])
                current_inputs = out_inputname[right] + [left]
            else:
                axis_for_input[left] = list(range(state_inputs[left].ndim))
                axis_for_input[right] = [
                    i + state_inputs[left].ndim for i in range(state_inputs[right].ndim)
                ]
                current_axis = axis_for_input[left] + axis_for_input[right]
                current_shape = tuple(
                    [i for i in state_inputs[left].shape + state_inputs[right].shape])
                current_inputs = [left, right]
            # adding partial output to the stack
            stack.append("OUT_{}".format(len(out_shape)))
            out_inputname["OUT_{}".format(len(out_shape))] = current_inputs
            out_axes["OUT_{}".format(len(out_shape))] = current_axis
            out_shape["OUT_{}".format(len(out_shape))] = current_shape

        # just a name of input
        else:
            stack.append(el)

    if len(stack) == 0:
        pass
    elif len(stack) > 1:
        raise Exception("exception from mapping_axis")
    elif not stack[0].startswith("OUT"):
        current_axis = [i for i in range(state_inputs[stack[0]].ndim)]
        axis_for_input[stack[0]] = current_axis

    if current_axis:
        ndim = max(current_axis) + 1
    else:
        ndim = 0
    return axis_for_input, ndim


def converting_axis2input(state_inputs, axis_for_input, ndim):
    """ Having axes for all the input fields, the function returns fields for each axis. """
    input_for_axis = []
    shape = []
    for i in range(ndim):
        input_for_axis.append([])
        shape.append(0)

    for inp, axis in axis_for_input.items():
        for (i, ax) in enumerate(axis):
            input_for_axis[ax].append(inp)
            shape[ax] = state_inputs[inp].shape[i]

    return input_for_axis, shape


#function used in State if combiner

def remove_inp_from_mapper_rpn(mapper_rpn, inputs_to_remove):
    """modifying mapper_rpn: removing inputs due to combining"""
    mapper_rpn_copy = mapper_rpn.copy()
    # reverting order
    mapper_rpn_copy.reverse()
    stack_inp = []
    stack_sgn = []
    from_last_sign = []
    for (ii, el) in enumerate(mapper_rpn_copy):
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
                stack_sgn.pop(-1*from_last_sign.pop())

    # creating the final mapper_rpn after combining
    remaining_elements = stack_sgn + stack_inp
    remaining_elements.sort(reverse=True)
    mapper_rpn_combined = [el for (i,el) in remaining_elements]
    return mapper_rpn_combined


def rpn2mapper(mapper_rpn):
    """recurrent algorithm to move from mapper_rpn to mapper.
       every time combines pairs of input in one input,
       ends when the length is one
     """
    if mapper_rpn == []:
        return None
    if len(mapper_rpn) == 1:
        return mapper_rpn[0]

    mapper_rpn_copy = mapper_rpn.copy()
    signs = [".", "*"]
    mapper_modified = []

    while mapper_rpn_copy:
        el = mapper_rpn_copy.pop()
        # element is a sign
        if el in signs:
            if mapper_rpn_copy[-1] not in signs and mapper_rpn_copy[-2] not in signs:
                right, left = mapper_rpn_copy.pop(), mapper_rpn_copy.pop()
                if el == ".":
                    mapper_modified.append((left, right))
                elif el == "*":
                    mapper_modified.append([left, right])
            else:
                mapper_modified.append(el)
        else:
            mapper_modified.append(el)

    # reversing the list and combining more
    mapper_modified.reverse()
    return rpn2mapper(mapper_modified)




# used in the Node to change names in a mapper


def change_mapper(mapper, name):
    """changing names of mapper: adding names of the node"""
    if isinstance(mapper, str):
        if "." in mapper or mapper.startswith("_"):
            return mapper
        else:
            return "{}.{}".format(name, mapper)
    elif isinstance(mapper, list):
        return _add_name(mapper, name)
    elif isinstance(mapper, tuple):
        mapper_l = list(mapper)
        return tuple(_add_name(mapper_l, name))


def _add_name(mlist, name):
    for i, elem in enumerate(mlist):
        if isinstance(elem, str):
            if "." in elem or elem.startswith("_"):
                pass
            else:
                mlist[i] = "{}.{}".format(name, mlist[i])
        elif isinstance(elem, list):
            mlist[i] = _add_name(elem, name)
        elif isinstance(elem, tuple):
            mlist[i] = list(elem)
            mlist[i] = _add_name(mlist[i], name)
            mlist[i] = tuple(mlist[i])
    return mlist


def search_mapper(element, mapper):
    """checking if an element is part of the mapper (used to check combiner elements)"""
    if type(mapper) in [list, tuple]:
        if any([search_mapper(element, mapper_el) for mapper_el in mapper]):
            return True
    elif element == mapper:
        return True
    else:
        return False


# want to use to access input as dot,
# but it doesnt work since im using "." within names (using my old syntax with - also cant work)
# https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


class CurrentInterface(object):
    def __init__(self, interface, name):
        self.nn = Nipype1Node(interface=interface, name=name)
        self.output = {}

    def run(self, inputs, base_dir, dir_nm_el):
        self.nn.base_dir = os.path.join(base_dir, dir_nm_el)
        for key, val in inputs.items():
            key = key.split(".")[-1]
            setattr(self.nn.inputs, key, val)
        #have to set again self._output_dir in case of mapper
        self.nn._output_dir = os.path.join(self.nn.base_dir, self.nn.name)
        res = self.nn.run()
        return res
