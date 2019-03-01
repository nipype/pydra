"""Basic compute graph elements"""
from collections import OrderedDict
import dataclasses as dc
import itertools
import json
import logging
import networkx as nx
import numpy as np
import os, pdb
from pathlib import Path
import typing as ty
import pickle as pk
from copy import copy, deepcopy

from . import state
from . import auxiliary as aux
from .specs import File, BaseSpec
from .helpers import (make_klass, create_checksum, print_help, load_result,
                      ensure_list, get_inputs)

logger = logging.getLogger('pydra')


class NodeBase:
    _api_version: str = "0.0.1"  # Should generally not be touched by subclasses
    _version: str  # Version of tool being wrapped
    _input_sets = None  # Dictionaries of predefined input settings

    # dj: do we need it??
    input_spec = BaseSpec
    output_spec = BaseSpec

    # TODO: write state should be removed
    def __init__(self, name, splitter=None, combiner=None,
                 inputs: ty.Union[ty.Text, File, ty.Dict, None] = None):
        """A base structure for nodes in the computational graph (i.e. both
        ``Node`` and ``Workflow``).

        Parameters
        ----------

        name : str
            Unique name of this node
        splitter : str or (list or tuple of (str or splitters))
            Whether inputs should be split at run time
        combiner: str or list of strings (names of variables)
            variables that should be used to combine results together
        inputs : dictionary (input name, input value or list of values)
            States this node's input names
        """
        self.name = name
        if not self.input_spec:
            raise Exception(
                'No input_spec in class: %s' % self.__class__.__name__)
        klass = make_klass(self.input_spec)
        self.inputs = klass(**{f.name: (None if f.default is dc.MISSING
                                  else f.default) for f in dc.fields(klass)})
        self.input_names = [field.name for field in dc.fields(klass)
                            if field.name not in ["_func"]]

        self._needed_outputs = []
        if splitter:
            # adding name of the node to the input name within the splitter
            splitter = aux.change_splitter(splitter, self.name)
        self.state = self.set_state(splitter, combiner)
        self._output = {}
        self._result = {}
        # flag that says if node finished all jobs
        self._is_complete = False

        if self._input_sets is None:
            self._input_sets = {}
        if inputs:
            if isinstance(inputs, dict):
                inputs = {k:v for k,v in inputs.items() if k in self.input_names}
            elif Path(inputs).is_file():
                inputs = json.loads(Path(inputs).read_text())
            elif isinstance(inputs, str):
                if self._input_sets is None or inputs not in self._input_sets:
                    raise ValueError("Unknown input set {!r}".format(inputs))
                inputs = self._input_sets[inputs]
            self.inputs = dc.replace(self.inputs, **inputs)
            self.state_inputs = inputs

    def __getstate__(self):
        state = self.__dict__.copy()
        state['input_spec'] = pk.dumps(state['input_spec'])
        state['output_spec'] = pk.dumps(state['output_spec'])
        state['inputs'] = dc.asdict(state['inputs'])
        return state

    def __setstate__(self, state):
        state['input_spec'] = pk.loads(state['input_spec'])
        state['output_spec'] = pk.loads(state['output_spec'])
        state['inputs'] = make_klass(state['input_spec'])(**state['inputs'])
        self.__dict__.update(state)

    def help(self, returnhelp=False):
        """ Prints class help
        """
        help_obj = print_help(self)
        if returnhelp:
            return help_obj

    @property
    def version(self):
        return self._version

    def save_set(self, name, inputs, force=False):
        if name in self._input_sets and not force:
            raise KeyError('Key {} already saved. Use force=True to override.')
        self._input_sets[name] = inputs

    @property
    def checksum(self):
        return create_checksum(self.__class__.__name__, self.inputs)

    def ready2run(self, index=None):
        # flag that says if the node/wf is ready to run (has all input)
        for node, _, _ in self.needed_outputs:
            if not node.is_finished(index=index):
                return False
        return True

    def is_finished(self, index=None):
        # TODO: check local procs
        return False

    @property
    def needed_outputs(self):
        return self._needed_outputs

    @needed_outputs.setter
    def needed_outputs(self, requires):
        self._needed_outputs = ensure_list(requires)


    def set_state(self, splitter, combiner=None):
        incoming_states = []
        for node, _, _ in self.needed_outputs:
            if node.state is not None:
                incoming_states.append(node.state)
        if splitter is None:
            splitter = [state.name for state in incoming_states] or None
        elif len(incoming_states):
            rpn = aux.splitter2rpn(splitter)
            # TODO: check for keys instead of just names
            left_out = [state.name for state in incoming_states
                        if state.name not in rpn]

        if splitter is not None:
            self.state = state.State(name=self.name, splitter=splitter, combiner=combiner)
        else:
            self.state = None
        return self.state

    @property
    def output_names(self):
        return [f.name for f in dc.fields(make_klass(self.output_spec))]

    @property
    def output(self):
        return self._output

    def result(self, cache_locations=None,
               return_state=False):
        self._reading_results()
        return self._result
        # return self._reading_results(cache_locations=cache_locations,
        #                              return_state=return_state)


    # TODO: should change state!
    def split(self, splitter, **kwargs):
        if kwargs:
            self.inputs = dc.replace(self.inputs, **kwargs)
            #dj:??, check if I need it
            self.state_inputs = kwargs
        splitter = aux.change_splitter(splitter, self.name)
        if self.state:
            raise Exception("splitter has been already set")
            # TODOD: should I allow?
            # if splitter != self.state.splitter:
            #     combiner = self.state.combiner
            #     self.set_state(splitter,combiner)
            # else: # if the same splitter do nothing
            #     pass
        else:
            self.set_state(splitter)
        return self
    #
    # def combine(self, combiner):
    #     self.combiner = combiner
    #     return self

    def checking_input_el(self, ind):
        """checking if all inputs are available (for specific state element)"""
        try:
            self.get_input_el(ind)
            return True
        except:  #TODO specify
            return False

    def get_input_el(self, ind):
        """collecting all inputs required to run the node (for specific state element)"""
        state_dict = self.state.states_val[ind]
        if hasattr(self, "partial_split_input"):
            for inp, ax_shift in self.partial_split_input.items():
                ax_shift.sort(reverse=True)
                for (orig_ax, new_ax) in ax_shift:
                    state_dict[inp] = np.take(state_dict[inp], indices=ind[new_ax], axis=orig_ax)
        inputs_dict = {"{}.{}".format(self.name, k): state_dict["{}.{}".format(self.name, k)]
                        for k in self.input_names}

        # reading extra inputs that come from previous nodes
        for (from_node, from_socket, to_socket) in self.needed_outputs:
            # TODO update
            if from_node.state.combiner:
                inputs_dict["{}.{}".format(self.name, to_socket)] =\
                    self._get_input_comb(from_node, from_socket, state_dict)
            else:
                dir_nm_el_from, _ = from_node._directory_name_state_surv(state_dict)
                # TODO: do I need this if, what if this is wf?
                if is_node(from_node):
                    out_from = getattr(from_node.results_dict[dir_nm_el_from].output, from_socket)
                    if out_from:
                        inputs_dict["{}.{}".format(self.name, to_socket)] = out_from
                    else:
                        raise Exception("output from {} doesnt exist".format(from_node))

        return state_dict, inputs_dict

    # TODO: update
    def _get_input_comb(self, from_node, from_socket, state_dict):
        """collecting all outputs from previous node that has combiner"""
        state_dict_all = self._state_dict_all_comb(from_node, state_dict)
        inputs_all = []
        for state in state_dict_all:
            dir_nm_el_from = "_".join([
                "{}:{}".format(i, j) for i, j in list(state.items())])
            if is_node(from_node):
                out_from = getattr(from_node.results_dict[dir_nm_el_from].output, from_socket)
                if out_from:
                    inputs_all.append(out_from)
                else:
                    raise Exception("output from {} doesnt exist".format(from_node))
        return inputs_all

    def _state_dict_all_comb(self, from_node, state_dict):
        """collecting state dictionary for all elements that were combined together"""
        elements_per_axes = {}
        axis_for_input = {}
        all_axes = []
        for inp in from_node.combiner:
            axis_for_input[inp] = from_node.state._axis_for_input[inp]
            for (i, ax) in enumerate(axis_for_input[inp]):
                elements_per_axes[ax] = state_dict[inp].shape[i]
                all_axes.append(ax)
        all_axes = list(set(all_axes))
        all_axes.sort()
        # axes in axis_for_input have to be shifted, so they start in 0
        # they should fit all_elements format
        for inp, ax_l in axis_for_input.items():
            ax_new_l = [all_axes.index(ax) for ax in ax_l]
            axis_for_input[inp] = ax_new_l
        # collecting shapes for all axes of the combiner
        shape = [el for (ax, el) in sorted(elements_per_axes.items())]
        all_elements = [range(i) for i in shape]
        index_generator = itertools.product(*all_elements)
        state_dict_all = []
        for ind in index_generator:
            state_dict_all.append(self._state_dict_el_for_comb(ind, state_dict,
                                                               axis_for_input))
        return state_dict_all


    # similar to State.state_value (could be combined?)
    def _state_dict_el_for_comb(self, ind, state_inputs, axis_for_input, value=True):
        """state input for a specific ind (used for connection)"""
        state_dict_el = {}
        for input, ax in axis_for_input.items():
            # checking which axes are important for the input
            sl_ax = slice(ax[0], ax[-1] + 1)
            # taking the indexes for the axes
            ind_inp = tuple(ind[sl_ax])  # used to be list
            if value:
                state_dict_el[input] = state_inputs[input][ind_inp]
            else:  # using index instead of value
                ind_inp_str = "x".join([str(el) for el in ind_inp])
                state_dict_el[input] = ind_inp_str

        if hasattr(self, "partial_comb_input"):
            for input, ax_shift in self.partial_comb_input.items():
                ind_inp = []
                partial_input = state_inputs[input]
                for (inp_ax, comb_ax) in ax_shift:
                    ind_inp.append(ind[comb_ax])
                    partial_input = np.take(partial_input, indices=ind[comb_ax], axis=inp_ax)
                if value:
                    state_dict_el[input] = partial_input
                else:  # using index instead of value
                    ind_inp_str = "x".join([str(el) for el in ind_inp])
                    state_dict_el[input] = ind_inp_str
        # adding values from input that are not used in the splitter
        for input in set(state_inputs) - set(axis_for_input) - set(self.partial_comb_input):
            if value:
                state_dict_el[input] = state_inputs[input]
            else:
                state_dict_el[input] = None
        # in py3.7 we can skip OrderedDict
        return OrderedDict(sorted(state_dict_el.items(), key=lambda t: t[0]))


    def _directory_name_state_surv(self, state_dict):
        """eliminating all inputs from state dictionary that are not in
        the splitter of the node;
        creating a directory name
        """
        # should I be using self.state._splitter_rpn_comb?
        state_surv_dict = dict((key, val) for (key, val) in state_dict.items()
                               if key in self.state.splitter_rpn)
        dir_nm_el = "_".join(["{}:{}".format(i, j)
                              for i, j in list(state_surv_dict.items())])
        return dir_nm_el, state_surv_dict


    # checking if all outputs are saved
    @property
    def is_complete(self):
        # once _is_complete os True, this should not change
        logger.debug('is_complete {}'.format(self._is_complete))
        if self._is_complete:
            return self._is_complete
        else:
            return self._check_all_results()

    def get_output(self):
        raise NotImplementedError

    def _check_all_results(self):
        raise NotImplementedError

    def _reading_results(self):
        raise NotImplementedError

    def _state_dict_to_list(self, container):
        """creating a list of tuples from dictionary and changing key (state) from str to dict"""
        if type(container) is dict:
                val_l = list(container.items())
        else:
            raise Exception("{} has to be dict".format(container))
        val_dict_l = [(self.state.states_val[i[0]], i[1]) for i in val_l]
        return val_dict_l


    def _combined_output(self, key_out, state_dict, output_el):
        for inp in self.state.combiner_all:
            state_dict.pop(inp)
        state_tuple = tuple(state_dict.items())
        if state_tuple in self._output[key_out].keys():
            self._output[key_out][state_tuple].append(output_el)
        else:
            self._output[key_out][state_tuple] = [output_el]


    def _reading_results_one_output(self, key_out):
        """reading results for one specific output name"""
        if not self.state.splitter:
            if type(self.output[key_out]) is tuple:
                result = self.output[key_out]
            elif type(self.output[key_out]) is dict:
                val_l = self._state_dict_to_list(self.output[key_out])
                if len(val_l) == 1:
                    result = val_l[0]
                # this is used for wf (can be no splitter but multiple values from node splitter)
                else:
                    result = val_l
        elif (self.state.combiner and not self.state._splitter_rpn_comb):
            val_l = self._state_dict_to_list(self.output[key_out])
            result = val_l[0]
        elif self.state.splitter:
            val_l = self._state_dict_to_list(self._output[key_out])
            result = val_l
        return result



class Node(NodeBase):
    def __init__(self, name, inputs=None, splitter=None, workingdir=None,
                 other_splitters=None, combiner=None):
        super(Node, self).__init__(name=name, splitter=splitter, inputs=inputs,
                                   #other_splitters=other_splitters,
                                   combiner=combiner)

        # working directory for node, will be change if node is a part of a wf
        self.workingdir = workingdir
        # dictionary of results from tasks
        self.results_dict = {}


    def run_interface_el(self, ind):
        """ running interface one element generated from node_state."""
        if ind is not None:
            logger.debug("Run interface el, name={}, ind={}".format(self.name, ind))
            state_dict, inputs_dict = self.get_input_el(ind)
            os.makedirs(os.path.join(os.getcwd(), self.workingdir), exist_ok=True)
            self.cache_dir = os.path.join(os.getcwd(), self.workingdir)
            interf_inputs = dict((k.split(".")[1], v) for k,v in inputs_dict.items())
            res = self.run(**interf_inputs)
            return ind, res #TODO NOW: don't have to return


    def get_output(self):
        """collecting all outputs and updating self._output
        (assuming that file already exist and this was checked)
        """
        for key_out in self.output_names:
            self._output[key_out] = {}
            for (ii, val) in enumerate(self.state.states_val):
                state_dict = val
                if self.state.splitter:
                    output_el = getattr(self.results_dict[ii].output, key_out)
                    if not self.state.combiner: # only splitter
                        self._output[key_out][tuple(self.state.states_val[ii].items())] = output_el
                    else:
                        self._combined_output(key_out, deepcopy(state_dict), output_el)
                else:
                    raise Exception("not implemented, TODO")
        return self._output


    # dj: should I combine with get_output?
    def _check_all_results(self):
        """checking if all files that should be created are present
        if all files and outputs are present, self._is_complete is changed to True
        (the method does not collect the output)
        """
        for ii, val in enumerate(self.state.states_val):
            for key_out in self.output_names:
                if not getattr(self.results_dict[ii].output, key_out):
                    return False
        self._is_complete = True
        return True

    def _reading_results(self, ):
        """ collecting all results for all output names"""
        """
        return load_result(self.checksum,
                             ensure_list(cache_locations) +
                             ensure_list(self._cache_dir))

        """
        for key_out in self.output_names:
            output = self.output[key_out]
            self._result[key_out] = [(dict(k), v) for k,v in output.items()]



class Workflow(NodeBase):
    def __init__(self, name, inputs=None, wf_output_names=None, splitter=None,
                 nodes=None, workingdir=None, *args, **kwargs):
        super(Workflow, self).__init__(name=name, splitter=splitter, inputs=inputs,
                                       *args, **kwargs)

        self.graph = nx.DiGraph()
        # all nodes in the workflow (probably will be removed)
        self._nodes = []
        # saving all connection between nodes
        self.connected_var = {}
        # input that are expected by nodes to get from wf.inputs
        self.needed_inp_wf = []
        if nodes:
            self.add_nodes(nodes)
        for nn in self._nodes:
            self.connected_var[nn] = {}
        # key: name of a node, value: the node
        self._node_names = {}
        # key: name of a node, value: splitter of the node
        self._node_splitters = {}
        # dj: not sure if this should be different than base_dir
        self.workingdir = os.path.join(os.getcwd(), workingdir)
        # list of (nodename, output name in the name, output name in wf) or (nodename, output name in the name)
        # dj: using different name than for node, since this one it is defined by a user
        self.wf_output_names = wf_output_names

        # nodes that are created when the workflow has splitter (key: node name, value: list of nodes)
        self.inner_nodes = {}
        # in case of inner workflow this points to the main/parent workflow
        self.parent_wf = None


    @property
    def nodes(self):
        return self._nodes

    @property
    def graph_sorted(self):
        # TODO: should I always update the graph?
        return list(nx.topological_sort(self.graph))


    def split_node(self, splitter, node=None, inputs=None):
        """this is setting a splitter to the wf's nodes (not to the wf)"""
        if type(node) is str:
            node = self._node_names[node]
        elif node is None:
            node = self._last_added
        if node.splitter:
            raise Exception("Cannot assign two splitters to the same node")
        node.split(splitter=splitter, inputs=inputs)
        self._node_splitters[node.name] = node.splitter
        return self


    def combine_node(self, combiner, node=None):
        """this is setting a combiner to the wf's nodes (not to the wf)"""
        if type(node) is str:
            node = self._node_names[node]
        elif node is None:
            node = self._last_added
        if node.combiner:
            raise Exception("Cannot assign two combiners to the same node")
        node.combine(combiner=combiner)
        return self


    def get_output(self):
        # not sure, if I should collect output of all nodes or only the ones that are used in wf.output
        self.node_outputs = {}
        for nn in self.graph:
            if self.splitter:
                self.node_outputs[nn.name] = [ni.get_output() for ni in self.inner_nodes[nn.name]]
            else:
                self.node_outputs[nn.name] = nn.get_output()
        if self.wf_output_names:
            for out in self.wf_output_names:
                if len(out) == 2:
                    node_nm, out_nd_nm, out_wf_nm = out[0], out[1], out[1]
                elif len(out) == 3:
                    node_nm, out_nd_nm, out_wf_nm = out
                else:
                    raise Exception("wf_output_names should have 2 or 3 elements")
                if out_wf_nm not in self._output.keys():
                    if self.splitter:
                        self._output[out_wf_nm] = {}
                        for (i, val) in enumerate(self.state.states_val):
                            wf_inputs_dict = val
                            dir_nm_el, _ = self._directory_name_state_surv(wf_inputs_dict)
                            output_el = self.node_outputs[node_nm][i][out_nd_nm]
                            if not self.combiner: # splitter only
                                self._output[out_wf_nm][dir_nm_el] = output_el[1]
                            else:
                                self._combined_output(out_wf_nm, wf_inputs_dict, output_el[1])
                    else:
                        self._output[out_wf_nm] = self.node_outputs[node_nm][out_nd_nm]
                else:
                    raise Exception(
                        "the key {} is already used in workflow.result".format(out_wf_nm))
        return self._output


    # TODO: might merge with the function from Node
    def _check_all_results(self):
        """checking if all files that should be created are present"""
        for nn in self.graph_sorted:
            if nn.name in self.inner_nodes.keys():
                if not all([ni.is_complete for ni in self.inner_nodes[nn.name]]):
                    return False
            else:
                if not nn.is_complete:
                    return False
        self._is_complete = True
        return True


    def _reading_results(self):
        """reading all results for the workflow,
        nodes/outputs names specified in self.wf_output_names
        """
        if self.wf_output_names:
            for out in self.wf_output_names:
                key_out = out[-1]
                self._result[key_out] = self._reading_results_one_output(key_out)


    # TODO: this should be probably using add method, but might be also removed completely
    def add_nodes(self, nodes):
        """adding nodes without defining connections
            most likely it will be removed at the end
        """
        self.graph.add_nodes_from(nodes)
        for nn in nodes:
            self._nodes.append(nn)
            self.connected_var[nn] = {}
            self._node_names[nn.name] = nn


    # TODO: workingir shouldn't have None
    def add(self, runnable, name=None, workingdir=None, inputs=None,
            output_names=None, splitter=None, combiner=None, **kwargs):
        # TODO: should I also accept normal function?
        if is_node(runnable):
            node = runnable
            node.other_splitters = self._node_splitters
        elif is_workflow(runnable):
            node = runnable
        elif is_function(runnable):
            if not output_names:
                output_names = ["out"]
            if not name:
                raise Exception("you have to specify name for the node")
            if not workingdir:
                workingdir = name
            from .task import to_task
            node = to_task(runnable, workingdir=workingdir,
                        name=name,
                        inputs=inputs, splitter=splitter,
                        other_splitters=self._node_splitters,
                        combiner=combiner, output_names=output_names)
        else:
            raise ValueError("Unknown workflow element: {!r}".format(runnable))
        self.add_nodes([node])
        self._last_added = node
        # connecting inputs from other nodes outputs
        # (assuming that all kwargs provide connections)
        for (inp, source) in kwargs.items():
            try:
                from_node_nm, from_socket = source.split(".")
                self.connect(from_node_nm, from_socket, node.name, inp)
            # TODO not sure if i need it, just check if from_node_nm is not None??
            except (ValueError):
                self.connect_wf_input(source, node.name, inp)
        return self


    def connect(self, from_node_nm, from_socket, to_node_nm, to_socket):
        from_node = self._node_names[from_node_nm]
        to_node = self._node_names[to_node_nm]
        self.graph.add_edges_from([(from_node, to_node)])
        if not to_node in self.nodes:
            self.add_nodes(to_node)
        self.connected_var[to_node][to_socket] = (from_node, from_socket)
        # from_node.sending_output.append((from_socket, to_node, to_socket))
        logger.debug('connecting {} and {}'.format(from_node, to_node))


    def connect_wf_input(self, inp_wf, node_nm, inp_nd):
        self.needed_inp_wf.append((node_nm, inp_wf, inp_nd))


    def preparing(self, wf_inputs=None, wf_inputs_ind=None, st_inputs=None):
        """preparing nodes which are connected: setting the final splitter and state_inputs"""
        for node_nm, inp_wf, inp_nd in self.needed_inp_wf:
            node = self._node_names[node_nm]
            if "{}.{}".format(self.name, inp_wf) in wf_inputs:
                node.state_inputs.update({
                    "{}.{}".format(node_nm, inp_nd):
                    wf_inputs["{}.{}".format(self.name, inp_wf)]
                })
                node.inputs.update({
                    "{}.{}".format(node_nm, inp_nd):
                    wf_inputs["{}.{}".format(self.name, inp_wf)]
                })
            else:
                raise Exception("{}.{} not in the workflow inputs".format(self.name, inp_wf))
        for nn in self.graph_sorted:
            if not st_inputs:
                st_inputs=wf_inputs
            dir_nm_el, _ = self._directory_name_state_surv(st_inputs)
            if not self.splitter:
                dir_nm_el = ""
            nn.workingdir = os.path.join(self.workingdir, dir_nm_el, nn.name)
            nn._is_complete = False  # helps when mp is used
            try:
                for inp, (out_node, out_var) in self.connected_var[nn].items():
                    nn.ready2run = False  #it has some history (doesnt have to be in the loop)
                    nn.state_inputs.update(out_node.state_inputs)
                    nn.needed_outputs.append((out_node, out_var, inp))
                    #if there is no splitter provided, i'm assuming that splitter is taken from the previous node
                    if (not nn.splitter or nn.splitter == out_node.splitter) and out_node.splitter:
                        # TODO!!: what if I have more connections, not only from one node
                        if out_node.combiner:
                            nn.splitter = out_node.state.splitter_comb
                            # adding information about partially combined input from previous nodes
                            nn.partial_split_input = out_node.state.partial_comb_input_rem_axes
                            nn.partial_comb_input = out_node.state.partial_comb_input_comb_axes
                        else:
                            nn.splitter = out_node.splitter
                    else:
                        pass
            except (KeyError):
                # tmp: we don't care about nn that are not in self.connected_var
                pass
            nn.prepare_state_input()


    # removing temp. from Workflow
    # def run(self, plugin="serial"):
    #     #self.preparing(wf_inputs=self.inputs) # moved to submitter
    #     self.prepare_state_input()
    #     logger.debug('the sorted graph is: {}'.format(self.graph_sorted))
    #     submitter = sub.SubmitterWorkflow(workflow=self, plugin=plugin)
    #     submitter.run_workflow()
    #     submitter.close()
    #     self.collecting_output()


def is_function(obj):
    return hasattr(obj, '__call__')


def is_node(obj):
    return isinstance(obj, Node)


def is_workflow(obj):
    return isinstance(obj, Workflow)
