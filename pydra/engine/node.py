"""Basic compute graph elements"""
import os
import itertools
import pdb
import networkx as nx
import numpy as np
from collections import OrderedDict

from nipype.utils.filemanip import loadpkl
from nipype import logging, Function

from . import state
from . import auxiliary as aux
logger = logging.getLogger('nipype.workflow')


class NodeBase(object):
    def __init__(self, name, splitter=None, combiner=None, inputs=None,
                 other_splitters=None, write_state=True, *args, **kwargs):
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
        other_splitters : dictionary (name of a node, splitter of the node)
            information about other nodes' splitters from workflow (in case the splitter
            from previous node is used)
        write_state : True
            flag that says if value of state input should be written out to output
            and directories (otherwise indices are used)



        """
        self.name = name
        self._inputs = {}
        self._state_inputs = {}

        if inputs:
            self.inputs = inputs

        if splitter:
            # adding name of the node to the input name within the splitter
            splitter = aux.change_splitter(splitter, self.name)
        self._splitter = splitter
        if other_splitters:
            self._other_splitters = other_splitters
        else:
            self._other_splitters = {}
        self._combiner = None
        if combiner:
            self.combiner = combiner
        self._output = {}
        self._result = {}
        # flag that says if the node/wf is ready to run (has all input)
        self.ready2run = True
        # needed outputs from other nodes if the node part of a wf
        self.needed_outputs = []
        # flag that says if node finished all jobs
        self._is_complete = False
        self.write_state = write_state



    @property
    def state(self):
        return self._state

    @property
    def splitter(self):
        return self._splitter

    @splitter.setter
    def splitter(self, splitter):
        if self._splitter:
            raise Exception("splitter is already set")
        self._splitter = aux.change_splitter(splitter, self.name)


    @property
    def combiner(self):
        return self._combiner

    @combiner.setter
    def combiner(self, combiner):
        if self._combiner:
            raise Exception("combiner is already set")
        if not self.splitter:
            raise Exception("splitter has to be set before setting combiner")
        if type(combiner) is str:
            combiner = [combiner]
        elif type(combiner) is not list:
            raise Exception("combiner should be a string or a list")
        self._combiner = aux.change_splitter(combiner, self.name)
        # TODO: this check should be moved somewhere
        # for el in self._combiner:
        #     if el not in self.state._splitter_rpn:
        #         raise Exception("element {} of combiner is not found in the splitter {}".format(
        #             el, self.splitter))

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        # Massage inputs dict
        inputs = {
            ".".join((self.name, key)): value if not isinstance(value, list) else np.array(value)
            for key, value in inputs.items()
        }
        self._inputs.update(inputs)
        self._state_inputs.update(inputs)

    @property
    def state_inputs(self):
        return self._state_inputs

    @state_inputs.setter
    def state_inputs(self, state_inputs):
        self._state_inputs.update(state_inputs)

    @property
    def output(self):
        return self._output

    @property
    def result(self):
        if not self._result:
            self._reading_results()
        return self._result

    def prepare_state_input(self):
        self._state = state.State(splitter=self._splitter, node_name=self.name, other_splitters=self._other_splitters,
                                  combiner=self.combiner)
        self._state.prepare_state_input(state_inputs=self.state_inputs)


    def split(self, splitter, inputs=None):
        self.splitter = splitter
        if inputs:
            self.inputs = inputs
            self._state_inputs.update(self.inputs)
        return self


    def combine(self, combiner):
        self.combiner = combiner
        return self


    def checking_input_el(self, ind):
        """checking if all inputs are available (for specific state element)"""
        try:
            self.get_input_el(ind)
            return True
        except:  #TODO specify
            return False

    # dj: this is not used for a single node
    def get_input_el(self, ind):
        """collecting all inputs required to run the node (for specific state element)"""
        state_dict = self.state.state_values(ind)
        if hasattr(self, "partial_split_input"):
            for inp, ax_shift in self.partial_split_input.items():
                ax_shift.sort(reverse=True)
                for (orig_ax, new_ax) in ax_shift:
                    state_dict[inp] = np.take(state_dict[inp], indices=ind[new_ax], axis=orig_ax)
        inputs_dict = {k: state_dict[k] for k in self._inputs.keys()}
        if not self.write_state:
            state_dict = self.state.state_ind(ind)

         # reading extra inputs that come from previous nodes
        for (from_node, from_socket, to_socket) in self.needed_outputs:
            # if the previous node has combiner I have to collect all elements
            if from_node.state.combiner:
                inputs_dict["{}.{}".format(self.name, to_socket)] =\
                    self._get_input_comb(from_node, from_socket, state_dict)
            else:
                dir_nm_el_from, _ = from_node._directory_name_state_surv(state_dict)
                # TODO: do I need this if, what if this is wf?
                if is_node(from_node):
                    out_from = self._reading_ci_output(
                        node=from_node, dir_nm_el=dir_nm_el_from, out_nm=from_socket)
                    if out_from:
                        inputs_dict["{}.{}".format(self.name, to_socket)] = out_from
                    else:
                        raise Exception("output from {} doesnt exist".format(from_node))
        return state_dict, inputs_dict


    def _get_input_comb(self, from_node, from_socket, state_dict):
        """collecting all outputs from previous node that has combiner"""
        state_dict_all = self._state_dict_all_comb(from_node, state_dict)
        inputs_all = []
        for state in state_dict_all:
            dir_nm_el_from = "_".join([
                "{}:{}".format(i, j) for i, j in list(state.items())])
            if is_node(from_node):
                out_from = self._reading_ci_output(
                    node=from_node, dir_nm_el=dir_nm_el_from, out_nm=from_socket)
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


    def _reading_ci_output(self, dir_nm_el, out_nm, node=None):
        """used for current interfaces: checking if the output exists and returns the path if it does"""
        if not node:
            node = self
        result_pklfile = os.path.join(os.getcwd(), node.workingdir, dir_nm_el,
                                      node.interface.nn.name, "result_{}.pklz".format(
                                          node.interface.nn.name))
        if os.path.exists(result_pklfile) and os.stat(result_pklfile).st_size > 0:
            out = getattr(loadpkl(result_pklfile).outputs, out_nm)
            if out:
                return out
        return False


    def _directory_name_state_surv(self, state_dict):
        """eliminating all inputs from state dictionary that are not in
        the splitter;
        creating a directory name
        """
        # should I be using self.state._splitter_rpn_comb?
        state_surv_dict = dict((key, val) for (key, val) in state_dict.items()
                               if key in self.state._splitter_rpn)
        dir_nm_el = "_".join(["{}:{}".format(i, j)
                              for i, j in list(state_surv_dict.items())])
        if not self.splitter:
            dir_nm_el = ""
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
        val_dict_l = self._state_str_to_dict(val_l)
        return val_dict_l


    def _state_str_to_dict(self, values_list):
        """taking a list of tuples (state, value)
        and converting state from string to dictionary.
        string has format "FirstInputName:Value_SecondInputName:Value"
        """
        values_dict_list = []
        for val_el in values_list:
            val_dict = {}
            for val_str in val_el[0].split("_"):
                if val_str:
                    key, val = val_str.split(":")
                    try:
                        val = float(val)
                        if val.is_integer():
                            val = int(val)
                    except Exception:
                        pass
                    val_dict[key] = val
            values_dict_list.append((val_dict, val_el[1]))
        return values_dict_list


    def _combined_output(self, key_out, state_dict, output_el):
        dir_nm_comb = "_".join(["{}:{}".format(i, j)
                                for i, j in list(state_dict.items())
                                if i not in self.state.comb_inp_to_remove])
        if dir_nm_comb in self._output[key_out].keys():
            self._output[key_out][dir_nm_comb].append(output_el)
        else:
            self._output[key_out][dir_nm_comb] = [output_el]


    def _reading_results_one_output(self, key_out):
        """reading results for one specifc output name"""
        if not self.splitter:
            if type(self.output[key_out]) is tuple:
                result = self.output[key_out]
            elif type(self.output[key_out]) is dict:
                val_l = self._state_dict_to_list(self.output[key_out])
                if len(val_l) == 1:
                    result = val_l[0]
                # this is used for wf (can be no splitter but multiple values from node splitter)
                else:
                    result = val_l
        elif (self.combiner and not self.state._splitter_rpn_comb):
            val_l = self._state_dict_to_list(self.output[key_out])
            result = val_l[0]
        elif self.splitter:
            val_l = self._state_dict_to_list(self._output[key_out])
            result = val_l
        return result


class Node(NodeBase):
    def __init__(self, name, interface, inputs=None, splitter=None, workingdir=None,
                 other_splitters=None, output_names=None, write_state=True,
                 combiner=None, *args, **kwargs):
        super(Node, self).__init__(name=name, splitter=splitter, inputs=inputs,
                                   other_splitters=other_splitters, write_state=write_state,
                                   combiner=combiner, *args, **kwargs)

        # working directory for node, will be change if node is a part of a wf
        self.workingdir = workingdir
        self.interface = interface

        # list of  interf_key_out
        self.output_names = output_names
        if not self.output_names:
            self.output_names = []


    def run_interface_el(self, i, ind):
        """ running interface one element generated from node_state."""
        logger.debug("Run interface el, name={}, i={}, ind={}".format(self.name, i, ind))
        state_dict, inputs_dict = self.get_input_el(ind)
        if not self.write_state:
            state_dict = self.state.state_ind(ind)
        dir_nm_el, state_surv_dict = self._directory_name_state_surv(state_dict)
        print("Run interface el, dict={}".format(state_surv_dict))
        logger.debug("Run interface el, name={}, inputs_dict={}, state_dict={}".format(
            self.name, inputs_dict, state_surv_dict))
        res = self.interface.run(
            inputs=inputs_dict,
            base_dir=os.path.join(os.getcwd(), self.workingdir),
            dir_nm_el=dir_nm_el)
        return res


    def get_output(self):
        """collecting all outputs and updating self._output"""
        for key_out in self.output_names:
            self._output[key_out] = {}
            for (i, ind) in enumerate(itertools.product(*self.state.all_elements)):
                if self.write_state:
                    state_dict = self.state.state_values(ind)
                else:
                    state_dict = self.state.state_ind(ind)
                dir_nm_el, state_surv_dict = self._directory_name_state_surv(state_dict)
                if self.splitter:
                    output_el = self._reading_ci_output(dir_nm_el, out_nm=key_out)
                    if not self.combiner: # only splitter
                        self._output[key_out][dir_nm_el] = output_el
                    else:
                        self._combined_output(key_out, state_dict, output_el)
                else:
                    self._output[key_out] = \
                        (state_surv_dict, self._reading_ci_output(dir_nm_el, out_nm=key_out))
        return self._output


    def _check_all_results(self):
        """checking if all files that should be created are present"""
        for ind in itertools.product(*self.state.all_elements):
            if self.write_state:
                state_dict = self.state.state_values(ind)
            else:
                state_dict = self.state.state_ind(ind)
            dir_nm_el, _ = self._directory_name_state_surv(state_dict)
            for key_out in self.output_names:
                if not self._reading_ci_output(dir_nm_el, key_out):
                    return False
        self._is_complete = True
        return True


    def _reading_results(self):
        """ collecting all results for all output names"""
        for key_out in self.output_names:
            self._result[key_out] = self._reading_results_one_output(key_out)


class Workflow(NodeBase):
    def __init__(self, name, inputs=None, wf_output_names=None, splitter=None,
                 nodes=None, workingdir=None, write_state=True, *args, **kwargs):
        super(Workflow, self).__init__(name=name, splitter=splitter, inputs=inputs,
                                       write_state=write_state, *args, **kwargs)

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
        # dj not sure what was the motivation, wf_klasses gives an empty list
        #mro = self.__class__.mro()
        #wf_klasses = mro[:mro.index(Workflow)][::-1]
        #items = {}
        #for klass in wf_klasses:
        #    items.update(klass.__dict__)
        #for name, runnable in items.items():
        #    if name in ('__module__', '__doc__'):
        #        continue

        #    self.add(name, value)

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
        # not sure, if I should collecto output of all nodes or only the ones that are used in wf.output
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
                        for (i, ind) in enumerate(itertools.product(*self.state.all_elements)):
                            if self.write_state:
                                wf_inputs_dict = self.state.state_values(ind)
                            else:
                                wf_inputs_dict = self.state.state_ind(ind)
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


    # TODO: this should be probably using add method
    def add_nodes(self, nodes):
        """adding nodes without defining connections
            most likely it will be removed at the end
        """
        self.graph.add_nodes_from(nodes)
        for nn in nodes:
            self._nodes.append(nn)
            #self._inputs.update(nn.inputs)
            self.connected_var[nn] = {}
            self._node_names[nn.name] = nn
            #TODO i think this is not needed at the end
            # when we have a combiner in a previous node, we have to pass the final splitter
            # if nn.combiner: pdb.set_trace()
            #     self._node_splitters[nn.name] = nn.state.splitter_comb
            # else:
            #     self._node_splitters[nn.name] = nn.splitter
            # nn.other_splitters = self._node_splitters


    # TODO: workingir shouldn't have None
    def add(self, runnable, name=None, workingdir=None, inputs=None, input_names=None,
            output_names=None, splitter=None, combiner=None, write_state=True, **kwargs):
        if is_function(runnable):
            if not output_names:
                output_names = ["out"]
            if input_names is None:
                raise Exception("you need to specify input_names")
            if not name:
                raise Exception("you have to specify name for the node")
            nipype1_interf = Function(function=runnable, input_names=input_names,
                                      output_names=output_names)
            interface = aux.CurrentInterface(interface=nipype1_interf, name="addtwo")
            if not workingdir:
                workingdir = name
            node = Node(interface=interface, workingdir=workingdir, name=name,
                        inputs=inputs, splitter=splitter, other_splitters=self._node_splitters,
                        combiner=combiner, output_names=output_names,
                        write_state=write_state)
        elif is_current_interface(runnable):
            if not name:
                raise Exception("you have to specify name for the node")
            if not workingdir:
                workingdir = name
            node = Node(interface=runnable, workingdir=workingdir, name=name,
                        inputs=inputs, splitter=splitter, other_splitters=self._node_splitters,
                        combiner=combiner, output_names=output_names,
                        write_state=write_state)
        elif is_nipype_interface(runnable):
            ci = aux.CurrentInterface(interface=runnable, name=name)
            if not name:
                raise Exception("you have to specify name for the node")
            if not workingdir:
                workingdir = name
            node = Node(interface=ci, workingdir=workingdir, name=name, inputs=inputs,
                        splitter=splitter, other_splitters=self._node_splitters,
                        combiner=combiner, output_names=output_names,
                        write_state=write_state)
        elif is_node(runnable):
            node = runnable
            node.other_splitters = self._node_splitters
        elif is_workflow(runnable):
            node = runnable
        else:
            raise ValueError("Unknown workflow element: {!r}".format(runnable))
        self.add_nodes([node])
        self._last_added = node
        # connecting inputs to other nodes outputs
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
        #pdb.set_trace()
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
            if self.write_state:
                if not st_inputs: st_inputs=wf_inputs
                dir_nm_el, _ = self._directory_name_state_surv(st_inputs)
            else:
                # wf_inputs_ind is already ok, doesn't need st_inputs_ind
                  dir_nm_el, _ = self._directory_name_state_surv(wf_inputs_ind)
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
                    #TODO: implement inner splitter
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


def is_current_interface(obj):
    return type(obj) is aux.CurrentInterface


def is_nipype_interface(obj):
    return hasattr(obj, "_run_interface")


def is_node(obj):
    return type(obj) is Node


def is_workflow(obj):
    return type(obj) is Workflow
