from __future__ import (print_function, division, unicode_literals,
                        absolute_import)
from builtins import str, bytes, open

import os
import networkx as nx
import itertools
import numpy as np

from . import state
from . import auxiliary as aux
from . import submitter as sub

from nipype.utils.filemanip import loadpkl

import logging
logger = logging.getLogger('nipype.workflow')

import pdb

# dj ??: should I use EngineBase?
class NewBase(object):
    def __init__(self, name, mapper=None, inputs=None, other_mappers=None, mem_gb=None,
                 cache_location=None, print_val=True, *args, **kwargs):
        self.name = name
        #dj TODO: I should think what is needed in the __init__ (I redefine some of rhe attributes anyway)
        if inputs:
            # adding name of the node to the input name
            self._inputs = dict(("{}.{}".format(self.name, key), value) for (key, value) in inputs.items())
            self._inputs = dict((key, np.array(val)) if type(val) is list else (key, val)
                                for (key, val) in self._inputs.items())
            self._state_inputs = self._inputs.copy()
        else:
            self._inputs = {}
            self._state_inputs = {}
        if mapper:
            # adding name of the node to the input name within the mapper
            mapper = aux.change_mapper(mapper, self.name)
        self._mapper = mapper
        # information about other nodes' mappers from workflow (in case the mapper from previous node is used)
        self._other_mappers = other_mappers
        # create state (takes care of mapper, connects inputs with axes, so we can ask for specifc element)
        self._state = state.State(mapper=self._mapper, node_name=self.name, other_mappers=self._other_mappers)
        self._output = {}
        self._result = {}
        # flag that says if the node/wf is ready to run (has all input)
        self.ready2run = True
        # needed outputs from other nodes if the node part of a wf
        self.needed_outputs = []
        # flag that says if node finished all jobs
        self._is_complete = False
        # flag that says if value of state input should be printed in output and directories (otherwise indices)
        self.print_val = print_val

        # TODO: don't use it yet
        self.mem_gb = mem_gb
        self.cache_location = cache_location


    # TBD
    def join(self, field):
        pass

    @property
    def state(self):
        return self._state

    @property
    def mapper(self):
        return self._mapper

    @mapper.setter
    def mapper(self, mapper):
        self._mapper = mapper
        # updating state
        self._state = state.State(mapper=self._mapper, node_name=self.name, other_mappers=self._other_mappers)

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
        self._state.prepare_state_input(state_inputs=self.state_inputs)


    def map(self, mapper, inputs=None):
        if self._mapper:
            raise Exception("mapper is already set")
        else:
            self._mapper = aux.change_mapper(mapper, self.name)

        if inputs:
            inputs = dict(("{}.{}".format(self.name, key), value) for (key, value) in inputs.items())
            inputs = dict((key, np.array(val)) if type(val) is list else (key, val)
                          for (key, val) in inputs.items())
            self._inputs.update(inputs)
            self._state_inputs.update(inputs)
        if mapper:
            # updating state if we have a new mapper
            self._state = state.State(mapper=self._mapper, node_name=self.name, other_mappers=self._other_mappers)


    def join(self, field, node=None):
        # TBD
        pass


    def checking_input_el(self, ind):
        """checking if all inputs are available (for specific state element)"""
        try:
            self.get_input_el(ind)
            return True
        except: #TODO specify
            return False


    # dj: this is not used for a single node
    def get_input_el(self, ind):
        """collecting all inputs required to run the node (for specific state element)"""
        state_dict = self.state.state_values(ind)
        inputs_dict = {k: state_dict[k] for k in self._inputs.keys()}
        if not self.print_val:
            state_dict = self.state.state_ind(ind)
        # reading extra inputs that come from previous nodes
        for (from_node, from_socket, to_socket) in self.needed_outputs:
            dir_nm_el_from = "_".join(["{}:{}".format(i, j) for i, j in list(state_dict.items())
                                       if i in list(from_node._state_inputs.keys())])
            if not from_node.mapper:
                dir_nm_el_from = ""

            if is_node(from_node) and is_current_interface(from_node.interface):
                file_from = self._reading_ci_output(node=from_node, dir_nm_el=dir_nm_el_from, out_nm=from_socket)
                if file_from and os.path.exists(file_from):
                    inputs_dict["{}.{}".format(self.name, to_socket)] = file_from
                else:
                    raise Exception("{} doesnt exist".format(file_from))
            else: # assuming here that I want to read the file (will not be used with the current interfaces)
                file_from = os.path.join(from_node.workingdir, dir_nm_el_from, from_socket+".txt")
                with open(file_from) as f:
                    content = f.readline()
                    try:
                        inputs_dict["{}.{}".format(self.name, to_socket)] = eval(content)
                    except NameError:
                        inputs_dict["{}.{}".format(self.name, to_socket)] = content

        return state_dict, inputs_dict

    def _reading_ci_output(self, dir_nm_el, out_nm, node=None):
        """used for current interfaces: checking if the output exists and returns the path if it does"""
        if not node:
            node = self
        result_pklfile = os.path.join(os.getcwd(), node.workingdir, dir_nm_el,
                                      node.interface.nn.name, "result_{}.pklz".format(node.interface.nn.name))
        if os.path.exists(result_pklfile):
            out_file = getattr(loadpkl(result_pklfile).outputs, out_nm)
            if os.path.exists(out_file):
                return out_file
            else:
                return False
        else:
            return False


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


    def _dict_tuple2list(self, container):
        if type(container) is dict:
            val_l = [val for (_, val) in container.items()]
        elif type(container) is tuple:
            val_l = [container]
        else:
            raise Exception("{} has to be dict or tuple".format(container))
        return val_l


class NewNode(NewBase):
    def __init__(self, name, interface, inputs=None, mapper=None, join_by=None,
                 workingdir=None, other_mappers=None, mem_gb=None, cache_location=None,
                 output_names=None, print_val=True, *args, **kwargs):
        super(NewNode, self).__init__(name=name, mapper=mapper, inputs=inputs,
                                      other_mappers=other_mappers, mem_gb=mem_gb,
                                      cache_location=cache_location, print_val=print_val,
                                      *args, **kwargs)

        # working directory for node, will be change if node is a part of a wf
        self.workingdir = workingdir
        self.interface = interface

        if is_function_interface(self.interface):
            # adding node name to the interface's name mapping
            self.interface.input_map = dict((key, "{}.{}".format(self.name, value))
                                             for (key, value) in self.interface.input_map.items())
            # list of output names taken from interface output name
            self.output_names = self.interface._output_nm
        elif is_current_interface(self.interface):
            # list of  interf_key_out
            self.output_names = output_names
        if not self.output_names:
            self.output_names = []



    # dj: not sure if I need it
    # def __deepcopy__(self, memo): # memo is a dict of id's to copies
    #     id_self = id(self)        # memoization avoids unnecesary recursion
    #     _copy = memo.get(id_self)
    #     if _copy is None:
    #         # changing names of inputs and input_map, so it doesnt contain node.name
    #         inputs_copy = dict((key[len(self.name)+1:], deepcopy(value))
    #                            for (key, value) in self.inputs.items())
    #         interface_copy = deepcopy(self.interface)
    #         interface_copy.input_map = dict((key, val[len(self.name)+1:])
    #                                         for (key, val) in interface_copy.input_map.items())
    #         _copy = type(self)(
    #             name=deepcopy(self.name), interface=interface_copy,
    #             inputs=inputs_copy, mapper=deepcopy(self.mapper),
    #             base_dir=deepcopy(self.nodedir), other_mappers=deepcopy(self._other_mappers))
    #         memo[id_self] = _copy
    #     return _copy


    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs.update(inputs)


    def run_interface_el(self, i, ind):
        """ running interface one element generated from node_state."""
        logger.debug("Run interface el, name={}, i={}, ind={}".format(self.name, i, ind))
        state_dict, inputs_dict = self.get_input_el(ind)
        if not self.print_val:
            state_dict = self.state.state_ind(ind)
        dir_nm_el = "_".join(["{}:{}".format(i, j) for i, j in list(state_dict.items())])
        print("Run interface el, dict={}".format(state_dict))
        logger.debug("Run interface el, name={}, inputs_dict={}, state_dict={}".format(
                                                            self.name, inputs_dict, state_dict))
        if is_function_interface(self.interface):
            res = self.interface.run(inputs_dict)
            output = self.interface.output
            print("Run fun interface el, output={}".format(output))
            logger.debug("Run fun interface el, output={}".format(output))
            self._writting_results_tmp(state_dict, dir_nm_el, output)
        elif is_current_interface(self.interface):
            if not self.mapper:
                dir_nm_el = ""
            res = self.interface.run(inputs=inputs_dict, base_dir=os.path.join(os.getcwd(), self.workingdir),
                                     dir_nm_el=dir_nm_el)

        # TODO when join
        #if self._joinByKey:
        #    dir_join = "join_" + "_".join(["{}.{}".format(i, j) for i, j in list(state_dict.items()) if i not in self._joinByKey])
        #elif self._join:
        #    dir_join = "join_"
        #if self._joinByKey or self._join:
        #    os.makedirs(os.path.join(self.nodedir, dir_join), exist_ok=True)
        #    dir_nm_el = os.path.join(dir_join, dir_nm_el)
        return res


    def _writting_results_tmp(self, state_dict, dir_nm_el, output):
        """temporary method to write the results in the files (this is usually part of a interface)"""
        if not self.mapper:
            dir_nm_el = ''
        os.makedirs(os.path.join(self.workingdir, dir_nm_el), exist_ok=True)
        for key_out, val_out in output.items():
            with open(os.path.join(self.workingdir, dir_nm_el, key_out+".txt"), "w") as fout:
                fout.write(str(val_out))


    def get_output(self):
        """collecting all outputs and updating self._output"""
        for key_out in self.output_names:
            self._output[key_out] = {}
            for (i, ind) in enumerate(itertools.product(*self.state.all_elements)):
                if self.print_val:
                    state_dict = self.state.state_values(ind)
                else:
                    state_dict = self.state.state_ind(ind)
                dir_nm_el = "_".join(["{}:{}".format(i, j) for i, j in list(state_dict.items())])
                if self.mapper:
                    if is_function_interface(self.interface):
                        output = os.path.join(self.workingdir, dir_nm_el, key_out + ".txt")
                        if self.interface.out_read:
                            with open(output) as fout:
                                content = fout.readline()
                                try:
                                    output = eval(content)
                                except NameError:
                                    output = content
                        self._output[key_out][dir_nm_el] = (state_dict, output)
                    elif is_current_interface(self.interface):
                        self._output[key_out][dir_nm_el] = \
                            (state_dict, (state_dict, self._reading_ci_output(dir_nm_el=dir_nm_el, out_nm=key_out)))
                else:
                    if is_function_interface(self.interface):
                        output = os.path.join(self.workingdir, key_out + ".txt")
                        if self.interface.out_read:
                            with open(output) as fout:
                                try:
                                    output = eval(fout.readline())
                                except NewWorkflow:
                                    output = fout.readline()
                        self._output[key_out] = (state_dict, output)
                    elif is_current_interface(self.interface):
                        self._output[key_out] = \
                            (state_dict, self._reading_ci_output(dir_nm_el="", out_nm=key_out))
        return self._output


    # dj: version without join
    def _check_all_results(self):
        """checking if all files that should be created are present"""
        for ind in itertools.product(*self.state.all_elements):
            if self.print_val:
                state_dict = self.state.state_values(ind)
            else:
                state_dict = self.state.state_ind(ind)
            dir_nm_el = "_".join(["{}:{}".format(i, j) for i, j in list(state_dict.items())])
            if not self.mapper:
                dir_nm_el = ""

            for key_out in self.output_names:
                if is_function_interface(self.interface):
                    if not os.path.isfile(os.path.join(self.workingdir, dir_nm_el, key_out+".txt")):
                        return False
                elif is_current_interface(self.interface):
                    if not self._reading_ci_output(dir_nm_el, key_out):
                        return False
        self._is_complete = True
        return True


    def _reading_results(self):
        """temporary: reading results from output files (that is now just txt)
            should be probably just reading output for self.output_names
        """
        for key_out in self.output_names:
            self._result[key_out] = []
            #pdb.set_trace()
            if self._state_inputs:
                val_l = self._dict_tuple2list(self._output[key_out])
                for (st_dict, filename) in val_l:
                    with open(filename) as fout:
                        self._result[key_out].append((st_dict, eval(fout.readline())))
            else:
                # st_dict should be {}
                # not sure if this is used (not tested)
                (st_dict, filename) = self._output[key_out][None]
                with open(filename) as fout:
                    self._result[key_out].append(({}, eval(fout.readline())))

    # dj: removing temp. from NewNode class
    # def run(self, plugin="serial"):
    #     """preparing the node to run and run the interface"""
    #     self.prepare_state_input()
    #     submitter = sub.SubmitterNode(plugin, node=self)
    #     submitter.run_node()
    #     submitter.close()
    #     self.collecting_output()


class NewWorkflow(NewBase):
    def __init__(self, name, inputs=None, wf_output_names=None, mapper=None, #join_by=None,
                 nodes=None, workingdir=None, mem_gb=None, cache_location=None, print_val=True, *args, **kwargs):
        super(NewWorkflow, self).__init__(name=name, mapper=mapper, inputs=inputs, mem_gb=mem_gb,
                                          cache_location=cache_location, print_val=print_val, *args, **kwargs)

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
        # key: name of a node, value: mapper of the node
        self._node_mappers = {}
        # dj: not sure if this should be different than base_dir
        self.workingdir = os.path.join(os.getcwd(), workingdir)
        # list of (nodename, output name in the name, output name in wf) or (nodename, output name in the name)
        # dj: using different name than for node, since this one it is defined by a user
        self.wf_output_names = wf_output_names

        # nodes that are created when the workflow has mapper (key: node name, value: list of nodes)
        self.inner_nodes = {}
        # in case of inner workflow this points to the main/parent workflow
        self.parent_wf = None
        # dj not sure what was the motivation, wf_klasses gives an empty list
        #mro = self.__class__.mro()
        #wf_klasses = mro[:mro.index(NewWorkflow)][::-1]
        #items = {}
        #for klass in wf_klasses:
        #    items.update(klass.__dict__)
        #for name, runnable in items.items():
        #    if name in ('__module__', '__doc__'):
        #        continue

        #    self.add(name, value)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs.update(dict(("{}.{}".format(self.name, key), value) for (key, value) in inputs.items()))


    @property
    def nodes(self):
        return self._nodes

    @property
    def graph_sorted(self):
        # TODO: should I always update the graph?
        return list(nx.topological_sort(self.graph))


    def map_node(self, mapper, node=None, inputs=None):
        """this is setting a mapper to the wf's nodes (not to the wf)"""
        if not node:
            node = self._last_added
        if node.mapper:
            raise Exception("Cannot assign two mappings to the same input")
        node.map(mapper=mapper, inputs=inputs)
        self._node_mappers[node.name] = node.mapper


    def get_output(self):
        # not sure, if I should collecto output of all nodes or only the ones that are used in wf.output
        self.node_outputs = {}
        for nn in self.graph:
            if self.mapper:
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
                    if self.mapper:
                        self._output[out_wf_nm] = {}
                        for (i, ind) in enumerate(itertools.product(*self.state.all_elements)):
                            if self.print_val:
                                wf_inputs_dict = self.state.state_values(ind)
                                dir_nm_el = "_".join(["{}:{}".format(i, j) for i, j in list(wf_inputs_dict.items())])
                            else:
                                wf_ind_dict = self.state.state_ind(ind)
                                dir_nm_el = "_".join(["{}:{}".format(i, j) for i, j in list(wf_ind_dict.items())])
                            self._output[out_wf_nm][dir_nm_el] = self.node_outputs[node_nm][i][out_nd_nm]
                    else:
                        self._output[out_wf_nm] = self.node_outputs[node_nm][out_nd_nm]
                else:
                    raise Exception("the key {} is already used in workflow.result".format(out_wf_nm))
        return self._output


    # dj: version without join
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


    # TODO: should try to merge with the function from Node
    def _reading_results(self):
        """reading all results of the workflow
           using temporary Node._reading_results that reads txt files
        """
        if self.wf_output_names:
            for out in self.wf_output_names:
                key_out = out[2] if len(out)==3 else out[1]
                self._result[key_out] = []
                if self.mapper:
                    for (i, ind) in enumerate(itertools.product(*self.state.all_elements)):
                        if self.print_val:
                            wf_inputs_dict = self.state.state_values(ind)
                        else:
                            wf_inputs_dict = self.state.state_ind(ind)
                        dir_nm_el = "_".join(["{}:{}".format(i, j) for i, j in list(wf_inputs_dict.items())])
                        res_l= []
                        val_l = self._dict_tuple2list(self.output[key_out][dir_nm_el])
                        for val in val_l:
                            with open(val[1]) as fout:
                                logger.debug('Reading Results: file={}, st_dict={}'.format(val[1], val[0]))
                                res_l.append((val[0], eval(fout.readline())))
                        self._result[key_out].append((wf_inputs_dict, res_l))
                else:
                    val_l = self._dict_tuple2list(self.output[key_out])
                    for val in val_l:
                        #TODO: I think that val shouldn't be dict here...
                        # TMP solution
                        if type(val) is dict:
                            val = [v for k,v in val.items()][0]
                        with open(val[1]) as fout:
                            logger.debug('Reading Results: file={}, st_dict={}'.format(val[1], val[0]))
                            self._result[key_out].append((val[0], eval(fout.readline())))


    def add_nodes(self, nodes):
        """adding nodes without defining connections"""
        self.graph.add_nodes_from(nodes)
        for nn in nodes:
            self._nodes.append(nn)
            #self._inputs.update(nn.inputs)
            self.connected_var[nn] = {}
            self._node_names[nn.name] = nn
            self._node_mappers[nn.name] = nn.mapper


    def add(self, runnable, name=None, workingdir=None, inputs=None, output_names=None, mapper=None,
            mem_gb=None, print_val=True, out_read=False, **kwargs):
        if is_function(runnable):
            if not output_names:
                output_names = ["out"]
            interface = aux.FunctionInterface(function=runnable, output_nm=output_names, out_read=out_read)
            if not name:
                raise Exception("you have to specify name for the node")
            if not workingdir:
                workingdir = name
            node = NewNode(interface=interface, workingdir=workingdir, name=name, inputs=inputs, mapper=mapper,
                           other_mappers=self._node_mappers, mem_gb=mem_gb, print_val=print_val)
        elif is_function_interface(runnable) or is_current_interface(runnable):
            if not name:
                raise Exception("you have to specify name for the node")
            if not workingdir:
                workingdir = name
            node = NewNode(interface=runnable, workingdir=workingdir, name=name, inputs=inputs, mapper=mapper,
                           other_mappers=self._node_mappers, mem_gb_node=mem_gb, output_names=output_names,
                           print_val=print_val)
        elif is_nipype_interface(runnable):
            ci = aux.CurrentInterface(interface=runnable, name=name)
            if not name:
                raise Exception("you have to specify name for the node")
            if not workingdir:
                workingdir = name
            node = NewNode(interface=ci, workingdir=workingdir, name=name, inputs=inputs, mapper=mapper,
                           other_mappers=self._node_mappers, mem_gb_node=mem_gb, output_names=output_names,
                           print_val=print_val)
        elif is_node(runnable):
            node = runnable
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
            except(ValueError):
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


    def preparing(self, wf_inputs=None, wf_inputs_ind=None):
        """preparing nodes which are connected: setting the final mapper and state_inputs"""
        #pdb.set_trace()
        for node_nm, inp_wf, inp_nd in self.needed_inp_wf:
            node = self._node_names[node_nm]
            if "{}.{}".format(self.name, inp_wf) in wf_inputs:
                node.state_inputs.update({"{}.{}".format(node_nm, inp_nd): wf_inputs["{}.{}".format(self.name, inp_wf)]})
                node.inputs.update({"{}.{}".format(node_nm, inp_nd): wf_inputs["{}.{}".format(self.name, inp_wf)]})
            else:
                raise Exception("{}.{} not in the workflow inputs".format(self.name, inp_wf))
        for nn in self.graph_sorted:
            if self.print_val:
                dir_nm_el = "_".join(["{}:{}".format(i, j) for i, j in list(wf_inputs.items())])
            else:
                dir_nm_el = "_".join(["{}:{}".format(i, j) for i, j in list(wf_inputs_ind.items())])
            if not self.mapper:
                dir_nm_el = ""
            nn.workingdir = os.path.join(self.workingdir, dir_nm_el, nn.name)
            nn._is_complete = False # helps when mp is used
            try:
                for inp, (out_node, out_var) in self.connected_var[nn].items():
                    nn.ready2run = False #it has some history (doesnt have to be in the loop)
                    nn.state_inputs.update(out_node.state_inputs)
                    nn.needed_outputs.append((out_node, out_var, inp))
                    #if there is no mapper provided, i'm assuming that mapper is taken from the previous node
                    if (not nn.mapper or nn.mapper == out_node.mapper) and out_node.mapper:
                        nn.mapper = out_node.mapper
                    else:
                        pass
                    #TODO: implement inner mapper
            except(KeyError):
                # tmp: we don't care about nn that are not in self.connected_var
                pass

            nn.prepare_state_input()

    # removing temp. from NewWorkflow
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

def is_function_interface(obj):
    return type(obj) is aux.FunctionInterface

def is_current_interface(obj):
    return type(obj) is aux.CurrentInterface

def is_nipype_interface(obj):
    return hasattr(obj, "_run_interface")

def is_node(obj):
    return type(obj) is NewNode

def is_workflow(obj):
    return type(obj) is NewWorkflow
