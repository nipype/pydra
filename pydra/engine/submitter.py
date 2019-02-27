import os, time, pdb
from copy import deepcopy

from .workers import MpWorker, SerialWorker, DaskWorker, ConcurrentFuturesWorker
from .node import NodeBase

import logging
logger = logging.getLogger('nipype.workflow')


class Submitter(object):
    # TODO: runnable in init or run
    def __init__(self, plugin, runnable):
        self.plugin = plugin
        self.node_line = []
        self._to_finish = []  # used only for wf
        if self.plugin == "mp":
            self.worker = MpWorker()
        elif self.plugin == "serial":
            self.worker = SerialWorker()
        elif self.plugin == "dask":
            self.worker = DaskWorker()
        elif self.plugin == "cf":
            self.worker = ConcurrentFuturesWorker()
        else:
            raise Exception("plugin {} not available".format(self.plugin))

        if isinstance(runnable, NodeBase):  # a node/task
            self.node = runnable
        elif hasattr(runnable, "graph"):  # a workflow
            self.workflow = runnable
        else:
            raise Exception("runnable has to be a Node or Workflow")

    def run(self):
        """main running method, checks if submitter id for Node or Workflow"""
        if hasattr(self, "node"):
            self.run_node()
        elif hasattr(self, "workflow"):
            self.run_workflow()

    def run_node(self):
        """the main method to run a Node"""
        if self.node.state:
            self.node.state.prepare_states(self.node.inputs)
        self._submit_node(self.node)
        while not self.node.is_complete:
            logger.debug("Submitter, in while, to_finish: {}".format(self.node))
            time.sleep(3)
        self.node.get_output()


    def _submit_node(self, node):
        """submitting nodes's interface for all states"""
        if node.state:
            for ii, ind in enumerate(node.state.states_val):
                # this is run only for a single node or the first node in a wf
                self._submit_node_el(node, ii)
        else:
            self._submit_node_el(node, ind=None)

    def _submit_node_el(self, node, ind):
        """submitting node's interface for one element of states"""
        logger.debug("SUBMIT WORKER, node: {}, ind: {}".format(node, ind))
        res = self.worker.run_el(node.run_interface_el, ind)
        # saving results in a node dictionary
        node.results_dict[res[0]] = res[1]


    def run_workflow(self, workflow=None, ready=True):
        """the main function to run Workflow"""
        if not workflow:
            workflow = self.workflow
        workflow.prepare_state_input()

        # TODO: should I have inner_nodes for all workflow (to avoid if wf.splitter)??
        if workflow.splitter:
            for key in workflow._node_names.keys():
                workflow.inner_nodes[key] = []
            for ind in workflow.state.index_generator:
                new_workflow = deepcopy(workflow)
                new_workflow.parent_wf = workflow
                # adding all nodes to the parent workflow
                for (i_n, node) in enumerate(new_workflow.graph_sorted):
                    workflow.inner_nodes[node.name].append(node)
                if ready:
                    self._run_workflow_el(new_workflow, ind)
                else:
                    self.node_line.append((new_workflow, ind))
        else:
            if ready:
                workflow.preparing(wf_inputs=workflow.inputs)
                self._run_workflow_nd(workflow=workflow)
            else:
                self.node_line.append((workflow, ()))

        # this parts submits nodes that are waiting to be run
        # it should stop when nothing is waiting
        while self._nodes_check():
            logger.debug("Submitter, in while, node_line: {}".format(self.node_line))
            time.sleep(3)

        # this part simply waiting for all "last nodes" to finish
        while self._output_check():
            logger.debug("Submitter, in while, to_finish: {}".format(self._to_finish))
            time.sleep(3)

        # calling only for the main wf (other wf will be called inside the function)
        if workflow is self.workflow:
            workflow.get_output()

    def _run_workflow_el(self, workflow, ind, collect_inp=False):
        """running one internal workflow (if workflow has a splitter)"""
        # TODO: can I simplify and remove collect inp? where should it be?
        if collect_inp:
            st_inputs, wf_inputs = workflow.get_input_el(ind)
        else:
            wf_inputs = workflow.state.state_values(ind)
            st_inputs = wf_inputs
        workflow.preparing(wf_inputs=wf_inputs, st_inputs=st_inputs)
        self._run_workflow_nd(workflow=workflow)


    def _run_workflow_nd(self, workflow):
        """iterating over all nodes from a workflow and submitting them or adding to the node_line"""
        for (i_n, node) in enumerate(workflow.graph_sorted):
            node.prepare_state_input()
            self._to_finish.append(node)
            # submitting all the nodes who are self sufficient (self.workflow.graph is already sorted)
            if node.ready2run:
                if hasattr(node, 'interface'):
                    self._submit_node(node)
                else:  # it's workflow
                    self.run_workflow(workflow=node)
            # if its not, its been added to a line
            else:
                break
            # in case there is no element in the graph that goes to the break
            # i want to be sure that not calculating the last node again in the next for loop
            if i_n == len(workflow.graph_sorted) - 1:
                i_n += 1

        # all nodes that are not self sufficient (not ready to run) will go to the line
        # iterating over all elements
        for nn in list(workflow.graph_sorted)[i_n:]:
            if hasattr(nn, 'interface'):
                for ind in nn.state.index_generator:
                    self._to_finish.append(nn)
                    self.node_line.append((nn, ind))
            else:  #wf
                self.run_workflow(workflow=nn, ready=False)

    def _nodes_check(self):
        """checking which nodes-states are ready to run and running the ones that are ready"""
        _to_remove = []
        for (to_node, ind) in self.node_line:
            if hasattr(to_node, 'interface'):
                print("_NODES_CHECK INPUT", to_node.name, to_node.checking_input_el(ind))
                if to_node.checking_input_el(ind):
                    self._submit_node_el(to_node, ind)
                    _to_remove.append((to_node, ind))
            else:  #wf
                if to_node.checking_input_el(ind):
                    self._run_workflow_el(workflow=to_node, ind=ind, collect_inp=True)
                    _to_remove.append((to_node, ind))
                else:
                    pass

        for rn in _to_remove:
            self.node_line.remove(rn)
        return self.node_line

    # this I believe can be done for entire node
    def _output_check(self):
        """"checking if all nodes are done"""
        _to_remove = []
        for node in self._to_finish:
            print("_output check node", node, node.name, node.is_complete)
            if node.is_complete:
                _to_remove.append(node)
        for rn in _to_remove:
            self._to_finish.remove(rn)
        return self._to_finish

    def close(self):
        self.worker.close()
