"""Basic compute graph elements"""
import abc
from collections import OrderedDict
import dataclasses as dc
import itertools
import json
import logging
import networkx as nx
import os
import pdb
from pathlib import Path
import typing as ty
import pickle as pk
from copy import deepcopy

import cloudpickle as cp
from filelock import FileLock
import shutil
from tempfile import mkdtemp

from . import state
from . import auxiliary as aux
from .specs import File, BaseSpec, RuntimeSpec, Result, SpecInfo, LazyField
from .helpers import (
    make_klass,
    create_checksum,
    print_help,
    load_result,
    gather_runtime_info,
    save_result,
    ensure_list,
    get_inputs,
)
from ..utils.messenger import send_message, make_message, gen_uuid, now, AuditFlag

logger = logging.getLogger("pydra")

develop = True


class NodeBase:
    _api_version: str = "0.0.1"  # Should generally not be touched by subclasses
    _version: str  # Version of tool being wrapped
    _task_version: ty.Optional[
        str
    ] = None  # Task writers encouraged to define and increment when implementation changes sufficiently
    _input_sets = None  # Dictionaries of predefined input settings

    audit_flags: AuditFlag = AuditFlag.NONE  # What to audit. See audit flags for details

    _can_resume = False  # Does the task allow resuming from previous state
    _redirect_x = False  # Whether an X session should be created/directed

    _runtime_requirements = RuntimeSpec()
    _runtime_hints = None

    _cache_dir = None  # Working directory in which to operate
    _references = None  # List of references for a task

    # dj: do we need it??
    input_spec = BaseSpec
    output_spec = BaseSpec

    # TODO: write state should be removed
    def __init__(
        self,
        name,
        inputs: ty.Union[ty.Text, File, ty.Dict, None] = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers=None,
        messenger_args=None,
        cache_dir=None,
    ):
        """A base structure for nodes in the computational graph (i.e. both
        ``Node`` and ``Workflow``).

        Parameters
        ----------

        name : str
            Unique name of this node
        inputs : dictionary (input name, input value or list of values)
            States this node's input names
        """
        self.name = name
        if not self.input_spec:
            raise Exception("No input_spec in class: %s" % self.__class__.__name__)
        klass = make_klass(self.input_spec)
        self.inputs = klass(
            **{
                f.name: (None if f.default is dc.MISSING else f.default)
                for f in dc.fields(klass)
            }
        )
        self.input_names = [
            field.name for field in dc.fields(klass) if field.name not in ["_func"]
        ]

        self._needed_outputs = []
        self.state = None
        self._output = {}
        self._result = {}
        # flag that says if node finished all jobs
        self._done = False

        if self._input_sets is None:
            self._input_sets = {}
        if inputs:
            if isinstance(inputs, dict):
                inputs = {k: v for k, v in inputs.items() if k in self.input_names}
            elif Path(inputs).is_file():
                inputs = json.loads(Path(inputs).read_text())
            elif isinstance(inputs, str):
                if self._input_sets is None or inputs not in self._input_sets:
                    raise ValueError("Unknown input set {!r}".format(inputs))
                inputs = self._input_sets[inputs]
            self.inputs = dc.replace(self.inputs, **inputs)
            self.state_inputs = inputs

        self.audit_flags = audit_flags
        self.messengers = ensure_list(messengers)
        self.messenger_args = messenger_args
        self.cache_dir = cache_dir

        # dictionary of results from tasks
        self.results_dict = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["input_spec"] = pk.dumps(state["input_spec"])
        state["output_spec"] = pk.dumps(state["output_spec"])
        state["inputs"] = dc.asdict(state["inputs"])
        return state

    def __setstate__(self, state):
        state["input_spec"] = pk.loads(state["input_spec"])
        state["output_spec"] = pk.loads(state["output_spec"])
        state["inputs"] = make_klass(state["input_spec"])(**state["inputs"])
        self.__dict__.update(state)

    def __getattr__(self, name):
        if name == "lzout":  # lazy output
            return LazyField(self, "output")
        return self.__getattribute__(name)

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
            raise KeyError("Key {} already saved. Use force=True to override.")
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
            left_out = [
                state.name for state in incoming_states if state.name not in rpn
            ]

        if splitter is not None:
            self.state = state.State(
                name=self.name, splitter=splitter, combiner=combiner
            )
        else:
            self.state = None
        return self.state

    @property
    def output_names(self):
        return [f.name for f in dc.fields(make_klass(self.output_spec))]

    def audit(self, message, flags=None):
        if develop:
            with open(
                Path(os.path.dirname(__file__)) / ".." / "schema/context.jsonld", "rt"
            ) as fp:
                context = json.load(fp)
        else:
            context = {
                "@context": "https://raw.githubusercontent.com/satra/pydra/enh/task/pydra/schema/context.jsonld"
            }
        if self.audit_flags & flags:
            if self.messenger_args:
                send_message(
                    make_message(message, context=context),
                    messengers=self.messengers,
                    **self.messenger_args
                )
            else:
                send_message(
                    make_message(message, context=context), messengers=self.messengers
                )

    @property
    def can_resume(self):
        """Task can reuse partial results after interruption
        """
        return self._can_resume

    @abc.abstractmethod
    def _run_task(self):
        pass

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, location):
        if location is not None:
            self._cache_dir = Path(location)
            self._cache_dir.mkdir(parents=False, exist_ok=True)
        else:
            self._cache_dir = mkdtemp()
            self._cache_dir = Path(self._cache_dir)

    @property
    def output_dir(self):
        return self._cache_dir / self.checksum

    def audit_check(self, flag):
        return self.audit_flags & flag

    def __call__(self, cache_locations=None, **kwargs):
        return self.run(cache_locations=cache_locations, **kwargs)

    def run(self, cache_locations=None, **kwargs):
        self.inputs = dc.replace(self.inputs, **kwargs)
        checksum = self.checksum
        lockfile = self.cache_dir / (checksum + ".lock")
        """
        Concurrent execution scenarios

        1. prior cache exists -> return result
        2. other process running -> wait
           a. finishes (with or without exception) -> return result
           b. gets killed -> restart
        3. no cache or other process -> start
        4. two or more concurrent new processes get to start
        """
        # TODO add signal handler for processes killed after lock acquisition
        with FileLock(lockfile):
            # Let only one equivalent process run
            # Eagerly retrieve cached
            if self.results_dict:  # if run method called without submitter
                result = self.result(cache_locations=cache_locations)
                if result is not None:
                    return result
            odir = self.output_dir
            if not self.can_resume and odir.exists():
                shutil.rmtree(odir)
            cwd = os.getcwd()
            odir.mkdir(parents=False, exist_ok=True if self.can_resume else False)
            # start recording provenance, but don't send till directory is created
            # in case message directory is inside task output directory
            if self.audit_check(AuditFlag.PROV):
                aid = "uid:{}".format(gen_uuid())
                start_message = {"@id": aid, "@type": "task", "startedAtTime": now()}
            os.chdir(odir)
            if self.audit_check(AuditFlag.PROV):
                self.audit(start_message, AuditFlag.PROV)
                # audit inputs
            # check_runtime(self._runtime_requirements)
            # isolate inputs if files
            # cwd = os.getcwd()
            if self.audit_check(AuditFlag.RESOURCE):
                from ..utils.profiler import ResourceMonitor

                resource_monitor = ResourceMonitor(os.getpid(), logdir=odir)
            result = Result(output=None, runtime=None)
            try:
                if self.audit_check(AuditFlag.RESOURCE):
                    resource_monitor.start()
                    if self.audit_check(AuditFlag.PROV):
                        mid = "uid:{}".format(gen_uuid())
                        self.audit(
                            {
                                "@id": mid,
                                "@type": "monitor",
                                "startedAtTime": now(),
                                "wasStartedBy": aid,
                            },
                            AuditFlag.PROV,
                        )
                self._run_task()
                result.output = self._collect_outputs()
            except Exception as e:
                print(e)
                # record_error(self, e)
                raise
            finally:
                if self.audit_check(AuditFlag.RESOURCE):
                    resource_monitor.stop()
                    result.runtime = gather_runtime_info(resource_monitor.fname)
                    if self.audit_check(AuditFlag.PROV):
                        self.audit(
                            {"@id": mid, "endedAtTime": now(), "wasEndedBy": aid},
                            AuditFlag.PROV,
                        )
                        # audit resources/runtime information
                        eid = "uid:{}".format(gen_uuid())
                        entity = dc.asdict(result.runtime)
                        entity.update(
                            **{
                                "@id": eid,
                                "@type": "runtime",
                                "prov:wasGeneratedBy": aid,
                            }
                        )
                        self.audit(entity, AuditFlag.PROV)
                        self.audit(
                            {
                                "@type": "prov:Generation",
                                "entity_generated": eid,
                                "hadActivity": mid,
                            },
                            AuditFlag.PROV,
                        )
                save_result(odir, result)
                with open(odir / "_node.pklz", "wb") as fp:
                    cp.dump(self, fp)
                os.chdir(cwd)
                if self.audit_check(AuditFlag.PROV):
                    # audit outputs
                    self.audit({"@id": aid, "endedAtTime": now()}, AuditFlag.PROV)
            return result

    # TODO: Decide if the following two functions should be separated
    @abc.abstractmethod
    def _list_outputs(self):
        pass

    def _collect_outputs(self):
        run_output = ensure_list(self._list_outputs())
        output_klass = make_klass(self.output_spec)
        output = output_klass(**{f.name: None for f in dc.fields(output_klass)})
        return dc.replace(output, **dict(zip(self.output_names, run_output)))

    # TODO: should change state!
    def split(self, splitter, **kwargs):
        if kwargs:
            self.inputs = dc.replace(self.inputs, **kwargs)
            # dj:??, check if I need it
            self.state_inputs = kwargs
        splitter = aux.change_splitter(splitter, self.name)
        if self.state:
            raise Exception("splitter has been already set")
        else:
            self.set_state(splitter)
        return self

    def combine(self, combiner):
        if not self.state:
            raise Exception("splitter has to be set first")
        elif self.state.combiner:
            raise Exception("combiner has been already set")
        self.combiner = combiner
        self.set_state(splitter=self.state.splitter, combiner=self.combiner)
        return self

    def checking_input_el(self, ind):
        """checking if all inputs are available (for specific state element)"""
        try:
            self.get_input_el(ind)
            return True
        except:  # TODO specify
            return False

    def get_input_el(self, ind):
        """collecting all inputs required to run the node (for specific state element)"""
        if ind is not None:
            # TODO: check if the current version requires both state_dict and inputs_dict
            state_dict = self.state.states_val[ind]
            inputs_dict = {
                "{}.{}".format(self.name, k): state_dict["{}.{}".format(self.name, k)]
                for k in self.input_names
            }

            # reading extra inputs that come from previous nodes
            for (from_node, from_socket, to_socket) in self.needed_outputs:
                # TODO update to new version: if previous has state, it would have to be combined
                # if the current node has no state
                if from_node.state.combiner:
                    pdb.set_trace()
                    inputs_dict[
                        "{}.{}".format(self.name, to_socket)
                    ] = self._get_input_comb(from_node, from_socket, state_dict)
                else:
                    dir_nm_el_from, _ = from_node._directory_name_state_surv(state_dict)
                    # TODO: do I need this if, what if this is wf?
                    if is_node(from_node):
                        out_from = getattr(
                            from_node.results_dict[dir_nm_el_from].output, from_socket
                        )
                        if out_from:
                            inputs_dict["{}.{}".format(self.name, to_socket)] = out_from
                        else:
                            raise Exception(
                                "output from {} doesnt exist".format(from_node)
                            )
            return state_dict, inputs_dict

        else:
            inputs_dict = {
                "{}.{}".format(self.name, inp): getattr(self.inputs, inp)
                for inp in self.input_names
            }
            # TODO: adding parts from self.needed_outputs
            return None, inputs_dict

    # TODO: update
    def _get_input_comb(self, from_node, from_socket, state_dict):
        """collecting all outputs from previous node that has combiner"""
        state_dict_all = self._state_dict_all_comb(from_node, state_dict)
        inputs_all = []
        for state in state_dict_all:
            dir_nm_el_from = "_".join(
                ["{}:{}".format(i, j) for i, j in list(state.items())]
            )
            if is_node(from_node):
                out_from = getattr(
                    from_node.results_dict[dir_nm_el_from].output, from_socket
                )
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
            state_dict_all.append(
                self._state_dict_el_for_comb(ind, state_dict, axis_for_input)
            )
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
        # adding values from input that are not used in the splitter
        for input in set(state_inputs) - set(axis_for_input):
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
        state_surv_dict = dict(
            (key, val)
            for (key, val) in state_dict.items()
            if key in self.state.splitter_rpn
        )
        dir_nm_el = "_".join(
            ["{}:{}".format(i, j) for i, j in list(state_surv_dict.items())]
        )
        return dir_nm_el, state_surv_dict

    def to_job(self, ind):
        """ running interface one element generated from node_state."""
        logger.debug("Run interface el, name={}, ind={}".format(self.name, ind))
        el = deepcopy(self)
        el.state = None
        _, inputs_dict = self.get_input_el(ind)
        interf_inputs = dict((k.split(".")[1], v) for k, v in inputs_dict.items())
        el.inputs = dc.replace(el.inputs, **interf_inputs)
        return el

    # checking if all outputs are saved
    @property
    def done(self):
        if self.results_dict:
            return all([future.done() for _, (future, _) in self.results_dict.items()])

    def _state_dict_to_list(self, container):
        """creating a list of tuples from dictionary and changing key (state) from str to dict"""
        if type(container) is dict:
            val_l = list(container.items())
        else:
            raise Exception("{} has to be dict".format(container))
        val_dict_l = [(self.state.states_val[i[0]], i[1]) for i in val_l]
        return val_dict_l

    def _combined_output(self, cache_locations=None):
        combined_results = []
        for (gr, ind_l) in self.state.final_groups_mapping.items():
            combined_results.append([])
            for ind in ind_l:
                result = load_result(
                    self.results_dict[ind][1],
                    ensure_list(cache_locations) + ensure_list(self._cache_dir),
                )
                combined_results[gr].append(result)
        return combined_results

    def result(self, state_index=None, cache_locations=None):
        """

        :param state_index:
        :param cache_locations:
        :return:
        """
        # TODO: check if result is available in load_result and
        # return a future if not
        if self.state:
            if state_index is not None:
                if self.state.combiner:
                    return self._combined_output()[state_index]
                result = load_result(
                    self.results_dict[state_index][1],
                    ensure_list(cache_locations) + ensure_list(self._cache_dir),
                )
                return result
            if self.state.combiner:
                return self._combined_output()
            # if state_index=None, collecting all results
            results = []
            for (ii, val) in enumerate(self.state.states_val):
                result = load_result(
                    self.results_dict[ii][1],
                    ensure_list(cache_locations) + ensure_list(self._cache_dir),
                )
                results.append(result)
            return results
        else:
            if state_index is not None:
                raise ValueError("Task does not have a state")
            if self.results_dict:
                checksum = self.results_dict[None][1]
            else:
                checksum = self.checksum
            result = load_result(
                checksum, ensure_list(cache_locations) + ensure_list(self._cache_dir)
            )
            return result


class Workflow(NodeBase):
    def __init__(
        self,
        name,
        inputs: ty.Union[ty.Text, File, ty.Dict, None] = None,
        input_spec: ty.Union[ty.List[ty.Text], BaseSpec, None] = None,
        output_spec: ty.Optional[BaseSpec] = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers=None,
        messenger_args=None,
        cache_dir=None,
    ):
        if input_spec:
            if isinstance(input_spec, BaseSpec):
                self.input_spec = input_spec
            else:
                self.input_spec = SpecInfo(
                    name="Inputs",
                    fields=[(name, ty.Any) for name in input_spec],
                    bases=(BaseSpec,),
                )
        if output_spec is None:
            output_spec = SpecInfo(
                name="Output", fields=[("out", ty.Any)], bases=(BaseSpec,)
            )
        self.output_spec = output_spec

        super(Workflow, self).__init__(
            name=name,
            inputs=inputs,
            cache_dir=cache_dir,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
        )

        self.graph = nx.DiGraph()

        self.name2obj = {}

        # store output connections
        self._connections = None

        # TODO: this may be used in run task
        self.submitter = None

    def __getattr__(self, name):
        if name == "lzin":  # lazy output
            return LazyField(self, "input")
        if name in self.name2obj:
            return self.name2obj[name]
        return self.__getattribute__(name)

    @property
    def nodes(self):
        return self._nodes

    @property
    def graph_sorted(self):
        return list(nx.topological_sort(self.graph))

    def add(self, task):
        if not is_task(task):
            raise ValueError("Unknown workflow element: {!r}".format(task))
        self.graph.add_nodes_from([task])
        self.name2obj[task.name] = task
        self._last_added = task
        for field in dc.fields(task.inputs):
            val = getattr(task.inputs, field.name)
            if isinstance(val, LazyField):
                if val.name != self.name:
                    self.graph.add_edge(
                        getattr(self, val.name),
                        task,
                        from_field=val.field,
                        to_field=field.name,
                    )
        return self

    def _run_task(self):
        for task in self.graph_sorted:
            # TODO: this next line will not work with split and combine
            task.inputs.retrieve_values(self)
            result = task.run()

    def set_output(self, connections):
        self._connections = connections
        fields = [(name, ty.Any) for name, _ in connections]
        self.output_spec = SpecInfo(name="Output", fields=fields, bases=(BaseSpec,))

    def _list_outputs(self):
        output = []
        for name, val in self._connections:
            if not isinstance(val, LazyField):
                raise ValueError("all connections must be lazy")
            output.append(val.get_value(self))
        return output


# TODO: task has also call
def is_function(obj):
    return hasattr(obj, "__call__")


def is_task(obj):
    return hasattr(obj, "_run_task")


def is_workflow(obj):
    return isinstance(obj, Workflow)
