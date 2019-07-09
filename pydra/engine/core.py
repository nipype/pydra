"""Basic compute graph elements"""
import abc
import dataclasses as dc
import json
import logging
import os
from pathlib import Path
import typing as ty
import pickle as pk
from copy import deepcopy

import cloudpickle as cp
from filelock import SoftFileLock
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
    save_result,
    ensure_list,
    record_error,
)
from .graph import Graph
from .audit import Audit
from ..utils.messenger import AuditFlag

logger = logging.getLogger("pydra")

develop = True


class TaskBase:
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
        cache_locations=None,
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
            field.name
            for field in dc.fields(klass)
            if field.name not in ["_func", "_graph"]
        ]
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
        self.audit = Audit(
            AuditFlag=AuditFlag,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            develop=develop,
        )
        self.cache_dir = cache_dir
        self.cache_locations = cache_locations
        self._checksum = None

        # dictionary of results from tasks
        self.results_dict = {}
        self.plugin = None

    def __repr__(self):
        return self.name

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

    @property
    def checksum(self):
        """calculating checksum if self._checksum is None only
            (avoiding recomputing during the execution)
        """
        if self._checksum is None:
            if self.state is None:
                self._checksum = create_checksum(self.__class__.__name__, self.inputs)
            else:
                return {sidx: res[1] for (sidx, res) in self.results_dict.items()}
        return self._checksum

    def set_state(self, splitter, combiner=None):
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
            self._cache_dir = Path(location).resolve()
            self._cache_dir.mkdir(parents=False, exist_ok=True)
        else:
            self._cache_dir = mkdtemp()
            self._cache_dir = Path(self._cache_dir)

    @property
    def cache_locations(self):
        return self._cache_locations + ensure_list(self._cache_dir)

    @cache_locations.setter
    def cache_locations(self, locations):
        if locations is not None:
            self._cache_locations = [Path(loc) for loc in ensure_list(locations)]
        else:
            self._cache_locations = []

    @property
    def output_dir(self):
        if self.state:
            return {
                sidx: self._cache_dir / checksum
                for (sidx, checksum) in self.checksum.items()
            }
        else:
            return self._cache_dir / self.checksum

    def __call__(self, submitter=None, plugin=None, **kwargs):
        if submitter and plugin:
            raise Exception("you can specify submitter OR plugin, not both")
        elif submitter:
            submitter(self)
        elif plugin:
            from .submitter import Submitter

            with Submitter(plugin=plugin) as sub:
                sub(self)
        else:
            return self._run(**kwargs)

    def _run(self, **kwargs):
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
        with SoftFileLock(lockfile):
            # Let only one equivalent process run
            # Eagerly retrieve cached
            if self.results_dict:  # should be skipped if run called without submitter
                result = self.result()
                if result is not None:
                    return result
            odir = self.output_dir
            if not self.can_resume and odir.exists():
                shutil.rmtree(odir)
            cwd = os.getcwd()
            odir.mkdir(parents=False, exist_ok=True if self.can_resume else False)
            self.audit.start_audit(odir)
            result = Result(output=None, runtime=None, errored=False)
            try:
                self.audit.monitor()
                self._run_task()
                result.output = self._collect_outputs()
            except Exception as e:
                record_error(self.output_dir, e)
                result.errored = True
                raise
            finally:
                self.audit.finalize_audit(result)
                save_result(odir, result)
                with open(odir / "_node.pklz", "wb") as fp:
                    cp.dump(self, fp)
                os.chdir(cwd)
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
            self.split(splitter=None)
            # a task can have a combiner without a splitter
            # if is connected to one with a splitter;
            # self.fut_combiner will be used later as a combiner
            self.fut_combiner = combiner
            return self
        elif self.state.combiner:
            raise Exception("combiner has been already set")
        else:  # self.state and not self.state.combiner
            self.combiner = combiner
            self.set_state(splitter=self.state.splitter, combiner=self.combiner)
            return self

    def get_input_el(self, ind):
        """collecting all inputs required to run the node (for specific state element)"""
        if ind is not None:
            # TODO: doesnt work properly for more cmplicated wf
            state_dict = self.state.states_val[ind]
            input_ind = self.state.inputs_ind[ind]
            inputs_dict = {}
            for inp in set(self.input_names):
                if f"{self.name}.{inp}" in input_ind:
                    inputs_dict[inp] = getattr(self.inputs, inp)[
                        input_ind[f"{self.name}.{inp}"]
                    ]
                else:
                    inputs_dict[inp] = getattr(self.inputs, inp)
            return state_dict, inputs_dict
        else:
            inputs_dict = {inp: getattr(self.inputs, inp) for inp in self.input_names}
            return None, inputs_dict

    def to_job(self, ind):
        """ running interface one element generated from node_state."""
        # logger.debug("Run interface el, name={}, ind={}".format(self.name, ind))
        el = deepcopy(self)
        el.state = None
        # dj might be needed
        # el._checksum = None
        _, inputs_dict = self.get_input_el(ind)
        el.inputs = dc.replace(el.inputs, **inputs_dict)
        return el

    # checking if all outputs are saved
    @property
    def done(self):
        if self.state:
            # TODO: only check for needed state result
            if self.result() and all(self.result()):
                return True
        else:
            if self.result():
                return True
        return False

    def _combined_output(self):
        combined_results = []
        for (gr, ind_l) in self.state.final_groups_mapping.items():
            combined_results.append([])
            for ind in ind_l:
                result = load_result(self.results_dict[ind][1], self.cache_locations)
                if result is None:
                    return None
                combined_results[gr].append(result)
        return combined_results

    def result(self, state_index=None):
        """
        :param state_index:
        :return:
        """
        # TODO: check if result is available in load_result and
        # return a future if not
        if self.state:
            if state_index is None:
                # if state_index=None, collecting all results
                if self.state.combiner:
                    return self._combined_output()
                else:
                    results = []
                    for (ii, val) in enumerate(self.state.states_val):
                        result = load_result(
                            self.results_dict[ii][1], self.cache_locations
                        )
                        if result is None:
                            return None
                        results.append(result)
                    return results
            else:  # state_index is not None
                if self.state.combiner:
                    return self._combined_output()[state_index]
                result = load_result(
                    self.results_dict[state_index][1], self.cache_locations
                )
                return result
        else:
            if state_index is not None:
                raise ValueError("Task does not have a state")
            if self.results_dict:
                checksum = self.results_dict[None][1]
            else:
                checksum = self.checksum
            result = load_result(checksum, self.cache_locations)
            return result


class Workflow(TaskBase):
    def __init__(
        self,
        name,
        input_spec: ty.Union[ty.List[ty.Text], BaseSpec, None] = None,
        output_spec: ty.Optional[BaseSpec] = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers=None,
        messenger_args=None,
        cache_dir=None,
        cache_locations=None,
        **kwargs,
    ):
        if input_spec:
            if isinstance(input_spec, BaseSpec):
                self.input_spec = input_spec
            else:
                self.input_spec = SpecInfo(
                    name="Inputs",
                    fields=[(name, ty.Any) for name in input_spec]
                    + [("_graph", ty.Any)],
                    bases=(BaseSpec,),
                )
        if output_spec is None:
            output_spec = SpecInfo(
                name="Output", fields=[("out", ty.Any)], bases=(BaseSpec,)
            )
        self.output_spec = output_spec

        super(Workflow, self).__init__(
            name=name,
            inputs=kwargs,
            cache_dir=cache_dir,
            cache_locations=cache_locations,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
        )

        self.graph = Graph()
        self.name2obj = {}

        # store output connections
        self._connections = None
        self.node_names = []

    def __getattr__(self, name):
        if name == "lzin":
            return LazyField(self, "input")
        if name == "lzout":
            return super().__getattr__(name)
        if name in self.name2obj:
            return self.name2obj[name]
        return self.__getattribute__(name)

    @property
    def done_all_tasks(self):
        """ checking if all tasks from the graph are done;
            (it doesn't mean that results of the wf are available,
            this can be checked with self.done)
        """
        for task in self.graph:
            if not task.done:
                return False
        return True

    @property
    def nodes(self):
        return self.name2obj.values()

    @property
    def graph_sorted(self):
        return self.graph.sorted_nodes

    def add(self, task):
        if not is_task(task):
            raise ValueError("Unknown workflow element: {!r}".format(task))
        self.graph.add_nodes(task)
        self.name2obj[task.name] = task
        self._last_added = task
        other_states = {}
        for field in dc.fields(task.inputs):
            val = getattr(task.inputs, field.name)
            if isinstance(val, LazyField):
                # adding an edge to the graph if task id expecting output from a different task
                if val.name != self.name:
                    self.graph.add_edges((getattr(self, val.name), task))
                    logger.debug("Connecting %s to %s", val.name, task.name)
                if val.name in self.node_names and getattr(self, val.name).state:
                    # adding a state from the previous task to other_states
                    other_states[val.name] = (getattr(self, val.name).state, field.name)
        # if task has connections state has to be recalculated
        if other_states:
            if hasattr(task, "fut_combiner"):
                task.state = state.State(
                    task.name, other_states=other_states, combiner=task.fut_combiner
                )
            else:
                task.state = state.State(task.name, other_states=other_states)
        self.node_names.append(task.name)
        logger.debug(f"Added {task}")
        self.inputs._graph = self.graph_sorted
        return self

    async def _run(self, submitter=None, **kwargs):
        self.inputs = dc.replace(self.inputs, **kwargs)
        checksum = self.checksum
        lockfile = self.cache_dir / (checksum + ".lock")
        # Eagerly retrieve cached
        result = self.result()
        if result is not None:
            return result

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
        with SoftFileLock(lockfile):
            # # Let only one equivalent process run
            odir = self.output_dir
            if not self.can_resume and odir.exists():
                shutil.rmtree(odir)
            cwd = os.getcwd()
            odir.mkdir(parents=False, exist_ok=True if self.can_resume else False)
            self.audit.start_audit(odir=odir)
            result = Result(output=None, runtime=None, errored=False)
            try:
                self.audit.monitor()
                await self._run_task(submitter)
                result.output = self._collect_outputs()
            except Exception as e:
                record_error(self.output_dir, e)
                result.errored = True
                raise
            finally:
                self.audit.finalize_audit(result=result)
                save_result(odir, result)
                with open(odir / "_node.pklz", "wb") as fp:
                    cp.dump(self, fp)
                os.chdir(cwd)
            return result

    async def _run_task(self, submitter):
        if not submitter:
            raise Exception("Submitter should already be set.")
        # at this point Workflow is stateless so this should be fine
        await submitter.submit(self, return_task=True)

    def set_output(self, connections):
        self._connections = connections
        fields = [(name, ty.Any) for name, _ in connections]
        self.output_spec = SpecInfo(name="Output", fields=fields, bases=(BaseSpec,))
        logger.info("Added %s to %s", self.output_spec, self)

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


def is_runnable(graph, obj):
    """Check if a task within a graph is runnable"""
    if graph.predecessors(obj):
        for pred in graph.predecessors(obj):
            if not pred.done:
                return False
    return True
