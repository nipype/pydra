"""Basic compute graph elements"""
import abc
import dataclasses as dc
import json
import logging
import os
from pathlib import Path
import typing as ty
from copy import deepcopy

import cloudpickle as cp
from filelock import SoftFileLock
import shutil
from tempfile import mkdtemp

from . import state
from . import helpers_state as hlpst
from .specs import File, BaseSpec, RuntimeSpec, Result, SpecInfo, LazyField, TaskHook
from .helpers import (
    make_klass,
    create_checksum,
    print_help,
    load_result,
    save,
    ensure_list,
    record_error,
    hash_function,
    output_from_inputfields,
    output_names_from_inputfields,
)
from .graph import DiGraph
from .audit import Audit
from ..utils.messenger import AuditFlag

logger = logging.getLogger("pydra")

develop = False


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

    def __init__(
        self,
        name: str,
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
        # todo should be used to input_check in spec??
        self.inputs = klass(
            **{
                f.name: (None if f.default is dc.MISSING else f.default)
                for f in dc.fields(klass)
            }
        )
        self.input_names = [
            field.name
            for field in dc.fields(klass)
            if field.name not in ["_func", "_graph_checksums"]
        ]
        # dictionary to save the connections with lazy fields
        self.inp_lf = {}
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
            self.inputs.check_metadata()
            self.state_inputs = inputs

        self.audit = Audit(
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            develop=develop,
        )
        self.cache_dir = cache_dir
        self.cache_locations = cache_locations
        self.allow_cache_override = True
        self._checksum = None

        self.plugin = None
        self.hooks = TaskHook()

    def __repr__(self):
        return self.name

    def __getstate__(self):
        state = self.__dict__.copy()
        state["input_spec"] = cp.dumps(state["input_spec"])
        state["output_spec"] = cp.dumps(state["output_spec"])
        state["inputs"] = dc.asdict(state["inputs"])
        return state

    def __setstate__(self, state):
        state["input_spec"] = cp.loads(state["input_spec"])
        state["output_spec"] = cp.loads(state["output_spec"])
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
        """calculating checksum
        """
        # if checksum is called before run the _graph_checksums is not ready
        if is_workflow(self) and self.inputs._graph_checksums is None:
            self.inputs._graph_checksums = [nd.checksum for nd in self.graph_sorted]

        input_hash = self.inputs.hash
        if self.state is None:
            self._checksum = create_checksum(self.__class__.__name__, input_hash)
        else:
            # including splitter in the hash
            splitter_hash = hash_function(self.state.splitter)
            self._checksum = create_checksum(
                self.__class__.__name__, hash_function([input_hash, splitter_hash])
            )
        return self._checksum

    def checksum_states(self, state_index=None):
        """ calculating checksum for the specific state or all of the states
            replace lists in the inputs fields with a specific values for states
            can be used only for tasks with a state
        """
        if state_index is not None:
            if self.state is None:
                raise Exception("can't use state_index if no splitter is used")
            inputs_copy = deepcopy(self.inputs)
            for key, ind in self.state.inputs_ind[state_index].items():
                setattr(
                    inputs_copy,
                    key.split(".")[1],
                    getattr(inputs_copy, key.split(".")[1])[ind],
                )
            input_hash = inputs_copy.hash
            checksum_ind = create_checksum(self.__class__.__name__, input_hash)
            return checksum_ind
        else:
            checksum_list = []
            for ind in range(len(self.state.inputs_ind)):
                checksum_list.append(self.checksum_states(state_index=ind))
            return checksum_list

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
        output_spec_names = [f.name for f in dc.fields(make_klass(self.output_spec))]
        from_input_spec_names = output_names_from_inputfields(self.inputs)
        return output_spec_names + from_input_spec_names

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
            self._cache_dir = Path(self._cache_dir).resolve()

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
            return [self._cache_dir / checksum for checksum in self.checksum_states()]
        else:
            return self._cache_dir / self.checksum

    def __call__(self, submitter=None, plugin=None, **kwargs):
        from .submitter import Submitter

        if submitter and plugin:
            raise Exception("Specify submitter OR plugin, not both")
        plugin = plugin or self.plugin
        if plugin:
            submitter = Submitter(plugin=plugin)
        elif self.state:
            submitter = Submitter()

        if submitter:
            with submitter as sub:
                res = sub(self)
        else:
            if is_workflow(self):
                raise NotImplementedError(
                    "TODO: linear workflow execution - assign submitter or plugin for now"
                )
            res = self._run(**kwargs)
        return res

    def _run(self, **kwargs):
        self.inputs = dc.replace(self.inputs, **kwargs)
        self.inputs.check_fields_input_spec()
        checksum = self.checksum
        lockfile = self.cache_dir / (checksum + ".lock")
        # Eagerly retrieve cached
        """
        Concurrent execution scenarios

        1. prior cache exists -> return result
        2. other process running -> wait
           a. finishes (with or without exception) -> return result
           b. gets killed -> restart
        3. no cache or other process -> start
        4. two or more concurrent new processes get to start
        """
        self.hooks.pre_run(self)
        # TODO add signal handler for processes killed after lock acquisition
        with SoftFileLock(lockfile):
            result = self.result()
            if result is not None:
                return result
            # Let only one equivalent process run
            odir = self.output_dir
            if not self.can_resume and odir.exists():
                shutil.rmtree(odir)
            cwd = os.getcwd()
            odir.mkdir(parents=False, exist_ok=True if self.can_resume else False)
            self.inputs.copyfile_input(self.output_dir)
            self.inputs.template_update()
            self.audit.start_audit(odir)
            result = Result(output=None, runtime=None, errored=False)
            self.hooks.pre_run_task(self)
            try:
                self.audit.monitor()
                self._run_task()
                result.output = self._collect_outputs()
            except Exception as e:
                record_error(self.output_dir, e)
                result.errored = True
                raise
            finally:
                self.hooks.post_run_task(self, result)
                self.audit.finalize_audit(result)
                save(odir, result=result, task=self)
                os.chdir(cwd)
        self.hooks.post_run(self, result)
        return result

    def _collect_outputs(self):
        run_output = self.output_
        self.output_spec = output_from_inputfields(self.output_spec, self.inputs)
        output_klass = make_klass(self.output_spec)
        output = output_klass(**{f.name: None for f in dc.fields(output_klass)})
        other_output = output.collect_additional_outputs(
            self.input_spec, self.inputs, self.output_dir
        )
        return dc.replace(output, **run_output, **other_output)

    def split(self, splitter, overwrite=False, **kwargs):
        splitter = hlpst.add_name_splitter(splitter, self.name)
        # if user want to update the splitter, overwrite has to be True
        if self.state and not overwrite and self.state.splitter != splitter:
            raise Exception(
                "splitter has been already set, "
                "if you want to overwrite it - use overwrite=True"
            )
        if kwargs:
            self.inputs = dc.replace(self.inputs, **kwargs)
            self.state_inputs = kwargs
        if not self.state or splitter != self.state.splitter:
            self.set_state(splitter)
        return self

    def combine(self, combiner, overwrite=False):
        if not isinstance(combiner, (str, list)):
            raise Exception("combiner has to be a string or a list")
        combiner = hlpst.add_name_combiner(ensure_list(combiner), self.name)
        if (
            self.state
            and self.state.combiner
            and combiner != self.state.combiner
            and not overwrite
        ):
            raise Exception(
                "combiner has been already set, "
                "if you want to overwrite it - use overwrite=True"
            )
        if not self.state:
            self.split(splitter=None)
            # a task can have a combiner without a splitter
            # if is connected to one with a splitter;
            # self.fut_combiner will be used later as a combiner
            self.fut_combiner = combiner
            return self
        else:  # self.state and not self.state.combiner
            self.combiner = combiner
            self.set_state(splitter=self.state.splitter, combiner=self.combiner)
            return self

    def get_input_el(self, ind):
        """collecting all inputs required to run the node (for specific state element)"""
        if ind is not None:
            # TODO: doesnt work properly for more cmplicated wf (check if still an issue)
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
            # todo it never gets here
            breakpoint()
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
        for (gr, ind_l) in self.state.final_combined_ind_mapping.items():
            combined_results.append([])
            for ind in ind_l:
                result = load_result(self.checksum_states(ind), self.cache_locations)
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
                    for checksum in self.checksum_states():
                        result = load_result(checksum, self.cache_locations)
                        if result is None:
                            return None
                        results.append(result)
                    return results
            else:  # state_index is not None
                if self.state.combiner:
                    return self._combined_output()[state_index]
                result = load_result(
                    self.checksum_states(state_index), self.cache_locations
                )
                return result
        else:
            if state_index is not None:
                raise ValueError("Task does not have a state")
            checksum = self.checksum
            result = load_result(checksum, self.cache_locations)
            return result

    def _reset(self):
        """resetting the connections between inputs and LazyFields"""
        for field in dc.fields(self.inputs):
            if field.name in self.inp_lf:
                setattr(self.inputs, field.name, self.inp_lf[field.name])
        if is_workflow(self):
            for task in self.graph.nodes:
                task._reset()


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
                    fields=[("_graph_checksums", ty.Any)]
                    + [
                        (
                            nm,
                            ty.Any,
                            dc.field(
                                metadata={
                                    "help_string": f"{nm} input from {name} workflow"
                                }
                            ),
                        )
                        for nm in input_spec
                    ],
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

        self.graph = DiGraph()
        self.name2obj = {}

        # store output connections
        self._connections = None

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
        for task in self.graph.nodes:
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
        """adding a task to the workflow"""
        if not is_task(task):
            raise ValueError("Unknown workflow element: {!r}".format(task))
        self.graph.add_nodes(task)
        self.name2obj[task.name] = task
        self._last_added = task
        logger.debug(f"Added {task}")
        return self

    def create_connections(self, task):
        """creating connections between tasks"""
        other_states = {}
        for field in dc.fields(task.inputs):
            val = getattr(task.inputs, field.name)
            if isinstance(val, LazyField):
                # saving all connections with LazyFields
                task.inp_lf[field.name] = val
                # adding an edge to the graph if task id expecting output from a different task
                if val.name != self.name:
                    # checking if the connection is already in the graph
                    if (getattr(self, val.name), task) in self.graph.edges:
                        continue
                    self.graph.add_edges((getattr(self, val.name), task))
                    logger.debug("Connecting %s to %s", val.name, task.name)

                    if getattr(self, val.name).state:
                        # adding a state from the previous task to other_states
                        other_states[val.name] = (
                            getattr(self, val.name).state,
                            field.name,
                        )
        # if task has connections state has to be recalculated
        if other_states:
            if task.state:
                old_splitter = task.state.splitter
            else:
                old_splitter = None
            if hasattr(task, "fut_combiner"):
                task.state = state.State(
                    task.name,
                    splitter=old_splitter,
                    other_states=other_states,
                    combiner=task.fut_combiner,
                )
            else:
                task.state = state.State(
                    task.name, splitter=old_splitter, other_states=other_states
                )

    async def _run(self, submitter=None, **kwargs):
        # self.inputs = dc.replace(self.inputs, **kwargs) don't need it?
        checksum = self.checksum
        lockfile = self.cache_dir / (checksum + ".lock")
        # Eagerly retrieve cached
        result = self.result()
        if result is not None:
            return result
        # creating connections that were defined after adding tasks to the wf
        for task in self.graph.nodes:
            self.create_connections(task)
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
        self.hooks.pre_run(self)
        with SoftFileLock(lockfile):
            # # Let only one equivalent process run
            odir = self.output_dir
            if not self.can_resume and odir.exists():
                shutil.rmtree(odir)
            cwd = os.getcwd()
            odir.mkdir(parents=False, exist_ok=True if self.can_resume else False)
            self.audit.start_audit(odir=odir)
            result = Result(output=None, runtime=None, errored=False)
            self.hooks.pre_run_task(self)
            try:
                self.audit.monitor()
                await self._run_task(submitter)
                result.output = self._collect_outputs()
            except Exception as e:
                record_error(self.output_dir, e)
                result.errored = True
                raise
            finally:
                self.hooks.post_run_task(self, result)
                self.audit.finalize_audit(result=result)
                save(odir, result=result, task=self)
                os.chdir(cwd)
        self.hooks.post_run(self, result)
        return result

    async def _run_task(self, submitter):
        if not submitter:
            raise Exception("Submitter should already be set.")
        # at this point Workflow is stateless so this should be fine
        await submitter._run_workflow(self)

    def set_output(self, connections):
        if isinstance(connections, tuple) and len(connections) == 2:
            self._connections = [connections]
        elif isinstance(connections, list) and all(
            [len(el) == 2 for el in connections]
        ):
            self._connections = connections
        elif isinstance(connections, dict):
            self._connections = list(connections.items())
        else:
            raise Exception(
                "Connections can be a 2-elements tuple, a list of these tuples, or dictionary"
            )
        fields = [(name, ty.Any) for name, _ in self._connections]
        self.output_spec = SpecInfo(name="Output", fields=fields, bases=(BaseSpec,))
        logger.info("Added %s to %s", self.output_spec, self)

    def _collect_outputs(self):
        output_klass = make_klass(self.output_spec)
        output = output_klass(**{f.name: None for f in dc.fields(output_klass)})
        # collecting outputs from tasks
        output_wf = {}
        for name, val in self._connections:
            if not isinstance(val, LazyField):
                raise ValueError("all connections must be lazy")
            output_wf[name] = val.get_value(self)
        return dc.replace(output, **output_wf)


def is_task(obj):
    return hasattr(obj, "_run_task")


def is_workflow(obj):
    return isinstance(obj, Workflow)
