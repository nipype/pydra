"""Basic processing graph elements."""
import abc
import attr
import json
import logging
import os
from pathlib import Path
import typing as ty
from copy import deepcopy, copy

import cloudpickle as cp
from filelock import SoftFileLock
import shutil
from tempfile import mkdtemp

from . import state
from . import helpers_state as hlpst
from .specs import (
    File,
    BaseSpec,
    RuntimeSpec,
    Result,
    SpecInfo,
    LazyField,
    TaskHook,
    attr_fields,
)
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
from .helpers_file import copyfile_input, template_update
from .graph import DiGraph
from .audit import Audit
from ..utils.messenger import AuditFlag

logger = logging.getLogger("pydra")

develop = False


class TaskBase:
    """
    A base structure for the nodes in the processing graph.

    Tasks are a generic compute step from which both elemntary tasks and
    :class:`Workflow` instances inherit.

    """

    _api_version: str = "0.0.1"  # Should generally not be touched by subclasses
    _etelemetry_version_data = None  # class variable to store etelemetry information
    _version: str  # Version of tool being wrapped
    _task_version: ty.Optional[str] = None
    # Task writers encouraged to define and increment when implementation changes sufficiently
    _input_sets = None  # Dictionaries of predefined input settings

    audit_flags: AuditFlag = AuditFlag.NONE
    """What to audit -- available flags: :class:`~pydra.utils.messenger.AuditFlag`."""

    _can_resume = False  # Does the task allow resuming from previous state
    _redirect_x = False  # Whether an X session should be created/directed

    _runtime_requirements = RuntimeSpec()
    _runtime_hints = None

    _cache_dir = None  # Working directory in which to operate
    _references = None  # List of references for a task

    def __init__(
        self,
        name: str,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        inputs: ty.Union[ty.Text, File, ty.Dict, None] = None,
        messenger_args=None,
        messengers=None,
        rerun=False,
    ):
        """
        Initialize a task.

        Tasks allow for caching (retrieving a previous result of the same
        task definition and inputs), and concurrent execution.
        Running tasks follows a decision flow:

            1. Check whether prior cache exists --
               if ``True``, return cached result
            2. Check whether other process is running this task --
               wait if ``True``:
               a. Finishes (with or without exception) -> return result
               b. Gets killed -> restart
            3. No cache or other process -> start
            4. Two or more concurrent new processes get to start

        Parameters
        ----------
        name : :obj:`str`
            Unique name of this node
        audit_flags : :class:`AuditFlag`, optional
            Configure provenance tracking. Default is no provenance tracking.
            See available flags at :class:`~pydra.utils.messenger.AuditFlag`.
        cache_dir : :obj:`os.pathlike`
            Set a custom directory of previously computed nodes.
        cache_locations :
            TODO
        inputs : :obj:`typing.Text`, or :class:`File`, or :obj:`dict`, or `None`.
            Set particular inputs to this node.
        messenger_args :
            TODO
        messengers :
            TODO

        """
        from .. import check_latest_version

        if TaskBase._etelemetry_version_data is None:
            TaskBase._etelemetry_version_data = check_latest_version()

        self.name = name
        if not self.input_spec:
            raise Exception("No input_spec in class: %s" % self.__class__.__name__)
        klass = make_klass(self.input_spec)
        # todo should be used to input_check in spec??
        self.inputs = klass(
            **{
                (f.name[1:] if f.name.startswith("_") else f.name): f.default
                for f in attr.fields(klass)
            }
        )
        self.input_names = [
            field.name
            for field in attr.fields(klass)
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
            self.inputs = attr.evolve(self.inputs, **inputs)
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
        # if True the results are not checked (does not propagate to nodes)
        self.task_rerun = rerun

        self.plugin = None
        self.hooks = TaskHook()

    def __str__(self):
        return self.name

    def __getstate__(self):
        state = self.__dict__.copy()
        state["input_spec"] = cp.dumps(state["input_spec"])
        state["output_spec"] = cp.dumps(state["output_spec"])
        inputs = {}
        for k, v in attr.asdict(state["inputs"]).items():
            if k.startswith("_"):
                k = k[1:]
            inputs[k] = v
        state["inputs"] = inputs
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
        """Print class help."""
        help_obj = print_help(self)
        if returnhelp:
            return help_obj

    @property
    def version(self):
        """Get version of this task structure."""
        return self._version

    @property
    def checksum(self):
        """ Calculates the unique checksum of the task.
            Used to create specific directory name for task that are run;
            and to create nodes checksums needed for graph checkums
            (before the tasks have inputs etc.)
        """
        input_hash = self.inputs.hash
        if self.state is None:
            self._checksum = create_checksum(self.__class__.__name__, input_hash)
        else:
            splitter_hash = hash_function(self.state.splitter)
            self._checksum = create_checksum(
                self.__class__.__name__, hash_function([input_hash, splitter_hash])
            )
        return self._checksum

    def checksum_states(self, state_index=None):
        """
        Calculate a checksum for the specific state or all of the states of the task.
        Replaces lists in the inputs fields with a specific values for states.
        Used to recreate names of the task directories,

        Parameters
        ----------
        state_index :
            TODO

        """
        self.state.prepare_states(self.inputs)
        self.state.prepare_inputs()
        if state_index is not None:
            inputs_copy = deepcopy(self.inputs)
            for key, ind in self.state.inputs_ind[state_index].items():
                setattr(
                    inputs_copy,
                    key.split(".")[1],
                    getattr(inputs_copy, key.split(".")[1])[ind],
                )
            input_hash = inputs_copy.hash
            if is_workflow(self):
                con_hash = hash_function(self._connections)
                hash_list = [input_hash, con_hash]
                checksum_ind = create_checksum(
                    self.__class__.__name__, self._checksum_wf(input_hash)
                )
            else:
                checksum_ind = create_checksum(self.__class__.__name__, input_hash)
            return checksum_ind
        else:
            checksum_list = []
            for ind in range(len(self.state.inputs_ind)):
                checksum_list.append(self.checksum_states(state_index=ind))
            return checksum_list

    def set_state(self, splitter, combiner=None):
        """
        Set a particular state on this task.

        Parameters
        ----------
        splitter :
            TODO
        combiner :
            TODO

        """
        if splitter is not None:
            self.state = state.State(
                name=self.name, splitter=splitter, combiner=combiner
            )
        else:
            self.state = None
        return self.state

    @property
    def output_names(self):
        """Get the names of the parameters generated by the task."""
        output_spec_names = [f.name for f in attr.fields(make_klass(self.output_spec))]
        from_input_spec_names = output_names_from_inputfields(self.inputs)
        return output_spec_names + from_input_spec_names

    @property
    def can_resume(self):
        """Whether the task accepts checkpoint-restart."""
        return self._can_resume

    @abc.abstractmethod
    def _run_task(self):
        pass

    @property
    def cache_dir(self):
        """Get the location of the cache directory."""
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
        """Get the list of cache sources."""
        return self._cache_locations + ensure_list(self._cache_dir)

    @cache_locations.setter
    def cache_locations(self, locations):
        if locations is not None:
            self._cache_locations = [Path(loc) for loc in ensure_list(locations)]
        else:
            self._cache_locations = []

    @property
    def output_dir(self):
        """Get the filesystem path where outputs will be written."""
        if self.state:
            return [self._cache_dir / checksum for checksum in self.checksum_states()]
        return self._cache_dir / self.checksum

    def __call__(self, submitter=None, plugin=None, rerun=False, **kwargs):
        """Make tasks callable themselves."""
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
            res = self._run(rerun=rerun, **kwargs)
        return res

    def _run(self, rerun=False, **kwargs):
        self.inputs = attr.evolve(self.inputs, **kwargs)
        self.inputs.check_fields_input_spec()
        checksum = self.checksum
        lockfile = self.cache_dir / (checksum + ".lock")
        # Eagerly retrieve cached - see scenarios in __init__()
        self.hooks.pre_run(self)
        # TODO add signal handler for processes killed after lock acquisition
        with SoftFileLock(lockfile):
            if not (rerun or self.task_rerun):
                result = self.result()
                if result is not None:
                    return result
            # Let only one equivalent process run
            odir = self.output_dir
            if not self.can_resume and odir.exists():
                shutil.rmtree(odir)
            cwd = os.getcwd()
            odir.mkdir(parents=False, exist_ok=True if self.can_resume else False)
            orig_inputs = attr.asdict(self.inputs)
            map_copyfiles = copyfile_input(self.inputs, self.output_dir)
            modified_inputs = template_update(self.inputs, map_copyfiles)
            if modified_inputs:
                self.inputs = attr.evolve(self.inputs, **modified_inputs)
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
                for k, v in orig_inputs.items():
                    setattr(self.inputs, k, v)
                os.chdir(cwd)
        self.hooks.post_run(self, result)
        return result

    def _collect_outputs(self):
        run_output = self.output_
        self.output_spec = output_from_inputfields(self.output_spec, self.inputs)
        output_klass = make_klass(self.output_spec)
        output = output_klass(**{f.name: None for f in attr.fields(output_klass)})
        other_output = output.collect_additional_outputs(
            self.input_spec, self.inputs, self.output_dir
        )
        return attr.evolve(output, **run_output, **other_output)

    def split(self, splitter, overwrite=False, **kwargs):
        """
        Run this task parametrically over lists of splitted inputs.

        Parameters
        ----------
        splitter :
            TODO
        overwrite : :obj:`bool`
            TODO

        """
        splitter = hlpst.add_name_splitter(splitter, self.name)
        # if user want to update the splitter, overwrite has to be True
        if self.state and not overwrite and self.state.splitter != splitter:
            raise Exception(
                "splitter has been already set, "
                "if you want to overwrite it - use overwrite=True"
            )
        if kwargs:
            self.inputs = attr.evolve(self.inputs, **kwargs)
            self.state_inputs = kwargs
        if not self.state or splitter != self.state.splitter:
            self.set_state(splitter)
        return self

    def combine(self, combiner, overwrite=False):
        """
        Combine inputs parameterized by one or more previous tasks.

        Parameters
        ----------
        combiner :
            TODO
        overwrite : :obj:`bool`
            TODO

        """
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
        """Collect all inputs required to run the node (for specific state element)."""
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

    def pickle_task(self):
        """ Pickling the tasks with full inputs"""
        pkl_files = self.cache_dir / "pkl_files"
        pkl_files.mkdir(exist_ok=True, parents=True)
        task_main_path = pkl_files / f"{self.name}_{self.checksum}_task.pklz"
        save(task_path=pkl_files, task=self, name_prefix=f"{self.name}_{self.checksum}")
        return task_main_path

    @property
    def done(self):
        """Check whether the tasks has been finalized and all outputs are stored."""
        # if any of the field is lazy, there is no need to check results
        if is_lazy(self.inputs):
            return False
        if self.state:
            # TODO: only check for needed state result
            if self.result() and all(self.result()):
                return True
            # checking if self.result() is not an empty list only because
            # the states_ind is an empty list (input field might be an empty list)
            elif (
                self.result() == []
                and hasattr(self.state, "states_ind")
                and self.state.states_ind == []
            ):
                return True
        else:
            if self.result():
                return True
        return False

    def _combined_output(self, return_inputs=False):
        combined_results = []
        for (gr, ind_l) in self.state.final_combined_ind_mapping.items():
            combined_results_gr = []
            for ind in ind_l:
                result = load_result(self.checksum_states(ind), self.cache_locations)
                if result is None:
                    return None
                if return_inputs is True or return_inputs == "val":
                    result = (self.state.states_val[ind], result)
                elif return_inputs == "ind":
                    result = (self.state.states_ind[ind], result)
                combined_results_gr.append(result)
            combined_results.append(combined_results_gr)
        if len(combined_results) == 1 and self.state.splitter_rpn_final == []:
            # in case it's full combiner, removing the nested structure
            return combined_results[0]
        else:
            return combined_results

    def result(self, state_index=None, return_inputs=False):
        """
        Retrieve the outcomes of this particular task.

        Parameters
        ----------
        state_index : :obj: `int`
            index of the element for task with splitter and multiple states
        return_inputs : :obj: `bool`, :obj:`str`
            if True or "val" result is returned together with values of the input fields,
            if "ind" result is returned together with indices of the input fields

        Returns
        -------
        result :

        """
        # TODO: check if result is available in load_result and
        # return a future if not
        if self.state:
            if state_index is None:
                # if state_index=None, collecting all results
                if self.state.combiner:
                    return self._combined_output(return_inputs=return_inputs)
                else:
                    results = []
                    for checksum in self.checksum_states():
                        result = load_result(checksum, self.cache_locations)
                        if result is None:
                            return None
                        results.append(result)
                    if return_inputs is True or return_inputs == "val":
                        return list(zip(self.state.states_val, results))
                    elif return_inputs == "ind":
                        return list(zip(self.state.states_ind, results))
                    else:
                        return results
            else:  # state_index is not None
                if self.state.combiner:
                    return self._combined_output(return_inputs=return_inputs)[
                        state_index
                    ]
                result = load_result(
                    self.checksum_states(state_index), self.cache_locations
                )
                if return_inputs is True or return_inputs == "val":
                    return (self.state.states_val[state_index], result)
                elif return_inputs == "ind":
                    return (self.state.states_ind[state_index], result)
                else:
                    return result
        else:
            if state_index is not None:
                raise ValueError("Task does not have a state")
            checksum = self.checksum
            result = load_result(checksum, self.cache_locations)
            if return_inputs is True or return_inputs == "val":
                inputs_val = {
                    f"{self.name}.{inp}": getattr(self.inputs, inp)
                    for inp in self.input_names
                }
                return (inputs_val, result)
            elif return_inputs == "ind":
                inputs_ind = {f"{self.name}.{inp}": None for inp in self.input_names}
                return (inputs_ind, result)
            else:
                return result

    def _reset(self):
        """Reset the connections between inputs and LazyFields."""
        for field in attr_fields(self.inputs):
            if field.name in self.inp_lf:
                setattr(self.inputs, field.name, self.inp_lf[field.name])
        if is_workflow(self):
            for task in self.graph.nodes:
                task._reset()


class Workflow(TaskBase):
    """A composite task with structure of computational graph."""

    def __init__(
        self,
        name,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        input_spec: ty.Union[ty.List[ty.Text], BaseSpec, None] = None,
        messenger_args=None,
        messengers=None,
        output_spec: ty.Optional[BaseSpec] = None,
        rerun=False,
        propagate_rerun=True,
        **kwargs,
    ):
        """
        Initialize a workflow.

        Parameters
        ----------
        name : :obj:`str`
            Unique name of this node
        audit_flags : :class:`AuditFlag`, optional
            Configure provenance tracking. Default is no provenance tracking.
            See available flags at :class:`~pydra.utils.messenger.AuditFlag`.
        cache_dir : :obj:`os.pathlike`
            Set a custom directory of previously computed nodes.
        cache_locations :
            TODO
        inputs : :obj:`typing.Text`, or :class:`File`, or :obj:`dict`, or `None`.
            Set particular inputs to this node.
        messenger_args :
            TODO
        messengers :
            TODO
        output_spec :
            TODO

        """
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
                            attr.ib(
                                type=ty.Any,
                                metadata={
                                    "help_string": f"{nm} input from {name} workflow"
                                },
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
            rerun=rerun,
        )

        self.graph = DiGraph()
        self.name2obj = {}

        # store output connections
        self._connections = None
        # propagating rerun if task_rerun=True
        self.propagate_rerun = propagate_rerun

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
        """
        Check if all tasks from the graph are done.

        .. important ::
            The fact that all tasks are reported as done
            doesn't mean that results of the workflow
            are available.
            That can be checked with :py:meth:`~Workflow.done`.

        """
        for task in self.graph.nodes:
            if not task.done:
                return False
        return True

    @property
    def nodes(self):
        """Get the list of node names."""
        return self.name2obj.values()

    @property
    def graph_sorted(self):
        """Get a sorted graph representation of the workflow."""
        return self.graph.sorted_nodes

    @property
    def checksum(self):
        """ Calculates the unique checksum of the task.
            Used to create specific directory name for task that are run;
            and to create nodes checksums needed for graph checkums
            (before the tasks have inputs etc.)
        """
        # if checksum is called before run the _graph_checksums is not ready
        if is_workflow(self) and self.inputs._graph_checksums is attr.NOTHING:
            self.inputs._graph_checksums = [nd.checksum for nd in self.graph_sorted]

        input_hash = self.inputs.hash
        if not self.state:
            self._checksum = create_checksum(
                self.__class__.__name__, self._checksum_wf(input_hash)
            )
        else:
            self._checksum = create_checksum(
                self.__class__.__name__,
                self._checksum_wf(input_hash, with_splitter=True),
            )
        return self._checksum

    def _checksum_wf(self, input_hash, with_splitter=False):
        """ creating hash value for workflows
            includes connections and splitter if with_splitter is True
        """
        connection_hash = hash_function(self._connections)
        hash_list = [input_hash, connection_hash]
        if with_splitter and self.state:
            # including splitter in the hash
            splitter_hash = hash_function(self.state.splitter)
            hash_list.append(splitter_hash)
        return hash_function(hash_list)

    def add(self, task):
        """
        Add a task to the workflow.

        Parameters
        ----------
        task : :class:`TaskBase`
            The task to be added.

        """
        if not is_task(task):
            raise ValueError("Unknown workflow element: {!r}".format(task))
        self.graph.add_nodes(task)
        self.name2obj[task.name] = task
        self._last_added = task
        logger.debug(f"Added {task}")
        return self

    def create_connections(self, task):
        """
        Add and connect a particular task to existing nodes in the workflow.

        Parameters
        ----------
        task : :class:`TaskBase`
            The task to be added.

        """
        other_states = {}
        for field in attr_fields(task.inputs):
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

                    if (
                        getattr(self, val.name).state
                        and getattr(self, val.name).state.splitter_rpn_final
                    ):
                        # adding a state from the previous task to other_states
                        other_states[val.name] = (
                            getattr(self, val.name).state,
                            field.name,
                        )
        # if task has connections state has to be recalculated
        if other_states:
            if hasattr(task, "fut_combiner"):
                combiner = task.fut_combiner
            else:
                combiner = None

            if task.state:
                task.state.update_connections(
                    new_other_states=other_states, new_combiner=combiner
                )
            else:
                task.state = state.State(
                    task.name,
                    splitter=None,
                    other_states=other_states,
                    combiner=combiner,
                )

    async def _run(self, submitter=None, rerun=False, **kwargs):
        # self.inputs = dc.replace(self.inputs, **kwargs) don't need it?
        checksum = self.checksum
        lockfile = self.cache_dir / (checksum + ".lock")
        # Eagerly retrieve cached
        if not (rerun or self.task_rerun):
            result = self.result()
            if result is not None:
                return result
        # creating connections that were defined after adding tasks to the wf
        for task in self.graph.nodes:
            # if workflow has task_rerun=True and propagate_rerun=True,
            # it should be passed to the tasks
            if self.task_rerun and self.propagate_rerun:
                task.task_rerun = self.task_rerun
                # if the task is a wf, than the propagate_rerun should be also set
                if is_workflow(task):
                    task.propagate_rerun = self.propagate_rerun
            task.cache_locations = task._cache_locations + self.cache_locations
            self.create_connections(task)
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
                await self._run_task(submitter, rerun=rerun)
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

    async def _run_task(self, submitter, rerun=False):
        if not submitter:
            raise Exception("Submitter should already be set.")
        # at this point Workflow is stateless so this should be fine
        await submitter._run_workflow(self, rerun=rerun)

    def set_output(self, connections):
        """
        Write outputs.

        Parameters
        ----------
        connections :
            TODO

        """
        if self._connections is None:
            self._connections = []
        if isinstance(connections, tuple) and len(connections) == 2:
            new_connections = [connections]
        elif isinstance(connections, list) and all(
            [len(el) == 2 for el in connections]
        ):
            new_connections = connections
        elif isinstance(connections, dict):
            new_connections = list(connections.items())
        else:
            raise Exception(
                "Connections can be a 2-elements tuple, a list of these tuples, or dictionary"
            )
        # checking if a new output name is already in the connections
        connection_names = [name for name, _ in self._connections]
        new_names = [name for name, _ in new_connections]
        if set(connection_names).intersection(new_names):
            raise Exception(
                f"output name {set(connection_names).intersection(new_names)} is already set"
            )

        self._connections += new_connections
        fields = [(name, ty.Any) for name, _ in self._connections]
        self.output_spec = SpecInfo(name="Output", fields=fields, bases=(BaseSpec,))
        logger.info("Added %s to %s", self.output_spec, self)

    def _collect_outputs(self):
        output_klass = make_klass(self.output_spec)
        output = output_klass(**{f.name: None for f in attr.fields(output_klass)})
        # collecting outputs from tasks
        output_wf = {}
        for name, val in self._connections:
            if not isinstance(val, LazyField):
                raise ValueError("all connections must be lazy")
            output_wf[name] = val.get_value(self)
        return attr.evolve(output, **output_wf)


def is_task(obj):
    """Check whether an object looks like a task."""
    return hasattr(obj, "_run_task")


def is_workflow(obj):
    """Check whether an object is a :class:`Workflow` instance."""
    return isinstance(obj, Workflow)


def is_lazy(obj):
    """Check whether an object has any field that is a Lazy Field"""
    for f in attr_fields(obj):
        if isinstance(getattr(obj, f.name), LazyField):
            return True
    return False
