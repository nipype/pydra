"""Basic processing graph elements."""

import abc
import json
import logging
import os
import sys
from pathlib import Path
import typing as ty
from copy import deepcopy
from uuid import uuid4
from filelock import SoftFileLock
import shutil
from tempfile import mkdtemp
from traceback import format_exception
import attr
import cloudpickle as cp
from . import state
from . import helpers_state as hlpst
from .specs import (
    File,
    RuntimeSpec,
    Result,
    TaskHook,
    TaskSpec,
)
from .helpers import (
    create_checksum,
    attrs_fields,
    attrs_values,
    print_help,
    load_result,
    save,
    ensure_list,
    record_error,
    PydraFileLock,
    parse_copyfile,
    is_lazy,
)
from pydra.utils.hash import hash_function
from .helpers_file import copy_nested_files, template_update
from .graph import DiGraph
from .audit import Audit
from pydra.utils.messenger import AuditFlag
from fileformats.core import FileSet

logger = logging.getLogger("pydra")

develop = False


class Task:
    """
    A base structure for the nodes in the processing graph.

    Tasks are a generic compute step from which both elementary tasks and
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

    name: str
    spec: TaskSpec

    def __init__(
        self,
        spec,
        name: str | None = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        inputs: ty.Optional[ty.Union[ty.Text, File, ty.Dict]] = None,
        cont_dim=None,
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
        cont_dim : :obj:`dict`, or `None`
            Container dimensions for input fields,
            if any of the container should be treated as a container
        messenger_args :
            TODO
        messengers :
            TODO
        """
        from . import check_latest_version

        if Task._etelemetry_version_data is None:
            Task._etelemetry_version_data = check_latest_version()

        self.spec = spec
        self.name = name

        self.input_names = [
            field.name
            for field in attr.fields(type(self.spec))
            if field.name not in ["_func", "_graph_checksums"]
        ]

        if inputs:
            if isinstance(inputs, dict):
                # selecting items that are in input_names (ignoring fields that are not in input_spec)
                inputs = {k: v for k, v in inputs.items() if k in self.input_names}
            # TODO: this needs to finished and tested after #305
            elif Path(inputs).is_file():
                inputs = json.loads(Path(inputs).read_text())
            # TODO: this needs to finished and tested after #305
            elif isinstance(inputs, str):
                if self._input_sets is None or inputs not in self._input_sets:
                    raise ValueError(f"Unknown input set {inputs!r}")
                inputs = self._input_sets[inputs]

        self.spec = attr.evolve(self.spec, **inputs)

        # checking if metadata is set properly
        self.spec._check_resolved()
        self.spec._check_rules()
        self._output = {}
        self._result = {}
        # flag that says if node finished all jobs
        self._done = False
        if self._input_sets is None:
            self._input_sets = {}

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
        self._uid = uuid4().hex
        # if True the results are not checked (does not propagate to nodes)
        self.task_rerun = rerun

        self.plugin = None
        self.hooks = TaskHook()
        self._errored = False
        self._lzout = None

    def __str__(self):
        return self.name

    def __getstate__(self):
        state = self.__dict__.copy()
        state["spec"] = cp.dumps(state["spec"])
        return state

    def __setstate__(self, state):
        state["spec"] = cp.loads(state["spec"])
        self.__dict__.update(state)

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
    def errored(self):
        """Check if the task has raised an error"""
        return self._errored

    @property
    def checksum(self):
        """Calculates the unique checksum of the task.
        Used to create specific directory name for task that are run;
        and to create nodes checksums needed for graph checksums
        (before the tasks have inputs etc.)
        """
        input_hash = self.spec._hash
        self._checksum = create_checksum(self.__class__.__name__, input_hash)
        return self._checksum

    @property
    def uid(self):
        """the unique id number for the task
        It will be used to create unique names for slurm scripts etc.
        without a need to run checksum
        """
        return self._uid

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
        """Get the names of the outputs from the task's output_spec
        (not everything has to be generated, see generated_output_names).
        """
        return [f.name for f in attr.fields(self.spec.Outputs)]

    @property
    def generated_output_names(self):
        """Get the names of the outputs generated by the task.
        If the spec doesn't have generated_output_names method,
        it uses output_names.
        The results depends on the input provided to the task
        """
        output_klass = self.spec.Outputs
        if hasattr(output_klass, "_generated_output_names"):
            output = output_klass(
                **{f.name: attr.NOTHING for f in attr.fields(output_klass)}
            )
            # using updated input (after filing the templates)
            _inputs = deepcopy(self.spec)
            modified_inputs = template_update(_inputs, self.output_dir)
            if modified_inputs:
                _inputs = attr.evolve(_inputs, **modified_inputs)

            return output._generated_output_names(
                inputs=_inputs, output_dir=self.output_dir
            )
        else:
            return self.output_names

    @property
    def can_resume(self):
        """Whether the task accepts checkpoint-restart."""
        return self._can_resume

    @abc.abstractmethod
    def _run_task(self, environment=None):
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
        return self._cache_dir / self.checksum

    @property
    def cont_dim(self):
        # adding inner_cont_dim to the general container_dimension provided by the users
        cont_dim_all = deepcopy(self._cont_dim)
        for k, v in self._inner_cont_dim.items():
            cont_dim_all[k] = cont_dim_all.get(k, 1) + v
        return cont_dim_all

    @cont_dim.setter
    def cont_dim(self, cont_dim):
        if cont_dim is None:
            self._cont_dim = {}
        else:
            self._cont_dim = cont_dim

    def __call__(
        self,
        submitter=None,
        plugin=None,
        plugin_kwargs=None,
        rerun=False,
        environment=None,
        **kwargs,
    ):
        """Make tasks callable themselves."""
        from .submitter import Submitter

        if submitter and plugin:
            raise Exception("Specify submitter OR plugin, not both")
        elif submitter:
            pass
        # if there is plugin provided or the task is a Workflow or has a state,
        # the submitter will be created using provided plugin, self.plugin or "cf"
        elif plugin:
            plugin = plugin or self.plugin or "cf"
            if plugin_kwargs is None:
                plugin_kwargs = {}
            submitter = Submitter(plugin=plugin, **plugin_kwargs)

        if submitter:
            with submitter as sub:
                self.spec = attr.evolve(self.spec, **kwargs)
                res = sub(self, environment=environment)
        else:  # tasks without state could be run without a submitter
            res = self._run(rerun=rerun, environment=environment, **kwargs)
        return res

    def _modify_inputs(self):
        """This method modifies the inputs of the task ahead of its execution:
        - links/copies upstream files and directories into the destination tasks
          working directory as required select state array values corresponding to
          state index (it will try to leave them where they are unless specified or
          they are on different file systems)
        - resolve template values (e.g. output_file_template)
        - deepcopy all inputs to guard against in-place changes during the task's
          execution (they will be replaced after the task's execution with the
          original inputs to ensure the tasks checksums are consistent)
        """
        from pydra.utils.typing import TypeParser

        orig_inputs = {
            k: v for k, v in attrs_values(self.spec).items() if not k.startswith("_")
        }
        map_copyfiles = {}
        input_fields = attr.fields(type(self.spec))
        for name, value in orig_inputs.items():
            fld = getattr(input_fields, name)
            copy_mode, copy_collation = parse_copyfile(
                fld, default_collation=self.DEFAULT_COPY_COLLATION
            )
            if value is not attr.NOTHING and TypeParser.contains_type(
                FileSet, fld.type
            ):
                copied_value = copy_nested_files(
                    value=value,
                    dest_dir=self.output_dir,
                    mode=copy_mode,
                    collation=copy_collation,
                    supported_modes=self.SUPPORTED_COPY_MODES,
                )
                if value is not copied_value:
                    map_copyfiles[name] = copied_value
        modified_inputs = template_update(
            self.spec, self.output_dir, map_copyfiles=map_copyfiles
        )
        assert all(m in orig_inputs for m in modified_inputs), (
            "Modified inputs contain fields not present in original inputs. "
            "This is likely a bug."
        )
        for name, orig_value in orig_inputs.items():
            try:
                value = modified_inputs[name]
            except KeyError:
                # Ensure we pass a copy not the original just in case inner
                # attributes are modified during execution
                value = deepcopy(orig_value)
            setattr(self.spec, name, value)
        return orig_inputs

    def _populate_filesystem(self, checksum, output_dir):
        """
        Invoked immediately after the lockfile is generated, this function:
        - Creates the cache file
        - Clears existing outputs if `can_resume` is False
        - Generates a fresh output directory

        Created as an attempt to simplify overlapping `Task`|`Workflow` behaviors.
        """
        # adding info file with the checksum in case the task was cancelled
        # and the lockfile has to be removed
        with open(self.cache_dir / f"{self.uid}_info.json", "w") as jsonfile:
            json.dump({"checksum": checksum}, jsonfile)
        if not self.can_resume and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=False, exist_ok=self.can_resume)

    def _run(self, rerun=False, environment=None):
        checksum = self.checksum
        output_dir = self.output_dir
        lockfile = self.cache_dir / (checksum + ".lock")
        # Eagerly retrieve cached - see scenarios in __init__()
        self.hooks.pre_run(self)
        logger.debug("'%s' is attempting to acquire lock on %s", self.name, lockfile)
        with SoftFileLock(lockfile):
            if not (rerun or self.task_rerun):
                result = self.result()
                if result is not None and not result.errored:
                    return result
            cwd = os.getcwd()
            self._populate_filesystem(checksum, output_dir)
            os.chdir(output_dir)
            result = Result(output=None, runtime=None, errored=False)
            self.hooks.pre_run_task(self)
            self.audit.start_audit(odir=output_dir)
            if self.audit.audit_check(AuditFlag.PROV):
                self.audit.audit_task(task=self)
            try:
                self.audit.monitor()
                self._run_task(environment=environment)
                result.output = self.spec.Outputs.from_task(self)
            except Exception:
                etype, eval, etr = sys.exc_info()
                traceback = format_exception(etype, eval, etr)
                record_error(output_dir, error=traceback)
                result.errored = True
                raise
            finally:
                self.hooks.post_run_task(self, result)
                self.audit.finalize_audit(result)
                save(output_dir, result=result, task=self)
                # removing the additional file with the checksum
                (self.cache_dir / f"{self.uid}_info.json").unlink()
                # Restore original values to inputs
                os.chdir(cwd)
        self.hooks.post_run(self, result)
        # Check for any changes to the input hashes that have occurred during the execution
        # of the task
        self._check_for_hash_changes()
        return result

    def _extract_input_el(self, inputs, inp_nm, ind):
        """
        Extracting element of the inputs taking into account
        container dimension of the specific element that can be set in self.cont_dim.
        If input name is not in cont_dim, it is assumed that the input values has
        a container dimension of 1, so only the most outer dim will be used for splitting.
        If
        """
        if f"{self.name}.{inp_nm}" in self.cont_dim:
            return list(
                hlpst.flatten(
                    ensure_list(getattr(inputs, inp_nm)),
                    max_depth=self.cont_dim[f"{self.name}.{inp_nm}"],
                )
            )[ind]
        else:
            return getattr(inputs, inp_nm)[ind]

    def get_input_el(self, ind):
        """Collect all inputs required to run the node (for specific state element)."""
        # TODO: doesn't work properly for more cmplicated wf (check if still an issue)
        input_ind = self.state.inputs_ind[ind]
        inputs_dict = {}
        for inp in set(self.input_names):
            if f"{self.name}.{inp}" in input_ind:
                inputs_dict[inp] = self._extract_input_el(
                    inputs=self.spec,
                    inp_nm=inp,
                    ind=input_ind[f"{self.name}.{inp}"],
                )
        return inputs_dict
        # else:
        #     # todo it never gets here
        #     breakpoint()
        #     inputs_dict = {inp: getattr(self.inputs, inp) for inp in self.input_names}
        #     return None, inputs_dict

    def pickle_task(self):
        """Pickling the tasks with full inputs"""
        pkl_files = self.cache_dir / "pkl_files"
        pkl_files.mkdir(exist_ok=True, parents=True)
        task_main_path = pkl_files / f"{self.name}_{self.uid}_task.pklz"
        save(task_path=pkl_files, task=self, name_prefix=f"{self.name}_{self.uid}")
        return task_main_path

    @property
    def done(self):
        """Check whether the tasks has been finalized and all outputs are stored."""
        # if any of the field is lazy, there is no need to check results
        if has_lazy(self.spec):
            return False
        _result = self.result()
        if self.state:
            # TODO: only check for needed state result
            if _result and all(_result):
                if self.state.combiner and isinstance(_result[0], list):
                    for res_l in _result:
                        if any([res.errored for res in res_l]):
                            raise ValueError(f"Task {self.name} raised an error")
                    return True
                else:
                    if any([res.errored for res in _result]):
                        raise ValueError(f"Task {self.name} raised an error")
                    return True
            # checking if self.result() is not an empty list only because
            # the states_ind is an empty list (input field might be an empty list)
            elif (
                _result == []
                and hasattr(self.state, "states_ind")
                and self.state.states_ind == []
            ):
                return True
        else:
            if _result:
                if _result.errored:
                    self._errored = True
                    raise ValueError(f"Task {self.name} raised an error")
                else:
                    return True
        return False

    def _combined_output(self, return_inputs=False):
        combined_results = []
        for gr, ind_l in self.state.final_combined_ind_mapping.items():
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
        result : Result
            the result of the task
        """
        # TODO: check if result is available in load_result and
        # return a future if not
        if self.errored:
            return Result(output=None, runtime=None, errored=True)

        if state_index is not None:
            raise ValueError("Task does not have a state")
        checksum = self.checksum
        result = load_result(checksum, self.cache_locations)
        if result and result.errored:
            self._errored = True
        if return_inputs is True or return_inputs == "val":
            inputs_val = {
                f"{self.name}.{inp}": getattr(self.spec, inp)
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
        for field in attrs_fields(self.spec):
            if field.name in self.inp_lf:
                setattr(self.spec, field.name, self.inp_lf[field.name])
        if is_workflow(self):
            for task in self.graph.nodes:
                task._reset()

    def _check_for_hash_changes(self):
        hash_changes = self.spec._hash_changes()
        details = ""
        for changed in hash_changes:
            field = getattr(attr.fields(type(self.spec)), changed)
            val = getattr(self.spec, changed)
            field_type = type(val)
            if issubclass(field.type, FileSet):
                details += (
                    f"- {changed}: value passed to the {field.type} field is of type "
                    f"{field_type} ('{val}'). If it is intended to contain output data "
                    "then the type of the field in the interface class should be changed "
                    "to `pathlib.Path`. Otherwise, if the field is intended to be an "
                    "input field but it gets altered by the task in some way, then the "
                    "'copyfile' flag should be set to 'copy' in the field metadata of "
                    "the task interface class so copies of the files/directories in it "
                    "are passed to the task instead.\n"
                )
            else:
                details += (
                    f"- {changed}: the {field_type} object passed to the {field.type}"
                    f"field appears to have an unstable hash. This could be due to "
                    "a stochastic/non-thread-safe attribute(s) of the object\n\n"
                    f"The {field.type}.__bytes_repr__() method can be implemented to "
                    "bespoke hashing methods based only on the stable attributes for "
                    f"the `{field_type.__module__}.{field_type.__name__}` type. "
                    f"See pydra/utils/hash.py for examples. Value: {val}\n"
                )
        if hash_changes:
            raise RuntimeError(
                f"Input field hashes have changed during the execution of the "
                f"'{self.name}' {type(self).__name__}.\n\n{details}"
            )
        logger.debug(
            "Input values and hashes for '%s' %s node:\n%s\n%s",
            self.name,
            type(self).__name__,
            self.spec,
            self.spec._hashes,
        )

    SUPPORTED_COPY_MODES = FileSet.CopyMode.any
    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.any


class WorkflowTask(Task):
    """A composite task with structure of computational graph."""

    def __init__(
        self,
        name,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        input_spec: ty.Optional[
            ty.Union[ty.List[ty.Text], ty.Dict[ty.Text, ty.Type[ty.Any]]]
        ] = None,
        cont_dim=None,
        messenger_args=None,
        messengers=None,
        output_spec: ty.Optional[ty.Union[ty.List[str], ty.Dict[str, type]]] = None,
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
        cont_dim : :obj:`dict`, or `None`
            Container dimensions for input fields,
            if any of the container should be treated as a container
        messenger_args :
            TODO
        messengers :
            TODO
        output_spec :
            TODO

        """

        if name in dir(self):
            raise ValueError(
                "Cannot use names of attributes or methods as workflow name"
            )
        self.name = name

        super().__init__(
            name=name,
            inputs=kwargs,
            cont_dim=cont_dim,
            cache_dir=cache_dir,
            cache_locations=cache_locations,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            rerun=rerun,
        )

        self.graph = DiGraph(name=name)
        self.name2obj = {}
        self._lzin = None
        self._pre_split = (
            False  # To signify if the workflow has been split on task load or not
        )

        # store output connections
        self._connections = None
        # propagating rerun if task_rerun=True
        self.propagate_rerun = propagate_rerun

    def __getattr__(self, name):
        if name in self.name2obj:
            return self.name2obj[name]
        return self.__getattribute__(name)

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
        """Calculates the unique checksum of the task.
        Used to create specific directory name for task that are run;
        and to create nodes checksums needed for graph checksums
        (before the tasks have inputs etc.)
        """
        # if checksum is called before run the _graph_checksums is not ready
        if is_workflow(self) and self.spec._graph_checksums is attr.NOTHING:
            self.spec._graph_checksums = {
                nd.name: nd.checksum for nd in self.graph_sorted
            }

        input_hash = self.spec.hash
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
        """creating hash value for workflows
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
        if task.name in dir(self):
            raise ValueError(
                "Cannot use names of workflow attributes or methods as task name"
            )
        if task.name in self.name2obj:
            raise ValueError(
                "Another task named {} is already added to the workflow".format(
                    task.name
                )
            )
        self.name2obj[task.name] = task

        if not is_task(task):
            raise ValueError(f"Unknown workflow element: {task!r}")
        self.graph.add_nodes(task)
        self._last_added = task
        logger.debug(f"Added {task}")
        return self

    def create_connections(self, task, detailed=False):
        """
        Add and connect a particular task to existing nodes in the workflow.

        Parameters
        ----------
        task : :class:`TaskBase`
            The task to be added.
        detailed : :obj:`bool`
            If True, `add_edges_description` is run for self.graph to add
            a detailed descriptions of the connections (input/output fields names)
        """
        # TODO: create connection is run twice
        other_states = {}
        for field in attrs_fields(task.inputs):
            val = getattr(task.inputs, field.name)
            if is_lazy(val):
                # saving all connections with LazyFields
                task.inp_lf[field.name] = val
                # adding an edge to the graph if task id expecting output from a different task
                if val.name != self.name:
                    # checking if the connection is already in the graph
                    if (getattr(self, val.name), task) not in self.graph.edges:
                        self.graph.add_edges((getattr(self, val.name), task))
                    if detailed:
                        self.graph.add_edges_description(
                            (task.name, field.name, val.name, val.field)
                        )
                    logger.debug("Connecting %s to %s", val.name, task.name)
                    # adding a state from the previous task to other_states
                    if (
                        getattr(self, val.name).state
                        and getattr(self, val.name).state.splitter_rpn_final
                    ):
                        # variables that are part of inner splitters should be treated as a containers
                        if (
                            task.state
                            and f"{task.name}.{field.name}" in task.state.splitter
                        ):
                            task._inner_cont_dim[f"{task.name}.{field.name}"] = 1
                        # adding task_name: (task.state, [a field from the connection]
                        if val.name not in other_states:
                            other_states[val.name] = (
                                getattr(self, val.name).state,
                                [field.name],
                            )
                        else:
                            # if the task already exist in other_state,
                            # additional field name should be added to the list of fields
                            other_states[val.name][1].append(field.name)
                else:  # LazyField with the wf input
                    # connections with wf input should be added to the detailed graph description
                    if detailed:
                        self.graph.add_edges_description(
                            (task.name, field.name, val.name, val.field)
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
        # output_spec needs to be set using set_output or at workflow initialization
        if self.output_spec is None:
            raise ValueError(
                "Workflow output cannot be None, use set_output to define output(s)"
            )
        # creating connections that were defined after adding tasks to the wf
        self._connect_and_propagate_to_tasks(
            propagate_rerun=self.task_rerun and self.propagate_rerun
        )

        checksum = self.checksum
        output_dir = self.output_dir
        lockfile = self.cache_dir / (checksum + ".lock")
        self.hooks.pre_run(self)
        logger.debug(
            "'%s' is attempting to acquire lock on %s with Pydra lock",
            self.name,
            lockfile,
        )
        async with PydraFileLock(lockfile):
            if not (rerun or self.task_rerun):
                result = self.result()
                if result is not None and not result.errored:
                    return result
            cwd = os.getcwd()
            self._populate_filesystem(checksum, output_dir)
            result = Result(output=None, runtime=None, errored=False)
            self.hooks.pre_run_task(self)
            self.audit.start_audit(odir=output_dir)
            try:
                self.audit.monitor()
                await self._run_task(submitter, rerun=rerun)
                result.output = self._collect_outputs()
            except Exception:
                etype, eval, etr = sys.exc_info()
                traceback = format_exception(etype, eval, etr)
                record_error(output_dir, error=traceback)
                result.errored = True
                self._errored = True
                raise
            finally:
                self.hooks.post_run_task(self, result)
                self.audit.finalize_audit(result=result)
                save(output_dir, result=result, task=self)
                # removing the additional file with the checksum
                (self.cache_dir / f"{self.uid}_info.json").unlink()
                os.chdir(cwd)
        self.hooks.post_run(self, result)
        # Check for any changes to the input hashes that have occurred during the execution
        # of the task
        self._check_for_hash_changes()
        return result

    async def _run_task(self, submitter, rerun=False, environment=None):
        if not submitter:
            raise Exception("Submitter should already be set.")
        for nd in self.graph.nodes:
            if nd.allow_cache_override:
                nd.cache_dir = self.cache_dir
        # at this point Workflow is stateless so this should be fine
        await submitter.expand_workflow(self, rerun=rerun)

    # def set_output(
    #     self,
    #     connections: ty.Union[
    #         ty.Tuple[str, LazyField], ty.List[ty.Tuple[str, LazyField]]
    #     ],
    # ):
    #     """
    #     Set outputs of the workflow by linking them with lazy outputs of tasks

    #     Parameters
    #     ----------
    #     connections : tuple[str, LazyField] or list[tuple[str, LazyField]] or None
    #         single or list of tuples linking the name of the output to a lazy output
    #         of a task in the workflow.
    #     """
    #     from pydra.utils.typing import TypeParser

    #     if self._connections is None:
    #         self._connections = []
    #     if isinstance(connections, tuple) and len(connections) == 2:
    #         new_connections = [connections]
    #     elif isinstance(connections, list) and all(
    #         [len(el) == 2 for el in connections]
    #     ):
    #         new_connections = connections
    #     elif isinstance(connections, dict):
    #         new_connections = list(connections.items())
    #     else:
    #         raise TypeError(
    #             "Connections can be a 2-elements tuple, a list of these tuples, or dictionary"
    #         )
    #     # checking if a new output name is already in the connections
    #     connection_names = [name for name, _ in self._connections]
    #     if self.output_spec:
    #         output_types = {a.name: a.type for a in attr.fields(self.interface.Outputs)}
    #     else:
    #         output_types = {}
    #     # Check for type matches with explicitly defined outputs
    #     conflicting = []
    #     type_mismatches = []
    #     for conn_name, lazy_field in new_connections:
    #         if conn_name in connection_names:
    #             conflicting.append(conn_name)
    #         try:
    #             output_type = output_types[conn_name]
    #         except KeyError:
    #             pass
    #         else:
    #             if not TypeParser.matches_type(lazy_field.type, output_type):
    #                 type_mismatches.append((conn_name, output_type, lazy_field.type))
    #     if conflicting:
    #         raise ValueError(f"the output names {conflicting} are already set")
    #     if type_mismatches:
    #         raise TypeError(
    #             f"the types of the following outputs of {self} don't match their declared types: "
    #             + ", ".join(
    #                 f"{n} (expected: {ex}, provided: {p})"
    #                 for n, ex, p in type_mismatches
    #             )
    #         )
    #     self._connections += new_connections
    #     fields = []
    #     for con in self._connections:
    #         wf_out_nm, lf = con
    #         task_nm, task_out_nm = lf.name, lf.field
    #         if task_out_nm == "all_":
    #             help_string = f"all outputs from {task_nm}"
    #             fields.append((wf_out_nm, dict, {"help_string": help_string}))
    #         else:
    #             from pydra.utils.typing import TypeParser

    #             # getting information about the output field from the task output_spec
    #             # providing proper type and some help string
    #             task_output_spec = getattr(self, task_nm).output_spec
    #             out_fld = attr.fields_dict(task_output_spec)[task_out_nm]
    #             help_string = (
    #                 f"{out_fld.metadata.get('help_string', '')} (from {task_nm})"
    #             )
    #             if TypeParser.get_origin(lf.type) is StateArray:
    #                 type_ = TypeParser.get_item_type(lf.type)
    #             else:
    #                 type_ = lf.type
    #             fields.append((wf_out_nm, type_, {"help_string": help_string}))
    #     self.output_spec = SpecInfo(name="Output", fields=fields, bases=(BaseSpec,))
    #     logger.info("Added %s to %s", self.output_spec, self)

    def _collect_outputs(self):
        output_klass = self.spec.Outputs
        output = output_klass(
            **{f.name: attr.NOTHING for f in attr.fields(output_klass)}
        )
        # collecting outputs from tasks
        output_wf = {}
        for name, val in self._connections:
            if not is_lazy(val):
                raise ValueError("all connections must be lazy")
            try:
                val_out = val.get_value(self)
                output_wf[name] = val_out
            except (ValueError, AttributeError) as e:
                output_wf[name] = None
                # checking if the tasks has predecessors that raises error
                if isinstance(getattr(self, val.name)._errored, list):
                    raise ValueError(
                        f"Tasks {getattr(self, val.name)._errored} raised an error"
                    )
                else:
                    if isinstance(getattr(self, val.name).output_dir, list):
                        err_file = [
                            el / "_error.pklz"
                            for el in getattr(self, val.name).output_dir
                        ]
                        if not all(e.exists() for e in err_file):
                            raise e
                    else:
                        err_file = getattr(self, val.name).output_dir / "_error.pklz"
                        if not Path(err_file).exists():
                            raise e
                    raise ValueError(
                        f"Task {val.name} raised an error, full crash report is here: "
                        f"{err_file}"
                    )
        return attr.evolve(output, **output_wf)

    def create_dotfile(self, type="simple", export=None, name=None, output_dir=None):
        """creating a graph - dotfile and optionally exporting to other formats"""
        outdir = output_dir if output_dir is not None else self.cache_dir
        if not name:
            name = f"graph_{self.name}"
        if type == "simple":
            for task in self.graph.nodes:
                self.create_connections(task)
            dotfile = self.graph.create_dotfile_simple(outdir=outdir, name=name)
        elif type == "nested":
            for task in self.graph.nodes:
                self.create_connections(task)
            dotfile = self.graph.create_dotfile_nested(outdir=outdir, name=name)
        elif type == "detailed":
            # create connections with detailed=True
            for task in self.graph.nodes:
                self.create_connections(task, detailed=True)
            # adding wf outputs
            for wf_out, lf in self._connections:
                self.graph.add_edges_description((self.name, wf_out, lf.name, lf.field))
            dotfile = self.graph.create_dotfile_detailed(outdir=outdir, name=name)
        else:
            raise Exception(
                f"type of the graph can be simple, detailed or nested, "
                f"but {type} provided"
            )
        if not export:
            return dotfile
        else:
            if export is True:
                export = ["png"]
            elif isinstance(export, str):
                export = [export]
            formatted_dot = []
            for ext in export:
                formatted_dot.append(self.graph.export_graph(dotfile=dotfile, ext=ext))
            return dotfile, formatted_dot

    def _connect_and_propagate_to_tasks(
        self,
        *,
        propagate_rerun=False,
        override_task_caches=False,
    ):
        """
        Visit each node in the graph and create the connections.
        Additionally checks if all tasks should be rerun.
        """
        for task in self.graph.nodes:
            self.create_connections(task)
            # if workflow has task_rerun=True and propagate_rerun=True,
            # it should be passed to the tasks
            if propagate_rerun:
                task.task_rerun = True
                # if the task is a wf, than the propagate_rerun should be also set
                if is_workflow(task):
                    task.propagate_rerun = True

            # ported from Submitter.__call__
            # TODO: no prepare state ?
            if override_task_caches and task.allow_cache_override:
                task.cache_dir = self.cache_dir
            task.cache_locations = task._cache_locations + self.cache_locations


def is_task(obj):
    """Check whether an object looks like a task."""
    return hasattr(obj, "_run_task")


def is_workflow(obj):
    """Check whether an object is a :class:`Workflow` instance."""
    return isinstance(obj, WorkflowTask)


def has_lazy(obj):
    """Check whether an object has lazy fields."""
    for f in attrs_fields(obj):
        if is_lazy(getattr(obj, f.name)):
            return True
    return False
