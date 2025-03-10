"""Basic processing graph elements."""

import json
import logging
import os
import inspect
import sys
from pathlib import Path
import typing as ty
from uuid import uuid4
import shutil
from traceback import format_exception
import attr
import cloudpickle as cp
from copy import copy
from collections import defaultdict
from typing import Self
import attrs
from filelock import SoftFileLock
from pydra.engine.specs import TaskDef, WorkflowDef, TaskOutputs, WorkflowOutputs
from pydra.engine.graph import DiGraph
from pydra.engine import state
from .lazy import LazyInField, LazyOutField
from pydra.utils.hash import hash_function, Cache
from pydra.engine.state import State
from .node import Node
from datetime import datetime
from fileformats.core import FileSet
from .specs import (
    RuntimeSpec,
    Result,
    TaskHooks,
)
from .helpers import (
    attrs_fields,
    attrs_values,
    load_result,
    save,
    record_error,
    PydraFileLock,
    list_fields,
    is_lazy,
    ensure_list,
)
from .helpers_file import copy_nested_files, template_update
from pydra.utils.messenger import AuditFlag
from pydra.engine.environments import Environment

logger = logging.getLogger("pydra")

develop = False

if ty.TYPE_CHECKING:
    from pydra.engine.submitter import Submitter, NodeExecution
    from pydra.design.base import Arg

DefType = ty.TypeVar("DefType", bound=TaskDef)


class Task(ty.Generic[DefType]):
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
    definition: DefType
    submitter: "Submitter | None"
    environment: "Environment | None"
    state_index: state.StateIndex
    bindings: dict[str, ty.Any] | None = None  # Bindings for the task environment

    _inputs: dict[str, ty.Any] | None = None

    def __init__(
        self,
        definition: DefType,
        submitter: "Submitter",
        name: str,
        environment: "Environment | None" = None,
        state_index: "state.StateIndex | None" = None,
        hooks: TaskHooks | None = None,
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
        """

        if state_index is None:
            state_index = state.StateIndex()

        if not isinstance(definition, TaskDef):
            raise ValueError(
                f"Task definition ({definition!r}) must be a TaskDef, not {type(definition)}"
            )
        # Check that the definition is fully resolved and ready to run
        definition._check_resolved()
        definition._check_rules()
        self.definition = definition
        # We save the submitter is the definition is a workflow otherwise we don't
        # so the task can be pickled
        self.submitter = submitter
        self.environment = (
            environment if environment is not None else submitter.environment
        )
        self.name = name
        self.state_index = state_index

        self.return_values = {}
        self._result = {}
        # flag that says if node finished all jobs
        self._done = False
        if self._input_sets is None:
            self._input_sets = {}

        self.allow_cache_override = True
        self._checksum = None
        self._uid = uuid4().hex
        self.hooks = hooks if hooks is not None else TaskHooks()
        self._errored = False
        self._lzout = None

        # Save the submitter attributes needed to run the task later
        self.audit = submitter.audit
        self.cache_dir = submitter.cache_dir
        self.cache_locations = submitter.cache_locations

    @property
    def cache_dir(self):
        return self._cache_dir

    @property
    def is_async(self) -> bool:
        """Check to see if the task should be run asynchronously."""
        return self.submitter.worker.is_async and is_workflow(self.definition)

    @cache_dir.setter
    def cache_dir(self, path: os.PathLike):
        self._cache_dir = Path(path)

    @property
    def cache_locations(self):
        """Get the list of cache sources."""
        return self._cache_locations + ensure_list(self.cache_dir)

    @cache_locations.setter
    def cache_locations(self, locations):
        if locations is not None:
            self._cache_locations = [Path(loc) for loc in ensure_list(locations)]
        else:
            self._cache_locations = []

    def __str__(self):
        return self.name

    def __getstate__(self):
        state = self.__dict__.copy()
        state["definition"] = cp.dumps(state["definition"])
        return state

    def __setstate__(self, state):
        state["definition"] = cp.loads(state["definition"])
        self.__dict__.update(state)

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
        if self._checksum is not None:
            return self._checksum
        self._checksum = self.definition._checksum
        return self._checksum

    @property
    def lockfile(self):
        return self.output_dir.with_suffix(".lock")

    @property
    def uid(self):
        """the unique id number for the task
        It will be used to create unique names for slurm scripts etc.
        without a need to run checksum
        """
        return self._uid

    @property
    def output_names(self):
        """Get the names of the outputs from the task's output_spec"""
        return [f.name for f in attr.fields(self.definition.Outputs)]

    @property
    def can_resume(self):
        """Whether the task accepts checkpoint-restart."""
        return self._can_resume

    @property
    def output_dir(self):
        """Get the filesystem path where outputs will be written."""
        return self.cache_dir / self.checksum

    @property
    def inputs(self) -> dict[str, ty.Any]:
        """Resolve any template inputs of the task ahead of its execution:

        - links/copies upstream files and directories into the destination tasks
          working directory as required select state array values corresponding to
          state index (it will try to leave them where they are unless specified or
          they are on different file systems)
        - resolve template values (e.g. output_file_template)
        - deepcopy all inputs to guard against in-place changes during the task's
          execution (they will be replaced after the task's execution with the
          original inputs to ensure the tasks checksums are consistent)
        """
        if self._inputs is not None:
            return self._inputs

        from pydra.utils.typing import TypeParser

        self._inputs = {
            k: v
            for k, v in attrs_values(self.definition).items()
            if not k.startswith("_")
        }
        map_copyfiles = {}
        fld: "Arg"
        for fld in list_fields(self.definition):
            name = fld.name
            value = self._inputs[name]
            if value and TypeParser.contains_type(FileSet, fld.type):
                copied_value = copy_nested_files(
                    value=value,
                    dest_dir=self.output_dir,
                    mode=fld.copy_mode,
                    collation=fld.copy_collation,
                    supported_modes=self.SUPPORTED_COPY_MODES,
                )
                if value is not copied_value:
                    map_copyfiles[name] = copied_value
        self._inputs.update(
            template_update(
                self.definition, output_dir=self.output_dir, map_copyfiles=map_copyfiles
            )
        )
        return self._inputs

    def _populate_filesystem(self):
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
            json.dump({"checksum": self.checksum}, jsonfile)
        if not self.can_resume and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=False, exist_ok=self.can_resume)
        # Save task pkl into the output directory for future reference
        save(self.output_dir, task=self)

    def run(self, rerun: bool = False):
        """Prepare the task working directory, execute the task definition, and save the
        results.

        Parameters
        ----------
        rerun : bool
            If True, the task will be re-run even if a result already exists. Will
            propagated to all tasks within workflow tasks.
        """
        # TODO: After these changes have been merged, will refactor this function and
        # run_async to use common helper methods for pre/post run tasks

        # checking if the definition is fully resolved and ready to run
        self.hooks.pre_run(self)
        logger.debug(
            "'%s' is attempting to acquire lock on %s", self.name, self.lockfile
        )
        with SoftFileLock(self.lockfile):
            if not (rerun):
                result = self.result()
                if result is not None and not result.errored:
                    return result
            cwd = os.getcwd()
            self._populate_filesystem()
            os.chdir(self.output_dir)
            result = Result(
                outputs=None,
                runtime=None,
                errored=False,
                output_dir=self.output_dir,
                definition=self.definition,
            )
            self.hooks.pre_run_task(self)
            self.audit.start_audit(odir=self.output_dir)
            if self.audit.audit_check(AuditFlag.PROV):
                self.audit.audit_task(task=self)
            try:
                self.audit.monitor()
                self.definition._run(self, rerun)
                result.outputs = self.definition.Outputs._from_task(self)
            except Exception:
                etype, eval, etr = sys.exc_info()
                traceback = format_exception(etype, eval, etr)
                record_error(self.output_dir, error=traceback)
                result.errored = True
                raise
            finally:
                self.hooks.post_run_task(self, result)
                self.audit.finalize_audit(result=result)
                save(self.output_dir, result=result, task=self)
                # removing the additional file with the checksum
                (self.cache_dir / f"{self.uid}_info.json").unlink()
                os.chdir(cwd)
        self.hooks.post_run(self, result)
        # Check for any changes to the input hashes that have occurred during the execution
        # of the task
        self._check_for_hash_changes()
        return result

    async def run_async(self, rerun: bool = False) -> Result:
        """Prepare the task working directory, execute the task definition asynchronously,
        and save the results. NB: only workflows are run asynchronously at the moment.

        Parameters
        ----------
        rerun : bool
            If True, the task will be re-run even if a result already exists. Will
            propagated to all tasks within workflow tasks.
        """
        # checking if the definition is fully resolved and ready to run
        self.hooks.pre_run(self)
        logger.debug(
            "'%s' is attempting to acquire lock on %s", self.name, self.lockfile
        )
        async with PydraFileLock(self.lockfile):
            if not rerun:
                result = self.result()
                if result is not None and not result.errored:
                    return result
            cwd = os.getcwd()
            self._populate_filesystem()
            result = Result(
                outputs=None,
                runtime=None,
                errored=False,
                output_dir=self.output_dir,
                definition=self.definition,
            )
            self.hooks.pre_run_task(self)
            self.audit.start_audit(odir=self.output_dir)
            try:
                self.audit.monitor()
                await self.definition._run_async(self, rerun)
                result.outputs = self.definition.Outputs._from_task(self)
            except Exception:
                etype, eval, etr = sys.exc_info()
                traceback = format_exception(etype, eval, etr)
                record_error(self.output_dir, error=traceback)
                result.errored = True
                self._errored = True
                raise
            finally:
                self.hooks.post_run_task(self, result)
                self.audit.finalize_audit(result=result)
                save(self.output_dir, result=result, task=self)
                # removing the additional file with the checksum
                (self.cache_dir / f"{self.uid}_info.json").unlink()
                os.chdir(cwd)
        self.hooks.post_run(self, result)
        # Check for any changes to the input hashes that have occurred during the execution
        # of the task
        self._check_for_hash_changes()
        return result

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
        if has_lazy(self.definition):
            return False
        _result = self.result()
        if _result:
            if _result.errored:
                self._errored = True
                raise ValueError(f"Task {self.name!r} raised an error")
            else:
                return True
        return False

    @property
    def run_start_time(self) -> datetime | None:
        """Check whether the task is currently running."""
        if not self.lockfile.exists():
            return None
        return datetime.fromtimestamp(self.lockfile.stat().st_ctime)

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
                elif return_inputs is True or return_inputs == "ind":
                    result = (self.state.states_ind[ind], result)
                combined_results_gr.append(result)
            combined_results.append(combined_results_gr)
        if len(combined_results) == 1 and self.state.splitter_rpn_final == []:
            # in case it's full combiner, removing the nested structure
            return combined_results[0]
        else:
            return combined_results

    def result(self, return_inputs=False):
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
        if self.errored:
            return Result(
                outputs=None,
                runtime=None,
                errored=True,
                output_dir=self.output_dir,
                definition=self.definition,
            )

        checksum = self.checksum
        result = load_result(checksum, self.cache_locations)
        if result and result.errored:
            self._errored = True
        if return_inputs is True or return_inputs == "val":
            inputs_val = {
                f"{self.name}.{inp}": getattr(self.definition, inp)
                for inp in self.input_names
            }
            return (inputs_val, result)
        elif return_inputs == "ind":
            inputs_ind = {f"{self.name}.{inp}": None for inp in self.input_names}
            return (inputs_ind, result)
        else:
            return result

    def _check_for_hash_changes(self):
        hash_changes = self.definition._hash_changes()
        details = ""
        for changed in hash_changes:
            field = getattr(attr.fields(type(self.definition)), changed)
            hash_function(getattr(self.definition, changed))
            val = getattr(self.definition, changed)
            field_type = type(val)
            if inspect.isclass(field.type) and issubclass(field.type, FileSet):
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
                    f'A "bytes_repr" method for {field.type!r} can be implemented to '
                    "bespoke hashing methods based only on the stable attributes for "
                    f"the `{field_type.__module__}.{field_type.__name__}` type. "
                    f"See pydra/utils/hash.py for examples. Value: {val}\n"
                )
        if hash_changes:
            raise RuntimeError(
                f"Input field hashes have changed during the execution of the "
                f"'{self.name}' task of {type(self)} type.\n\n{details}"
            )
        logger.debug(
            "Input values and hashes for '%s' %s node:\n%s\n%s",
            self.name,
            type(self).__name__,
            self.definition,
            self.definition._hashes,
        )

    def _write_notebook(self):
        """Writes a notebook into the"""
        raise NotImplementedError

    SUPPORTED_COPY_MODES = FileSet.CopyMode.any
    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.any


logger = logging.getLogger("pydra")

OutputsType = ty.TypeVar("OutputType", bound=TaskOutputs)
WorkflowOutputsType = ty.TypeVar("OutputType", bound=WorkflowOutputs)


@attrs.define(auto_attribs=False)
class Workflow(ty.Generic[WorkflowOutputsType]):
    """A workflow, constructed from a workflow definition

    Parameters
    ----------
    name : str
        The name of the workflow
    inputs : TaskDef
        The input definition of the workflow
    outputs : TaskDef
        The output definition of the workflow
    """

    name: str = attrs.field()
    inputs: WorkflowDef[WorkflowOutputsType] = attrs.field()
    outputs: WorkflowOutputsType = attrs.field()
    _nodes: dict[str, Node] = attrs.field(factory=dict)

    def __repr__(self):
        return f"Workflow(name={self.name!r}, defn={self.inputs!r})"

    @classmethod
    def clear_cache(
        cls, definition: WorkflowDef[WorkflowOutputsType] | None = None
    ) -> None:
        """Clear the cache of constructed workflows"""
        if definition is None:
            cls._constructed_cache = defaultdict(lambda: defaultdict(dict))
        else:
            cls._constructed_cache[hash_function(definition)] = defaultdict(dict)

    @classmethod
    def construct(
        cls, definition: WorkflowDef[WorkflowOutputsType], dont_cache: bool = False
    ) -> Self:
        """Construct a workflow from a definition, caching the constructed worklow"""

        # Check the previously constructed workflows to see if a workflow has been
        # constructed for the given set of inputs, or a less-specific set (i.e. with a
        # super-set of lazy inputs), and use that if it exists

        non_lazy_vals = {
            n: v for n, v in attrs_values(definition).items() if not is_lazy(v)
        }
        non_lazy_keys = frozenset(non_lazy_vals)
        hash_cache = Cache()  # share the hash cache to avoid recalculations
        non_lazy_hash = hash_function(non_lazy_vals, cache=hash_cache)
        defn_hash = hash_function(type(definition), cache=hash_cache)
        # Check for same non-lazy inputs
        try:
            defn_cache = cls._constructed_cache[defn_hash]
        except KeyError:
            pass
        else:
            if (
                non_lazy_keys in defn_cache
                and non_lazy_hash in defn_cache[non_lazy_keys]
            ):
                return defn_cache[non_lazy_keys][non_lazy_hash]
            # Check for supersets of lazy inputs
            for key_set, key_set_cache in defn_cache.items():
                if key_set.issubset(non_lazy_keys):
                    subset_vals = {
                        k: v for k, v in non_lazy_vals.items() if k in key_set
                    }
                    subset_hash = hash_function(subset_vals, cache=hash_cache)
                    if subset_hash in key_set_cache:
                        return key_set_cache[subset_hash]

        # Initialise the outputs of the workflow
        outputs = definition.Outputs(
            **{f.name: attrs.NOTHING for f in attrs.fields(definition.Outputs)}
        )

        # Initialise the lzin fields
        lazy_spec = copy(definition)
        workflow = Workflow(
            name=type(definition).__name__,
            inputs=lazy_spec,
            outputs=outputs,
        )
        # Set lazy inputs to the workflow, need to do it after the workflow is initialised
        # so a back ref to the workflow can be set in the lazy field
        for field in list_fields(definition):
            if field.name not in non_lazy_keys:
                setattr(
                    lazy_spec,
                    field.name,
                    LazyInField(
                        workflow=workflow,
                        field=field.name,
                        type=field.type,
                    ),
                )

        input_values = attrs_values(lazy_spec)
        constructor = input_values.pop("constructor")
        # Call the user defined constructor to set the outputs
        output_lazy_fields = constructor(**input_values)
        # Check to see whether any mandatory inputs are not set
        for node in workflow.nodes:
            node._definition._check_rules()
        # Check that the outputs are set correctly, either directly by the constructor
        # or via returned values that can be zipped with the output names
        if output_lazy_fields:
            if not isinstance(output_lazy_fields, (list, tuple)):
                output_lazy_fields = [output_lazy_fields]
            output_fields = list_fields(definition.Outputs)
            if len(output_lazy_fields) != len(output_fields):
                raise ValueError(
                    f"Expected {len(output_fields)} outputs, got "
                    f"{len(output_lazy_fields)} ({output_lazy_fields})"
                )
            for outpt, outpt_lf in zip(output_fields, output_lazy_fields):
                # Automatically combine any uncombined state arrays into a single lists
                outpt_lf._type = State.combine_state_arrays(outpt_lf._type)
                setattr(outputs, outpt.name, outpt_lf)
        else:
            if unset_outputs := [
                a for a, v in attrs_values(outputs).items() if v is attrs.NOTHING
            ]:
                raise ValueError(
                    f"Expected outputs {unset_outputs} to be set by the "
                    f"constructor of {workflow!r}"
                )
        if not dont_cache:
            cls._constructed_cache[defn_hash][non_lazy_keys][non_lazy_hash] = workflow

        return workflow

    @classmethod
    def under_construction(cls) -> "Workflow[ty.Any]":
        """Access the under_construction variable by iterating up through the call stack."""
        frame = inspect.currentframe()
        while frame:
            # Find the frame where the construct method was called
            if (
                frame.f_code.co_name == "construct"
                and frame.f_locals.get("cls") is cls
                and "workflow" in frame.f_locals
            ):
                return frame.f_locals["workflow"]  # local var "workflow" in construct
            frame = frame.f_back
        raise RuntimeError(
            "No workflow is currently under construction (i.e. did not find a "
            "`Workflow.construct` in the current call stack"
        )

    def add(
        self,
        task_def: TaskDef[OutputsType],
        name: str | None = None,
        environment: Environment | None = None,
        hooks: TaskHooks | None = None,
    ) -> OutputsType:
        """Add a node to the workflow

        Parameters
        ----------
        task_spec : TaskDef
            The definition of the task to add to the workflow as a node
        name : str, optional
            The name of the node, by default it will be the name of the task definition
            class
        environment : Environment, optional
            The environment to run the task in, such as the Docker or Singularity container,
            by default it will be the "native"
        hooks : TaskHooks, optional
            The hooks to run before or after the task, by default no hooks will be run

        Returns
        -------
        OutputType
            The outputs definition of the node
        """
        from pydra.engine.environments import Native

        if name is None:
            name = type(task_def).__name__
        if name in self._nodes:
            raise ValueError(f"Node with name {name!r} already exists in the workflow")
        if (
            environment
            and not isinstance(environment, Native)
            and task_def._task_type != "shell"
        ):
            raise ValueError(
                "Environments can only be used with 'shell' tasks not "
                f"{task_def._task_type!r} tasks ({task_def!r})"
            )
        node = Node[OutputsType](
            name=name,
            definition=task_def,
            workflow=self,
            environment=environment,
            hooks=hooks,
        )
        self._nodes[name] = node
        return node.lzout

    def __getitem__(self, key: str) -> Node:
        return self._nodes[key]

    @property
    def nodes(self) -> ty.Iterable[Node]:
        return self._nodes.values()

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes)

    # Used to cache the constructed workflows by their hashed input values
    _constructed_cache: dict[
        str, dict[frozenset[str], dict[str, "Workflow[ty.Any]"]]
    ] = defaultdict(lambda: defaultdict(dict))

    def execution_graph(self, submitter: "Submitter") -> DiGraph:
        from pydra.engine.submitter import NodeExecution

        exec_nodes = [NodeExecution(n, submitter, workflow=self) for n in self.nodes]
        graph = self._create_graph(exec_nodes)
        # Set the graph attribute of the nodes so lazy fields can be resolved as tasks
        # are created
        for node in exec_nodes:
            node.graph = graph
        return graph

    @property
    def graph(self) -> DiGraph:
        return self._create_graph(self.nodes, detailed=True)

    def _create_graph(
        self, nodes: "list[Node | NodeExecution]", detailed: bool = False
    ) -> DiGraph:
        """
        Connects a particular task to existing nodes in the workflow.

        Parameters
        ----------
        detailed : bool
            If True, `add_edges_description` is run a detailed descriptions of the
            connections (input/output fields names)
        node_klass : type, optional
            The class to use for the nodes in the workflow. If provided the node is
            wrapped by an instance of the class, if None the node is added as is,
            by default None

        Returns
        -------
        DiGraph
            The graph of the workflow
        """
        graph: DiGraph = DiGraph()
        for node in nodes:
            graph.add_nodes(node)
        # TODO: create connection is run twice
        for node in nodes:
            other_states = {}
            for field in attrs_fields(node.inputs):
                lf = node._definition[field.name]
                if isinstance(lf, LazyOutField):
                    # adding an edge to the graph if task id expecting output from a different task
                    if lf._node.name != self.name:
                        # checking if the connection is already in the graph
                        if (graph.node(lf._node.name), node) not in graph.edges:
                            graph.add_edges((graph.node(lf._node.name), node))
                        if detailed:
                            graph.add_edges_description(
                                (node.name, field.name, lf._node.name, lf._field)
                            )
                        logger.debug("Connecting %s to %s", lf._node.name, node.name)
                        # adding a state from the previous task to other_states
                        if (
                            graph.node(lf._node.name).state
                            and graph.node(lf._node.name).state.splitter_rpn_final
                        ):
                            # variables that are part of inner splitters should be
                            # treated as a containers
                            if (
                                node.state
                                and f"{node.name}.{field.name}" in node.state.splitter
                            ):
                                node.state._inner_cont_dim[
                                    f"{node.name}.{field.name}"
                                ] = 1
                            # adding task_name: (task.state, [a field from the connection]
                            if lf._node.name not in other_states:
                                other_states[lf._node.name] = (
                                    graph.node(lf._node.name).state,
                                    [field.name],
                                )
                            else:
                                # if the task already exist in other_state,
                                # additional field name should be added to the list of fields
                                other_states[lf._node.name][1].append(field.name)
                    else:  # LazyField with the wf input
                        # connections with wf input should be added to the detailed graph description
                        if detailed:
                            graph.add_edges_description(
                                (node.name, field.name, lf._node.name, lf._field)
                            )

            # if task has connections state has to be recalculated
            if other_states:
                if hasattr(node, "fut_combiner"):
                    combiner = node.fut_combiner
                else:
                    combiner = None

                if node.state:
                    node.state.update_connections(
                        new_other_states=other_states, new_combiner=combiner
                    )
                else:
                    node.state = state.State(
                        node.name,
                        splitter=None,
                        other_states=other_states,
                        combiner=combiner,
                    )
        return graph


def is_workflow(obj):
    """Check whether an object is a :class:`Workflow` instance."""
    from pydra.engine.specs import WorkflowDef
    from pydra.engine.core import Workflow

    return isinstance(obj, (WorkflowDef, Workflow))


def has_lazy(obj):
    """Check whether an object has lazy fields."""
    for f in attrs_fields(obj):
        if is_lazy(getattr(obj, f.name)):
            return True
    return False
