"""Basic processing graph elements."""

import json
import logging
import os
import sys
from pathlib import Path
import typing as ty
from uuid import uuid4
import shutil
from traceback import format_exception
import attr
import cloudpickle as cp
from copy import copy
from operator import itemgetter
from typing_extensions import Self
import attrs
from filelock import SoftFileLock
from pydra.engine.specs import TaskDef, WorkflowDef, TaskOutputs, WorkflowOutputs
from pydra.engine.graph import DiGraph
from pydra.engine import state
from .lazy import LazyInField, LazyOutField
from pydra.utils.hash import hash_function
from pydra.utils.typing import TypeParser, StateArray
from .node import Node
from fileformats.generic import FileSet
from .specs import (
    RuntimeSpec,
    Result,
    TaskHook,
)
from .helpers import (
    create_checksum,
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
    submitter: "Submitter"
    state_index: state.StateIndex

    _inputs: dict[str, ty.Any] | None = None

    def __init__(
        self,
        definition: DefType,
        submitter: "Submitter",
        name: str,
        state_index: "state.StateIndex | None" = None,
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
        from . import check_latest_version

        if Task._etelemetry_version_data is None:
            Task._etelemetry_version_data = check_latest_version()

        if state_index is None:
            state_index = state.StateIndex()

        self.definition = definition
        self.name = name
        self.state_index = state_index

        # checking if metadata is set properly
        self.definition._check_resolved()
        self.definition._check_rules()
        self._output = {}
        self._result = {}
        # flag that says if node finished all jobs
        self._done = False
        if self._input_sets is None:
            self._input_sets = {}

        self.allow_cache_override = True
        self._checksum = None
        self._uid = uuid4().hex
        self.hooks = TaskHook()
        self._errored = False
        self._lzout = None

        # Save the submitter attributes needed to run the task later
        self.audit = submitter.audit
        self.cache_dir = submitter.cache_dir
        self.cache_locations = submitter.cache_locations

    @property
    def cache_dir(self):
        return self._cache_dir

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
        input_hash = self.definition._hash
        self._checksum = create_checksum(self.__class__.__name__, input_hash)
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
        (not everything has to be generated, see _generated_output_names).
        """
        return [f.name for f in attr.fields(self.definition.Outputs)]

    @property
    def _generated_output_names(self):
        return self.output_names

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
            if value is not attr.NOTHING and TypeParser.contains_type(
                FileSet, fld.type
            ):
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
                self.definition, self.output_dir, map_copyfiles=map_copyfiles
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

    def run(self, rerun: bool = False):
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
            result = Result(outputs=None, runtime=None, errored=False, task=self)
            self.hooks.pre_run_task(self)
            self.audit.start_audit(odir=self.output_dir)
            if self.audit.audit_check(AuditFlag.PROV):
                self.audit.audit_task(task=self)
            try:
                self.audit.monitor()
                run_outputs = self.definition._run(self)
                result.outputs = self.definition.Outputs.from_task(self, run_outputs)
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
            return Result(outputs=None, runtime=None, errored=True, task=self)

        if state_index is not None:
            raise ValueError("Task does not have a state")
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
            val = getattr(self.definition, changed)
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
            self.definition,
            self.definition._hashes,
        )

    SUPPORTED_COPY_MODES = FileSet.CopyMode.any
    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.any


class WorkflowTask(Task):

    def __init__(
        self,
        definition: DefType,
        submitter: "Submitter",
        name: str,
        state_index: "state.StateIndex | None" = None,
    ):
        super().__init__(definition, submitter, name, state_index)
        self.submitter = submitter

    async def run(self, rerun: bool = False):
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
            result = Result(outputs=None, runtime=None, errored=False, task=self)
            self.hooks.pre_run_task(self)
            self.audit.start_audit(odir=self.output_dir)
            try:
                self.audit.monitor()
                await self.submitter.expand_workflow(self)
                result.outputs = self.definition.Outputs.from_task(self)
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

    @classmethod
    def construct(
        cls,
        definition: WorkflowDef[WorkflowOutputsType],
    ) -> Self:
        """Construct a workflow from a definition, caching the constructed worklow"""

        lazy_inputs = [f for f in list_fields(type(definition)) if f.lazy]

        # Create a cache key by hashing all the non-lazy input values in the definition
        # and use this to store the constructed workflow in case it is reused or nested
        # and split over within another workflow
        lazy_input_names = {f.name for f in lazy_inputs}
        non_lazy_vals = tuple(
            sorted(
                (
                    i
                    for i in attrs_values(definition).items()
                    if i[0] not in lazy_input_names
                ),
                key=itemgetter(0),
            )
        )
        if lazy_non_lazy_vals := [f for f in non_lazy_vals if is_lazy(f[1])]:
            raise ValueError(
                f"Lazy input fields {lazy_non_lazy_vals} found in non-lazy fields "
            )
        hash_key = hash_function(non_lazy_vals)
        if hash_key in cls._constructed:
            return cls._constructed[hash_key]

        # Initialise the outputs of the workflow
        outputs = definition.Outputs(
            **{f.name: attrs.NOTHING for f in attrs.fields(definition.Outputs)}
        )

        # Initialise the lzin fields
        lazy_spec = copy(definition)
        wf = cls.under_construction = Workflow(
            name=type(definition).__name__,
            inputs=lazy_spec,
            outputs=outputs,
        )
        for lzy_inpt in lazy_inputs:
            setattr(
                lazy_spec,
                lzy_inpt.name,
                LazyInField(
                    workflow=wf,
                    field=lzy_inpt.name,
                    type=lzy_inpt.type,
                ),
            )

        input_values = attrs_values(lazy_spec)
        constructor = input_values.pop("constructor")
        cls._under_construction = wf
        try:
            # Call the user defined constructor to set the outputs
            output_lazy_fields = constructor(**input_values)
            # Check to see whether any mandatory inputs are not set
            for node in wf.nodes:
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
                    # Automatically combine any uncombined state arrays into lists
                    if TypeParser.get_origin(outpt_lf.type) is StateArray:
                        outpt_lf.type = list[TypeParser.strip_splits(outpt_lf.type)[0]]
                    setattr(outputs, outpt.name, outpt_lf)
            else:
                if unset_outputs := [
                    a for a, v in attrs_values(outputs).items() if v is attrs.NOTHING
                ]:
                    raise ValueError(
                        f"Expected outputs {unset_outputs} to be set by the "
                        f"constructor of {wf!r}"
                    )
        finally:
            cls._under_construction = None

        cls._constructed[hash_key] = wf

        return wf

    @classmethod
    def clear_cache(cls):
        """Clear the cache of constructed workflows"""
        cls._constructed.clear()

    def add(self, task_spec: TaskDef[OutputsType], name=None) -> OutputsType:
        """Add a node to the workflow

        Parameters
        ----------
        task_spec : TaskDef
            The definition of the task to add to the workflow as a node
        name : str, optional
            The name of the node, by default it will be the name of the task definition
            class

        Returns
        -------
        OutputType
            The outputs definition of the node
        """
        if name is None:
            name = type(task_spec).__name__
        if name in self._nodes:
            raise ValueError(f"Node with name {name!r} already exists in the workflow")
        node = Node[OutputsType](name=name, definition=task_spec, workflow=self)
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

    @property
    @classmethod
    def under_construction(cls) -> "Workflow[ty.Any]":
        if cls._under_construction is None:
            raise ValueError(
                "pydra.design.workflow.this() can only be called from within a workflow "
                "constructor function (see 'pydra.design.workflow.define')"
            )
        return cls._under_construction

    # Used to store the workflow that is currently being constructed
    _under_construction: "Workflow[ty.Any]" = None
    # Used to cache the constructed workflows by their hashed input values
    _constructed: dict[int, "Workflow[ty.Any]"] = {}

    def execution_graph(self, submitter: "Submitter") -> DiGraph:
        from pydra.engine.submitter import NodeExecution

        return self._create_graph([NodeExecution(n, submitter) for n in self.nodes])

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
                    if lf.name != self.name:
                        # checking if the connection is already in the graph
                        if (self[lf.name], node) not in graph.edges:
                            graph.add_edges((self[lf.name], node))
                        if detailed:
                            graph.add_edges_description(
                                (node.name, field.name, lf.name, lf.field)
                            )
                        logger.debug("Connecting %s to %s", lf.name, node.name)
                        # adding a state from the previous task to other_states
                        if (
                            self[lf.name].state
                            and self[lf.name].state.splitter_rpn_final
                        ):
                            # variables that are part of inner splitters should be
                            # treated as a containers
                            if (
                                node.state
                                and f"{node.name}.{field.name}" in node.state.splitter
                            ):
                                node._inner_cont_dim[f"{node.name}.{field.name}"] = 1
                            # adding task_name: (task.state, [a field from the connection]
                            if lf.name not in other_states:
                                other_states[lf.name] = (
                                    self[lf.name].state,
                                    [field.name],
                                )
                            else:
                                # if the task already exist in other_state,
                                # additional field name should be added to the list of fields
                                other_states[lf.name][1].append(field.name)
                    else:  # LazyField with the wf input
                        # connections with wf input should be added to the detailed graph description
                        if detailed:
                            graph.add_edges_description(
                                (node.name, field.name, lf.name, lf.field)
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

    def create_dotfile(self, type="simple", export=None, name=None, output_dir=None):
        """creating a graph - dotfile and optionally exporting to other formats"""
        outdir = output_dir if output_dir is not None else self.cache_dir
        graph = self.graph
        if not name:
            name = f"graph_{self.name}"
        if type == "simple":
            for task in graph.nodes:
                self.create_connections(task)
            dotfile = graph.create_dotfile_simple(outdir=outdir, name=name)
        elif type == "nested":
            for task in graph.nodes:
                self.create_connections(task)
            dotfile = graph.create_dotfile_nested(outdir=outdir, name=name)
        elif type == "detailed":
            # create connections with detailed=True
            for task in graph.nodes:
                self.create_connections(task, detailed=True)
            # adding wf outputs
            for wf_out, lf in self._connections:
                graph.add_edges_description((self.name, wf_out, lf.name, lf.field))
            dotfile = graph.create_dotfile_detailed(outdir=outdir, name=name)
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
                formatted_dot.append(graph.export_graph(dotfile=dotfile, ext=ext))
            return dotfile, formatted_dot


def is_task(obj):
    """Check whether an object looks like a task."""
    return hasattr(obj, "_run_task")


def is_workflow(obj):
    """Check whether an object is a :class:`Workflow` instance."""
    from pydra.engine.specs import WorkflowDef
    from pydra.engine.core import Workflow
    from pydra.engine.core import WorkflowTask

    return isinstance(obj, (WorkflowDef, WorkflowTask, Workflow))


def has_lazy(obj):
    """Check whether an object has lazy fields."""
    for f in attrs_fields(obj):
        if is_lazy(getattr(obj, f.name)):
            return True
    return False
