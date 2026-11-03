"""Basic processing graph elements."""

import json
import logging
import os
import inspect
import sys
from pathlib import Path
import typing as ty
from uuid import uuid4
from traceback import format_exception
import attr
import cloudpickle as cp
from pydra.compose.base import Task
from pydra.utils.hash import hash_function
from datetime import datetime
from fileformats.core import FileSet
from pydra.engine.hooks import TaskHooks
from pydra.engine.context import job_context
from pydra.engine.store import remove_from_cache
from pydra.engine.result import (
    RuntimeSpec,
    Result,
    record_error,
)
from pydra.utils.general import (
    attrs_values,
    attrs_fields,
    get_fields,
    is_workflow,
)
from pydra.utils.typing import is_lazy
from pydra.engine.result import save
from pydra.utils.typing import copy_nested_files
from pydra.compose.shell.templating import template_update
from pydra.utils.messenger import AuditFlag
from pydra.environments.base import Environment
from pydra.utils.resources import ComputeResources
from pydra.utils.etelemetry import check_latest_version

logger = logging.getLogger("pydra")


if ty.TYPE_CHECKING:
    from pydra.engine.submitter import Submitter
    from pydra.compose.base import Arg

TaskType = ty.TypeVar("TaskType", bound=Task)


class Job(ty.Generic[TaskType]):
    """
    A base structure for the nodes in the processing graph.

    Tasks are a generic compute step from which both elementary tasks and
    :class:`Workflow` instances inherit.

    """

    _api_version: str = "0.0.1"  # Should generally not be touched by subclasses
    _etelemetry_checked: ty.ClassVar[bool] = False
    _version: str  # Version of tool being wrapped
    _task_version: ty.Optional[str] = None
    # Job writers encouraged to define and increment when implementation changes sufficiently
    _input_sets = None  # Dictionaries of predefined input settings

    audit_flags: AuditFlag = AuditFlag.NONE
    """What to audit -- available flags: :class:`~pydra.utils.messenger.AuditFlag`."""

    _can_resume = False  # Does the job allow resuming from previous state
    _redirect_x = False  # Whether an X session should be created/directed

    _runtime_requirements = RuntimeSpec()
    _runtime_hints = None

    _cache_root = None  # Working directory in which to operate
    _references = None  # List of references for a job

    name: str
    task: TaskType
    submitter: "Submitter | None"
    environment: "Environment | None"
    state_index: int
    bindings: dict[str, ty.Any] | None = None  # Bindings for the job environment

    _inputs: dict[str, ty.Any] | None = None
    _run_start_time: datetime | None

    def __init__(
        self,
        task: TaskType,
        submitter: "Submitter",
        name: str,
        environment: "Environment | None" = None,
        state_index: int | None = None,
        hooks: TaskHooks | None = None,
        resources: ComputeResources | None = None,
    ):
        """
        Initialize a job.

        Jobs allow for caching (retrieving a previous result of the same
        task and inputs), and concurrent execution.
        Running tasks follows a decision flow:

            1. Check whether prior cache exists --
               if ``True``, return cached result
            2. Check whether other process is running this job --
               wait if ``True``:
               a. Finishes (with or without exception) -> return result
               b. Gets killed -> restart
            3. No cache or other process -> start
            4. Two or more concurrent new processes get to start
        """

        if not isinstance(task, Task):
            raise ValueError(f"Job task ({task!r}) must be a Task, not {type(task)}")
        # Check that the task is fully resolved and ready to run
        task._check_resolved()
        task._check_rules()
        self.task = task
        # We save the submitter is the task is a workflow otherwise we don't
        # so the job can be pickled
        self.submitter = submitter
        self.environment = (
            environment if environment is not None else submitter.environment
        )
        self.name = name
        self.state_index = state_index
        self.resources = resources if resources is not None else task._resources

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
        # In-memory result cache: avoids reading back from disk the result that
        # was just written by run(). Not included in __getstate__ so it is
        # never pickled (subprocess workers will still use the disk path).
        self._cached_result = None
        # Set while the job is queued to run in this process's submitter and its run
        # has not finished
        self._awaiting_run = False
        self._context = None

        # Save the submitter attributes needed to run the job later
        self.audit = submitter.audit
        self.cache_root = submitter.cache_root
        self.store = submitter.store
        self._run_start_time = None

    @property
    def cache_root(self):
        return self._cache_root

    @property
    def is_async(self) -> bool:
        """Check to see if the job should be run asynchronously."""
        return is_workflow(self.task) and self.submitter.worker_for(self).is_async

    @cache_root.setter
    def cache_root(self, path: os.PathLike):
        self._cache_root = Path(path)

    def __str__(self):
        return self.name

    def __getstate__(self):
        # Carry the checksum, so a worker never rehashes the task's code
        self.checksum
        state = self.__dict__.copy()
        state["task"] = cp.dumps(state["task"])
        # Never serialise the in-memory result cache: subprocess workers must
        # go through the normal disk-based load_result() path.
        state.pop("_cached_result", None)
        state.pop("_awaiting_run", None)
        # Carry the submitting process's context, so a worker judges reuse as it would
        state["_context"] = self.context
        return state

    def __setstate__(self, state):
        state["task"] = cp.loads(state["task"])
        self.__dict__.update(state)
        # _cached_result is excluded from __getstate__; ensure it always exists
        # so that result() works correctly on deserialized jobs.
        if "_cached_result" not in self.__dict__:
            self._cached_result = None
        self.__dict__.setdefault("_awaiting_run", False)
        self.__dict__.setdefault("_context", None)

    @classmethod
    def check_etelemetry(cls) -> None:
        """Run the etelemetry version check at most once per session."""
        if not cls._etelemetry_checked:
            cls._etelemetry_checked = True
            check_latest_version()

    @property
    def errored(self):
        """Check if the job has raised an error"""
        return self._errored

    @property
    def checksum(self):
        """Calculates the unique checksum of the job.
        Used to create specific directory name for job that are run;
        and to create nodes checksums needed for graph checksums
        (before the tasks have inputs etc.)
        """
        if self._checksum is not None:
            return self._checksum
        self._checksum = self.task._checksum
        return self._checksum

    @property
    def context(self) -> dict[str, ty.Any]:
        """The execution context of the job (see `pydra.engine.context.job_context`),
        built once in the submitting process and carried with the job to the worker. It
        selects the job's slot in the store, is recorded on its result and is what a
        later job's context is compared with, so every process uses this one object."""
        if self._context is None:
            self.checksum  # collects the library modules the task's code reaches
            self._context = job_context(self.task, self.environment)
        return self._context

    @property
    def lockfile(self):
        return self.store.lockfile(self.checksum, self.context)

    @property
    def uid(self):
        """the unique id number for the job
        It will be used to create unique names for slurm scripts etc.
        without a need to run checksum
        """
        return self._uid

    @property
    def output_names(self):
        """Get the names of the outputs from the job's output_spec"""
        return [f.name for f in attr.fields(self.task.Outputs)]

    @property
    def can_resume(self):
        """Whether the job accepts checkpoint-restart."""
        return self._can_resume

    @property
    def cache_dir(self):
        """Get the filesystem path where outputs will be written."""
        return self.store.cache_dir(self.checksum, self.context)

    @property
    def inputs(self) -> dict[str, ty.Any]:
        """Resolve any template inputs of the job ahead of its execution:

        - links/copies upstream files and directories into the destination tasks
          working directory as required select state array values corresponding to
          state index (it will try to leave them where they are unless specified or
          they are on different file systems)
        - resolve template values (e.g. output_file_template)
        - deepcopy all inputs to guard against in-place changes during the job's
          execution (they will be replaced after the job's execution with the
          original inputs to ensure the tasks checksums are consistent)
        """
        if self._inputs is not None:
            return self._inputs

        from pydra.utils.typing import TypeParser

        self._inputs = {
            k: v for k, v in attrs_values(self.task).items() if not k.startswith("_")
        }
        map_copyfiles = {}
        fld: "Arg"
        for fld in get_fields(self.task):
            name = fld.name
            value = self._inputs[name]
            if value and TypeParser.contains_type(FileSet, fld.type):
                copied_value = copy_nested_files(
                    value=value,
                    dest_dir=self.cache_dir,
                    mode=fld.copy_mode,
                    collation=fld.copy_collation,
                    supported_modes=self.SUPPORTED_COPY_MODES,
                )
                if value is not copied_value:
                    map_copyfiles[name] = copied_value
        self._inputs.update(
            template_update(
                self.task, cache_dir=self.cache_dir, map_copyfiles=map_copyfiles
            )
        )
        return self._inputs

    def _populate_filesystem(self):
        """
        Invoked immediately after the lockfile is generated, this function:
        - Creates the cache file
        - Clears existing outputs if `can_resume` is False
        - Generates a fresh output directory

        Created as an attempt to simplify overlapping `Job`|`Workflow` behaviors.
        """
        # adding info file with the checksum in case the job was cancelled
        # and the lockfile has to be removed
        with open(self.cache_root / f"{self.uid}_info.json", "w") as jsonfile:
            json.dump({"checksum": self.checksum}, jsonfile)
        if not self.can_resume and self.cache_dir.exists():
            remove_from_cache(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=self.can_resume)
        # Save job pkl into the output directory for future reference
        save(self.cache_dir, job=self)

    def run(self, rerun: bool = False):
        """Prepare the job working directory, execute the task, and save the
        results.

        Parameters
        ----------
        rerun : bool
            If True, the job will be re-run even if a result already exists. Will
            propagated to all tasks within workflow tasks.
        """
        # TODO: After these changes have been merged, will refactor this function and
        # run_async to use common helper methods for pre/post run tasks

        # checking if the task is fully resolved and ready to run
        self.hooks.pre_run(self)
        lease = self.store.acquire(self.checksum, self.context)
        logger.debug("'%s' is attempting to acquire %s", self.name, lease)
        with lease:
            if not (rerun):
                result = self.result()
                if result is not None and not result.errored:
                    self._awaiting_run = False
                    return result
            self._errored = False
            cwd = os.getcwd()
            self._populate_filesystem()
            os.chdir(self.cache_dir)
            result = Result(
                outputs=None,
                runtime=None,
                errored=False,
                cache_dir=self.cache_dir,
                task=self.task,
                context=self.context,
            )
            self.hooks.pre_run_task(self)
            self.audit.start_audit(odir=self.cache_dir)
            if self.audit.audit_check(AuditFlag.PROV):
                self.audit.audit_task(job=self)
            try:
                self.audit.monitor()
                self.task._run(self, rerun)
                result.outputs = self.task.Outputs._from_job(self)
            except Exception:
                etype, eval, etr = sys.exc_info()
                traceback = format_exception(etype, eval, etr)
                record_error(self.cache_dir, error=traceback)
                result.errored = True
                raise
            finally:
                try:
                    self.hooks.post_run_task(self, result)
                    self.audit.finalize_audit(result=result)
                    self.store.commit(lease, result, job=self)
                    self._awaiting_run = False
                    self._remove_info_file()
                finally:
                    os.chdir(cwd)
        self.hooks.post_run(self, result)
        # Check for any changes to the input hashes that have occurred during the execution
        # of the job
        self._check_for_hash_changes()
        # Cache the completed result in memory so that callers who immediately
        # call job.result() (e.g. Submitter.__call__) do not need to deserialise
        # it back from disk.
        self._cached_result = result
        return result

    async def run_async(self, rerun: bool = False) -> Result:
        """Prepare the job working directory, execute the task asynchronously,
        and save the results. NB: only workflows are run asynchronously at the moment.

        Parameters
        ----------
        rerun : bool
            If True, the job will be re-run even if a result already exists. Will
            propagated to all tasks within workflow tasks.
        """
        # checking if the task is fully resolved and ready to run
        self.hooks.pre_run(self)
        lease = self.store.acquire(self.checksum, self.context)
        logger.debug("'%s' is attempting to acquire %s", self.name, lease)
        async with lease:
            if not rerun:
                result = self.result()
                if result is not None and not result.errored:
                    self._awaiting_run = False
                    return result
            self._errored = False
            cwd = os.getcwd()
            self._populate_filesystem()
            result = Result(
                outputs=None,
                runtime=None,
                errored=False,
                cache_dir=self.cache_dir,
                task=self.task,
                context=self.context,
            )
            self.hooks.pre_run_task(self)
            self.audit.start_audit(odir=self.cache_dir)
            try:
                self.audit.monitor()
                await self.task._run_async(self, rerun)
                result.outputs = self.task.Outputs._from_job(self)
            except Exception:
                etype, eval, etr = sys.exc_info()
                traceback = format_exception(etype, eval, etr)
                record_error(self.cache_dir, error=traceback)
                result.errored = True
                self._errored = True
                raise
            finally:
                try:
                    self.hooks.post_run_task(self, result)
                    self.audit.finalize_audit(result=result)
                    self.store.commit(lease, result, job=self)
                    self._awaiting_run = False
                    self._remove_info_file()
                finally:
                    os.chdir(cwd)
        self.hooks.post_run(self, result)
        # Check for any changes to the input hashes that have occurred during the execution
        # of the job
        self._check_for_hash_changes()
        return result

    def _remove_info_file(self):
        """Remove the job's ``<uid>_info.json``; one that cannot be removed is logged
        and left in place."""
        info_file = self.cache_root / f"{self.uid}_info.json"
        try:
            remove_from_cache(info_file)
        except OSError as e:
            logger.warning("Could not remove %s: %s", info_file, e)

    def pickle_task(self):
        """Pickling the tasks with full inputs"""
        pkl_files = self.cache_root / "pkl_files"
        pkl_files.mkdir(exist_ok=True, parents=True)
        task_main_path = pkl_files / f"{self.name}_{self.uid}_job.pklz"
        save(task_path=pkl_files, job=self, name_prefix=f"{self.name}_{self.uid}")
        return task_main_path

    @property
    def done(self):
        """Check whether the tasks has been finalized and all outputs are stored."""
        # if any of the field is lazy, there is no need to check results
        if has_lazy(self.task):
            return False
        if self._awaiting_run:
            return False
        _result = self.result()
        if _result:
            if _result.errored:
                self._errored = True
                raise ValueError(f"Job {self.name!r} failed")
            else:
                return True
        return False

    @property
    def run_start_time(self) -> datetime | None:
        """Check whether the job is currently running."""
        if self._run_start_time is not None:
            return self._run_start_time
        try:
            stat = self.lockfile.stat()
        except FileNotFoundError:
            return None
        self._run_start_time = datetime.fromtimestamp(stat.st_ctime)
        return self._run_start_time

    def result(self, return_inputs=False):
        """
        Retrieve the outcomes of this particular job.

        Parameters
        ----------
        state_index : :obj: `int`
            index of the element for job with splitter and multiple states
        return_inputs : :obj: `bool`, :obj:`str`
            if True or "val" result is returned together with values of the input fields,
            if "ind" result is returned together with indices of the input fields

        Returns
        -------
        result : Result
            the result of the job
        """
        if self.errored:
            return Result(
                outputs=None,
                runtime=None,
                errored=True,
                cache_dir=self.cache_dir,
                task=self.task,
            )

        # Fast path: return the in-memory result cached by run() to avoid
        # deserialising from disk for callers in the same process.
        if self._cached_result is not None and not return_inputs:
            return self._cached_result

        result = self.store.lookup(self.checksum, self.context, task=self.task)
        if result and result.errored:
            self._errored = True
        if return_inputs is True or return_inputs == "val":
            inputs_val = {
                f"{self.name}.{inp}": getattr(self.task, inp)
                for inp in self.input_names
            }
            return (inputs_val, result)
        elif return_inputs == "ind":
            inputs_ind = {f"{self.name}.{inp}": None for inp in self.input_names}
            return (inputs_ind, result)
        else:
            return result

    def _check_for_hash_changes(self):
        from pydra.utils.typing import TypeParser

        if hasattr(self.task, "__checksum__"):
            if (checksum := self.task._checksum) != self.checksum:
                raise RuntimeError(
                    f"The checksum supplied by the '{self.name}' job of {type(self)} type "
                    f"changed during its execution, from {self.checksum!r} to {checksum!r}"
                )
            return
        # For tasks whose input fields contain no FileSet types and no values
        # with custom __bytes_repr__ methods, hashes cannot change during
        # execution (scalar/pure-Python values are immutable from Pydra's
        # perspective). Skip the expensive full recomputation in that common case.
        if not any(
            TypeParser.contains_type(FileSet, f.type)
            or hasattr(getattr(self.task, f.name), "__bytes_repr__")
            for f in get_fields(self.task)
        ):
            logger.debug(
                "Input values and hashes for '%s' %s node:\n%s\n%s",
                self.name,
                type(self).__name__,
                self.task,
                self.task._hashes,
            )
            return
        hash_changes = self.task._hash_changes()
        details = ""
        for changed in hash_changes:
            field = getattr(attr.fields(type(self.task)), changed)
            hash_function(getattr(self.task, changed))
            val = getattr(self.task, changed)
            field_type = type(val)
            if inspect.isclass(field.type) and issubclass(field.type, FileSet):
                details += (
                    f"- {changed}: value passed to the {field.type} field is of type "
                    f"{field_type} ('{val}'). If it is intended to contain output data "
                    "then the type of the field in the interface class should be changed "
                    "to `pathlib.Path`. Otherwise, if the field is intended to be an "
                    "input field but it gets altered by the job in some way, then the "
                    "'copyfile' flag should be set to 'copy' in the field metadata of "
                    "the job interface class so copies of the files/directories in it "
                    "are passed to the job instead.\n"
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
                f"'{self.name}' job of {type(self)} type.\n\n{details}"
            )
        logger.debug(
            "Input values and hashes for '%s' %s node:\n%s\n%s",
            self.name,
            type(self).__name__,
            self.task,
            self.task._hashes,
        )

    def _write_notebook(self):
        """Writes a notebook into the"""
        raise NotImplementedError

    SUPPORTED_COPY_MODES = FileSet.CopyMode.any
    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.any


def has_lazy(obj):
    """Check whether an object has lazy fields."""
    for f in attrs_fields(obj):
        if is_lazy(getattr(obj, f.name)):
            return True
    return False


def load_and_run(job_pkl: Path, rerun: bool = False) -> Path:
    """
    loading a job from a pickle file, settings proper input
    and running the job

    Parameters
    ----------
    job_pkl : :obj:`Path`
        The path to pickled job file

    Returns
    -------
    resultfile : :obj:`Path`
        The path to the pickled result file
    """

    try:
        job: Job[TaskType] = load_job(job_pkl=job_pkl)
    except Exception:
        if job_pkl.parent.exists():
            etype, eval, etr = sys.exc_info()
            traceback = format_exception(etype, eval, etr)
            errorfile = record_error(job_pkl.parent, error=traceback)
            result = Result(output=None, runtime=None, errored=True, task=None)
            save(job_pkl.parent, result=result)
        raise

    resultfile = job.cache_dir / "_result.pklz"
    try:
        if job.is_async:
            job.submitter.submit(job, rerun=rerun)
        else:
            job.run(rerun=rerun)
    except Exception as e:
        # creating result and error files if missing
        errorfile = job.cache_dir / "_error.pklz"
        if not errorfile.exists():  # not sure if this is needed
            etype, eval, etr = sys.exc_info()
            traceback = format_exception(etype, eval, etr)
            errorfile = record_error(job.cache_dir, error=traceback)
        if not resultfile.exists():  # not sure if this is needed
            result = Result(output=None, runtime=None, errored=True, task=None)
            save(job.cache_dir, result=result)
        e.add_note(f" full crash report is here: {errorfile}")
        raise
    return resultfile


def load_job(job_pkl: os.PathLike) -> "Job[TaskType]":
    """loading a job from a pickle file, settings proper input for the specific ind"""
    with open(job_pkl, "rb") as fp:
        job = cp.load(fp)
    return job
