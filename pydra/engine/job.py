"""Basic processing graph elements."""

import json
import logging
import os
import inspect
import sys
import warnings
from pathlib import Path
import typing as ty
from uuid import uuid4
from traceback import format_exception
import attr
import cloudpickle as cp
from pydra.compose.base import Task
from datetime import datetime
from fileformats.core import FileSet
from pydra.engine.hooks import TaskHooks
from pydra.engine.absent import absence_suffix, absence_triples
from pydra.engine.context import job_context
from pydra.engine.lease import LeaseLost, LeaseTimeout
from pydra.engine.store import (
    check_declared_code,
    lease_still_held,
    remove_from_cache,
)
from pydra.utils.storage import StorageUnsupported
from pydra.engine.result import (
    RuntimeSpec,
    Result,
    record_error,
)
from pydra.utils.general import (
    capture_cwd,
    release_dir,
    atomic_write,
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
    from pydra.engine.portable.store import Committed
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
    # The flat index of the job's element within its split, None without a split, and
    # None on a partial node until the node is sealed
    state_index: int | None
    # The key of the job's element on a partial node (see ``NodeExecution``), else None
    state_key: tuple | None = None
    rerun: bool = False  # Set by the graph runner when it hands the job out
    # The names of the nested workflows that hold the job, outermost first, with the
    # element's index where one is split; set by the graph runner
    workflow_path: tuple[str, ...] = ()
    # The id of this attempt at the job (see the attribute's docstring below)
    attempt: str | None = None
    # Whether the audit of a workflow job was started by the runner, so that it is
    # finalised with the job
    _audit_started: bool = False
    bindings: dict[str, ty.Any] | None = None  # Bindings for the job environment

    _inputs: dict[str, ty.Any] | None = None
    _run_start_time: datetime | None

    attempt: str | None = None
    """The id of this attempt at the job (RFC 0003 section 5.4): a fresh UUID each time the
    runner hands it out (again after ``retry``), given to a workflow job when its expansion
    starts and to ``Submitter``'s outer job when it is created. Under a store that has
    ``work_dir`` it names the job's work directory, except for a task that can resume, whose
    directory is the stable ``work/resume-<hash>`` of its address; its outcome files there
    carry the attempt. None for a job never handed out (settled or stored)."""
    committed: "Committed | None" = None
    """What this process's `PortableStore` publish returned for the job."""

    def __init__(
        self,
        task: TaskType,
        submitter: "Submitter",
        name: str,
        environment: "Environment | None" = None,
        state_index: int | None = None,
        hooks: TaskHooks | None = None,
        resources: ComputeResources | None = None,
        state_key: tuple | None = None,
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
        self.state_key = state_key
        self.resources = resources if resources is not None else task._resources

        self.return_values = {}
        self._result = {}
        # flag that says if node finished all jobs
        self._done = False
        if self._input_sets is None:
            self._input_sets = {}

        self.allow_cache_override = True
        self._checksum = None
        self._unsuffixed_checksum = None
        # The (field, element index, cause) of every Absent an input field accepted or
        # dropped, set by the graph runner when it forms the job
        self._accepted_absent: list[tuple[str, int, str]] = []
        # The Absent the job completes with instead of running, set likewise
        self._propagated_absent = None
        # Whether that Absent is a gate's "not-routed" rather than a propagated one
        self._gated = False
        self._uid = uuid4().hex
        self.hooks = hooks if hooks is not None else TaskHooks()
        self._errored = False
        self._lzout = None
        # In-memory result cache: avoids reading back from disk the result that
        # was just written by run(). Not included in __getstate__ so it is
        # never pickled (subprocess workers will still use the disk path).
        self._cached_result = None
        # The result reported by ``GraphRunner.complete``, as given
        self._held = None
        # Set while the job is queued to run in this process's submitter and its run
        # has not finished
        self._awaiting_run = False
        self._context = None
        # Whether `_populate_filesystem` and `store.commit` write ``_job.pklz`` for a
        # job that succeeds. In-process workers clear it; a failed job always keeps
        # its pickle, which `pydracli crash --rerun` reads.
        self.persist_pickle = True
        # The stored contexts of the checksum read by `check_declared_code`, for the
        # job's first lookup
        self._stored_contexts = None
        self.attempt = None
        self.committed = None
        # The stored entry the job runs against, set from a `Verify` by `_reusable` or by a
        # runner (`Candidate | None`; RFC 0003 section 7.1)
        self.verify = None
        # A rerun under a store that replaces: set by the store's ``begin_rerun`` (RFC 0003
        # section 10); ``replaces`` is the output digest of the entry it evicts
        self.replace = False
        self.replaces = None
        self._work_dir_memo = None

        # Save the submitter attributes needed to run the job later
        self.audit = submitter.audit
        self.chdir = submitter.chdir
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
        # The task stays an object in the state, so it is serialised in the job's own
        # stream, and jobs pickled together (a workflow's node jobs) share one memo.
        # A Job therefore requires cloudpickle: the stdlib pickle cannot serialise the
        # task class, so a worker plugin must hand a Job over as ``cloudpickle.dumps(job)``
        state = self.__dict__.copy()
        # Never serialise the in-memory result cache: subprocess workers must
        # go through the normal disk-based load_result() path.
        state.pop("_cached_result", None)
        state.pop("_awaiting_run", None)
        if self._propagated_absent is not None:
            # completed absent: its result is in memory only, and it has no context
            return state
        state.pop("_held", None)
        # Carry the submitting process's context, so a worker judges reuse as it would
        state["_context"] = self.context
        return state

    def __setstate__(self, state):
        if isinstance(
            state["task"], bytes
        ):  # a job pickled before the task was inlined
            state["task"] = cp.loads(state["task"])
        self.__dict__.update(state)
        # _cached_result is excluded from __getstate__; ensure it always exists
        # so that result() works correctly on deserialized jobs.
        if "_cached_result" not in self.__dict__:
            self._cached_result = None
        self.__dict__.setdefault("_held", None)
        self.__dict__.setdefault("_awaiting_run", False)
        self.__dict__.setdefault("persist_pickle", True)
        self.__dict__.setdefault("_context", None)
        self.__dict__.setdefault("chdir", True)
        self.__dict__.setdefault("_stored_contexts", None)
        self.__dict__.setdefault("attempt", None)
        self.__dict__.setdefault("committed", None)
        self.__dict__.setdefault("verify", None)
        self.__dict__.setdefault("replace", False)
        self.__dict__.setdefault("replaces", None)
        self.__dict__.setdefault("_work_dir_memo", None)
        self.__dict__.setdefault("_unsuffixed_checksum", None)
        self.__dict__.setdefault("_accepted_absent", [])
        self.__dict__.setdefault("_propagated_absent", None)
        self.__dict__.setdefault("_gated", False)

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
        self._checksum = self.unsuffixed_checksum + absence_suffix(self.absent)
        return self._checksum

    @property
    def unsuffixed_checksum(self) -> str:
        """The checksum of the task, without the absence suffix of
        :attr:`checksum` (RFC 0002 section 2.3(a))"""
        if self._unsuffixed_checksum is None:
            self._unsuffixed_checksum = self.task._checksum
        return self._unsuffixed_checksum

    @property
    def absent(self) -> list[tuple[str, int, str]]:
        """The sorted ``(field, element index, cause)`` of every accepted or dropped
        ``Absent`` the task's checksum may not see (see
        :func:`~pydra.engine.absent.absence_triples`); their suffix is appended to
        :attr:`checksum`"""
        return absence_triples(self)

    @property
    def context(self) -> dict[str, ty.Any]:
        """The execution context of the job (see `pydra.engine.context.job_context`),
        built once in the submitting process and carried with the job to the worker. It
        selects the job's slot in the store, is recorded on its result and is what a
        later job's context is compared with, so every process uses this one object.
        Building it reports a declared version whose stored results were computed from
        other code (see `pydra.engine.store.check_declared_code`). A store with an
        ``identity`` supplies the context (``identity.context(job, "lookup")``)."""
        if self._context is None:
            self.checksum  # collects the library modules the task's code reaches
            identity = self.store.identity
            if identity is None:
                self._context = job_context(self.task, self.environment)
            else:
                self._context = identity.context(self, "lookup")
            self._stored_contexts = check_declared_code(
                self.store, self.checksum, self._context, self.task
            )
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
    def _has_work_dir(self) -> bool:
        """Whether the store runs jobs in work directories of their own, apart from the
        entries it commits (a `~pydra.engine.portable.store.PortableStore`)."""
        return self.store.has_work_dirs

    @property
    def cache_dir(self):
        """Get the filesystem path where outputs will be written: the store's work
        directory for the job if it defines one (a pure function of the checksum, the
        context and `attempt`), otherwise the job's slot."""
        if not self._has_work_dir:
            return self.store.cache_dir(self.checksum, self.context)
        memo = self._work_dir_memo
        key = (self.attempt, self.can_resume)
        if memo is None or memo[0] != key:
            memo = (key, self.store.work_dir(self))
            self._work_dir_memo = memo
        return memo[1]

    @property
    def record_dir(self) -> Path | None:
        """Where the job's audit messages and resource logs go if not its working directory:
        under a store with work directories, which empties the work directory once the result
        is published, the store's ``audit_dir`` (``<root>/audit/<cdir>/<fingerprint>/<attempt>``).
        """
        return self.store.audit_dir(self)

    def _make_record_dir(self) -> Path | None:
        if not self.audit.audit_flags:
            return None
        return self.store.make_audit_dir(self)

    @property
    def error_file(self) -> Path:
        """Where a failure of the job is recorded (the store's `error_file`)"""
        return self.store.error_file(self)

    def _may_record_error(self, lease) -> bool:
        """Whether a failure may be recorded beside the job: always in an attempt's own
        work directory, which no other run writes to; in a slot, or the work directory a
        resumable job shares across runs, only while ``lease`` is still held."""
        if self._has_work_dir and not self.can_resume:
            return True
        return lease_still_held(lease)

    def _record_error(self, error, exception: BaseException | None = None) -> Path:
        """Record ``error`` (a formatted traceback) and ``exception`` beside the job."""
        return self.store.record_error(self, error, exception)

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

    def _withdraw_success(self, lease) -> None:
        """Remove the success record of the job's slot, under ``lease``, before the
        slot is rewritten, so the submitter never reads it without the lease while it
        is rewritten (a store without records has nothing to remove)."""
        self.store.withdraw_success(lease)

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
        with atomic_write(self.cache_root / f"{self.uid}_info.json", "w") as jsonfile:
            json.dump({"checksum": self.checksum}, jsonfile)
        self.store.begin_run(self)
        # Save job pkl into the output directory for future reference. An in-process
        # worker never reads it back, so it is written only if the job fails.
        if self.persist_pickle:
            save(self.cache_dir, job=self)

    @property
    def _enters_cache_dir(self) -> bool:
        """Whether the job runs with the process working directory set to its cache
        directory: unless the submitter was given ``chdir=False``."""
        return self.chdir

    def _warn_if_outputs_outside_cache_dir(self, outputs, cwd) -> None:
        """Under ``chdir=False``, warn about FileSet outputs of a Python task that are
        in the caller's directory ``cwd`` rather than the cache directory: a relative
        path written by the function, which other jobs can overwrite."""
        if self.chdir or self.task._executor_name != "function":
            return
        root = self.cache_dir.resolve()
        caller = Path(cwd).resolve()
        for name in outputs:
            vals = getattr(outputs, name, None)
            for val in vals if isinstance(vals, (list, tuple)) else [vals]:
                if not isinstance(val, FileSet):
                    continue
                for path in val.fspaths:
                    resolved = Path(path).resolve()
                    if resolved.is_relative_to(caller) and not resolved.is_relative_to(
                        root
                    ):
                        warnings.warn(
                            f"Output {name!r} of {self.name!r} is {str(path)!r}, in the "
                            f"caller's directory {str(caller)!r} rather than its cache "
                            f"directory {str(root)!r}: with chdir=False a relative path "
                            "is written in the caller's directory, where other jobs can "
                            "overwrite it and corrupt cache hits"
                        )

    def _pickle_to_commit(self, result: Result) -> "Job | None":
        """The job to write beside ``result``: this job if it is to be persisted or
        has failed, otherwise None. A store with work directories is always given the
        job, which names the directory; it never writes the job to an entry."""
        if self._has_work_dir:
            return self
        return self if self.persist_pickle or result.errored else None

    def _take_attempt(self) -> None:
        """Give a job that is run without having been handed out by a runner (``job.run()``
        called directly) an attempt of its own, so that under a store with work directories
        it has one to run in."""
        if self.attempt is None and self._has_work_dir:
            self.attempt = uuid4().hex

    def _acquire(self):
        """The lease guarding the job; a resumable job's work directory is stable across
        runs, so a store with work directories guards it whatever its de-duplication."""
        if self.can_resume:
            return self.store.acquire(self.checksum, self.context, resumable=True)
        return self.store.acquire(self.checksum, self.context)

    def _reusable(self) -> Result | None:
        """The result a run under the lease can serve instead of computing: the stored
        result for the job, if any. A store with ``find`` answers through its reuse hook
        (RFC 0003 section 7.1): a `Verify` marks the job to verify against that entry (and
        the job runs), anything else clears the mark. A work directory's ``_committed.json``
        is the outcome of an earlier run and is never served to a new one."""
        if self.store.finds:
            result = self.accept_found(self.store.find(self), errored=True)
            if result is not None:  # a reader in another process finds what was served
                self.store.record_served(self, result)
            return result
        return self.result()

    def accept_found(self, found: ty.Any, *, errored: bool = False) -> Result | None:
        """Take what the store's ``find`` answered for the job (None: the store could not
        be read): the candidate of a `Verify` as `verify`, cleared otherwise, and the
        result of a `Hit`, keeping its `Committed`; an errored result only with
        ``errored``. None when there is no result to serve."""
        if found is None:
            self.verify = None
            return None
        self.verify = found.candidate
        result = found.result
        if result is None or (result.errored and not errored):
            return None
        if result.committed is not None:
            self.committed = result.committed
        return result

    def _begin_rerun(self, rerun: bool) -> None:
        """Under a store that replaces a stored result on a rerun (`PortableStore`), mark the
        job so that its commit evicts the entry it read and takes its place; a job that is not
        a rerun is not marked."""
        if rerun:
            self.store.begin_rerun(self)
        else:
            self.replace = False
            self.replaces = None

    def _published(self, result: Result) -> Result:
        """The result of a run, as the job reads it from here on: under a store with
        work directories its outputs are read from the entry (or the kept outputs), which
        is where their files are once the work directory is emptied."""
        if not self._has_work_dir or result.errored or result.committed is None:
            return result
        self.committed = result.committed
        if self.audit.audit_check(AuditFlag.PROV):
            ref = result.committed.entry
            self.audit.audit_message(
                {
                    "@type": "publication",
                    "checksum": self.checksum,
                    "attempt": self.attempt,
                    "entry": {
                        "root": ref.root,
                        "cdir": ref.cdir,
                        "fingerprint": ref.fingerprint,
                    },
                    "stored": result.committed.stored,
                },
                AuditFlag.PROV,
            )
        read = self.store.read_committed(self, result.committed)
        if read is None:
            return result
        read.runtime = result.runtime  # this run's measurement; an entry stores none
        return read

    def _start_workflow(self, lease) -> Result:
        """Begin the expansion of a workflow job whose lease is held and whose stored
        result, if any, is not to be used: withdraw its success record, prepare its
        directory, run the ``pre_run_task`` hook and start its audit. Returns the
        result the workflow's outputs are later put in (see `_finish_workflow`)."""
        self._errored = False
        self._audit_started = False
        self._begin_rerun(self.rerun)
        self._withdraw_success(lease)
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
        self.audit.start_audit(
            odir=self.cache_dir, chdir=False, record_dir=self._make_record_dir()
        )
        self._audit_started = True
        if self.audit.audit_check(AuditFlag.PROV):
            self.audit.audit_task(job=self)
        self.audit.monitor()
        return result

    def _finish_workflow(
        self, lease, result: Result | None, error: BaseException | None
    ) -> BaseException | None:
        """Commit the result of a workflow job whose nodes are done, under its lease,
        and run its ``post_run`` hook if it succeeded. Returns the error of the
        workflow (``error``, or the one raised building its outputs), None if it
        succeeded. Raises what the commit raises (`LeaseLost`), after which nothing is
        stored."""
        cwd = os.getcwd()
        if result is None:  # it failed while starting
            result = Result(
                outputs=None,
                runtime=None,
                errored=False,
                cache_dir=self.cache_dir,
                task=self.task,
                context=self.context,
            )
        if error is None:
            try:
                result.outputs = self.task.Outputs._from_job(self)
                self._warn_if_outputs_outside_cache_dir(result.outputs, cwd)
            except Exception as exc:
                error = exc
        if error is not None:
            traceback = format_exception(type(error), error, error.__traceback__)
            if self._may_record_error(lease):
                self._record_error(traceback, error)
            result.errored = True
            self._errored = True
        if self._audit_started:  # not if it failed before the audit
            self.hooks.post_run_task(self, result)
            self.audit.finalize_audit(result=result)
        self.store.commit(lease, result, job=self._pickle_to_commit(result))
        self._awaiting_run = False
        self._remove_info_file()
        return error

    def _complete_workflow(self, result: Result) -> None:
        """What follows the release of the lease of a workflow job that succeeded"""
        result = self._published(result)
        self.hooks.post_run(self, result)
        self._check_for_hash_changes()
        self._cached_result = result

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
        self._take_attempt()
        lease = self._acquire()
        logger.debug("'%s' is attempting to acquire %s", self.name, lease)
        with lease:
            if not (rerun):
                result = self._reusable()
                if result is not None and not result.errored:
                    self._awaiting_run = False
                    self._cached_result = result
                    return result
            self._begin_rerun(rerun)
            self._errored = False
            enters = self._enters_cache_dir
            # the directory restored to after the run, captured and held at once so that
            # nothing removes it meanwhile (jobs on threads share the working directory)
            cwd = capture_cwd() if enters else os.getcwd()
            try:
                self._withdraw_success(lease)
                self._populate_filesystem()
                if enters:
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
                self.audit.start_audit(
                    odir=self.cache_dir,
                    chdir=enters,
                    record_dir=self._make_record_dir(),
                )
                if self.audit.audit_check(AuditFlag.PROV):
                    self.audit.audit_task(job=self)
                try:
                    self.audit.monitor()
                    self.task._run(self, rerun)
                    result.outputs = self.task.Outputs._from_job(self)
                    self._warn_if_outputs_outside_cache_dir(result.outputs, cwd)
                except Exception:
                    etype, eval, etr = sys.exc_info()
                    traceback = format_exception(etype, eval, etr)
                    if self._may_record_error(lease):
                        self._record_error(traceback, eval)
                    result.errored = True
                    raise
                finally:
                    self.hooks.post_run_task(self, result)
                    self.audit.finalize_audit(result=result)
                    self.store.commit(lease, result, job=self._pickle_to_commit(result))
                    self._awaiting_run = False
                    self._remove_info_file()
            finally:
                if enters:
                    try:
                        os.chdir(cwd)
                    finally:
                        release_dir(cwd)
        result = self._published(result)
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
        self._take_attempt()
        lease = self._acquire()
        logger.debug("'%s' is attempting to acquire %s", self.name, lease)
        async with lease:
            if not rerun:
                result = self._reusable()
                if result is not None and not result.errored:
                    self._awaiting_run = False
                    self._cached_result = result
                    return result
            self._begin_rerun(rerun)
            self._errored = False
            enters = self._enters_cache_dir
            # the directory restored to after the run, captured and held at once so that
            # nothing removes it meanwhile (jobs on threads share the working directory)
            cwd = capture_cwd() if enters else os.getcwd()
            try:
                self._withdraw_success(lease)
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
                self.audit.start_audit(
                    odir=self.cache_dir,
                    chdir=enters,
                    record_dir=self._make_record_dir(),
                )
                try:
                    self.audit.monitor()
                    await self.task._run_async(self, rerun)
                    result.outputs = self.task.Outputs._from_job(self)
                    self._warn_if_outputs_outside_cache_dir(result.outputs, cwd)
                except Exception:
                    etype, eval, etr = sys.exc_info()
                    traceback = format_exception(etype, eval, etr)
                    if self._may_record_error(lease):
                        self._record_error(traceback, eval)
                    result.errored = True
                    self._errored = True
                    raise
                finally:
                    self.hooks.post_run_task(self, result)
                    self.audit.finalize_audit(result=result)
                    self.store.commit(lease, result, job=self._pickle_to_commit(result))
                    self._awaiting_run = False
                    self._remove_info_file()
            finally:
                if enters:
                    try:
                        os.chdir(cwd)
                    finally:
                        release_dir(cwd)
        result = self._published(result)
        self.hooks.post_run(self, result)
        # Check for any changes to the input hashes that have occurred during the execution
        # of the job
        self._check_for_hash_changes()
        if self._has_work_dir:
            self._cached_result = result
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

    def result(self, return_inputs=False, *, settled=False):
        """
        Retrieve the outcomes of this particular job.

        Parameters
        ----------
        state_index : :obj: `int`
            index of the element for job with splitter and multiple states
        return_inputs : :obj: `bool`, :obj:`str`
            if True or "val" result is returned together with values of the input fields,
            if "ind" result is returned together with indices of the input fields
        settled : bool
            Under a store with work directories: the job has been reported complete, so a
            result that is not found is a failure, reported as an errored result naming the
            work directory, after the store's visibility retry for ``_committed.json``
            (RFC 0003 section 7.2). Otherwise nothing found is None, a job not yet run.

        Returns
        -------
        result : Result
            the result of the job
        """
        if self._held is not None and not return_inputs:
            return self._held
        if self.errored:
            try:
                where = self.cache_dir
            except ValueError:  # a job never handed out has no work directory
                where = Path(self.store.path)
            return Result(
                outputs=None,
                runtime=None,
                errored=True,
                cache_dir=where,
                task=self.task,
            )

        # Fast path: return the in-memory result cached by run() to avoid
        # deserialising from disk for callers in the same process.
        if self._cached_result is not None and not return_inputs:
            return self._cached_result

        context, stored = self.context, self._stored_contexts
        if self._has_work_dir:
            result = self.store.read_result(self, settled=settled)
        elif stored is None:
            result = self.store.lookup(self.checksum, context, task=self.task)
        else:
            self._stored_contexts = None
            result = self.store.lookup(
                self.checksum, context, task=self.task, stored=stored
            )
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
            if (checksum := self.task._checksum) != self.unsuffixed_checksum:
                raise RuntimeError(
                    f"The checksum supplied by the '{self.name}' job of {type(self)} type "
                    f"changed during its execution, from {self.unsuffixed_checksum!r} to "
                    f"{checksum!r}"
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
            val = getattr(self.task, changed)
            field = getattr(attr.fields(type(self.task)), changed, None)
            if field is None:
                # Not an input field, e.g. the "Outputs" class, which is included in
                # the hash in case the names/types of the outputs change
                details += (
                    f"- {changed}: the hash of {val!r} appears to be unstable. This "
                    "could be due to a stochastic/non-thread-safe attribute(s) of it, "
                    "or a function/lambda referenced by it (e.g. an output callable) "
                    "that can't be consistently hashed\n"
                )
                continue
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
            errorfile = record_error(job_pkl.parent, error=traceback, exception=eval)
            result = Result(
                outputs=None,
                runtime=None,
                errored=True,
                task=None,
                cache_dir=job_pkl.parent,
            )
            save(job_pkl.parent, result=result)
        raise

    portable = job._has_work_dir
    job._take_attempt()  # a job pickled before a runner handed it out
    resultfile = job.cache_dir / "_result.pklz"
    try:
        if job.is_async:
            job.submitter.submit(job, rerun=rerun)
        else:
            job.run(rerun=rerun)
    except (LeaseTimeout, LeaseLost, StorageUnsupported):
        # the job's slot belongs to another process, or was never entered
        raise
    except Exception as e:
        # creating result and error files if missing
        errorfile = job.error_file
        if not errorfile.exists():  # not sure if this is needed
            etype, eval, etr = sys.exc_info()
            traceback = format_exception(etype, eval, etr)
            errorfile = job._record_error(traceback, eval)
        if not portable and not resultfile.exists():  # not sure if this is needed
            result = Result(
                outputs=None,
                runtime=None,
                errored=True,
                task=None,
                cache_dir=job.cache_dir,
            )
            save(job.cache_dir, result=result)
        e.add_note(f" full crash report is here: {errorfile}")
        raise
    return resultfile


def load_job(job_pkl: os.PathLike) -> "Job[TaskType]":
    """loading a job from a pickle file, settings proper input for the specific ind"""
    with open(job_pkl, "rb") as fp:
        job = cp.load(fp)
    return job
