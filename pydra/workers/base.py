"""Execution workers."""

import asyncio
import attrs
import sys
import abc
import inspect
import asyncio.subprocess as asp
import typing as ty
import logging
from pydra.engine.job import Job
from pydra.utils.general import get_plugin_classes
import pydra.workers

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result
    from pydra.compose import base

TaskType = ty.TypeVar("TaskType", bound="base.Task")


@attrs.define
class Worker(metaclass=abc.ABCMeta):
    """A base class for execution of tasks."""

    loop: asyncio.AbstractEventLoop = None
    batches: ty.ClassVar[bool] = False
    """Whether the worker has ``submit_many(jobs)``, which places a round of jobs together"""
    reports_done: ty.ClassVar[bool] = False
    """Whether ``submit_many`` also takes ``done(job, outcome)``, called as each job ends"""

    def __getstate__(self) -> dict[str, ty.Any]:
        """Return state for pickling."""
        state = attrs.asdict(self, recurse=False)
        state["loop"] = None
        return state

    def __setstate__(self, state: dict[str, ty.Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)
        self.loop = None
        # Loop will be restored by submitter __setstate__

    @abc.abstractmethod
    def run(self, job: "Job[TaskType]", rerun: bool = False) -> "Result":
        """Return coroutine for job execution."""
        pass

    def place(self, jobs: "list[Job]") -> ty.Any:
        """Where ``jobs`` would run if handed to the worker now (`GraphRunner.plan`);
        None: the worker gives no placement"""
        return None

    async def submit(self, job: "Job[TaskType]", rerun: bool = False) -> "Result":
        assert self.is_async, "Worker is not asynchronous, job should just be `run()`"
        if job.is_async:  # only for workflows at this stage and the foreseeable
            # These jobs are run in the primary process but potentially farm out
            # workflow jobs to other processes/job-schedulers
            return await job.run_async(rerun=rerun)
        else:
            return await self.run(job=job, rerun=rerun)

    def close(self):
        """Close this worker."""

    @property
    def is_async(self) -> bool:
        """Return whether the worker is asynchronous."""
        return inspect.iscoroutinefunction(self.run)

    @classmethod
    def available_plugins(cls) -> ty.Dict[str, ty.Type["Worker"]]:
        """Return all installed worker types"""
        return get_plugin_classes(pydra.workers, "Worker")

    @classmethod
    def plugin(cls, plugin_name: str) -> ty.Type["Worker"]:
        """Return a worker class by name."""
        try:
            return cls.available_plugins()[plugin_name.replace("-", "_")]
        except KeyError:
            raise ValueError(
                f"No worker matches {plugin_name!r}, check if there is a "
                f"plugin package called 'pydra-workers-{plugin_name}' that needs to be "
                "installed."
            )

    @classmethod
    def plugin_name(cls) -> str:
        """Return the name of the plugin."""
        try:
            plugin_name = cls._plugin_name
        except AttributeError:
            parts = cls.__module__.split(".")
            if parts[:-1] != ["pydra", "workers"]:
                raise ValueError(
                    f"Cannot infer plugin name of Worker ({cls}) from module path, as it "
                    f"isn't installed within `pydra.workers` ({cls.__module__}). "
                    "Please set the `_plugin_name` attribute on the class explicitly."
                )
            plugin_name = parts[-1].replace("_", "-")
        return plugin_name


@attrs.define
class ExecutorWorker(Worker):
    """An executor (`pydra.engine.executor.Executor`, such as a `WarmPool`) given to a `Submitter` as
    a worker. The submitter hands its runs' jobs to the executor itself, and a lone job through `run`.

    Parameters
    ----------
    executor : Executor
        The executor; bound to each run's driver in turn.
    owned : bool
        Whether the submitter made it (by name) and closes it with itself; an executor passed in
        belongs to its caller, who closes it, and `close` leaves it open.
    """

    executor: ty.Any = None
    owned: bool = False
    _plugin_name: ty.ClassVar[str] = "executor"

    def __getstate__(self) -> dict[str, ty.Any]:
        """The executor stays in the submitting process."""
        state = super().__getstate__()
        state["executor"] = None
        return state

    async def run(self, job: "Job[TaskType]", rerun: bool = False) -> "Result":
        """Hand ``job`` to the executor alone and wait for its outcome: its result, the error of a
        ``Failed`` job, or `RunIncomplete` for a job that did not run."""
        from pydra.engine.runner import Failed, NotRun
        from pydra.engine.submitter import RunIncomplete

        if self.executor is None:
            raise RuntimeError(
                "an executor worker copied to another process cannot run jobs"
            )
        loop = asyncio.get_running_loop()
        done: asyncio.Future = loop.create_future()

        def settle(outcome: ty.Any) -> None:
            if not done.done():
                done.set_result(outcome)

        def report(reported: "Job", outcome: ty.Any) -> None:
            if reported is job:
                loop.call_soon_threadsafe(settle, outcome)

        job.rerun = rerun
        self.executor.bind(report)
        try:
            handle = self.executor.submit([job])
            try:
                outcome = await done
            except BaseException:
                if handle.open:
                    self.executor.cancel(handle)
                raise
        finally:
            self.executor.unbind(report)
        if isinstance(outcome, Failed):
            raise outcome.error
        if isinstance(outcome, NotRun):
            held = ", held by infrastructure" if outcome.held else ""
            raise RunIncomplete(
                f"Job {job.name!r} did not run{held}: {outcome.reason}. A later run resumes "
                "it from the store.",
                [],
                [],
                [(job, outcome)],
            )
        return outcome

    def close(self) -> None:
        """Close the executor when the submitter made it; one passed in is its caller's."""
        if self.owned and self.executor is not None:
            self.executor.close()


async def read_and_display_async(*cmd, hide_display=False, strip=False):
    """
    Capture standard input and output of a process, displaying them as they arrive.

    Works line-by-line.

    """
    # start process
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asp.PIPE, stderr=asp.PIPE
    )

    stdout_display = sys.stdout.buffer.write if not hide_display else None
    stderr_display = sys.stderr.buffer.write if not hide_display else None
    # read child's stdout/stderr concurrently (capture and display)
    try:
        stdout, stderr = await asyncio.gather(
            read_stream_and_display(process.stdout, stdout_display),
            read_stream_and_display(process.stderr, stderr_display),
        )
    except BaseException:  # including cancellation, e.g. a timeout around the call
        await _stop_process(process)
        raise
    finally:
        # wait for the process to exit
        rc = await process.wait()
    if strip:
        return rc, stdout.strip(), stderr
    else:
        return rc, stdout, stderr


STOP_GRACE = 2.0  # seconds a terminated child has to exit before it is killed


async def _stop_process(process, grace: float = None):
    """Terminate a child process, killing it if it has not exited after ``grace`` s."""
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), STOP_GRACE if grace is None else grace)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass


async def read_stream_and_display(stream, display):
    """
    Read from stream line by line until EOF, display, and capture the lines.

    See Also
    --------
    This `discussion on StackOverflow
    <https://stackoverflow.com/questions/17190221>`__.

    """
    output = []
    while True:
        line = await stream.readline()
        if not line:
            break
        output.append(line)
        if display is not None:
            display(line)  # assume it doesn't block
    return b"".join(output).decode()


def ensure_non_negative(value: int) -> int:
    if not value or value < 0:
        return 0
    return value
