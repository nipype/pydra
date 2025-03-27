"""Execution workers."""

import asyncio
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


class Worker(metaclass=abc.ABCMeta):
    """A base class for execution of tasks."""

    plugin_name: str
    loop: asyncio.AbstractEventLoop

    def __init__(self, loop=None):
        """Initialize the worker."""
        logger.debug(f"Initializing {self.__class__.__name__}")
        self.loop = loop

    def __getstate__(self):
        """Return state for pickling."""
        state = super().__getstate__()
        state["loop"] = None
        return state

    @abc.abstractmethod
    def run(self, job: "Job[TaskType]", rerun: bool = False) -> "Result":
        """Return coroutine for job execution."""
        pass

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

    async def fetch_finished(
        self, futures
    ) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        """
        Awaits asyncio's :class:`asyncio.Task` until one is finished.

        Parameters
        ----------
        futures : set of asyncio awaitables
            Job execution coroutines or asyncio :class:`asyncio.Task`

        Returns
        -------
        pending : set
            Pending asyncio :class:`asyncio.Task`.
        done: set
            Completed asyncio :class:`asyncio.Task`

        """
        done = set()
        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(f) if not isinstance(f, asyncio.Task) else f
                    for f in futures
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except ValueError:
            # nothing pending!
            pending = set()
        logger.debug(f"Tasks finished: {len(done)}")
        return pending, done

    @classmethod
    def available_plugins(cls) -> ty.Dict[str, ty.Type["Worker"]]:
        """Return all installed worker types"""
        return get_plugin_classes(pydra.workers, "Worker")

    @classmethod
    def plugin(cls, plugin_name: str) -> ty.Type["Worker"]:
        """Return a worker class by name."""
        return cls.available_plugins()[plugin_name.replace("-", "_")]

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
            plugin_name = parts[-1]
        return plugin_name.replace("_", "-")


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
    except Exception:
        process.kill()
        raise
    finally:
        # wait for the process to exit
        rc = await process.wait()
    if strip:
        return rc, stdout.strip(), stderr
    else:
        return rc, stdout, stderr


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
