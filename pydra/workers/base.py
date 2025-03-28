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


def ensure_non_negative(value: int) -> int:
    if not value or value < 0:
        return 0
    return value
