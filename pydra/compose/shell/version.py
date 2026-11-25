"""The version of the tool a shell task runs (RFC 0004 section 5.2).

``shell.define(version_command=..., version_pattern=...)`` keys a task by the first
group of the first match of the pattern in the command's stdout, never the whole output.
The submitting
process runs the command once per resolved executable path and keeps the answer in
memory; a worker running the task natively runs it again, once per process, and fails
the job when the answer differs from the key.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import typing as ty
import warnings

if ty.TYPE_CHECKING:
    from pydra.compose.shell.task import Task

__all__ = ["VersionCommandError"]

TIMEOUT = 60
"""Seconds a version command may run before keying fails with `VersionCommandError`. A
version command prints and exits; the bound only ends one that hangs (a licence prompt, a
network wait), and is large against the start-up of a container image on a loaded node.
An unmeasured choice."""


class VersionCommandError(RuntimeError):
    """A ``version_command`` failed, or its output did not match ``version_pattern``,
    or a worker found another version of the tool than the one the job was keyed by"""


# The answers of the submitting process and of a worker process, by resolved executable
# path, command and pattern. They are separate so a job run in the submitting process
# (the ``debug`` worker) still checks the tool again as a worker would.
_MEMO: dict[tuple[str, tuple[str, ...], str], str] = {}
_WORKER_MEMO: dict[tuple[str, tuple[str, ...], str], str] = {}


def check_arguments(
    version: str | None,
    version_command: ty.Sequence[str] | None,
    version_pattern: str | None,
) -> None:
    """Raise ValueError for a combination `shell.define` does not accept"""
    if version_command is None and version_pattern is None:
        return
    if version_command is None:
        raise ValueError("version_pattern needs a version_command to run")
    if version_pattern is None:
        raise ValueError(
            "version_command needs a version_pattern, whose first capture group is "
            "the version"
        )
    if version is not None:
        raise ValueError(
            "give one of version= and version_command=, not both: a declared version "
            "is the version, a command asks the tool"
        )
    if (
        isinstance(version_command, str)
        or not version_command
        or not all(isinstance(c, str) for c in version_command)
    ):
        raise ValueError(
            f"version_command must be a non-empty list of strings, not {version_command!r}"
        )
    try:
        compiled = re.compile(version_pattern)
    except re.error as e:
        raise ValueError(f"version_pattern {version_pattern!r} is not valid: {e}")
    if compiled.groups < 1:
        raise ValueError(
            f"version_pattern {version_pattern!r} needs a capture group, whose first "
            "match is the version"
        )


def run_version_command(
    command: ty.Sequence[str],
    pattern: str,
    memo: dict[tuple[str, tuple[str, ...], str], str],
) -> str | None:
    """The first group of the first match of `pattern` in the stdout of `command`, or
    None when its program is not found. The answer is kept in `memo` by the resolved path
    of the program: not on disk, as a shim keeps its stat while the tool behind it
    changes, and not when the program is not found."""
    path = shutil.which(command[0])
    if path is None:
        return None
    key = (os.path.realpath(path), tuple(command), pattern)
    if key in memo:
        return memo[key]
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=TIMEOUT,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise VersionCommandError(f"version_command {list(command)!r} failed: {e}")
    match = re.search(pattern, completed.stdout)
    if match is None or match.group(1) is None:
        shown = completed.stdout[:500]
        if not completed.stdout.strip():
            shown = (
                f"(empty stdout, return code {completed.returncode}; "
                f"stderr: {completed.stderr[:500]!r})"
            )
        raise VersionCommandError(
            f"version_pattern {pattern!r} does not match the stdout of version_command "
            f"{list(command)!r}:\n{shown}"
        )
    memo[key] = match.group(1)
    return memo[key]


def tool_version(task: "Task") -> str | dict[str, ty.Any] | None:
    """The record's ``job.tool_version`` for a job of `task` run natively: the version
    the command reported, None for a task without a ``version_command``, and
    ``{"command", "state": "unresolved"}`` when its program is not found here (the
    result is then used within the process and not served to another, see
    `pydra.engine.context.job_context`)."""
    command = getattr(task, "_version_command", None)
    if command is None:
        return None
    version = run_version_command(command, task._version_pattern, _MEMO)
    if version is None:
        warnings.warn(
            f"the program of version_command {list(command)!r} was not found, so the "
            f"version of the tool is unresolved and results are reused only within this "
            "process; give version= to key the task by a declared version",
            UserWarning,
            stacklevel=2,
        )
        return {"command": list(command), "state": "unresolved"}
    return version


def check_in_worker(task: "Task", expected: ty.Any) -> None:
    """Run the version command in this (worker) process, once, and raise
    `VersionCommandError` with both values if the tool's version is not the one the
    job was keyed by (`expected`, the record's ``job.tool_version``). A key that is
    unresolved, or a task without a command, is not checked."""
    command = getattr(task, "_version_command", None)
    if command is None or not isinstance(expected, str):
        return
    actual = run_version_command(command, task._version_pattern, _WORKER_MEMO)
    if actual is None:
        raise VersionCommandError(
            f"the program of version_command {list(command)!r} is not found by the "
            f"worker, the job was keyed by version {expected!r}"
        )
    if actual != expected:
        raise VersionCommandError(
            f"the worker found version {actual!r} of the tool where the job was keyed by "
            f"version {expected!r} (version_command {list(command)!r})"
        )
