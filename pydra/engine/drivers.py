"""Drivers: the loops that run a :class:`~pydra.engine.runner.GraphRunner` to completion"""

import typing as ty
from pydra.engine.runner import GraphRunner, OnError

if ty.TYPE_CHECKING:
    from pydra.engine.executor import InlineExecutor


def drive_inline(
    runner: GraphRunner,
    executor: "InlineExecutor",
    *,
    on_error: OnError,
    limit: int | None = None,
) -> None:
    """Run every job of the runner in the calling thread, one at a time

    With ``on_error="fail_fast"`` the first failure stops the run and its exception is
    raised as the task raised it; no job is started after it. Otherwise jobs that do not
    read a failed job still run, and the failures are left on ``runner.errors``.

    Parameters
    ----------
    runner : GraphRunner
        The runner to drive
    executor : InlineExecutor
        Runs each job
    on_error : "raise_at_end" | "fail_fast"
        Whether to stop at the first failure
    limit : int, optional
        The most jobs to take from the runner per round, None if unlimited
    """
    executor.bind(runner.complete)
    fail_fast = on_error == "fail_fast"
    try:
        dispatch = runner.ready(limit).dispatch
        while dispatch or not runner.done:
            for job in dispatch:
                executor.submit([job])
                if fail_fast and runner.first_error is not None:
                    raise runner.first_error
            if fail_fast and runner.errors:
                break
            dispatch = runner.ready(limit).dispatch
            if not dispatch and not runner.done:
                raise RuntimeError(runner.stuck())
    finally:
        executor.close()
