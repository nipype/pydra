"""Drivers: the loops that run a :class:`~pydra.engine.runner.GraphRunner` to completion"""

import asyncio
import time
import typing as ty
from collections import defaultdict, deque
from pydra.engine.runner import Failed, GraphRunner, OnError

if ty.TYPE_CHECKING:
    from pydra.engine.executor import Executor, Handle, InlineExecutor
    from pydra.engine.job import Job
    from pydra.engine.runner import Outcome
    from pydra.utils.routing import Router


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
        rnd = runner.ready(limit)
        dispatch = rnd.dispatch
        while dispatch or not runner.done:
            for job in dispatch:
                executor.submit([job])
                if fail_fast and runner.first_error is not None:
                    raise runner.first_error
            if fail_fast and runner.failed:
                break
            rnd = runner.ready(limit)
            dispatch = rnd.dispatch
            if not dispatch and not runner.done:
                if (
                    rnd.retry_after is not None
                ):  # a workflow lease another process holds
                    time.sleep(rnd.retry_after)
                    rnd = runner.ready(limit)
                    dispatch = rnd.dispatch
                elif diagnosis := runner.stuck():
                    raise RuntimeError(diagnosis)
    finally:
        runner.close()
        executor.close()


async def drive(
    runner: GraphRunner,
    executors: "ty.Mapping[str, Executor]",
    router: "Router",
    *,
    max_concurrent: float,
    on_error: OnError,
) -> None:
    """Run every job of the runner on the executors, on the running event loop

    Under ``fail_fast`` an executor whose ``serial`` attribute is true (a synchronous
    worker) has one job in flight at a time, so none starts after the first failure.

    The jobs of each round are grouped by the executor the router names, and handed to it
    in node order, one call per job unless the executor batches. Outcomes are reported
    onto this loop. With ``on_error="fail_fast"`` the first failure stops the run. On the
    way out, whether complete, failed or interrupted, every open handle is cancelled and
    the executors are closed. Failures are left on ``runner.errors``.

    Parameters
    ----------
    runner : GraphRunner
        The runner to drive
    executors : Mapping[str, Executor]
        The executors, by the names the router returns
    router : Router
        Called with each job; the name of its executor. An exception it raises is the
        job's failure
    max_concurrent : float
        The most jobs in flight at once; a float because ``inf`` means unlimited
    on_error : "raise_at_end" | "fail_fast"
        Whether to stop at the first failure
    """
    loop = asyncio.get_running_loop()
    fail_fast = on_error == "fail_fast"
    progressed = asyncio.Event()
    open_handles: "list[Handle]" = []
    in_flight: "set[Job]" = set()
    # the jobs not run because a job with the same checksum is being run, by checksum
    sharing: "dict[str, set[Job]]" = defaultdict(set)
    futured: "dict[str, Job]" = {}
    # jobs waiting for an executor that runs one at a time under fail_fast, and the
    # number it has in flight
    waiting: "dict[str, deque[Job]]" = defaultdict(deque)
    busy: "dict[str, int]" = defaultdict(int)
    sent_to: "dict[Job, str]" = {}
    # the outcome of each job whose run has ended, by checksum
    finished: "dict[str, Outcome]" = {}

    def limit() -> int | None:
        if max_concurrent == float("inf"):
            return None
        return max(0, int(max_concurrent) - len(in_flight))

    def send(name: str, jobs: "list[Job]") -> None:
        executor = executors[name]
        for job in jobs:
            sent_to[job] = name
        busy[name] += len(jobs)
        for batch in [jobs] if executor.batches else [[j] for j in jobs]:
            open_handles.append(executor.submit(batch))

    def pump() -> None:
        """Hand each one-at-a-time executor its next waiting job once it is idle"""
        for name, queue in waiting.items():
            if queue and not busy[name] and not (fail_fast and runner.failed):
                send(name, [queue.popleft()])

    def on_outcome(job: "Job", outcome: "Outcome") -> None:
        if job in sent_to:
            busy[sent_to.pop(job)] -= 1
        in_flight.discard(job)
        finished[job.checksum] = outcome
        runner.complete(job, outcome)
        for ran in sharing.pop(job.checksum, ()):
            share(ran, outcome)
        open_handles[:] = [h for h in open_handles if h.open]
        pump()
        progressed.set()

    def share(job: "Job", outcome: "Outcome") -> None:
        """Complete a job that was not run because another with its checksum was, with
        that job's outcome; the failure itself is recorded for the job that ran"""
        if isinstance(outcome, Failed) or outcome.errored:
            job._awaiting_run = False
            job._errored = True
            if not isinstance(outcome, Failed):
                job._held = outcome
        else:
            runner.complete(job, outcome)

    def report(job: "Job", outcome: "Outcome") -> None:
        loop.call_soon_threadsafe(on_outcome, job, outcome)

    for executor in executors.values():
        executor.bind(report)
    try:
        rnd = runner.ready(limit())
        dispatch = rnd.dispatch
        while dispatch or in_flight or not runner.done:
            if not dispatch and not in_flight and rnd.retry_after is None:
                raise RuntimeError(runner.stuck())
            by_executor: "dict[str, list[Job]]" = defaultdict(list)
            for job in dispatch:
                if job.checksum in finished:
                    share(job, finished[job.checksum])
                elif job.checksum in futured:
                    if job is not futured[job.checksum]:
                        sharing[job.checksum].add(job)
                else:
                    futured[job.checksum] = job
                    try:
                        by_executor[router(job)].append(job)
                    except Exception as exc:
                        in_flight.add(job)
                        on_outcome(job, Failed(exc))
            for name, jobs in by_executor.items():
                in_flight.update(jobs)
                if fail_fast and executors[name].serial:
                    waiting[name].extend(jobs)
                else:
                    send(name, jobs)
            pump()
            if in_flight or rnd.retry_after is not None:
                # the next outcome, or the next attempt at a workflow lease; no timeout
                # context is entered, which needs a current task (none under
                # nest_asyncio)
                wake = None
                if rnd.retry_after is not None:
                    wake = loop.call_later(rnd.retry_after, progressed.set)
                try:
                    await progressed.wait()
                finally:
                    if wake is not None:
                        wake.cancel()
                progressed.clear()
            if fail_fast and runner.failed:
                break
            rnd = runner.ready(limit())
            dispatch = rnd.dispatch
            if fail_fast and runner.failed:
                break
    finally:
        runner.close()
        for handle in open_handles:
            handle.executor.cancel(handle)
        for executor in executors.values():
            executor.close()
