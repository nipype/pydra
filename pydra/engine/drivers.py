"""Drivers: the loops that run a :class:`~pydra.engine.runner.GraphRunner` to completion"""

import asyncio
import time
import typing as ty
from collections import defaultdict, deque
from pydra.engine.runner import Failed, GraphRunner, OnError

if ty.TYPE_CHECKING:
    from pydra.engine.audit import Audit
    from pydra.engine.executor import Executor, Handle, InlineExecutor
    from pydra.engine.job import Job
    from pydra.engine.runner import Outcome
    from pydra.utils.routing import Router


def drive_inline(
    runner: GraphRunner,
    executor: "InlineExecutor",
    *,
    on_error: OnError,
    audit: "Audit",
    limit: int | None = None,
) -> None:
    """Run every job of the runner in the calling thread, one at a time

    With ``on_error="fail_fast"`` the first failure stops the run and its exception is
    raised as the task raised it; no job is started after it. Otherwise jobs that do not
    read a failed job still run, and the failures are left on ``runner.errors``. The
    driver keeps no checksum table: the runner settles twins itself.

    Parameters
    ----------
    runner : GraphRunner
        The runner to drive
    executor : InlineExecutor
        Runs each job; left open, for whoever made it to close
    on_error : "raise_at_end" | "fail_fast"
        Whether to stop at the first failure
    audit : Audit
        Not used: jobs report to the audit of their submitter
    limit : int, optional
        The most jobs to take from the runner per round, None if unlimited
    """
    executor.bind(runner.complete)
    fail_fast = on_error == "fail_fast"
    try:
        while True:
            rnd = runner.ready(limit)
            for job in rnd.dispatch:
                executor.submit([job])
                if fail_fast and runner.first_error is not None:
                    raise runner.first_error
            if fail_fast and runner.failed:
                break
            if rnd.dispatch or rnd.settled or rnd.unrunnable:
                continue  # their readers may be ready now
            if runner.done:
                break
            if rnd.retry_after is not None:  # a workflow lease another process holds
                time.sleep(rnd.retry_after)
                continue
            raise RuntimeError(
                runner.stuck() or "Nothing can progress, but the run is not done"
            )
    finally:
        runner.close()


async def drive(
    runner: GraphRunner,
    executors: "ty.Mapping[str, Executor]",
    router: "Router",
    *,
    max_concurrent: float,
    on_error: OnError,
    audit: "Audit",
) -> None:
    """Run every job of the runner on the executors, on the running event loop

    The jobs of each round are grouped by the executor the router names, and handed to it
    in node order, one call per job unless the executor batches. Outcomes are reported
    onto this loop. The driver keeps no checksum table: the runner settles twins itself.
    With ``on_error="fail_fast"`` the first failure stops the run. On the way out, whether
    complete, failed or interrupted, every open handle is cancelled and the runner's leases
    are released (off the loop). The executors are never closed: whoever made them closes
    them, so an executor that holds warm workers (a ``WarmPool``) keeps them from one run to
    the next. Failures are left on ``runner.errors``.

    ``max_concurrent`` caps the jobs in flight through ``runner.ready(limit)``, unless an
    executor batches (``batches = True``): the runner is then asked for every ready job, so
    a node's elements stay in one array or batch, and the batching executor caps inside
    itself, each to its own jobs. Jobs routed to the other executors then wait in the
    driver for the cap, which counts only theirs, so the run as a whole can have more than
    ``max_concurrent`` jobs in flight. Under
    ``fail_fast`` an executor whose ``serial`` attribute is true (a synchronous worker) has
    one job in flight at a time, so none starts after the first failure.

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
    audit : Audit
        Not used: jobs report to the audit of their submitter
    """
    loop = asyncio.get_running_loop()
    fail_fast = on_error == "fail_fast"
    progressed = asyncio.Event()
    open_handles: "list[Handle]" = []
    # the jobs handed out by the runner whose outcome has not been reported
    in_flight: "set[Job]" = set()
    # jobs waiting for an executor that runs one at a time under fail_fast, and the
    # number each executor has in flight
    waiting: "dict[str, deque[Job]]" = defaultdict(deque)
    busy: "dict[str, int]" = defaultdict(int)
    sent_to: "dict[Job, str]" = {}
    batching = any(ex.batches for ex in executors.values())
    capped = max_concurrent != float("inf")
    # with a batching executor, the jobs for the other executors beyond the cap
    backlog: "deque[tuple[str, Job]]" = deque()

    def limit() -> int | None:
        if batching or not capped:
            return None
        return max(0, int(max_concurrent) - len(in_flight))

    def unbatched() -> int:
        return sum(busy[n] for n, ex in executors.items() if not ex.batches)

    def send(name: str, jobs: "list[Job]") -> None:
        executor = executors[name]
        for job in jobs:
            sent_to[job] = name
        busy[name] += len(jobs)
        for batch in [jobs] if executor.batches else [[j] for j in jobs]:
            open_handles.append(executor.submit(batch))

    def place(name: str, jobs: "list[Job]") -> None:
        if fail_fast and getattr(executors[name], "serial", False):
            waiting[name].extend(jobs)
        else:
            send(name, jobs)

    def pump() -> None:
        """Hand each one-at-a-time executor its next waiting job once it is idle, and
        the jobs held for the cap as it frees"""
        if fail_fast and runner.failed:
            return
        while backlog and unbatched() + sum(map(len, waiting.values())) < int(
            max_concurrent
        ):
            name, job = backlog.popleft()
            place(name, [job])
        for name, queue in waiting.items():
            if queue and not busy[name]:
                send(name, [queue.popleft()])

    def on_outcome(job: "Job", outcome: "Outcome") -> None:
        if job in sent_to:
            busy[sent_to.pop(job)] -= 1
        in_flight.discard(job)
        runner.complete(job, outcome)
        open_handles[:] = [h for h in open_handles if h.open]
        pump()
        progressed.set()

    def report(job: "Job", outcome: "Outcome") -> None:
        loop.call_soon_threadsafe(on_outcome, job, outcome)

    for executor in executors.values():
        executor.bind(report)
    runner.release_off_thread()
    runner.read_off_thread(lambda: loop.call_soon_threadsafe(progressed.set))
    try:
        while True:
            rnd = runner.ready(limit())
            if fail_fast and runner.failed:
                break
            by_executor: "dict[str, list[Job]]" = defaultdict(list)
            for job in rnd.dispatch:
                in_flight.add(job)
                try:
                    by_executor[router(job)].append(job)
                except Exception as exc:
                    on_outcome(job, Failed(exc))
            for name, jobs in by_executor.items():
                if batching and capped and not executors[name].batches:
                    backlog.extend((name, j) for j in jobs)
                else:
                    place(name, jobs)
            pump()
            if fail_fast and runner.failed:
                break
            if not in_flight and runner.done:
                break
            if (rnd.settled or rnd.unrunnable) and not rnd.dispatch:
                continue  # their readers may be ready now
            reading = getattr(runner, "reading", 0)
            if (
                not in_flight
                and not rnd.dispatch
                and rnd.retry_after is None
                and not reading
            ):
                raise RuntimeError(
                    runner.stuck() or "Nothing can progress, but the run is not done"
                )
            if in_flight or rnd.retry_after is not None or reading:
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
    finally:
        runner.close()
        for handle in open_handles:
            handle.executor.cancel(handle)
        await runner.released()
