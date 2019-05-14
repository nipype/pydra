import asyncio
import time

from .workers import MpWorker, SerialWorker, DaskWorker, ConcurrentFuturesWorker
from .core import is_workflow, is_task, is_runnable

import logging
logging.basicConfig(level=logging.DEBUG)  # TODO: RF
logger = logging.getLogger("pydra.submitter")


class Submitter:
    # TODO: runnable in init or run
    def __init__(self, plugin):
        self.plugin = plugin
        if self.plugin == "mp":
            self.worker = MpWorker()
        elif self.plugin == "serial":
            self.worker = SerialWorker()
        elif self.plugin == "dask":
            self.worker = DaskWorker()
        elif self.plugin == "cf":
            self.worker = ConcurrentFuturesWorker()
        else:
            raise Exception("plugin {} not available".format(self.plugin))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    async def fetch_pending(self, wf=None):
        """
        Fetch any completed Futures from worker and remove from worker's pending tasks.
        If wf is defined and a completed Future is a ``Workflow`` object, wf.Workflow
        is replaced with Future.result().

        Parameters
        ----------
        wf : Workflow (optional)
            The preceding workflow
        """
        done = await self.worker.fetch_finished()
        for fut in done:
            sidx, task = await fut
            self.worker._remove_pending(fut)
            msg = "Task: %s%s completed" % (
                task, " (with state index %s)" if sidx is not None else ""
            )
            logger.debug(msg)

            # handle completition vs states differently
            if sidx is None:
                if is_workflow(task):
                    wf.name2obj.get(task.name).__dict__.update(task.__dict__)
            else:
                # additional operations to ensure joining of state results
                # grab task master and add results to it

                # sidx is state index task.result() should pertain to
                # check if all state index have finished to update master task
                breakpoint()
        return wf

    async def _run_task(self, task):
        """
        Submits a ``Task`` across all states to a worker.
        """
        if task.state is None:
            job = task.to_job(None)
            checksum = job.checksum
            job.results_dict[None] = (None, checksum)
            self.worker.run_el(job)
        else:
            task.state.prepare_states(task.inputs)
            task.state.prepare_inputs()
            # submit each state as a separate job
            for sidx in range(len(task.state.states_val)):
                job = task.to_job(sidx)
                checksum = job.checksum
                job.results_dict[None] = (sidx, checksum)
                self.worker.run_el(job)

    async def _run_workflow(self, wf):
        """
        Submits a workflow and awaits the completion.
        """
        # some notion of topological sorting
        # regardless of DFS/BFS, will not always be absolute order
        # (no notion of job duration)
        self.worker.loop = asyncio.get_event_loop()
        logger.debug("Executing %s in event loop %s", wf, hex(id(self.worker.loop)))
        remaining_tasks = wf.graph_sorted
        while not wf.done:
            remaining_tasks, tasks = await get_runnable_tasks(
                wf.graph, remaining_tasks
            )
            logger.debug("Runnable tasks: %s", tasks)
            if not tasks and not self.worker._pending:
                raise Exception("Worker is stuck - something went wrong")
                # something is up
            for task in tasks:
                # grab inputs if needed
                logger.debug("Retrieving inputs for %s", task)
                task.inputs.retrieve_values(wf)  # what if state?
                if is_workflow(task):
                    # recurse into previous run and halt execution
                    logger.debug("Task %s is a workflow, expanding out and executing", task)
                    task.plugin = self.plugin
                # do not treat workflow tasks differently
                # instead, allow them to spawn a job
                await self._run_task(task)
            # wait for at least one of the tasks to finish
            wf = await self.fetch_pending(wf)

        logger.debug("Finished workflow %s", wf)
        return wf  # return the run workflow

    def run(self, runnable, cache_locations=None):
        """Submitter for ``Task``s and ``Workflow``s."""
        if not is_task(runnable):
            raise Exception("runnable has to be a Task or Workflow")

        runnable.plugin = self.plugin  # assign in case of downstream execution
        coro = self._run_workflow if is_workflow(runnable) else self._run_task

        startt = time.time()
        loop = asyncio.new_event_loop()  # create new loop for every workflow
        loop.set_debug(True)
        completed = loop.run_until_complete(coro(runnable))
        logger.debug("Closing event loop %s, total running time: %s",
                     hex(id(loop)), time.time() - startt)
        loop.close()
        return completed

    def close(self):
        self.worker.close()


async def get_runnable_tasks(graph, remaining_tasks):
    """Parse a graph and return all runnable tasks"""
    didx, tasks = [], []
    for idx, task in enumerate(remaining_tasks):
        # are all predecessors finished
        if is_runnable(graph, task):
            didx.append(idx)
            tasks.append(task)
    for i in sorted(didx, reverse=True):
        del remaining_tasks[i]

    return remaining_tasks, tasks
