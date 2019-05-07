import asyncio

from .workers import MpWorker, SerialWorker, DaskWorker, ConcurrentFuturesWorker
from .core import is_workflow, is_task, is_runnable

import logging
logging.basicConfig(level=logging.DEBUG)  # TODO: RF
logger = logging.getLogger("pydra.submitter")


class Submitter:
    # TODO: runnable in init or run
    def __init__(self, plugin, loop=None):
        self.plugin = plugin
        self.loop = loop or asyncio.get_event_loop()
        if self.plugin == "mp":
            self.worker = MpWorker()
        elif self.plugin == "serial":
            self.worker = SerialWorker()
        elif self.plugin == "dask":
            self.worker = DaskWorker()
        elif self.plugin == "cf":
            self.worker = ConcurrentFuturesWorker(loop=self.loop)
        else:
            raise Exception("plugin {} not available".format(self.plugin))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    async def fetch_pending(self):
        done = await self.worker.fetch_finished()
        for fut in done:
            task, res = await fut
            logger.debug("Task: %s completed", task)
            self.worker._remove_pending(fut)

    def _run_task(self, task, wait_on_results=True):
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
        # results should be waited for by default
        # avoid awaiting per job in workflow context
        if wait_on_results:
            asyncio.run(self.fetch_pending())
            # TODO: ensure these results are joined together?

    async def _run_workflow(self, wf):
        """
        Submits a workflow and awaits the completion.
        """
        # some notion of topological sorting
        # regardless of DFS/BFS, will not always be absolute order
        # (no notion of job duration)
        remaining_tasks = wf.graph_sorted
        while not wf.done:
            remaining_tasks, tasks = await get_runnable_tasks(
                wf.graph, remaining_tasks
            )
            logger.debug("Runnable tasks: %s", tasks)
            for task in tasks:
                # grab inputs if needed
                logger.debug("Retrieving inputs for %s", task)
                task.inputs.retrieve_values(wf)  # what if state?
                if is_workflow(task):
                    # recurse into previous run and halt execution
                    logger.debug("Task %s is a workflow, expanding out and executing", task)
                    task.plugin = self.plugin
                    # submit as job
                    task.run(self.loop)  # ensure this handles workflow states
                else:
                    # pass the future off to the worker
                    # state??: ensure downstream do not start until all states finish?
                    # but ensure job starts!!
                    self._run_task(task, wait_on_results=False)
            # wait for one of the tasks to finish
            await self.fetch_pending()

        logger.debug("Finished workflow %s", wf)
        return wf  # return the run workflow

    def run(self, runnable, cache_locations=None):
        """Submitter for ``Task``s and ``Workflow``s."""
        if not is_task(runnable):
            raise Exception("runnable has to be a Task or Workflow")

        runnable.plugin = self.plugin  # assign in case of downstream execution
        coro = self._run_workflow if is_workflow(runnable) else self._run_task
        completed = self.loop.run_until_complete(coro(runnable))
        return completed

    def close(self):
        self.worker.close()


async def get_runnable_tasks(graph, remaining_tasks, polling=1):
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
