import asyncio

from .workers import MpWorker, SerialWorker, DaskWorker, ConcurrentFuturesWorker
from .core import is_workflow, is_task, is_runnable

import logging
logging.basicConfig(level=logging.DEBUG)  # TODO: RF
logger = logging.getLogger("pydra.submitter")


class Submitter(object):
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

    async def check_pending(self):
        done = await self.worker.fetch_finished()
        if done:
            for fut in done:
                task, res = await fut
                logger.debug("Task: %s completed with result: %s", task, res)
                self.worker._remove_pending(fut)

    async def _run_workflow(self, wf):
        # some notion of topological sorting
        # regardless of DFS/BFS, will not always be absolute order
        # (no notion of job duration)
        remaining_tasks = wf.graph_sorted

        # TODO: remove after debug
        timeout = 5
        iters = 0

        while remaining_tasks:
            iters += 1
            # TODO: remove after debug endless loop
            if timeout < iters:
                breakpoint()

            remaining_tasks, tasks = await get_runnable_tasks(
                wf.graph, remaining_tasks
            )
            logger.debug("Runnable tasks: %s", tasks)
            if tasks:
                for task in tasks:
                    # grab inputs if needed
                    logger.debug("Retrieving inputs for %s", task)
                    task.inputs.retrieve_values(wf)
                    if is_workflow(task):
                        # recurse into previous run
                        # this is blocking
                        logger.debug("Task %s is a workflow, expanding out and executing", task)
                        self.run(task)
                    else:
                        # pass the future off to the worker
                        self.worker.run_el(task)
                # wait for one of the tasks to finish
                await self.check_pending()
        return wf

        # no more tasks to queue, but some may still be running
        if self.worker._pending:
            await self.check_pending()

    def run(self, runnable, cache_locations=None):
        """main running method, checks if submitter id for Task or Workflow"""
        if not is_task(runnable):
            raise Exception("runnable has to be a Task or Workflow")
        runnable.plugin = self.plugin  # assign in case of downstream execution

        if runnable.state:
            runnable.state.prepare_state(runnable.inputs)
            runnable.state.prepare_inputs()
            for ii, ind in enumerate(runnable.state.states_val):
                # creating a taskFunction for every element of state
                # this job will run interface (and will not have state)
                job = runnable.to_job(ii)
                checksum = job.checksum
                # run method has to have checksum to check the existing results
                job.results_dict[None] = (None, checksum)
                if cache_locations:
                    job.cache_locations = cache_locations
                # res = self.worker.run_el(job)
        else:
            # job = runnable
            job = runnable.to_job(None)
            checksum = job.checksum
            job.results_dict[None] = (None, checksum)
            if cache_locations:
                job.cache_locations = cache_locations

        if is_workflow(job):
            # blocking
            # however, this does not call wf.run()
            # no lockfile / cache is generated
            asyncio.run(self._run_workflow(job))
        else:
            self.worker.run_el(job)

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
    if len(tasks):
        return remaining_tasks, tasks
    return remaining_tasks, None
