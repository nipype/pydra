import time
from copy import deepcopy
import dataclasses as dc
import asyncio

from .workers import MpWorker, SerialWorker, DaskWorker, ConcurrentFuturesWorker
from .node import NodeBase, is_workflow, is_task, is_runnable

import logging
logger = logging.getLogger("pydra.workflow")


class Submitter(object):
    # TODO: runnable in init or run
    def __init__(self, plugin):
        self.plugin = plugin
        self.remaining_tasks = []
        self.submitted = set()
        self.completed = set()
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


    async def _fetch_completed(self, futures):
        pass

    async def _submit_task(self, wf, remaining=None, running=None):
        while remaining or running:
            taskinfo = await self.get_runnable_task(wf.graph)
            if taskinfo:
                # remove from remaining tasks
                tidx, task = taskinfo
                print("Starting task", task)
                del self.remaining_tasks[tidx]

                # pass the future off to the worker
                fut = self.worker.run_el(task)
                return

                done, _ = await asyncio.wait(
                    running_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # completed futures
                for fut in done:
                    print(fut)
                    running_tasks.discard(fut)
                    pending.discard(fut)
                    task, res = await fut

                    # deals with state index also
                    task.inputs.retrieve_values(self, wf)
                


    async def _run_workflow(self, wf):

        # some notion of topological sorting
        # regardless of DFS/BFS, will not always be absolute order
        # (no notion of job duration)
        remaining_tasks = wf.graph_sorted
        while remaining_tasks or self.worker._pending:
            remaining_tasks, tasks = await get_runnable_tasks(wf.graph, remaining_tasks)
            if tasks:
                for task in tasks:
                    # remove from remaining tasks
                    # tidx, task = taskinfo
                    task.inputs.retrieve_values(wf)
                    # pass the future off to the worker
                    self.worker.run_el(task)

                done = await self.worker.fetch_finished()

                for fut in done:
                    task, res = await fut
                    self.worker._remove_pending(fut)

    def run(self, runnable, cache_locations=None):
        """main running method, checks if submitter id for Task or Workflow"""
        if not is_task(runnable):
            raise Exception("runnable has to be a Task or Workflow")
        runnable.plugin = self.plugin  # assign in case of downstream execution

        job = runnable.to_job(None)
        checksum = job.checksum
        job.results_dict[None] = (None, checksum)
        if cache_locations:
            job.cache_locations = cache_locations

        if is_workflow(runnable):  # expand out
            asyncio.run(self._run_workflow(job))

        # asyncio.run(self._run_async(runnable, cache_locations))

    def close(self):
        self.worker.close()


async def get_runnable_tasks(graph, remaining_tasks, polling=3):
    """Parse a graph and return all runnable tasks"""
    didx = []
    tasks = []
    for idx, task in enumerate(remaining_tasks):
        if is_runnable(graph, task):
            didx.append(idx)
            tasks.append(task)
    for i in sorted(didx, reverse=True):
        del remaining_tasks[i]
    if len(tasks):
        return remaining_tasks, tasks
    else:
        # wait for a task to become runnable
        await asyncio.sleep(polling)
    return remaining_tasks, None
