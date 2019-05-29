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
        self.loop = None
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

    def __call__(self, runnable):
        if is_workflow(runnable):
            runnable.submit_async(self)
        else:
            raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    async def fetch_from_worker(self, wf=None, pending_futures=None):
        """
        Fetch any completed ``Future``s from worker and update pending futures.
        If wf is defined and a completed Future is a ``Workflow`` object, wf.Workflow
        is replaced with Future.result().

        Parameters
        ----------
        wf : Workflow (optional)
            The preceding workflow
        """
        done, pending = await self.worker.fetch_finished(pending_futures)
        for fut in done:
            job, res, sidx = await fut
            logger.debug(
                f"{job.name}{str(sidx) if sidx is not None else ''} completed"
            )
            # within workflow tasks can still have state
            if sidx is not None:
                master = wf.name2obj.get(job.name)
                # ensure there is no overwriting
                master.results_dict[sidx] = (job.result(), job.checksum)
        return pending

    async def _run_workflow(self, wf, state_idx=None):
        """
        Expands and executes a stateless ``Workflow``.

        Parameters
        ----------
        wf : Workflow
            Workflow Task object
        state_idx : int or None
            Job state index

        Returns
        -------
        wf : Workflow
            The computed workflow
        """
        remaining_tasks = wf.graph_sorted
        # keep track of local futures
        running_futures = set()
        while not wf.done:
            remaining_tasks, tasks = await get_runnable_tasks(
                wf.graph, remaining_tasks
            )
            if not tasks and not task_futures:
                raise Exception("Nothing queued or todo - something went wrong")
            for task in tasks:
                # grab inputs if needed
                logger.debug(f"Retrieving inputs for {task}")
                # TODO: add state idx to retrieve values to reduce waiting
                task.inputs.retrieve_values(wf)

                # a few choice options here
                # 1. workflow with state --> await _submit (recurse)
                # 2. workflow with no state --> await _submit (recurse)
                # 3. task with state --> _expand and submit to worker
                # 4. task with no state --> submit to worker
                if is_workflow(task):
                    await task.run(submitter=self)
                else:
                    task_futures = await self.submit(task)
                    running_futures.union(task_futures)

            # TODO: ensure wf is updating
            task_futures = await self.fetch_from_worker(wf, task_futures)
        return wf, state_idx

    async def submit(self, runnable, return_task=False):
        """
        Coroutine entrypoint for task submission.

        Removes state from task and adds one or more
        asyncio ``Task``s to the running loop.

        Parameters
        ----------
        runnable : Task
            Task instance (``Task``, ``Workflow``)
        return_task : bool (False)
            Option to return runnable once all states have finished
        """
        runnables = []
        futures = set()

        if runnable.state:
            runnable.state.prepare_states(runnable.inputs)
            runnable.state.prepare_inputs()
            logger.debug(
                f"Expanding {runnable} into {len(runnable.state.states_val)} states"
            )
            for sidx in range(len(runnable.state.states_val)):
                job = runnable.to_job(sidx)
                checksum = job.checksum
                job.results_dict[None] = (sidx, checksum)
                runnable.results_dict[sidx] = (None, checksum)
                if is_workflow(runnable):
                    runnables.append(
                        asyncio.create_task(self._run_workflow(job, state_idx=sidx))
                    )
                else:
                    # tasks are submitted to worker for execution
                    futures.add(self.worker.run_el(job, sidx))
        else:
            job = runnable.to_job(None)
            checksum = job.checksum
            job.results_dict[None] = (None, checksum)
            if is_workflow(runnable):
                # runnables = [asyncio.create_task(self._run_workflow(job))]
                runnable, _ = await self._run_workflow(job)
            else:
                # submit task to worker
                futures.add(self.worker.run_el(job))

        if return_task:
            # run coroutines concurrently and wait for execution
            # TODO: ensure unification of states
            done = await asyncio.gather(*runnables)
            return runnable
        else:
            return futures

    # async def submit(self, runnable):
    #     """
    #     Entrypoint for ``Task`` submission

    #     Parameters
    #     ----------
    #     runnable : Task
    #         Task instance (``Task``, ``Workflow``)
    #     sync : bool
    #         Run outside of loop
    #     """
    #     res = await self.run(runnable)
    #     return res

    def close(self):
        self.worker.close()


async def get_runnable_tasks(graph, remaining_tasks):
    """Parse a graph and return all runnable tasks"""
    didx, tasks = [], []
    for idx, task in enumerate(remaining_tasks):
        # are all predecessors finished
        if is_runnable(graph, task):  # consider states
            didx.append(idx)
            tasks.append(task)
    for i in sorted(didx, reverse=True):
        del remaining_tasks[i]

    return remaining_tasks, tasks
