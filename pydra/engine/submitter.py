import asyncio

from .workers import SerialWorker, ConcurrentFuturesWorker
from .core import is_workflow
from .helpers import get_open_loop

import logging

logger = logging.getLogger("pydra.submitter")


class Submitter:
    # TODO: runnable in init or run
    def __init__(self, plugin):
        self.loop = get_open_loop()
        self._own_loop = not self.loop.is_running()
        self.plugin = plugin
        if self.plugin == "serial":
            self.worker = SerialWorker()
        elif self.plugin == "cf":
            self.worker = ConcurrentFuturesWorker()
        else:
            raise Exception("plugin {} not available".format(self.plugin))
        self.worker.loop = self.loop

    def __call__(self, runnable, cache_locations=None):
        if cache_locations is not None:
            runnable.cache_locations = cache_locations
        # creating all connections and calculating the checksum of the graph before running
        if is_workflow(runnable):
            for nd in runnable.graph.nodes:
                runnable.create_connections(nd)
                if runnable._custom_cache_dir and nd.allow_cache_override:
                    nd.cache_dir = runnable.cache_dir
            runnable.inputs._graph_checksums = [
                nd.checksum for nd in runnable.graph_sorted
            ]
        # Start event loop and run workflow/task
        if is_workflow(runnable) and runnable.state is None:
            self.loop.run_until_complete(runnable._run(self))
        else:
            self.loop.run_until_complete(self.submit(runnable, return_task=True))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    async def _run_workflow(self, wf):
        """
        Expands and executes a stateless ``Workflow``.

        Parameters
        ----------
        wf : Workflow
            Workflow Task object

        Returns
        -------
        wf : Workflow
            The computed workflow
        """
        # creating a copy of the graph that will be modified
        # the copy contains new lists with original runnable objects
        graph_copy = wf.graph.copy()
        # keep track of local futures
        task_futures = set()
        while not wf.done_all_tasks:
            tasks = get_runnable_tasks(graph_copy)
            if not tasks and not task_futures:
                raise Exception("Nothing queued or todo - something went wrong")
            for task in tasks:
                # grab inputs if needed
                logger.debug(f"Retrieving inputs for {task}")
                # TODO: add state idx to retrieve values to reduce waiting
                task.inputs.retrieve_values(wf)
                # checksum has to be updated, so resetting
                task._checksum = None
                if is_workflow(task) and not task.state:
                    # ensure workflow is executed
                    await task._run(self)
                else:
                    for fut in await self.submit(task):
                        task_futures.add(fut)

            done, task_futures = await self.worker.fetch_finished(task_futures)
            for fut in done:
                # let caching take care of results
                await fut
        return wf

    async def submit(self, runnable, return_task=False):
        """
        Coroutine entrypoint for task submission.

        Removes state from task and adds one or more
        asyncio ``Task``s to the running loop.

         Possible routes for the runnable
         1. ``Workflow`` w/ state: separate states into individual jobs and run() each
         2. ``Workflow`` w/o state: await graph expansion
         3. ``Task`` w/ state: separate states and submit to worker
         4. ``Task`` w/o state: submit to worker

        Parameters
        ----------
        runnable : Task
            Task instance (``Task``, ``Workflow``)
        return_task : bool (False)
            Option to return runnable instead of futures once all states have finished

        Returns
        -------
        futures : set
            Asyncio tasks
        runnable : ``Task``
            Signals the end of submission
        """
        # ensure worker is using same loop
        futures = set()
        if runnable.state:
            runnable.state.prepare_states(runnable.inputs)
            runnable.state.prepare_inputs()
            logger.debug(
                f"Expanding {runnable} into {len(runnable.state.states_val)} states"
            )
            for sidx in range(len(runnable.state.states_val)):
                job = runnable.to_job(sidx)
                job.results_dict[None] = (sidx, job.checksum)
                runnable.results_dict[sidx] = (None, job.checksum)
                logger.debug(
                    f'Submitting runnable {job}{str(sidx) if sidx is not None else ""}'
                )
                if is_workflow(runnable):
                    # job has no state anymore
                    futures.add(asyncio.create_task(job._run(self)))
                else:
                    # tasks are submitted to worker for execution
                    futures.add(self.worker.run_el(job))
        else:
            runnable.results_dict[None] = (None, runnable.checksum)
            if is_workflow(runnable):
                # this should only be reached through the job's `run()` method
                await self._run_workflow(runnable)
            else:
                # submit task to worker
                futures.add(self.worker.run_el(runnable))

        if return_task:
            # run coroutines concurrently and wait for execution
            if futures:
                # wait until all states complete or error
                await asyncio.gather(*futures)
        # otherwise pass along futures to be awaited independently
        else:
            return futures

    def close(self):
        # do not close previously running loop
        if self._own_loop:
            self.loop.close()
        self.worker.close()


def get_runnable_tasks(graph):
    """Parse a graph and return all runnable tasks"""
    tasks = []
    to_remove = []
    for tsk in graph.sorted_nodes:
        # since the list is sorted (breadth-first) we can stop
        # when we find a task that depends on any task that is already in tasks
        if set(graph.predecessors[tsk.name]).intersection(set(tasks)):
            break
        if is_runnable(graph, tsk):
            tasks.append(tsk)
            to_remove.append(tsk)
    # removing tasks that are ready to run from the graph
    for nd in to_remove:
        graph.remove_nodes(nd)
    return tasks


def is_runnable(graph, obj):
    """Check if a task within a graph is runnable"""
    connections_to_remove = []
    for pred in graph.predecessors[obj.name]:
        is_done = pred.done
        if not is_done:
            return False
        else:
            connections_to_remove.append(pred)
    # removing nodes that are done from connections
    for nd in connections_to_remove:
        graph.remove_nodes_connections(nd)
    return True
