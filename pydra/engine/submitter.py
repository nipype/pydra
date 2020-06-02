"""Handle execution backends."""
import asyncio
from .workers import SerialWorker, ConcurrentFuturesWorker, SlurmWorker, DaskWorker
from .core import is_workflow
from .helpers import get_open_loop, load_and_run_async

import logging

logger = logging.getLogger("pydra.submitter")


# TODO: runnable in init or run
class Submitter:
    """Send a task to the execution backend."""

    def __init__(self, plugin="cf", **kwargs):
        """
        Initialize task submission.

        Parameters
        ----------
        plugin : :obj:`str`
            The identifier of the execution backend.
            Default is ``cf`` (Concurrent Futures).

        """
        self.loop = get_open_loop()
        self._own_loop = not self.loop.is_running()
        self.plugin = plugin
        if self.plugin == "serial":
            self.worker = SerialWorker()
        elif self.plugin == "cf":
            self.worker = ConcurrentFuturesWorker(**kwargs)
        elif self.plugin == "slurm":
            self.worker = SlurmWorker(**kwargs)
        elif self.plugin == "dask":
            self.worker = DaskWorker(**kwargs)
        else:
            raise Exception("plugin {} not available".format(self.plugin))
        self.worker.loop = self.loop

    def __call__(self, runnable, cache_locations=None, rerun=False):
        """Submit."""
        if cache_locations is not None:
            runnable.cache_locations = cache_locations
        # creating all connections and calculating the checksum of the graph before running
        if is_workflow(runnable):
            # TODO: no prepare state ?
            for nd in runnable.graph.nodes:
                runnable.create_connections(nd)
                if nd.allow_cache_override:
                    nd.cache_dir = runnable.cache_dir
            runnable.inputs._graph_checksums = [
                nd.checksum for nd in runnable.graph_sorted
            ]
        if is_workflow(runnable) and runnable.state is None:
            self.loop.run_until_complete(self.submit_workflow(runnable, rerun=rerun))
        else:
            self.loop.run_until_complete(self.submit(runnable, wait=True, rerun=rerun))
        if is_workflow(runnable):
            # resetting all connections with LazyFields
            runnable._reset()
        return runnable.result()

    async def submit_workflow(self, workflow, rerun=False):
        """Distribute or initiate workflow execution."""
        if is_workflow(workflow):
            if workflow.plugin and workflow.plugin != self.plugin:
                # dj: this is not tested!!! TODO
                await self.worker.run_el(workflow, rerun=rerun)
            else:
                await workflow._run(self, rerun=rerun)
        else:  # could be a tuple with paths to pickle files wiith tasks and inputs
            ind, wf_main_pkl, wf_orig = workflow
            if wf_orig.plugin and wf_orig.plugin != self.plugin:
                # dj: this is not tested!!! TODO
                await self.worker.run_el(workflow, rerun=rerun)
            else:
                await load_and_run_async(
                    task_pkl=wf_main_pkl, ind=ind, submitter=self, rerun=rerun
                )

    async def submit(self, runnable, wait=False, rerun=False):
        """
        Coroutine entrypoint for task submission.

        Removes any states from `runnable`. If `wait` is
        set to False (default), aggregates all worker
        execution coroutines and returns them. If `wait` is
        True, waits for all coroutines to complete / error
        and returns None.

        Parameters
        ----------
        runnable : pydra Task
            Task instance (`Task`, `Workflow`)
        wait : bool (False)
            Await all futures before completing

        Returns
        -------
        futures : set or None
            Coroutines for :class:`~pydra.engine.core.TaskBase` execution.

        """
        futures = set()
        if runnable.state:
            runnable.state.prepare_states(runnable.inputs)
            runnable.state.prepare_inputs()
            logger.debug(
                f"Expanding {runnable} into {len(runnable.state.states_val)} states"
            )
            task_pkl = runnable.pickle_task()

            for sidx in range(len(runnable.state.states_val)):
                job_tuple = (sidx, task_pkl, runnable)
                if is_workflow(runnable):
                    # job has no state anymore
                    futures.add(self.submit_workflow(job_tuple, rerun=rerun))
                else:
                    # tasks are submitted to worker for execution
                    futures.add(self.worker.run_el(job_tuple, rerun=rerun))
        else:
            if is_workflow(runnable):
                await self._run_workflow(runnable, rerun=rerun)
            else:
                # submit task to worker
                futures.add(self.worker.run_el(runnable, rerun=rerun))

        if wait and futures:
            # run coroutines concurrently and wait for execution
            # wait until all states complete or error
            await asyncio.gather(*futures)
            return
        # pass along futures to be awaited independently
        return futures

    async def _run_workflow(self, wf, rerun=False):
        """
        Expand and execute a stateless :class:`~pydra.engine.core.Workflow`.

        Parameters
        ----------
        wf : :obj:`~pydra.engine.core.Workflow`
            Workflow Task object

        Returns
        -------
        wf : :obj:`pydra.engine.core.Workflow`
            The computed workflow

        """
        # creating a copy of the graph that will be modified
        # the copy contains new lists with original runnable objects
        graph_copy = wf.graph.copy()
        # keep track of pending futures
        task_futures = set()
        while not wf.done_all_tasks or len(task_futures):
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
                    await self.submit_workflow(task, rerun=rerun)
                else:
                    for fut in await self.submit(task, rerun=rerun):
                        task_futures.add(fut)
            task_futures = await self.worker.fetch_finished(task_futures)
        return wf

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """
        Close submitter.

        Do not close previously running loop.

        """
        self.worker.close()
        if self._own_loop:
            self.loop.close()


def get_runnable_tasks(graph):
    """Parse a graph and return all runnable tasks."""
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
    """Check if a task within a graph is runnable."""
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
