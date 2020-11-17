"""Handle execution backends."""
import asyncio
import time
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
            raise Exception(f"plugin {self.plugin} not available")
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
        for nd in wf.graph.nodes:
            if nd.allow_cache_override:
                nd.cache_dir = wf.cache_dir

        # creating a copy of the graph that will be modified
        # the copy contains new lists with original runnable objects
        graph_copy = wf.graph.copy()
        # keep track of pending futures
        task_futures = set()
        tasks, tasks_follow_errored = get_runnable_tasks(graph_copy)
        while tasks or task_futures or graph_copy.nodes:
            if not tasks and not task_futures:
                # it's possible that task_futures is empty, but not able to get any
                # tasks from graph_copy (using get_runnable_tasks)
                # this might be related to some delays saving the files
                # so try to get_runnable_tasks for another minut
                ii = 0
                while not tasks and graph_copy.nodes:
                    tasks, follow_err = get_runnable_tasks(graph_copy)
                    ii += 1
                    time.sleep(1)
                    if ii > 60:
                        raise Exception(
                            "graph is not empty, but not able to get more tasks - something is wrong (e.g. with the filesystem)"
                        )
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
            tasks, follow_err = get_runnable_tasks(graph_copy)
            # updating tasks_errored
            for key, val in follow_err.items():
                tasks_follow_errored.setdefault(key, [])
                tasks_follow_errored[key] += val

        for key, val in tasks_follow_errored.items():
            setattr(getattr(wf, key), "_errored", val)
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
    # tasks that follow task that raises an error
    following_err = dict()
    for tsk in graph.sorted_nodes:
        if tsk not in graph.sorted_nodes:
            continue
        # since the list is sorted (breadth-first) we can stop
        # when we find a task that depends on any task that is already in tasks
        if set(graph.predecessors[tsk.name]).intersection(set(tasks)):
            break
        _is_runnable = is_runnable(graph, tsk)
        if _is_runnable is True:
            tasks.append(tsk)
            to_remove.append(tsk)
        elif _is_runnable is False:
            continue
        else:  # a previous task had an error
            errored_task = _is_runnable
            # removing all successors of the errored task
            for task_err in errored_task:
                task_to_remove = graph.remove_successors_nodes(task_err)
                for tsk in task_to_remove:
                    # adding tasks that were removed from the graph
                    # due to the error in the errored_task
                    following_err.setdefault(tsk, [])
                    following_err[tsk].append(task_err.name)

    # removing tasks that are ready to run from the graph
    for nd in to_remove:
        graph.remove_nodes(nd)
    return tasks, following_err


def is_runnable(graph, obj):
    """Check if a task within a graph is runnable."""
    connections_to_remove = []
    pred_errored = []
    is_done = None
    for pred in graph.predecessors[obj.name]:
        try:
            is_done = pred.done
        except ValueError:
            pred_errored.append(pred)

        if is_done is True:
            connections_to_remove.append(pred)
        elif is_done is False:
            return False

    if pred_errored:
        return pred_errored

    # removing nodes that are done from connections
    for nd in connections_to_remove:
        graph.remove_nodes_connections(nd)

    return True
