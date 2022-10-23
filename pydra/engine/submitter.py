"""Handle execution backends."""
import asyncio
import pickle
from uuid import uuid4
from .workers import WORKERS
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
        try:
            self.worker = WORKERS[self.plugin](**kwargs)
        except KeyError:
            raise NotImplementedError(f"No worker for {self.plugin}")
        self.worker.loop = self.loop

    def __call__(self, runnable, cache_locations=None, rerun=False):
        """Submitter run function."""
        if cache_locations is not None:
            runnable.cache_locations = cache_locations
        self.loop.run_until_complete(self.submit_from_call(runnable, rerun))
        return runnable.result()

    async def submit_from_call(self, runnable, rerun):
        """
        This coroutine should only be called once per Submitter call,
        and serves as the bridge between sync/async lands.

        There are 4 potential paths based on the type of runnable:
        0) Workflow has a different plugin than a submitter
        1) Workflow without State
        2) Task without State
        3) (Workflow or Task) with State

        Once Python 3.10 is the minimum, this should probably be refactored into using
        structural pattern matching.
        """
        if is_workflow(runnable):
            # connect and calculate the checksum of the graph before running
            runnable._connect_and_propagate_to_tasks(override_task_caches=True)
            # 0
            if runnable.plugin and runnable.plugin != self.plugin:
                # if workflow has a different plugin it's treated as a single element
                await self.worker.run_el(runnable, rerun=rerun)
            # 1
            if runnable.state is None:
                await runnable._run(self, rerun=rerun)
            # 3
            else:
                await self.expand_runnable(runnable, wait=True, rerun=rerun)
            runnable._reset()
        else:
            # 2
            if runnable.state is None:
                # run_el should always return a coroutine
                await self.worker.run_el(runnable, rerun=rerun)
            # 3
            else:
                await self.expand_runnable(runnable, wait=True, rerun=rerun)
        return True

    async def expand_runnable(self, runnable, wait=False, rerun=False):
        """
        This coroutine handles state expansion.

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
        if runnable.plugin and runnable.plugin != self.plugin:
            raise NotImplementedError()

        futures = set()
        if runnable.state is None:
            raise Exception("Only runnables with state should reach here")

        task_pkl = await prepare_runnable_with_state(runnable)

        for sidx in range(len(runnable.state.states_val)):
            if is_workflow(runnable):
                # job has no state anymore
                futures.add(
                    # This unpickles and runs workflow - why are we pickling?
                    asyncio.create_task(load_and_run_async(task_pkl, sidx, self, rerun))
                )
            else:
                futures.add(self.worker.run_el((sidx, task_pkl, runnable), rerun=rerun))

        if wait and futures:
            # if wait is True, we are at the end of the graph / state expansion.
            # Once the remaining jobs end, we will exit `submit_from_call`
            await asyncio.gather(*futures)
            return
        # pass along futures to be awaited independently
        return futures

    async def expand_workflow(self, wf, rerun=False):
        """
        Expand and execute a stateless :class:`~pydra.engine.core.Workflow`.
        This method is only reached by `Workflow._run_task`.

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
        # resetting uid for nodes in the copied workflows
        for nd in graph_copy.nodes:
            nd._uid = uuid4().hex
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
                    # don't block the event loop!
                    await asyncio.sleep(1)
                    if ii > 60:
                        blocked = _list_blocked_tasks(graph_copy)
                        get_runnable_tasks(graph_copy)
                        raise Exception(
                            "graph is not empty, but not able to get more tasks "
                            "- something may have gone wrong when retrieving the results "
                            "of predecessor tasks caused by a file-system error or a bug "
                            "in the internal workflow logic.\n\nBlocked tasks\n-------------\n"
                            + "\n".join(blocked)
                        )
            for task in tasks:
                # grab inputs if needed
                logger.debug(f"Retrieving inputs for {task}")
                # TODO: add state idx to retrieve values to reduce waiting
                task.inputs.retrieve_values(wf)
                if task.state:
                    for fut in await self.expand_runnable(task, rerun=rerun):
                        task_futures.add(fut)
                # expand that workflow
                elif is_workflow(task):
                    await task._run(self, rerun=rerun)
                # single task
                else:
                    task_futures.add(self.worker.run_el(task, rerun=rerun))
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


async def prepare_runnable_with_state(runnable):
    runnable.state.prepare_states(runnable.inputs, cont_dim=runnable.cont_dim)
    runnable.state.prepare_inputs()
    logger.debug(f"Expanding {runnable} into {len(runnable.state.states_val)} states")
    return runnable.pickle_task()


def _list_blocked_tasks(graph):
    """Generates a list of tasks that can't be run and predecessors that are blocking
    them to help debugging of broken workflows"""
    blocked = []
    for tsk in graph.sorted_nodes:
        blocking = []
        for pred in graph.predecessors[tsk.name]:
            if not pred.done:
                matching_name = []
                for cache_loc in tsk.cache_locations:
                    for tsk_work_dir in cache_loc.iterdir():
                        if (tsk_work_dir / "_task.pklz").exists():
                            with open(tsk_work_dir / "_task.pklz", "rb") as f:
                                saved_tsk = pickle.load(f)
                            if saved_tsk.name == pred.name:
                                matching_name.append(
                                    f"{saved_tsk.name} ({tsk_work_dir.name})"
                                )
                blocking.append(pred, ", ".join(matching_name))
        if blocking:
            blocked.append(
                f"\n{tsk.name} ({tsk.checksum}) is blocked by "
                + "; ".join(
                    f"{pred.name} ({pred.checksum}), which matches names of [{matching}]"
                    for pred, matching in blocking
                )
            )
    return blocked
