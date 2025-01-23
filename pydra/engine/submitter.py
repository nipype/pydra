"""Handle execution backends."""

import asyncio
import typing as ty
import pickle
import os
from pathlib import Path
from tempfile import mkdtemp
from copy import copy
from collections import defaultdict
from .workers import Worker, WORKERS
from .core import is_workflow
from .graph import DiGraph
from .helpers import (
    get_open_loop,
    list_fields,
)
from pydra.utils.hash import PersistentCache
from .state import StateIndex
from pydra.utils.typing import StateArray
from .audit import Audit
from .core import Task
from pydra.utils.messenger import AuditFlag, Messenger
from pydra.utils import user_cache_dir

import logging

logger = logging.getLogger("pydra.submitter")

if ty.TYPE_CHECKING:
    from .node import Node
    from .specs import TaskDef, WorkflowDef
    from .environments import Environment

# Used to flag development mode of Audit
develop = False

WORKER_KWARG_FAIL_NOTE = "Attempting to instantiate worker submitter"


class Submitter:
    """Send a task to the execution backend.

    Parameters
    ----------
    cache_dir : os.PathLike, optional
        Cache directory where the working directory/results for the task will be
        stored, by default None
    worker : str or Worker, optional
        The worker to use, by default "cf"
    environment: Environment, optional
        The execution environment to use, by default None
    rerun : bool, optional
        Whether to force the re-computation of the task results even if existing
        results are found, by default False
    cache_locations : list[os.PathLike], optional
        Alternate cache locations to check for pre-computed results, by default None
    audit_flags : AuditFlag, optional
        Auditing configuration, by default AuditFlag.NONE
    messengers : list, optional
        Messengers, by default None
    messenger_args : dict, optional
        Messenger arguments, by default None
    **kwargs : dict
        Keyword arguments to pass on to the worker initialisation
    """

    def __init__(
        self,
        cache_dir: os.PathLike | None = None,
        worker: ty.Union[str, ty.Type[Worker]] = "cf",
        environment: "Environment | None" = None,
        rerun: bool = False,
        cache_locations: list[os.PathLike] | None = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers: ty.Iterable[Messenger] | None = None,
        messenger_args: dict[str, ty.Any] | None = None,
        **kwargs,
    ):

        self.audit = Audit(
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            develop=develop,
        )
        if cache_dir is None:
            cache_dir = user_cache_dir / "run-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
        elif not cache_dir.exists():
            raise ValueError(f"Cache directory {str(cache_dir)!r} does not exist")
        self.cache_dir = cache_dir
        self.cache_locations = cache_locations
        self.environment = environment
        self.rerun = rerun
        self.loop = get_open_loop()
        self._own_loop = not self.loop.is_running()
        if isinstance(worker, str):
            self.worker_name = worker
            try:
                worker_cls = WORKERS[self.worker_name]
            except KeyError:
                raise NotImplementedError(f"No worker for '{self.worker_name}' plugin")
        else:
            try:
                self.worker_name = worker.plugin_name
            except AttributeError:
                raise ValueError("Worker class must have a 'plugin_name' str attribute")
            worker_cls = worker
        try:
            self._worker = worker_cls(**kwargs)
        except TypeError as e:
            e.add_note(WORKER_KWARG_FAIL_NOTE)
            raise
        self.worker_kwargs = kwargs
        self._worker.loop = self.loop

    @property
    def worker(self):
        if self._worker is None:
            raise RuntimeError(
                "Cannot access worker of unpickeld submitter (typically in subprocess)"
            )
        return self._worker

    def __call__(
        self,
        task_def: "TaskDef",
    ):
        """Submitter run function."""

        task_def._check_rules()
        # If the outer task is split, create an implicit workflow to hold the split nodes
        if task_def._splitter:

            from pydra.design import workflow

            output_types = {o.name: list[o.type] for o in list_fields(task_def.Outputs)}

            # We need to use a new variable as task_def will be overwritten by the time
            # the Split workflow constructor is called
            node_def = task_def

            @workflow.define(outputs=output_types)
            def Split():
                node = workflow.add(node_def)
                return tuple(getattr(node, o) for o in output_types)

            task_def = Split()

        elif task_def._combiner:
            raise ValueError(
                f"Task {self} is marked for combining, but not splitting. "
                "Use the `split` method to split the task before combining."
            )
        task = Task(task_def, submitter=self, name="task", environment=self.environment)
        if task.is_async:
            self.loop.run_until_complete(self.expand_runnable_async(task))
        else:
            self.expand_runnable(task)
        PersistentCache().clean_up()
        result = task.result()
        if result is None:
            if task.lockfile.exists():
                raise RuntimeError(
                    f"Task {task} has a lockfile, but no result was found. "
                    "This may be due to another submission process running, or the hard "
                    "interrupt (e.g. a debugging abortion) interrupting a previous run. "
                    f"In the case of an interrupted run, please remove {str(task.lockfile)!r} "
                    "and resubmit."
                )
            raise RuntimeError(f"Task {task} has no result in {str(task.output_dir)!r}")
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries or those that should not be pickled
        # When unpickled (in another process) the submitter can't be called
        state["loop"] = None
        state["_worker"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the loop and worker
        self.loop = get_open_loop()
        self._worker = WORKERS[self.worker_name](**self.worker_kwargs)
        self.worker.loop = self.loop

    def expand_runnable(self, task: "Task"):
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
        task.run(rerun=self.rerun)

    async def expand_runnable_async(
        self, task: "Task", wait=False
    ) -> set[ty.Coroutine] | None:
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
        futures = set()

        if is_workflow(task.definition):
            futures.add(asyncio.create_task(task.run(self.rerun)))
        else:
            task_pkl = await prepare_runnable(task)
            futures.add(self.worker.run((task_pkl, task), rerun=self.rerun))

        if wait and futures:
            # if wait is True, we are at the end of the graph / state expansion.
            # Once the remaining jobs end, we will exit `submit_from_call`
            await asyncio.gather(*futures)
            return
        # pass along futures to be awaited independently
        return futures

    def expand_workflow(self, workflow_task: "Task[WorkflowDef]") -> None:
        """Expands and executes a workflow task synchronously. Typically only used during
        debugging and testing, as the asynchronous version is more efficient.

        Parameters
        ----------
        task : :obj:`~pydra.engine.core.Task[WorkflowDef]`
            Workflow Task object

        """
        # Construct the workflow
        wf = workflow_task.definition.construct()
        # Generate the execution graph
        exec_graph = wf.execution_graph(submitter=self)
        tasks = self.get_runnable_tasks(exec_graph)
        while tasks or any(not n.done for n in exec_graph.nodes):
            for task in tasks:
                # grab inputs if needed
                logger.debug(f"Retrieving inputs for {task}")
                # TODO: add state idx to retrieve values to reduce waiting
                task.definition._retrieve_values(wf)
                self.worker.run(task, rerun=self.rerun)
            tasks = self.get_runnable_tasks(exec_graph)
        workflow_task.return_values = {"workflow": wf, "exec_graph": exec_graph}

    async def expand_workflow_async(self, task: "Task[WorkflowDef]") -> None:
        """
        Expand and execute a workflow task asynchronously.

        Parameters
        ----------
        task : :obj:`~pydra.engine.core.Task[WorkflowDef]`
            Workflow Task object
        """
        wf = task.definition.construct()
        # Generate the execution graph
        exec_graph = wf.execution_graph(submitter=self)
        # keep track of pending futures
        task_futures = set()
        tasks = self.get_runnable_tasks(exec_graph)
        while tasks or task_futures or any(not n.done for n in exec_graph.nodes):
            if not tasks and not task_futures:
                # it's possible that task_futures is empty, but not able to get any
                # tasks from graph_copy (using get_runnable_tasks)
                # this might be related to some delays saving the files
                # so try to get_runnable_tasks for another minute
                ii = 0
                while not tasks and exec_graph.nodes:
                    tasks, follow_err = self.get_runnable_tasks(exec_graph)
                    ii += 1
                    # don't block the event loop!
                    await asyncio.sleep(1)
                    if ii > 60:
                        msg = (
                            f"Graph of '{wf}' workflow is not empty, but not able to get "
                            "more tasks - something has gone wrong when retrieving the "
                            "results predecessors:\n\n"
                        )
                        # Get blocked tasks and the predecessors they are waiting on
                        outstanding: dict[Task, list[Task]] = {
                            t: [
                                p for p in exec_graph.predecessors[t.name] if not p.done
                            ]
                            for t in exec_graph.sorted_nodes
                        }

                        hashes_have_changed = False
                        for task, waiting_on in outstanding.items():
                            if not waiting_on:
                                continue
                            msg += f"- '{task.name}' node blocked due to\n"
                            for pred in waiting_on:
                                if (
                                    pred.checksum
                                    != wf.inputs._graph_checksums[pred.name]
                                ):
                                    msg += (
                                        f"    - hash changes in '{pred.name}' node inputs. "
                                        f"Current values and hashes: {pred.inputs}, "
                                        f"{pred.inputs._hashes}\n"
                                    )
                                    hashes_have_changed = True
                                elif pred not in outstanding:
                                    msg += (
                                        f"    - undiagnosed issues in '{pred.name}' node, "
                                        "potentially related to file-system access issues "
                                    )
                            msg += "\n"
                        if hashes_have_changed:
                            msg += (
                                "Set loglevel to 'debug' in order to track hash changes "
                                "throughout the execution of the workflow.\n\n "
                                "These issues may have been caused by `bytes_repr()` methods "
                                "that don't return stable hash values for specific object "
                                "types across multiple processes (see bytes_repr() "
                                '"singledispatch "function in pydra/utils/hash.py).'
                                "You may need to write specific `bytes_repr()` "
                                "implementations (see `pydra.utils.hash.register_serializer`) "
                                "or `__bytes_repr__()` dunder methods to handle one "
                                "or more types in your interface inputs."
                            )
                        raise RuntimeError(msg)
            for task in tasks:
                # grab inputs if needed
                logger.debug(f"Retrieving inputs for {task}")
                # TODO: add state idx to retrieve values to reduce waiting
                task.definition._retrieve_values(wf)
                if is_workflow(task):
                    await task.run(self)
                # single task
                else:
                    task_futures.add(self.worker.run(task, rerun=self.rerun))
            task_futures = await self.worker.fetch_finished(task_futures)
            tasks = self.get_runnable_tasks(exec_graph)
        task.return_values = {"workflow": wf, "exec_graph": exec_graph}

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

    def get_runnable_tasks(
        self,
        graph: DiGraph,
    ) -> tuple[list["Task"], dict["NodeExecution", list[str]]]:
        """Parse a graph and return all runnable tasks.

        Parameters
        ----------
        graph : :obj:`~pydra.engine.graph.DiGraph`
            Graph object

        Returns
        -------
        tasks : list of :obj:`~pydra.engine.core.Task`
            List of runnable tasks
        following_err : dict[NodeToExecute, list[str]]
            Dictionary of tasks that are blocked by errored tasks
        """
        tasks = []
        not_started = set()
        node: NodeExecution
        for node in graph.sorted_nodes:
            if node.done:
                continue
            # since the list is sorted (breadth-first) we can stop
            # when we find a task that depends on any task that is already in tasks
            if set(graph.predecessors[node.name]).intersection(not_started):
                break
            # Record if the node has not been started
            if not node.started:
                not_started.add(node)
            tasks.extend(node.get_runnable_tasks(graph))
        return tasks

    @property
    def cache_dir(self):
        """Get the location of the cache directory."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, location):
        if location is not None:
            self._cache_dir = Path(location).resolve()
            self._cache_dir.mkdir(parents=False, exist_ok=True)
        else:
            self._cache_dir = mkdtemp()
            self._cache_dir = Path(self._cache_dir).resolve()


class NodeExecution:
    """A wrapper around a workflow node containing the execution state of the tasks that
    are generated from it"""

    name: str
    node: "Node"
    submitter: Submitter

    # List of tasks that were completed successfully
    successful: dict[StateIndex | None, list["Task"]]
    # List of tasks that failed
    errored: dict[StateIndex | None, "Task"]
    # List of tasks that couldn't be run due to upstream errors
    unrunnable: dict[StateIndex | None, list["Task"]]
    # List of tasks that are running
    running: dict[StateIndex | None, "Task"]
    # List of tasks that are waiting on other tasks to complete before they can be run
    waiting: dict[StateIndex | None, "Task"]

    _tasks: dict[StateIndex | None, "Task"] | None

    def __init__(self, node: "Node", submitter: Submitter):
        self.name = node.name
        self.node = node
        self.submitter = submitter
        # Initialize the state dictionaries
        self._tasks = None
        self.waiting = {}
        self.successful = {}
        self.errored = {}
        self.running = {}
        self.unrunnable = defaultdict(list)
        self.state_names = self.node.state.names

    @property
    def inputs(self) -> "Node.Inputs":
        return self.node.inputs

    @property
    def _definition(self) -> "Node":
        return self.node._definition

    @property
    def tasks(self) -> ty.Iterable["Task"]:
        if self._tasks is None:
            self._tasks = {t.state_index: t for t in self._generate_tasks()}
        return self._tasks.values()

    def task(self, index: StateIndex | None = None) -> "Task | list[Task]":
        """Get a task object for a given state index."""
        self.tasks  # Ensure tasks are loaded
        try:
            return self._tasks[index]
        except KeyError:
            if index is None:
                return StateArray(self._tasks.values())
            raise

    @property
    def started(self) -> bool:
        return (
            self.successful
            or self.errored
            or self.unrunnable
            or self.running
            or self.waiting
        )

    @property
    def done(self) -> bool:
        return self.started and not (self.running or self.waiting)

    @property
    def all_failed(self) -> bool:
        return (self.unrunnable or self.errored) and not (
            self.successful or self.waiting or self.running
        )

    def _generate_tasks(self) -> ty.Iterable["Task"]:
        if self.node.state is None:
            yield Task(
                definition=self.node._definition,
                submitter=self.submitter,
                name=self.node.name,
            )
        else:
            for index, split_defn in self.node._split_definition().items():
                yield Task(
                    definition=split_defn,
                    submitter=self.submitter,
                    name=self.node.name,
                    state_index=index,
                )

        #     if state_index is None:
        #         # if state_index=None, collecting all results
        #         if self.node.state.combiner:
        #             return self._combined_output(return_inputs=return_inputs)
        #         else:
        #             results = []
        #             for ind in range(len(self.node.state.inputs_ind)):
        #                 checksum = self.checksum_states(state_index=ind)
        #                 result = load_result(checksum, cache_locations)
        #                 if result is None:
        #                     return None
        #                 results.append(result)
        #             if return_inputs is True or return_inputs == "val":
        #                 return list(zip(self.node.state.states_val, results))
        #             elif return_inputs == "ind":
        #                 return list(zip(self.node.state.states_ind, results))
        #             else:
        #                 return results
        #     else:  # state_index is not None
        #         if self.node.state.combiner:
        #             return self._combined_output(return_inputs=return_inputs)[
        #                 state_index
        #             ]
        #         result = load_result(self.checksum_states(state_index), cache_locations)
        #         if return_inputs is True or return_inputs == "val":
        #             return (self.node.state.states_val[state_index], result)
        #         elif return_inputs == "ind":
        #             return (self.node.state.states_ind[state_index], result)
        #         else:
        #             return result
        # else:
        #     return load_result(self._definition._checksum, cache_locations)

    def get_runnable_tasks(self, graph: DiGraph) -> list["Task"]:
        """For a given node, check to see which tasks have been successfully run, are ready
        to run, can't be run due to upstream errors, or are waiting on other tasks to complete.

        Parameters
        ----------
        node : :obj:`~pydra.engine.node.Node`
            The node object to get the tasks for
        graph : :obj:`~pydra.engine.graph.DiGraph`
            Graph object


        Returns
        -------
        runnable : list[NodeExecution]
            List of tasks that are ready to run
        """
        runnable: list["Task"] = []
        self.tasks  # Ensure tasks are loaded
        if not self.started:
            self.waiting = copy(self._tasks)
        # Check to see if any previously running tasks have completed
        for index, task in list(self.running.items()):
            if task.done:
                self.successful[task.state_index] = self.running.pop(index)
            elif task.errored:
                self.errored[task.state_index] = self.running.pop(index)
        # Check to see if any waiting tasks are now runnable/unrunnable
        for index, task in list(self.waiting.items()):
            pred: NodeExecution
            is_runnable = True
            for pred in graph.predecessors[self.node.name]:
                if index not in pred.successful:
                    is_runnable = False
                    if index in pred.errored:
                        self.unrunnable[index].append(self.waiting.pop(index))
                    if index in pred.unrunnable:
                        self.unrunnable[index].extend(pred.unrunnable[index])
                        self.waiting.pop(index)
                    break
            if is_runnable:
                runnable.append(self.waiting.pop(index))
        self.running.update({t.state_index: t for t in runnable})
        return runnable


async def prepare_runnable(runnable):
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
                blocking.append((pred, ", ".join(matching_name)))
        if blocking:
            blocked.append(
                f"\n{tsk.name} ({tsk.checksum}) is blocked by "
                + "; ".join(
                    f"{pred.name} ({pred.checksum}), which matches names of [{matching}]"
                    for pred, matching in blocking
                )
            )
    return blocked
