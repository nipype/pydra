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
from .helpers import get_open_loop, load_and_run_async
from pydra.utils.hash import PersistentCache
from .state import StateIndex
from .audit import Audit
from .core import Task
from pydra.utils.messenger import AuditFlag, Messenger

import logging

logger = logging.getLogger("pydra.submitter")

if ty.TYPE_CHECKING:
    from .node import Node
    from .specs import TaskDef, WorkflowDef
    from .environments import Environment

# Used to flag development mode of Audit
develop = False


class Submitter:
    """Send a task to the execution backend."""

    def __init__(
        self,
        worker: ty.Union[str, ty.Type[Worker]] = "cf",
        cache_dir: os.PathLike | None = None,
        cache_locations: list[os.PathLike] | None = None,
        environment: "Environment | None" = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers: ty.Iterable[Messenger] | None = None,
        messenger_args: dict[str, ty.Any] | None = None,
        rerun: bool = False,
        **kwargs,
    ):
        """
        Initialize task submission.

        Parameters
        ----------
        plugin : :obj:`str` or :obj:`ty.Type[pydra.engine.core.Worker]`
            Either the identifier of the execution backend or the worker class itself.
            Default is ``cf`` (Concurrent Futures).
        **kwargs
            Additional keyword arguments to pass to the worker.

        """

        self.audit = Audit(
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            develop=develop,
        )
        self.cache_dir = cache_dir
        self.cache_locations = cache_locations
        self.environment = environment
        self.rerun = rerun
        self.loop = get_open_loop()
        self._own_loop = not self.loop.is_running()
        if isinstance(worker, str):
            self.plugin = worker
            try:
                worker_cls = WORKERS[self.plugin]
            except KeyError:
                raise NotImplementedError(f"No worker for '{self.plugin}' plugin")
        else:
            try:
                self.plugin = worker.plugin_name
            except AttributeError:
                raise ValueError("Worker class must have a 'plugin_name' str attribute")
            worker_cls = worker
        self.worker = worker_cls(**kwargs)
        self.worker.loop = self.loop

    def __call__(
        self,
        task_def: "TaskDef",
    ):
        """Submitter run function."""

        task = Task(task_def, submitter=self, name="task")
        self.loop.run_until_complete(self.submit_from_call(task))
        PersistentCache().clean_up()
        return task.result()

    async def submit_from_call(self, task: "Task"):
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
        if is_workflow(task):  # TODO: env to wf
            # connect and calculate the checksum of the graph before running
            task._create_graph_connections()  # override_task_caches=True)
            # 0
            if task.plugin and task.plugin != self.plugin:
                # if workflow has a different plugin it's treated as a single element
                await self.worker.run(task, rerun=self.rerun)
            # 1
            # if runnable.state is None:
            #     await runnable._run(self, rerun=rerun)
            # # 3
            # else:
            await self.expand_runnable(task, wait=True)
        else:
            # 2
            await self.expand_runnable(task, wait=True)  # TODO
        return True

    async def expand_runnable(self, runnable: "Task", wait=False):
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

        task_pkl = await prepare_runnable(runnable)

        if is_workflow(runnable):
            # job has no state anymore
            futures.add(
                # This unpickles and runs workflow - why are we pickling?
                asyncio.create_task(load_and_run_async(task_pkl, self, self.rerun))
            )
        else:
            futures.add(self.worker.run((task_pkl, runnable), rerun=self.rerun))

        if wait and futures:
            # if wait is True, we are at the end of the graph / state expansion.
            # Once the remaining jobs end, we will exit `submit_from_call`
            await asyncio.gather(*futures)
            return
        # pass along futures to be awaited independently
        return futures

    async def expand_workflow(self, task: "Task[WorkflowDef]"):
        """
        Expand and execute a stateless :class:`~pydra.engine.core.Workflow`.
        This method is only reached by `Workflow._run_task`.

        Parameters
        ----------
        task : :obj:`~pydra.engine.core.WorkflowTask`
            Workflow Task object

        Returns
        -------
        wf : :obj:`pydra.engine.workflow.Workflow`
            The computed workflow

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
                if task.state:
                    for fut in await self.expand_runnable(task):
                        task_futures.add(fut)
                # expand that workflow
                elif is_workflow(task):
                    await task.run(self)
                # single task
                else:
                    task_futures.add(self.worker.run(task, rerun=self.rerun))
            task_futures = await self.worker.fetch_finished(task_futures)
            tasks = self.get_runnable_tasks(exec_graph)
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
            tasks.extend(node.get_runnable_tasks(graph, self))
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
        self.waiting = []
        self.successful = []
        self.errored = []
        self.running = []
        self.unrunnable = defaultdict(list)
        self.state_names = self.node.state_names

    def __getattr__(self, name: str) -> ty.Any:
        """Delegate attribute access to the underlying node."""
        return getattr(self.node, name)

    @property
    def tasks(self) -> ty.Iterable["Task"]:
        if self._tasks is None:
            self._tasks = {t.state_index: t for t in self._generate_tasks()}
        return self._tasks.values()

    def task(self, index: StateIndex | None = None) -> "Task":
        """Get a task object for a given state index."""
        self.tasks  # Ensure tasks are loaded
        try:
            return self._tasks[index]
        except KeyError:
            if index is None:
                raise KeyError(
                    f"{self!r} has been split, so a state index must be provided"
                ) from None
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
        if not self.started:
            self.waiting = copy(self._tasks)
        # Check to see if any previously running tasks have completed
        for index, task in copy(self.running.items()):
            if task.done:
                self.successful[task.state_index] = self.running.pop(index)
            elif task.errored:
                self.errored[task.state_index] = self.running.pop(index)
        # Check to see if any waiting tasks are now runnable/unrunnable
        for index, task in copy(self.waiting.items()):
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
