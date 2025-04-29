import logging
import inspect
import typing as ty
from copy import copy
from collections import defaultdict
from typing import Self
import attrs
from pydra.compose import workflow
from pydra.compose.base import Task, Outputs
from pydra.engine.graph import DiGraph, INPUTS_NODE_NAME, OUTPUTS_NODE_NAME
from pydra.engine import state
from pydra.engine.lazy import LazyInField, LazyOutField
from pydra.utils.hash import hash_function, Cache
from pydra.engine.state import State
from pydra.engine.node import Node
from pydra.engine.hooks import (
    TaskHooks,
)
from pydra.engine.submitter import Submitter, NodeExecution
from pydra.utils.general import (
    attrs_values,
    asdict,
    get_fields,
)
from pydra.utils.typing import is_lazy
from pydra.environments.base import Environment

logger = logging.getLogger("pydra")

OutputsType = ty.TypeVar("OutputType", bound=Outputs)
WorkflowOutputsType = ty.TypeVar("OutputType", bound=workflow.Outputs)


@attrs.define(auto_attribs=False)
class Workflow(ty.Generic[WorkflowOutputsType]):
    """A workflow, constructed from a workflow task

    Parameters
    ----------
    name : str
        The name of the workflow
    inputs : Task
        The input task of the workflow
    outputs : Task
        The output task of the workflow
    """

    name: str = attrs.field()
    inputs: workflow.Task[WorkflowOutputsType] = attrs.field()
    outputs: WorkflowOutputsType = attrs.field()
    _nodes: dict[str, Node] = attrs.field(factory=dict)

    def __repr__(self):
        return f"Workflow(name={self.name!r}, defn={self.inputs!r})"

    @classmethod
    def clear_cache(
        cls, task: workflow.Task[WorkflowOutputsType] | None = None
    ) -> None:
        """Clear the cache of constructed workflows"""
        if task is None:
            cls._constructed_cache = defaultdict(lambda: defaultdict(dict))
        else:
            cls._constructed_cache[hash_function(task)] = defaultdict(dict)

    @classmethod
    def construct(
        cls,
        task: workflow.Task[WorkflowOutputsType],
        dont_cache: bool = False,
        lazy: ty.Sequence[str] = (),
    ) -> Self:
        """Construct a workflow from a task, caching the constructed worklow

        Parameters
        ----------
        task : workflow.Task
            The task of the workflow to construct
        dont_cache : bool, optional
            Whether to cache the constructed workflow, by default False
        lazy : Sequence[str], optional
            The names of the inputs to the workflow to be considered lazy even if they
            have values in the given task, by default ()
        """

        # Check the previously constructed workflows to see if a workflow has been
        # constructed for the given set of inputs, or a less-specific set (i.e. with a
        # super-set of lazy inputs), and use that if it exists

        non_lazy_vals = {
            n: v
            for n, v in attrs_values(task).items()
            if not is_lazy(v) and n not in lazy
        }
        non_lazy_keys = frozenset(non_lazy_vals)
        hash_cache = Cache()  # share the hash cache to avoid recalculations
        non_lazy_hash = hash_function(non_lazy_vals, cache=hash_cache)
        task_hash = hash_function(type(task), cache=hash_cache)
        # Check for same non-lazy inputs
        try:
            cached_tasks = cls._constructed_cache[task_hash]
        except KeyError:
            pass
        else:
            if (
                non_lazy_keys in cached_tasks
                and non_lazy_hash in cached_tasks[non_lazy_keys]
            ):
                return cached_tasks[non_lazy_keys][non_lazy_hash]
            # Check for supersets of lazy inputs
            for key_set, key_set_cache in cached_tasks.items():
                if key_set.issubset(non_lazy_keys):
                    subset_vals = {
                        k: v for k, v in non_lazy_vals.items() if k in key_set
                    }
                    subset_hash = hash_function(subset_vals, cache=hash_cache)
                    if subset_hash in key_set_cache:
                        return key_set_cache[subset_hash]

        # Initialise the outputs of the workflow
        outputs = task.Outputs(
            **{f.name: attrs.NOTHING for f in get_fields(task.Outputs)}
        )

        # Initialise the lzin fields
        lazy_spec = copy(task)
        workflow = Workflow(
            name=type(task).__name__,
            inputs=lazy_spec,
            outputs=outputs,
        )
        # Set lazy inputs to the workflow, need to do it after the workflow is initialised
        # so a back ref to the workflow can be set in the lazy field
        for field in get_fields(task):
            if field.name not in non_lazy_keys:
                setattr(
                    lazy_spec,
                    field.name,
                    LazyInField(
                        workflow=workflow,
                        field=field.name,
                        type=field.type,
                    ),
                )

        input_values = attrs_values(lazy_spec)
        constructor = input_values.pop("constructor")
        # Call the user defined constructor to set the outputs
        output_lazy_fields = constructor(**input_values)
        if all(v is attrs.NOTHING for v in asdict(outputs).values()):
            if output_lazy_fields is None:
                raise ValueError(
                    f"Constructor function for {task} returned None, must a lazy field "
                    "or a tuple of lazy fields"
                )
        else:  # Outputs are set explicitly in the outputs object
            if output_lazy_fields is not None:
                raise ValueError(
                    f"Constructor function for {task} must not return anything "
                    "if any of the outputs are already set explicitly"
                )
            if unset_outputs := [
                n for n, v in asdict(outputs).items() if v is attrs.NOTHING
            ]:
                raise ValueError(
                    f"Mandatory outputs {unset_outputs} are not set by the "
                    f"constructor of {workflow!r}"
                )
        # Check to see whether any mandatory inputs are not set
        for node in workflow.nodes:
            node._task._check_rules()
        # Check that the outputs are set correctly, either directly by the constructor
        # or via returned values that can be zipped with the output names
        if output_lazy_fields:
            if not isinstance(output_lazy_fields, (list, tuple)):
                output_lazy_fields = [output_lazy_fields]
            output_fields = get_fields(task.Outputs)
            if len(output_lazy_fields) != len(output_fields):
                raise ValueError(
                    f"Expected {len(output_fields)} outputs, got "
                    f"{len(output_lazy_fields)} ({output_lazy_fields})"
                )
            for outpt, outpt_lf in zip(output_fields, output_lazy_fields):
                # Automatically combine any uncombined state arrays into a single lists
                outpt_lf._type = State.combine_state_arrays(outpt_lf._type)
                setattr(outputs, outpt.name, outpt_lf)
        else:
            if unset_outputs := [
                a for a, v in attrs_values(outputs).items() if v is attrs.NOTHING
            ]:
                raise ValueError(
                    f"Expected outputs {unset_outputs} to be set by the "
                    f"constructor of {workflow!r}"
                )
        if not dont_cache:
            cls._constructed_cache[task_hash][non_lazy_keys][non_lazy_hash] = workflow

        return workflow

    @classmethod
    def under_construction(cls) -> "Workflow[ty.Any]":
        """Access the under_construction variable by iterating up through the call stack."""
        frame = inspect.currentframe()
        while frame:
            # Find the frame where the construct method was called
            if (
                frame.f_code.co_name == "construct"
                and frame.f_locals.get("cls") is cls
                and "workflow" in frame.f_locals
            ):
                return frame.f_locals["workflow"]  # local var "workflow" in construct
            frame = frame.f_back
        raise RuntimeError(
            "No workflow is currently under construction (i.e. did not find a "
            "`Workflow.construct` in the current call stack"
        )

    def add(
        self,
        task: Task[OutputsType],
        name: str | None = None,
        environment: Environment | None = None,
        hooks: TaskHooks | None = None,
    ) -> OutputsType:
        """Add a node to the workflow

        Parameters
        ----------
        task_spec : Task
            The task of the job to add to the workflow as a node
        name : str, optional
            The name of the node, by default it will be the name of the task
            class
        environment : Environment, optional
            The environment to run the job in, such as the Docker or Singularity container,
            by default it will be the "native"
        hooks : TaskHooks, optional
            The hooks to run before or after the job, by default no hooks will be run

        Returns
        -------
        OutputType
            The outputs of the node
        """
        from pydra.environments import native

        if name is None:
            name = type(task).__name__
        if name in self._nodes:
            raise ValueError(f"Node with name {name!r} already exists in the workflow")
        if (
            environment
            and not isinstance(environment, native.Environment)
            and task._task_type() != "shell"
        ):
            raise ValueError(
                "Environments can only be used with 'shell' tasks not "
                f"{task._task_type()!r} tasks ({task!r})"
            )
        node = Node[OutputsType](
            name=name,
            task=task,
            workflow=self,
            environment=environment,
            hooks=hooks,
        )
        self._nodes[name] = node
        return node.lzout

    def __getitem__(self, key: str) -> Node:
        return self._nodes[key]

    @property
    def nodes(self) -> ty.Iterable[Node]:
        return self._nodes.values()

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes)

    # Used to cache the constructed workflows by their hashed input values
    _constructed_cache: dict[
        str, dict[frozenset[str], dict[str, "Workflow[ty.Any]"]]
    ] = defaultdict(lambda: defaultdict(dict))

    def execution_graph(self, submitter: "Submitter") -> DiGraph:
        from pydra.engine.submitter import NodeExecution

        exec_nodes = [NodeExecution(n, submitter, workflow=self) for n in self.nodes]
        graph = self._create_graph(exec_nodes)
        # Set the graph attribute of the nodes so lazy fields can be resolved as tasks
        # are created
        for node in exec_nodes:
            node.graph = graph
        return graph

    def graph(self, detailed: bool = False) -> DiGraph:
        return self._create_graph(self.nodes, detailed=detailed)

    def _create_graph(
        self, nodes: "list[Node | NodeExecution]", detailed: bool = False
    ) -> DiGraph:
        """
        Connects a particular job to existing nodes in the workflow.

        Parameters
        ----------
        detailed : bool
            If True, `add_edges_description` is run a detailed descriptions of the
            connections (input/output fields names)
        node_klass : type, optional
            The class to use for the nodes in the workflow. If provided the node is
            wrapped by an instance of the class, if None the node is added as is,
            by default None

        Returns
        -------
        DiGraph
            The graph of the workflow
        """
        graph: DiGraph = DiGraph(name=self.name)
        for node in nodes:
            graph.add_nodes(node)
        # TODO: create connection is run twice
        for node in nodes:
            other_states = {}
            for field in get_fields(node._task):
                lf = node._task[field.name]
                if isinstance(lf, LazyOutField):
                    # adding an edge to the graph if job id expecting output from a different job

                    # checking if the connection is already in the graph
                    if (graph.node(lf._node.name), node) not in graph.edges:
                        graph.add_edges((graph.node(lf._node.name), node))
                    if detailed:
                        graph.add_edges_description(
                            (node.name, field.name, lf._node.name, lf._field)
                        )
                    logger.debug("Connecting %s to %s", lf._node.name, node.name)
                    # adding a state from the previous job to other_states
                    if (
                        graph.node(lf._node.name).state
                        and graph.node(lf._node.name).state.splitter_rpn_final
                    ):
                        # variables that are part of inner splitters should be
                        # treated as a containers
                        if (
                            node.state
                            and f"{node.name}.{field.name}"
                            in node.state._current_splitter_rpn
                        ):
                            node.state._inner_container_ndim[
                                f"{node.name}.{field.name}"
                            ] = 1
                        # adding task_name: (job.state, [a field from the connection]
                        if lf._node.name not in other_states:
                            other_states[lf._node.name] = (
                                graph.node(lf._node.name).state,
                                [field.name],
                            )
                        else:
                            # if the job already exist in other_state,
                            # additional field name should be added to the list of fields
                            other_states[lf._node.name][1].append(field.name)
                elif (
                    isinstance(lf, LazyInField) and detailed
                ):  # LazyField with the wf input
                    # connections with wf input should be added to the detailed graph description
                    graph.add_edges_description(
                        (node.name, field.name, INPUTS_NODE_NAME, lf._field)
                    )

            # if job has connections state has to be recalculated
            if other_states:
                if hasattr(node, "fut_combiner"):
                    combiner = node.fut_combiner
                else:
                    combiner = None

                if node.state:
                    node.state.update_connections(
                        new_other_states=other_states, new_combiner=combiner
                    )
                else:
                    node.state = state.State(
                        node.name,
                        splitter=None,
                        other_states=other_states,
                        combiner=combiner,
                    )
        if detailed:
            lf: LazyOutField
            for outpt_name, lf in attrs_values(self.outputs).items():
                graph.add_edges_description(
                    (OUTPUTS_NODE_NAME, outpt_name, lf._node.name, lf._field)
                )
        return graph
