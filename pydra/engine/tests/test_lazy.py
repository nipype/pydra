import pytest
from pydra.engine.lazy import (
    LazyInField,
    LazyOutField,
)
from pydra.engine.workflow import Workflow
from pydra.engine.node import Node
from pydra.engine.submitter import Submitter, NodeExecution, DiGraph
from pydra.compose import python, workflow
from pydra.engine.tests.utils import (
    Foo,
    FunAddTwo,
    FunAddVar,
    ListSum,
)


@workflow.define
def ATestWorkflow(x: int, y: list[int]) -> int:
    node_a = workflow.add(FunAddTwo(a=x), name="A")
    node_b = workflow.add(FunAddVar(a=node_a.out).split(b=y).combine("b"), name="B")
    node_c = workflow.add(ListSum(x=node_b.out), name="C")
    return node_c.out


@pytest.fixture
def workflow_task(submitter: Submitter) -> workflow.Task:
    wf = ATestWorkflow(x=1, y=[1, 2, 3])
    with submitter:
        submitter(wf)
    return wf


@pytest.fixture
def wf(workflow_task: workflow.Task) -> Workflow:
    wf = Workflow.construct(workflow_task)
    return wf


@pytest.fixture
def submitter(tmp_path) -> Submitter:
    return Submitter(tmp_path)


@pytest.fixture
def graph(wf: Workflow, submitter: Submitter) -> DiGraph[NodeExecution]:
    graph = wf.execution_graph(submitter=submitter)
    for node in graph.nodes:
        if node.state:
            node.state.prepare_states(inputs=node.node.state_values)
            node.state.prepare_inputs()
        node.start()
    return graph


@pytest.fixture
def node_a(wf) -> Node:
    return wf["A"]  # we can pick any node to retrieve the values to


def test_lazy_inp(wf: Workflow, graph: DiGraph[NodeExecution]):
    lf = LazyInField(field="x", type=int, workflow=wf)
    assert lf._get_value(workflow=wf, graph=graph) == 1

    lf = LazyInField(field="y", type=str, workflow=wf)
    assert lf._get_value(workflow=wf, graph=graph) == [1, 2, 3]


def test_lazy_out(node_a, wf, graph):
    lf = LazyOutField(field="out", type=int, node=node_a)
    assert lf._get_value(wf, graph) == 3


def test_lazy_field_cast(wf: Workflow):
    lzout = wf.add(Foo(a="a", b=1, c=2.0), name="foo")

    assert lzout.y._type is int
    assert workflow.cast(lzout.y, float)._type is float


def test_wf_lzin_split(tmp_path):
    @python.define
    def identity(x: int) -> int:
        return x

    @workflow.define
    def Inner(x):
        ident = workflow.add(identity(x=x))
        return ident.out

    @workflow.define
    def Outer(xs):
        inner = workflow.add(Inner().split(x=xs))
        return inner.out

    outer = Outer(xs=[1, 2, 3])

    outputs = outer(cache_dir=tmp_path)
    assert outputs.out == [1, 2, 3]
