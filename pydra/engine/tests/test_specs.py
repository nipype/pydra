from pathlib import Path
import typing as ty
import time
import pytest
from fileformats.generic import File
from pydra.engine.specs import (
    Runtime,
    Result,
    WorkflowDef,
)
from pydra.engine.lazy import (
    LazyInField,
    LazyOutField,
)
from pydra.engine.core import Workflow
from pydra.engine.node import Node
from pydra.engine.submitter import Submitter, NodeExecution, DiGraph
from pydra.design import python, workflow
from .utils import Foo, FunAddTwo, FunAddVar, ListSum


@workflow.define
def TestWorkflow(x: int, y: list[int]) -> int:
    node_a = workflow.add(FunAddTwo(a=x), name="A")
    node_b = workflow.add(FunAddVar(a=node_a.out).split(b=y).combine("b"), name="B")
    node_c = workflow.add(ListSum(x=node_b.out), name="C")
    return node_c.out


@pytest.fixture
def workflow_task(submitter: Submitter) -> WorkflowDef:
    wf = TestWorkflow(x=1, y=[1, 2, 3])
    with submitter:
        submitter(wf)
    return wf


@pytest.fixture
def wf(workflow_task: WorkflowDef) -> Workflow:
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


def test_runtime():
    runtime = Runtime()
    assert hasattr(runtime, "rss_peak_gb")
    assert hasattr(runtime, "vms_peak_gb")
    assert hasattr(runtime, "cpu_peak_percent")


def test_result(tmp_path):
    result = Result(output_dir=tmp_path)
    assert hasattr(result, "runtime")
    assert hasattr(result, "outputs")
    assert hasattr(result, "errored")
    assert getattr(result, "errored") is False


def test_lazy_inp(wf: Workflow, graph: DiGraph[NodeExecution]):
    lf = LazyInField(field="x", type=int, workflow=wf)
    assert lf._get_value(workflow=wf, graph=graph) == 1

    lf = LazyInField(field="y", type=str, workflow=wf)
    assert lf._get_value(workflow=wf, graph=graph) == [1, 2, 3]


def test_lazy_out(node_a, wf, graph):
    lf = LazyOutField(field="out", type=int, node=node_a)
    assert lf._get_value(wf, graph) == 3


def test_input_file_hash_1(tmp_path):

    outfile = tmp_path / "test.file"
    outfile.touch()

    @python.define
    def A(in_file: File) -> File:
        return in_file

    assert A(in_file=outfile)._hash == "e708da65b720212c5ce9ed2c65aae59c"

    with open(outfile, "w") as fp:
        fp.write("test")

    assert A(in_file=outfile)._hash == "f726a193430352bb3b92dccf5eccff3a"


def test_input_file_hash_2(tmp_path):
    """input definition with File types, checking when the checksum changes"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    @python.define
    def A(in_file: File) -> File:
        return in_file

    # checking specific hash value
    hash1 = A(in_file=file)._hash
    assert hash1 == "eba2fafb8df4bae94a7aa42bb159b778"

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = A(in_file=file_diffname)._hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = A(in_file=file_diffcontent)._hash
    assert hash1 != hash3


def test_input_file_hash_2a(tmp_path):
    """input definition with ty.Union[File, ...] type, checking when the checksum changes"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    @python.define
    def A(in_file: ty.Union[File, int]) -> File:
        return in_file

    # checking specific hash value
    hash1 = A(in_file=file)._hash
    assert hash1 == "eba2fafb8df4bae94a7aa42bb159b778"

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = A(in_file=file_diffname)._hash
    assert hash1 == hash2

    # checking if string is also accepted
    hash3 = A(in_file=str(file))._hash
    assert hash3 == hash1

    # checking if different content (the same name) affects the hash
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash4 = A(in_file=file_diffcontent)._hash
    assert hash1 != hash4


def test_input_file_hash_3(tmp_path):
    """input definition with File types, checking when the hash and file_hash change"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    @python.define
    def A(in_file: File, in_int: int) -> File:
        return in_file, in_int

    a = A(in_file=file, in_int=3)
    # original hash and files_hash (dictionary contains info about files)
    hash1 = a._hash
    # files_hash1 = deepcopy(my_inp.files_hash)
    # file name should be in files_hash1[in_file]
    filename = str(Path(file))
    # assert filename in files_hash1["in_file"]

    # changing int input
    a.in_int = 5
    hash2 = a._hash
    # files_hash2 = deepcopy(my_inp.files_hash)
    # hash should be different
    assert hash1 != hash2
    # files_hash should be the same, and the tuple for filename shouldn't be recomputed
    # assert files_hash1 == files_hash2
    # assert id(files_hash1["in_file"][filename]) == id(files_hash2["in_file"][filename])

    # recreating the file
    time.sleep(2)  # ensure mtime is different
    with open(file, "w") as f:
        f.write("hello")

    hash3 = a._hash
    # files_hash3 = deepcopy(my_inp.files_hash)
    # hash should be the same,
    # but the entry for in_file in files_hash should be different (modification time)
    assert hash3 == hash2
    # assert files_hash3["in_file"][filename] != files_hash2["in_file"][filename]
    # different timestamp
    # assert files_hash3["in_file"][filename][0] != files_hash2["in_file"][filename][0]
    # the same content hash
    # assert files_hash3["in_file"][filename][1] == files_hash2["in_file"][filename][1]

    # setting the in_file again
    a.in_file = file
    # filename should be removed from files_hash
    # assert my_inp.files_hash["in_file"] == {}
    # will be saved again when hash is calculated
    assert a._hash == hash3
    # assert filename in my_inp.files_hash["in_file"]


def test_input_file_hash_4(tmp_path):
    """input definition with nested list, that contain ints and Files,
    checking changes in checksums
    """
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    @python.define
    def A(in_file: ty.List[ty.List[ty.Union[int, File]]]) -> File:
        return in_file

    # checking specific hash value
    hash1 = A(in_file=[[file, 3]])._hash
    assert hash1 == "b583e0fd5501d3bed9bf510ce2a9e379"

    # the same file, but int field changes
    hash1a = A(in_file=[[file, 5]])._hash
    assert hash1 != hash1a

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = A(in_file=[[file_diffname, 3]])._hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    time.sleep(2)  # need the mtime to be different
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = A(in_file=[[file_diffcontent, 3]])._hash
    assert hash1 != hash3


def test_input_file_hash_5(tmp_path):
    """input definition with File in nested containers, checking changes in checksums"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    @python.define
    def A(in_file: ty.List[ty.Dict[ty.Any, ty.Union[File, int]]]) -> File:
        return in_file

    # checking specific hash value
    hash1 = A(in_file=[{"file": file, "int": 3}])._hash
    assert hash1 == "aa2d4b708ed0dd8340582a6514bfd5ce"

    # the same file, but int field changes
    hash1a = A(in_file=[{"file": file, "int": 5}])._hash
    assert hash1 != hash1a

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = A(in_file=[{"file": file_diffname, "int": 3}])._hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    time.sleep(2)  # ensure mtime is different
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = A(in_file=[{"file": file_diffcontent, "int": 3}])._hash
    assert hash1 != hash3


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
