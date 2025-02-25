from pathlib import Path
import typing as ty
import os
from unittest.mock import Mock
import time
import pytest
from fileformats.generic import File
from pydra.engine.specs import (
    Runtime,
    Result,
)
from pydra.engine.lazy import (
    LazyInField,
    LazyOutField,
)
from pydra.engine.core import Workflow
from pydra.engine.submitter import NodeExecution
from pydra.utils.typing import StateArray
from pydra.design import python, workflow
from .utils import Foo, FunAddTwo, FunAddVar, ListSum


@workflow.define
def TestWorkflow(x: int, y: list[int]) -> int:
    node_a = workflow.add(FunAddTwo(a=x), name="A")
    node_b = workflow.add(FunAddVar(a=node_a.out).split(b=y).combine("b"), name="B")
    node_c = workflow.add(ListSum(x=node_b.out), name="C")
    return node_c.out


@pytest.fixture
def workflow_task():
    return TestWorkflow(x=1, y=[1, 2, 3])


@pytest.fixture
def workflow_obj(workflow_task):
    wf = Workflow.construct(workflow_task)
    for n in wf.nodes:
        if n._state:
            n._state.prepare_states()
            n._state.prepare_inputs()
    return wf


@pytest.fixture
def node_a(workflow_obj):
    return workflow_obj["A"]


@pytest.fixture
def node_b(workflow_obj):
    return workflow_obj["B"]


@pytest.fixture
def node_c(workflow_obj):
    return workflow_obj["C"]


@pytest.fixture
def node_exec(node_c, workflow_task):
    # We only use this to resolve the upstream outputs from, can be any node
    return NodeExecution(node=node_c, workflow_inputs=workflow_task, submitter=Mock())


def test_lazy_inp(workflow_obj, node_exec):
    lf = LazyInField(field="x", type=int, workflow=workflow_obj)
    assert lf._get_value(node_exec) == 1

    lf = LazyInField(field="y", type=str, workflow=workflow_obj)
    assert lf._get_value(node_exec) == [1, 2, 3]


def test_lazy_out(node_a, node_exec):
    lf = LazyOutField(field="a", type=int, node=node_a)
    assert lf._get_value(node_exec) == 3


def test_lazy_getvale():
    tn = NodeTesting()
    lf = LazyIn(task=tn)
    with pytest.raises(Exception) as excinfo:
        lf.inp_c
    assert (
        str(excinfo.value)
        == "Task 'tn' has no input attribute 'inp_c', available: 'inp_a', 'inp_b'"
    )


def test_input_file_hash_1(tmp_path):
    os.chdir(tmp_path)
    outfile = "test.file"
    fields = [("in_file", ty.Any)]
    input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseDef,))
    inputs = make_klass(input_spec)
    assert inputs(in_file=outfile).hash == "9a106eb2830850834d9b5bf098d5fa85"

    with open(outfile, "w") as fp:
        fp.write("test")
    fields = [("in_file", File)]
    input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseDef,))
    inputs = make_klass(input_spec)
    assert inputs(in_file=outfile).hash == "02fa5f6f1bbde7f25349f54335e1adaf"


def test_input_file_hash_2(tmp_path):
    """input definition with File types, checking when the checksum changes"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(name="Inputs", fields=[("in_file", File)], bases=(BaseDef,))
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=file).hash
    assert hash1 == "aaa50d60ed33d3a316d58edc882a34c3"

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = inputs(in_file=file_diffname).hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    time.sleep(2)  # ensure mtime is different
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = inputs(in_file=file_diffcontent).hash
    assert hash1 != hash3


def test_input_file_hash_2a(tmp_path):
    """input definition with ty.Union[File, ...] type, checking when the checksum changes"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs", fields=[("in_file", ty.Union[File, int])], bases=(BaseDef,)
    )
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=file).hash
    assert hash1 == "aaa50d60ed33d3a316d58edc882a34c3"

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = inputs(in_file=file_diffname).hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    time.sleep(2)  # ensure mtime is different
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = inputs(in_file=file_diffcontent).hash
    assert hash1 != hash3

    # checking if string is also accepted
    hash4 = inputs(in_file=str(file)).hash
    assert hash4 == "800af2b5b334c9e3e5c40c0e49b7ffb5"


def test_input_file_hash_3(tmp_path):
    """input definition with File types, checking when the hash and file_hash change"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs", fields=[("in_file", File), ("in_int", int)], bases=(BaseDef,)
    )
    inputs = make_klass(input_spec)

    my_inp = inputs(in_file=file, in_int=3)
    # original hash and files_hash (dictionary contains info about files)
    hash1 = my_inp.hash
    # files_hash1 = deepcopy(my_inp.files_hash)
    # file name should be in files_hash1[in_file]
    filename = str(Path(file))
    # assert filename in files_hash1["in_file"]

    # changing int input
    my_inp.in_int = 5
    hash2 = my_inp.hash
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

    hash3 = my_inp.hash
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
    my_inp.in_file = file
    # filename should be removed from files_hash
    # assert my_inp.files_hash["in_file"] == {}
    # will be saved again when hash is calculated
    assert my_inp.hash == hash3
    # assert filename in my_inp.files_hash["in_file"]


def test_input_file_hash_4(tmp_path):
    """input definition with nested list, that contain ints and Files,
    checking changes in checksums
    """
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs",
        fields=[("in_file", ty.List[ty.List[ty.Union[int, File]]])],
        bases=(BaseDef,),
    )
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=[[file, 3]]).hash
    assert hash1 == "0693adbfac9f675af87e900065b1de00"

    # the same file, but int field changes
    hash1a = inputs(in_file=[[file, 5]]).hash
    assert hash1 != hash1a

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = inputs(in_file=[[file_diffname, 3]]).hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    time.sleep(2)  # need the mtime to be different
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = inputs(in_file=[[file_diffcontent, 3]]).hash
    assert hash1 != hash3


def test_input_file_hash_5(tmp_path):
    """input definition with File in nested containers, checking changes in checksums"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs",
        fields=[("in_file", ty.List[ty.Dict[ty.Any, ty.Union[File, int]]])],
        bases=(BaseDef,),
    )
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=[{"file": file, "int": 3}]).hash
    assert hash1 == "56e6e2c9f3bdf0cd5bd3060046dea480"

    # the same file, but int field changes
    hash1a = inputs(in_file=[{"file": file, "int": 5}]).hash
    assert hash1 != hash1a

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = inputs(in_file=[{"file": file_diffname, "int": 3}]).hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    time.sleep(2)  # ensure mtime is different
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = inputs(in_file=[{"file": file_diffcontent, "int": 3}]).hash
    assert hash1 != hash3


def test_lazy_field_cast():
    task = Foo(a="a", b=1, c=2.0, name="foo")

    assert task.lzout.y._type is int
    assert workflow.cast(task.lzout.y, float)._type is float


def test_lazy_field_multi_same_split():
    @python.define
    def f(x: ty.List[int]) -> ty.List[int]:
        return x

    task = f(x=[1, 2, 3], name="foo")

    lf = task.lzout.out.split("foo.x")

    assert lf.type == StateArray[int]
    assert lf.splits == set([(("foo.x",),)])

    lf2 = lf.split("foo.x")
    assert lf2.type == StateArray[int]
    assert lf2.splits == set([(("foo.x",),)])


def test_lazy_field_multi_diff_split():
    @python.define
    def F(x: ty.Any, y: ty.Any) -> ty.Any:
        return x

    task = F(x=[1, 2, 3], name="foo")

    lf = task.lzout.out.split("foo.x")

    assert lf.type == StateArray[ty.Any]
    assert lf.splits == set([(("foo.x",),)])

    lf2 = lf.split("foo.x")
    assert lf2.type == StateArray[ty.Any]
    assert lf2.splits == set([(("foo.x",),)])

    lf3 = lf.split("foo.y")
    assert lf3.type == StateArray[StateArray[ty.Any]]
    assert lf3.splits == set([(("foo.x",),), (("foo.y",),)])


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
