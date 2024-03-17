from pathlib import Path
import typing as ty
import os
import attrs
from copy import deepcopy
import time

from ..specs import (
    BaseSpec,
    SpecInfo,
    File,
    Runtime,
    Result,
    ShellSpec,
    # ContainerSpec,
    LazyIn,
    LazyOut,
    LazyField,
    StateArray,
)
from ..helpers import make_klass
from .utils import foo
from pydra import mark, Workflow
import pytest


def test_basespec():
    spec = BaseSpec()
    assert spec.hash == "0b1d98df22ecd1733562711c205abca2"


def test_runtime():
    runtime = Runtime()
    assert hasattr(runtime, "rss_peak_gb")
    assert hasattr(runtime, "vms_peak_gb")
    assert hasattr(runtime, "cpu_peak_percent")


def test_result():
    result = Result()
    assert hasattr(result, "runtime")
    assert hasattr(result, "output")
    assert hasattr(result, "errored")
    assert getattr(result, "errored") is False


def test_shellspec():
    with pytest.raises(TypeError):
        spec = ShellSpec()
    spec = ShellSpec(executable="ls")  # (executable, args)
    assert hasattr(spec, "executable")
    assert hasattr(spec, "args")


class NodeTesting:
    @attrs.define()
    class Input:
        inp_a: str = "A"
        inp_b: str = "B"

    def __init__(self):
        class InpSpec:
            def __init__(self):
                self.fields = [("inp_a", int), ("inp_b", int)]

        class OutSpec:
            def __init__(self):
                self.fields = [("out_a", int)]

        self.name = "tn"
        self.inputs = self.Input()
        self.input_spec = InpSpec()
        self.output_spec = OutSpec()
        self.output_names = ["out_a"]
        self.state = None

    def result(self, state_index=None):
        class Output:
            def __init__(self):
                self.out_a = "OUT_A"

        class Result:
            def __init__(self):
                self.output = Output()
                self.errored = False

            def get_output_field(self, field):
                return getattr(self.output, field)

        return Result()


class WorkflowTesting:
    def __init__(self):
        class Input:
            def __init__(self):
                self.inp_a = "A"
                self.inp_b = "B"

        self.inputs = Input()
        self.tn = NodeTesting()


def test_lazy_inp():
    tn = NodeTesting()
    lzin = LazyIn(task=tn)

    lf = lzin.inp_a
    assert lf.get_value(wf=WorkflowTesting()) == "A"

    lf = lzin.inp_b
    assert lf.get_value(wf=WorkflowTesting()) == "B"


def test_lazy_out():
    tn = NodeTesting()
    lzout = LazyOut(task=tn)
    lf = lzout.out_a
    assert lf.get_value(wf=WorkflowTesting()) == "OUT_A"


def test_lazy_getvale():
    tn = NodeTesting()
    lf = LazyIn(task=tn)
    with pytest.raises(Exception) as excinfo:
        lf.inp_c
    assert str(excinfo.value) == "Task tn has no input attribute inp_c"


def test_input_file_hash_1(tmp_path):
    os.chdir(tmp_path)
    outfile = "test.file"
    fields = [("in_file", ty.Any)]
    input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseSpec,))
    inputs = make_klass(input_spec)
    assert inputs(in_file=outfile).hash == "9a106eb2830850834d9b5bf098d5fa85"

    with open(outfile, "w") as fp:
        fp.write("test")
    fields = [("in_file", File)]
    input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseSpec,))
    inputs = make_klass(input_spec)
    assert inputs(in_file=outfile).hash == "0e9306e5cae1de1b4dff1f27cca03bce"


def test_input_file_hash_2(tmp_path):
    """input spec with File types, checking when the checksum changes"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(name="Inputs", fields=[("in_file", File)], bases=(BaseSpec,))
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=file).hash
    assert hash1 == "17e4e2b4d8ce8f36bf3fd65804958dbb"

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
    """input spec with ty.Union[File, ...] type, checking when the checksum changes"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs", fields=[("in_file", ty.Union[File, int])], bases=(BaseSpec,)
    )
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=file).hash
    assert hash1 == "17e4e2b4d8ce8f36bf3fd65804958dbb"

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
    assert hash4 == "aee7c7ae25509fb4c92a081d58d17a67"


def test_input_file_hash_3(tmp_path):
    """input spec with File types, checking when the hash and file_hash change"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs", fields=[("in_file", File), ("in_int", int)], bases=(BaseSpec,)
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
    """input spec with nested list, that contain ints and Files,
    checking changes in checksums
    """
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs",
        fields=[("in_file", ty.List[ty.List[ty.Union[int, File]]])],
        bases=(BaseSpec,),
    )
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=[[file, 3]]).hash
    assert hash1 == "11b7e9c90bc8d9dc5ccfc8d4526ba091"

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
    """input spec with File in nested containers, checking changes in checksums"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs",
        fields=[("in_file", ty.List[ty.Dict[ty.Any, ty.Union[File, int]]])],
        bases=(BaseSpec,),
    )
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=[{"file": file, "int": 3}]).hash
    assert hash1 == "5fd53b79e55bbf62a4bb3027eb753a2c"

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
    task = foo(a="a", b=1, c=2.0, name="foo")

    assert task.lzout.y.type == int
    assert task.lzout.y.cast(float).type == float


def test_lazy_field_multi_same_split():
    @mark.task
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
    @mark.task
    def f(x: ty.Any, y: ty.Any) -> ty.Any:
        return x

    task = f(x=[1, 2, 3], name="foo")

    lf = task.lzout.out.split("foo.x")

    assert lf.type == StateArray[ty.Any]
    assert lf.splits == set([(("foo.x",),)])

    lf2 = lf.split("foo.x")
    assert lf2.type == StateArray[ty.Any]
    assert lf2.splits == set([(("foo.x",),)])

    lf3 = lf.split("foo.y")
    assert lf3.type == StateArray[StateArray[ty.Any]]
    assert lf3.splits == set([(("foo.x",),), (("foo.y",),)])


def test_wf_lzin_split():
    @mark.task
    def identity(x: int) -> int:
        return x

    inner = Workflow(name="inner", input_spec=["x"])
    inner.add(identity(x=inner.lzin.x, name="f"))
    inner.set_output(("out", inner.f.lzout.out))

    outer = Workflow(name="outer", input_spec=["x"])
    outer.add(inner.split(x=outer.lzin.x))
    outer.set_output(("out", outer.inner.lzout.out))

    result = outer(x=[1, 2, 3])
    assert result.output.out == StateArray([1, 2, 3])
