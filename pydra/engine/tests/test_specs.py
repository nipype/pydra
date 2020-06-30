from pathlib import Path
import typing as ty

from ..specs import (
    BaseSpec,
    SpecInfo,
    File,
    RuntimeSpec,
    Runtime,
    Result,
    ShellSpec,
    ContainerSpec,
    DockerSpec,
    SingularitySpec,
    LazyField,
)
from ..helpers import make_klass
import pytest


def test_basespec():
    spec = BaseSpec()
    assert (
        spec.hash == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


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
    assert getattr(result, "errored") == False


def test_shellspec():
    with pytest.raises(TypeError):
        spec = ShellSpec()
    spec = ShellSpec(executable="ls")  # (executable, args)
    assert hasattr(spec, "executable")
    assert hasattr(spec, "args")


container_attrs = ["image", "container", "container_xargs", "bindings"]


def test_container():
    with pytest.raises(TypeError):
        spec = ContainerSpec()
    spec = ContainerSpec(
        executable="ls", image="busybox", container="docker"
    )  # (execut., args, image, cont)
    assert all([hasattr(spec, attr) for attr in container_attrs])
    assert hasattr(spec, "executable")


def test_docker():
    with pytest.raises(TypeError):
        spec = DockerSpec(executable="ls")
    spec = DockerSpec(executable="ls", image="busybox")
    assert all(hasattr(spec, attr) for attr in container_attrs)
    assert getattr(spec, "container") == "docker"


def test_singularity():
    with pytest.raises(TypeError):
        spec = SingularitySpec()
    spec = SingularitySpec(executable="ls", image="busybox")
    assert all(hasattr(spec, attr) for attr in container_attrs)
    assert getattr(spec, "container") == "singularity"


class NodeTesting:
    def __init__(self):
        class Input:
            def __init__(self):
                self.inp_a = "A"
                self.inp_b = "B"

        class InpSpec:
            def __init__(self):
                self.fields = [("inp_a", None), ("inp_b", None)]

        self.name = "tn"
        self.inputs = Input()
        self.input_spec = InpSpec()
        self.output_names = ["out_a"]

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
    lf = LazyField(node=tn, attr_type="input")

    with pytest.raises(Exception):
        lf.get_value(wf=WorkflowTesting())

    lf.inp_a
    assert lf.get_value(wf=WorkflowTesting()) == "A"

    lf.inp_b
    assert lf.get_value(wf=WorkflowTesting()) == "B"


def test_lazy_out():
    tn = NodeTesting()
    lf = LazyField(node=tn, attr_type="output")

    lf.out_a
    assert lf.get_value(wf=WorkflowTesting()) == "OUT_A"


def test_laxy_errorattr():
    with pytest.raises(Exception) as excinfo:
        tn = NodeTesting()
        lf = LazyField(node=tn, attr_type="out")
    assert "LazyField: Unknown attr_type:" in str(excinfo.value)


def test_lazy_getvale():
    tn = NodeTesting()
    lf = LazyField(node=tn, attr_type="input")
    with pytest.raises(Exception) as excinfo:
        lf.inp_c
    assert str(excinfo.value) == "Task tn has no input attribute inp_c"


def test_input_file_hash_1(tmpdir):
    tmpdir.chdir()
    outfile = "test.file"
    fields = [("in_file", ty.Any)]
    input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseSpec,))
    inputs = make_klass(input_spec)
    assert (
        inputs(in_file=outfile).hash
        == "1384a1eb11cd94a5b826a82b948313b9237a0956d406ccff59e79ec92b3c935f"
    )
    with open(outfile, "wt") as fp:
        fp.write("test")
    fields = [("in_file", File)]
    input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseSpec,))
    inputs = make_klass(input_spec)
    assert (
        inputs(in_file=outfile).hash
        == "088625131e6718a00170ad445a9c295244dffd4e5d847c8ee4b1606d623dacb1"
    )


def test_input_file_hash_2(tmpdir):
    """ input spec with File types, checking when the checksum changes"""
    file = tmpdir.join("in_file_1.txt")
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(name="Inputs", fields=[("in_file", File)], bases=(BaseSpec,))
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=file).hash
    assert hash1 == "5d2870a7376150274eac72115fbf211792a8e5f250f220b3cc11bfc1851e4b53"

    # checking if different name doesn't affect the hash
    file_diffname = tmpdir.join("in_file_2.txt")
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = inputs(in_file=file_diffname).hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    file_diffcontent = tmpdir.join("in_file_1.txt")
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = inputs(in_file=file_diffcontent).hash
    assert hash1 != hash3


def test_input_file_hash_2a(tmpdir):
    """ input spec with ty.Union[File, ...] type, checking when the checksum changes"""
    file = tmpdir.join("in_file_1.txt")
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs", fields=[("in_file", ty.Union[File, int])], bases=(BaseSpec,)
    )
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=file).hash
    assert hash1 == "5d2870a7376150274eac72115fbf211792a8e5f250f220b3cc11bfc1851e4b53"

    # checking if different name doesn't affect the hash
    file_diffname = tmpdir.join("in_file_2.txt")
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = inputs(in_file=file_diffname).hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    file_diffcontent = tmpdir.join("in_file_1.txt")
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = inputs(in_file=file_diffcontent).hash
    assert hash1 != hash3

    # checking if string is also accepted
    hash4 = inputs(in_file="ala").hash
    assert hash4 == "004060c4475e8874c5fa55c6fffbe67f9ec8a81d578ea1b407dd77186f4d61c2"


def test_input_file_hash_3(tmpdir):
    """ input spec with nested list, that contain ints and Files,
        checking changes in checksums
    """
    file = tmpdir.join("in_file_1.txt")
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
    assert hash1 == "507d81adc3f2f468e82c27ac800d16f6beae4f24f69daaab1d04f52b32b4514d"

    # the same file, but int field changes
    hash1a = inputs(in_file=[[file, 5]]).hash
    assert hash1 != hash1a

    # checking if different name doesn't affect the hash
    file_diffname = tmpdir.join("in_file_2.txt")
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = inputs(in_file=[[file_diffname, 3]]).hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    file_diffcontent = tmpdir.join("in_file_1.txt")
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = inputs(in_file=[[file_diffcontent, 3]]).hash
    assert hash1 != hash3


def test_input_file_hash_4(tmpdir):
    """ input spec with File in nested containers, checking changes in checksums"""
    file = tmpdir.join("in_file_1.txt")
    with open(file, "w") as f:
        f.write("hello")

    input_spec = SpecInfo(
        name="Inputs",
        fields=[("in_file", ty.List[ty.Dict[ty.Any, File]])],
        bases=(BaseSpec,),
    )
    inputs = make_klass(input_spec)

    # checking specific hash value
    hash1 = inputs(in_file=[{"file": file, "int": 3}]).hash
    assert hash1 == "e0555e78a40a02611674b0f48da97cdd28eee7e9885ecc17392b560c14826f06"

    # the same file, but int field changes
    hash1a = inputs(in_file=[{"file": file, "int": 5}]).hash
    assert hash1 != hash1a

    # checking if different name doesn't affect the hash
    file_diffname = tmpdir.join("in_file_2.txt")
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = inputs(in_file=[{"file": file_diffname, "int": 3}]).hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    file_diffcontent = tmpdir.join("in_file_1.txt")
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = inputs(in_file=[{"file": file_diffcontent, "int": 3}]).hash
    assert hash1 != hash3
