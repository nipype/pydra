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
    spec = ShellSpec("ls", None)  # (executable, args)
    assert hasattr(spec, "executable")
    assert hasattr(spec, "args")


container_attrs = ["image", "container", "container_xargs", "bindings"]


def test_container():
    with pytest.raises(TypeError):
        spec = ContainerSpec()
    spec = ContainerSpec("ls", None, "busybox", None)  # (execut., args, image, cont)
    assert all([hasattr(spec, attr) for attr in container_attrs])
    assert hasattr(spec, "executable")


def test_docker():
    with pytest.raises(TypeError):
        spec = DockerSpec("ls", None)
    spec = DockerSpec("ls", None, "busybox")
    assert all(hasattr(spec, attr) for attr in container_attrs)
    assert getattr(spec, "container") == "docker"


def test_singularity():
    with pytest.raises(TypeError):
        spec = SingularitySpec()
    spec = SingularitySpec("ls", None, "busybox")
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


def test_file_hash(tmpdir):
    tmpdir.chdir()
    outfile = "test.file"
    fields = [("in_file", ty.Any)]
    input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseSpec,))
    inputs = make_klass(input_spec)
    assert (
        inputs(str(outfile)).hash
        == "1384a1eb11cd94a5b826a82b948313b9237a0956d406ccff59e79ec92b3c935f"
    )
    with open(outfile, "wt") as fp:
        fp.write("test")
    fields = [("in_file", File)]
    input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseSpec,))
    inputs = make_klass(input_spec)
    assert (
        inputs(outfile).hash
        == "088625131e6718a00170ad445a9c295244dffd4e5d847c8ee4b1606d623dacb1"
    )
