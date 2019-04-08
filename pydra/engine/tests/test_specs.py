from ..specs import (
    BaseSpec,
    RuntimeSpec,
    Runtime,
    Result,
    ShellSpec,
    ContainerSpec,
    DockerSpec,
    SingularitySpec,
    LazyField
)
import pytest


def test_basespec():
    spec = BaseSpec()
    assert (
        spec.hash == "01c12e5004b5e311e6f37bc758727644c08a719e46ab794eba312338e1d38ab0"
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
    spec = ShellSpec("ls")
    assert hasattr(spec, "executable")


container_attrs = ["image", "container", "container_xargs", "bindings"]


def test_container():
    with pytest.raises(TypeError):
        spec = ContainerSpec()
    spec = ContainerSpec("ls", "busybox", None)
    assert all([hasattr(spec, attr) for attr in container_attrs])
    assert hasattr(spec, "executable")


def test_docker():
    with pytest.raises(TypeError):
        spec = DockerSpec("ls")
    spec = DockerSpec("ls", "busybox")
    assert all(hasattr(spec, attr) for attr in container_attrs)
    assert getattr(spec, "container") == "docker"


def test_singularity():
    with pytest.raises(TypeError):
        spec = SingularitySpec()
    spec = SingularitySpec("ls", "busybox")
    assert all(hasattr(spec, attr) for attr in container_attrs)
    assert getattr(spec, "container") == "singularity"


class TestNode:
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

class TestWorkflow:
    def __init__(self):
        class Input:
            def __init__(self):
                self.inp_a = "A"
                self.inp_b = "B"
        self.inputs = Input()
        self.tn = TestNode()


def test_lazy_inp():
    tn = TestNode()
    lf = LazyField(node=tn, attr_type="input")

    with pytest.raises(Exception):
        lf.get_value(wf=TestWorkflow())

    lf.inp_a
    assert lf.get_value(wf=TestWorkflow()) == "A"

    lf.inp_b
    assert lf.get_value(wf=TestWorkflow()) == "B"


def test_lazy_out():
    tn = TestNode()
    lf = LazyField(node=tn, attr_type="output")

    lf.out_a
    assert lf.get_value(wf=TestWorkflow()) == "OUT_A"


def test_laxy_errorattr():
    with pytest.raises(Exception) as excinfo:
        tn = TestNode()
        lf = LazyField(node=tn, attr_type="out")
    assert "LazyField: Unknown attr_type:" in str(excinfo.value)


def test_lazy_getvale():
    tn = TestNode()
    lf = LazyField(node=tn, attr_type="input")
    with pytest.raises(Exception) as excinfo:
        lf.inp_c
    assert str(excinfo.value) == 'Task tn has no input attribute inp_c'