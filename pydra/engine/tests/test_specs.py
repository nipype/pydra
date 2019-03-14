from ..specs import (
    BaseSpec,
    RuntimeSpec,
    Runtime,
    Result,
    ShellSpec,
    ContainerSpec,
    DockerSpec,
    SingularitySpec,
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
