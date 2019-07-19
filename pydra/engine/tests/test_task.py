# -*- coding: utf-8 -*-

import typing as ty
import os
import pytest

from ..task import to_task, AuditFlag, ShellCommandTask, ContainerTask, DockerTask
from ...utils.messenger import PrintMessenger, FileMessenger, collect_messages


def test_output():
    @to_task
    def funaddtwo(a):
        return a + 2

    nn = funaddtwo(a=3)
    res = nn._run()
    assert res.output.out == 5


@pytest.mark.xfail(reason="cp.dumps(func) depends on the system/setup, TODO!!")
def test_checksum():
    @to_task
    def funaddtwo(a):
        return a + 2

    nn = funaddtwo(a=3)
    assert (
        nn.checksum
        == "FunctionTask_abb4e7cc03b13d0e73884b87d142ed5deae6a312275187a9d8df54407317d7d3"
    )


def test_annotated_func():
    @to_task
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out1", float)]):
        return a + b

    funky = testfunc(a=1)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "b")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 1
    assert getattr(funky.inputs, "b") == 0.1
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == set(["out1"])
    # assert funky.inputs.hash == '17772c3aec9540a8dd3e187eecd2301a09c9a25c6e371ddd86e31e3a1ecfeefa'
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out1")
    assert result.output.out1 == 1.1

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")
    funky.result()  # should not recompute
    funky.inputs.a = 2
    # assert funky.checksum == '537d25885fd2ea5662b7701ba02c132c52a9078a3a2d56aa903a777ea90e5536'
    assert funky.result() is None
    funky()
    result = funky.result()
    assert result.output.out1 == 2.1

    help = funky.help(returnhelp=True)
    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: int",
        "- b: float (default: 0.1)",
        "- _func: str",
        "Output Parameters:",
        "- out1: float",
    ]


def test_halfannotated_func():
    @to_task
    def testfunc(a, b) -> (int, int):
        return a + 1, b + 1

    funky = testfunc(a=10, b=20)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "b")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 10
    assert getattr(funky.inputs, "b") == 20
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == set(["out1", "out2"])
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out1")
    assert result.output.out1 == 11

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")

    funky.result()  # should not recompute
    funky.inputs.a = 11
    assert funky.result() is None
    funky()
    result = funky.result()
    assert result.output.out1 == 12
    help = funky.help(returnhelp=True)

    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: _empty",
        "- b: _empty",
        "- _func: str",
        "Output Parameters:",
        "- out1: int",
        "- out2: int",
    ]


def test_notannotated_func():
    @to_task
    def no_annots(c, d):
        return c + d

    natask = no_annots(c=17, d=3.2)
    assert hasattr(natask.inputs, "c")
    assert hasattr(natask.inputs, "d")
    assert hasattr(natask.inputs, "_func")

    result = natask._run()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out")
    assert result.output.out == 20.2


def test_exception_func():
    @to_task
    def raise_exception(c, d):
        raise Exception()

    bad_funk = raise_exception(c=17, d=3.2)
    assert pytest.raises(Exception, bad_funk)


@pytest.mark.xfail(reason="errors from cloudpickle")
def test_audit(tmpdir):
    @to_task
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    funky = testfunc(a=1, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())
    funky.cache_dir = tmpdir
    funky()

    funky = testfunc(a=2, audit_flags=AuditFlag.ALL, messengers=FileMessenger())
    message_path = tmpdir / funky.checksum / "messages"
    funky.cache_dir = tmpdir
    funky.messenger_args = dict(message_dir=message_path)
    funky()
    from glob import glob

    assert len(glob(str(tmpdir / funky.checksum / "proc*.log"))) == 1
    assert len(glob(str(message_path / "*.jsonld"))) == 6

    # commented out to speed up testing
    collect_messages(tmpdir / funky.checksum, message_path, ld_op="compact")
    assert (tmpdir / funky.checksum / "messages.jsonld").exists()


def test_shell_cmd(tmpdir):
    cmd = ["echo", "hail", "pydra"]

    # all args given as executable
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly._run()
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"

    # separate command into exec + args
    shelly = ShellCommandTask(name="test", executable=cmd[0], args=cmd[1:])
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == " ".join(cmd)
    res = shelly._run()
    assert res.output.return_code == 0
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"


def test_container_cmds(tmpdir):
    containy = ContainerTask(name="containy", executable="pwd")
    with pytest.raises(AttributeError):
        containy.cmdline
    containy.inputs.container = "docker"
    with pytest.raises(AttributeError):
        containy.cmdline
    containy.inputs.image = "busybox"
    assert containy.cmdline


def test_docker_cmd(tmpdir):
    docky = DockerTask(name="docky", executable="pwd", image="busybox")
    assert docky.cmdline == "docker run busybox pwd"
    docky.inputs.container_xargs = ["--rm -it"]
    assert docky.cmdline == "docker run --rm -it busybox pwd"
    docky.inputs.bindings = [
        ("/local/path", "/container/path", "ro"),
        ("/local2", "/container2", None),
    ]
    assert docky.cmdline == (
        "docker run --rm -it -v /local/path:/container/path:ro"
        " -v /local2:/container2:rw busybox pwd"
    )
