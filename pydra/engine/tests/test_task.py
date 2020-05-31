# -*- coding: utf-8 -*-

import typing as ty
import os, sys
import pytest

from ... import mark
from ..task import AuditFlag, ShellCommandTask, DockerTask, SingularityTask
from ...utils.messenger import FileMessenger, PrintMessenger, collect_messages
from .utils import gen_basic_wf

no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker/singularity command not adjusted for windows",
)


@mark.task
def funaddtwo(a):
    return a + 2


def test_output():
    nn = funaddtwo(a=3)
    res = nn._run()
    assert res.output.out == 5


def test_numpy():
    """ checking if mark.task works for numpy functions"""
    np = pytest.importorskip("numpy")
    fft = mark.annotate({"a": np.ndarray, "return": float})(np.fft.fft)
    fft = mark.task(fft)()
    arr = np.array([[1, 10], [2, 20]])
    fft.inputs.a = arr
    res = fft()
    assert np.allclose(np.fft.fft(arr), res.output.out)


@pytest.mark.xfail(reason="cp.dumps(func) depends on the system/setup, TODO!!")
def test_checksum():
    nn = funaddtwo(a=3)
    assert (
        nn.checksum
        == "FunctionTask_abb4e7cc03b13d0e73884b87d142ed5deae6a312275187a9d8df54407317d7d3"
    )


def test_annotated_func():
    @mark.task
    def testfunc(
        a: int, b: float = 0.1
    ) -> ty.NamedTuple("Output", [("out_out", float)]):
        return a + b

    funky = testfunc(a=1)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "b")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 1
    assert getattr(funky.inputs, "b") == 0.1
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == set(["out_out"])
    # assert funky.inputs.hash == '17772c3aec9540a8dd3e187eecd2301a09c9a25c6e371ddd86e31e3a1ecfeefa'
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out_out")
    assert result.output.out_out == 1.1

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")
    funky.result()  # should not recompute
    funky.inputs.a = 2
    # assert funky.checksum == '537d25885fd2ea5662b7701ba02c132c52a9078a3a2d56aa903a777ea90e5536'
    assert funky.result() is None
    funky()
    result = funky.result()
    assert result.output.out_out == 2.1

    help = funky.help(returnhelp=True)
    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: int",
        "- b: float (default: 0.1)",
        "- _func: str",
        "Output Parameters:",
        "- out_out: float",
    ]


def test_annotated_func_multreturn():
    """ the function has two elements in the return statement"""

    @mark.task
    def testfunc(
        a: float,
    ) -> ty.NamedTuple("Output", [("fractional", float), ("integer", int)]):
        import math

        return math.modf(a)

    funky = testfunc(a=3.5)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 3.5
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == set(["fractional", "integer"])
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")
    assert hasattr(result, "output")
    assert hasattr(result.output, "fractional")
    assert result.output.fractional == 0.5
    assert hasattr(result.output, "integer")
    assert result.output.integer == 3

    help = funky.help(returnhelp=True)
    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: float",
        "- _func: str",
        "Output Parameters:",
        "- fractional: float",
        "- integer: int",
    ]


def test_annotated_func_multreturn_exception():
    """function has two elements in the return statement,
        but three element provided in the spec - should raise an error
    """

    @mark.task
    def testfunc(
        a: float,
    ) -> ty.NamedTuple(
        "Output", [("fractional", float), ("integer", int), ("whoknows", int)]
    ):
        import math

        return math.modf(a)

    funky = testfunc(a=3.5)
    with pytest.raises(Exception) as excinfo:
        funky()
    assert "expected 3 elements" in str(excinfo.value)


def test_halfannotated_func():
    @mark.task
    def testfunc(a, b) -> int:
        return a + b

    funky = testfunc(a=10, b=20)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "b")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 10
    assert getattr(funky.inputs, "b") == 20
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == set(["out"])
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out")
    assert result.output.out == 30

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")

    funky.result()  # should not recompute
    funky.inputs.a = 11
    assert funky.result() is None
    funky()
    result = funky.result()
    assert result.output.out == 31
    help = funky.help(returnhelp=True)

    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: _empty",
        "- b: _empty",
        "- _func: str",
        "Output Parameters:",
        "- out: int",
    ]


def test_halfannotated_func_multreturn():
    @mark.task
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
    @mark.task
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


def test_notannotated_func_returnlist():
    @mark.task
    def no_annots(c, d):
        return [c, d]

    natask = no_annots(c=17, d=3.2)
    result = natask._run()
    assert hasattr(result.output, "out")
    assert result.output.out == [17, 3.2]


def test_halfannotated_func_multrun_returnlist():
    @mark.task
    def no_annots(c, d) -> (list, float):
        return [c, d], c + d

    natask = no_annots(c=17, d=3.2)
    result = natask._run()

    assert hasattr(result.output, "out1")
    assert hasattr(result.output, "out2")
    assert result.output.out1 == [17, 3.2]
    assert result.output.out2 == 20.2


def test_notannotated_func_multreturn():
    """ no annotation and multiple values are returned
        all elements should be returned as a tuple ans set to "out"
    """

    @mark.task
    def no_annots(c, d):
        return c + d, c - d

    natask = no_annots(c=17, d=3.2)
    assert hasattr(natask.inputs, "c")
    assert hasattr(natask.inputs, "d")
    assert hasattr(natask.inputs, "_func")

    result = natask._run()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out")
    assert result.output.out == (20.2, 13.8)


def test_exception_func():
    @mark.task
    def raise_exception(c, d):
        raise Exception()

    bad_funk = raise_exception(c=17, d=3.2)
    assert pytest.raises(Exception, bad_funk)


def test_result_none_1():
    """ checking if None is properly returned as the result"""

    @mark.task
    def fun_none(x):
        return None

    task = fun_none(name="none", x=3)
    res = task()
    assert res.output.out is None


def test_result_none_2():
    """ checking if None is properly set for all outputs """

    @mark.task
    def fun_none(x) -> (ty.Any, ty.Any):
        return None

    task = fun_none(name="none", x=3)
    res = task()
    assert res.output.out1 is None
    assert res.output.out2 is None


def test_audit_prov(tmpdir):
    @mark.task
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    # printing the audit message
    funky = testfunc(a=1, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())
    funky.cache_dir = tmpdir
    funky()

    # saving the audit message into the file
    funky = testfunc(a=2, audit_flags=AuditFlag.PROV, messengers=FileMessenger())
    message_path = tmpdir / funky.checksum / "messages"
    funky.cache_dir = tmpdir
    funky.messenger_args = dict(message_dir=message_path)
    funky()

    collect_messages(tmpdir / funky.checksum, message_path, ld_op="compact")
    assert (tmpdir / funky.checksum / "messages.jsonld").exists()


def test_audit_all(tmpdir):
    @mark.task
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

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


@no_win
def test_shell_cmd(tmpdir):
    cmd = ["echo", "hail", "pydra"]

    # all args given as executable
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly._run()
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"

    # separate command into exec + args
    shelly = ShellCommandTask(executable=cmd[0], args=cmd[1:])
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == " ".join(cmd)
    res = shelly._run()
    assert res.output.return_code == 0
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"


def test_container_cmds(tmpdir):
    containy = DockerTask(name="containy", executable="pwd")
    with pytest.raises(AttributeError) as excinfo:
        containy.cmdline
    assert "not specified" in str(excinfo.value)
    containy.inputs.image = "busybox"
    assert containy.cmdline


@no_win
def test_docker_cmd(tmpdir):
    docky = DockerTask(name="docky", executable="pwd", image="busybox")
    assert (
        docky.cmdline
        == f"docker run --rm -v {docky.output_dir}:/output_pydra:rw -w /output_pydra busybox pwd"
    )
    docky.inputs.container_xargs = ["--rm -it"]
    assert (
        docky.cmdline
        == f"docker run --rm -it -v {docky.output_dir}:/output_pydra:rw -w /output_pydra busybox pwd"
    )
    docky.inputs.bindings = [
        ("/local/path", "/container/path", "ro"),
        ("/local2", "/container2", None),
    ]
    assert docky.cmdline == (
        "docker run --rm -it -v /local/path:/container/path:ro"
        f" -v /local2:/container2:rw -v {docky.output_dir}:/output_pydra:rw -w /output_pydra busybox pwd"
    )


@no_win
def test_singularity_cmd(tmpdir):
    # todo how this should be done?
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singi", executable="pwd", image=image)
    assert (
        singu.cmdline
        == f"singularity exec -B {singu.output_dir}:/output_pydra:rw {image} pwd"
    )
    singu.inputs.bindings = [
        ("/local/path", "/container/path", "ro"),
        ("/local2", "/container2", None),
    ]
    assert singu.cmdline == (
        "singularity exec -B /local/path:/container/path:ro"
        f" -B /local2:/container2:rw -B {singu.output_dir}:/output_pydra:rw {image} pwd"
    )


def test_functask_callable(tmpdir):
    # no submitter or plugin
    foo = funaddtwo(a=1)
    res = foo()
    assert res.output.out == 3
    assert foo.plugin is None

    # plugin
    bar = funaddtwo(a=2)
    res = bar(plugin="cf")
    assert res.output.out == 4
    assert bar.plugin is None

    foo2 = funaddtwo(a=3)
    foo2.plugin = "cf"
    res = foo2()
    assert res.output.out == 5
    assert foo2.plugin == "cf"


def test_taskhooks(tmpdir, capsys):
    foo = funaddtwo(name="foo", a=1, cache_dir=tmpdir)
    assert foo.hooks
    # ensure all hooks are defined
    for attr in ("pre_run", "post_run", "pre_run_task", "post_run_task"):
        hook = getattr(foo.hooks, attr)
        assert hook() is None

    def myhook(task, *args):
        print("I was called")

    foo.hooks.pre_run = myhook
    foo()
    captured = capsys.readouterr()
    assert "I was called\n" in captured.out
    del captured

    # setting unknown hook should not be allowed
    with pytest.raises(AttributeError):
        foo.hooks.mid_run = myhook

    # set all hooks
    foo.hooks.post_run = myhook
    foo.hooks.pre_run_task = myhook
    foo.hooks.post_run_task = myhook
    foo.inputs.a = 2  # ensure not pre-cached
    foo()
    captured = capsys.readouterr()
    assert captured.out.count("I was called\n") == 4
    del captured

    # hooks are independent across tasks by default
    bar = funaddtwo(name="bar", a=3, cache_dir=tmpdir)
    assert bar.hooks is not foo.hooks
    # but can be shared across tasks
    bar.hooks = foo.hooks
    # and workflows
    wf = gen_basic_wf()
    wf.tmpdir = tmpdir
    wf.hooks = bar.hooks
    assert foo.hooks == bar.hooks == wf.hooks

    wf(plugin="cf")
    captured = capsys.readouterr()
    assert captured.out.count("I was called\n") == 4
    del captured

    # reset all hooks
    foo.hooks.reset()
    for attr in ("pre_run", "post_run", "pre_run_task", "post_run_task"):
        hook = getattr(foo.hooks, attr)
        assert hook() is None
