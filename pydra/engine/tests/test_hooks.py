import sys
import shutil
import pytest
from pathlib import Path
import glob as glob
from pydra.compose import python
from pydra.engine.hooks import TaskHooks
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job


no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker/singularity command not adjusted for windows",
)


@python.define
def FunAddTwo(a):
    return a + 2


def test_taskhooks_1(tmpdir: Path, capsys):
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()

    foo = Job(task=FunAddTwo(a=1), submitter=Submitter(cache_dir=tmpdir), name="foo")
    assert foo.hooks
    # ensure all hooks are defined
    for attr in ("pre_run", "post_run", "pre_run_task", "post_run_task"):
        hook = getattr(foo.hooks, attr)
        assert hook() is None

    def myhook(task, *args):
        print("I was called")

    FunAddTwo(a=1)(cache_dir=cache_dir, hooks=TaskHooks(pre_run=myhook))
    captured = capsys.readouterr()
    assert "I was called\n" in captured.out
    del captured

    # setting unknown hook should not be allowed
    with pytest.raises(AttributeError):
        foo.hooks.mid_run = myhook

    # reset all hooks
    foo.hooks.reset()
    for attr in ("pre_run", "post_run", "pre_run_task", "post_run_task"):
        hook = getattr(foo.hooks, attr)
        assert hook() is None

    # clear cache
    shutil.rmtree(cache_dir)
    cache_dir.mkdir()

    # set all hooks
    FunAddTwo(a=1)(
        cache_dir=cache_dir,
        hooks=TaskHooks(
            pre_run=myhook,
            post_run=myhook,
            pre_run_task=myhook,
            post_run_task=myhook,
        ),
    )
    captured = capsys.readouterr()
    assert captured.out.count("I was called\n") == 4
    del captured


def test_taskhooks_2(tmpdir, capsys):
    """checking order of the hooks; using task's attributes"""

    def myhook_prerun(task, *args):
        print(f"i. prerun hook was called from {task.name}")

    def myhook_prerun_task(task, *args):
        print(f"ii. prerun task hook was called {task.name}")

    def myhook_postrun_task(task, *args):
        print(f"iii. postrun task hook was called {task.name}")

    def myhook_postrun(task, *args):
        print(f"iv. postrun hook was called {task.name}")

    FunAddTwo(a=1)(
        cache_dir=tmpdir,
        hooks=TaskHooks(
            pre_run=myhook_prerun,
            post_run=myhook_postrun,
            pre_run_task=myhook_prerun_task,
            post_run_task=myhook_postrun_task,
        ),
    )

    captured = capsys.readouterr()
    hook_messages = captured.out.strip().split("\n")
    # checking the order of the hooks
    assert "i. prerun hook" in hook_messages[0]
    assert "ii. prerun task hook" in hook_messages[1]
    assert "iii. postrun task hook" in hook_messages[2]
    assert "iv. postrun hook" in hook_messages[3]


def test_taskhooks_3(tmpdir, capsys):
    """checking results in the post run hooks"""
    foo = Job(task=FunAddTwo(a=1), name="foo", submitter=Submitter(cache_dir=tmpdir))

    def myhook_postrun_task(task, result, *args):
        print(f"postrun task hook, the result is {result.outputs.out}")

    def myhook_postrun(task, result, *args):
        print(f"postrun hook, the result is {result.outputs.out}")

    foo.hooks.post_run = myhook_postrun
    foo.hooks.post_run_task = myhook_postrun_task
    foo.run()

    captured = capsys.readouterr()
    hook_messages = captured.out.strip().split("\n")
    # checking that the postrun hooks have access to results
    assert "postrun task hook, the result is 3" in hook_messages[0]
    assert "postrun hook, the result is 3" in hook_messages[1]


def test_taskhooks_4(tmpdir, capsys):
    """task raises an error: postrun task should be called, postrun shouldn't be called"""

    def myhook_postrun_task(task, result, *args):
        print(f"postrun task hook was called, result object is {result}")

    def myhook_postrun(task, result, *args):
        print("postrun hook should not be called")

    with pytest.raises(Exception):
        FunAddTwo(a="one")(
            cache_dir=tmpdir,
            hooks=TaskHooks(post_run=myhook_postrun, post_run_task=myhook_postrun_task),
        )

    captured = capsys.readouterr()
    hook_messages = captured.out.strip().split("\n")
    # only post run task hook should be called
    assert len(hook_messages) == 1
    assert "postrun task hook was called" in hook_messages[0]
