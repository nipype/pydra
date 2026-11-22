"""Python tasks run in the caller's working directory when ``chdir=False``
(proposal 0001, H0)."""

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from pydra.compose import python, shell, workflow
from pydra.engine.submitter import Submitter
from pydra.utils.messenger import AuditFlag, FileMessenger


@python.define(outputs=["seen", "wrote"])
def Probe(x: int) -> tuple[str, str]:
    Path("rel.txt").write_text(str(x))
    return os.getcwd(), str(Path("rel.txt").resolve())


@pytest.fixture
def caller_dir(tmp_path, monkeypatch):
    d = tmp_path / "caller"
    d.mkdir()
    monkeypatch.chdir(d)
    return d.resolve()


@pytest.mark.parametrize("w", ["debug", "cf"])
def test_cwd_unchanged_across_run(caller_dir, tmp_path, w):
    with Submitter(worker=w, cache_root=tmp_path / "cache", chdir=False) as sub:
        res = sub(Probe(x=1))
    assert Path(os.getcwd()).resolve() == caller_dir
    assert Path(res.outputs.seen).resolve() == caller_dir


def test_cwd_never_changes_during_run(caller_dir, tmp_path):
    """The hooks that run around the task see the caller's directory too."""
    seen = []

    @python.define
    def Noop(x: int) -> int:
        return x

    from pydra.engine.hooks import TaskHooks

    hooks = TaskHooks(
        pre_run_task=lambda job: seen.append(os.getcwd()),
        post_run_task=lambda job, result: seen.append(os.getcwd()),
    )
    with Submitter(worker="debug", cache_root=tmp_path / "cache", chdir=False) as sub:
        sub(Noop(x=1), hooks=hooks)
    assert [Path(s).resolve() for s in seen] == [caller_dir, caller_dir]


def test_default_still_runs_in_cache_dir(caller_dir, tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        res = sub(Probe(x=1))
    assert Path(res.outputs.seen).resolve() != caller_dir
    assert (
        Path(res.outputs.seen).resolve().is_relative_to((tmp_path / "cache").resolve())
    )
    assert Path(os.getcwd()).resolve() == caller_dir


def test_relative_writes_land_in_caller_dir(caller_dir, tmp_path):
    """Documented behaviour change: with chdir=False a relative path a function writes
    is in the caller's directory, not the job's cache directory."""
    cache = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=cache, chdir=False) as sub:
        sub(Probe(x=7))
    assert (caller_dir / "rel.txt").read_text() == "7"
    assert not list(cache.rglob("rel.txt"))


def test_relative_writes_land_in_cache_dir_by_default(caller_dir, tmp_path):
    cache = tmp_path / "cache"
    with Submitter(worker="debug", store="pickle", cache_root=cache) as sub:
        sub(Probe(x=7))
    assert not (caller_dir / "rel.txt").exists()
    assert len(list(cache.rglob("rel.txt"))) == 1


def test_concurrent_python_tasks_do_not_race_on_cwd(caller_dir, tmp_path):
    @python.define
    def Slow(x: int) -> bool:
        before = os.getcwd()
        time.sleep(0.2)
        return os.getcwd() == before

    results, errors = {}, []

    def run(i):
        try:
            with Submitter(
                worker="debug", cache_root=tmp_path / f"cache{i}", chdir=False
            ) as sub:
                results[i] = sub(Slow(x=i)).outputs.out
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(results.values()) and len(results) == 4
    assert Path(os.getcwd()).resolve() == caller_dir


def test_shell_tasks_run_in_cache_dir_without_chdir(caller_dir, tmp_path, print_cwd):
    cache = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=cache, chdir=False) as sub:
        res = sub(shell.define(print_cwd)())
    assert Path(res.outputs.stdout.strip()).resolve().is_relative_to(cache.resolve())
    assert Path(os.getcwd()).resolve() == caller_dir


@pytest.mark.parametrize("w", ["debug", "cf"])
def test_workflow_of_python_tasks_keeps_cwd(caller_dir, tmp_path, w):
    @workflow.define
    def Wf(x: int) -> str:
        p = workflow.add(Probe(x=x))
        return p.seen

    with Submitter(worker=w, cache_root=tmp_path / "cache", chdir=False) as sub:
        res = sub(Wf(x=2))
    assert Path(res.outputs.out).resolve() == caller_dir
    assert Path(os.getcwd()).resolve() == caller_dir


def test_audit_messages_go_to_the_job_dir(caller_dir, tmp_path):
    with Submitter(
        worker="debug",
        store="pickle",
        cache_root=tmp_path / "cache",
        chdir=False,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    ) as sub:
        res = sub(Probe(x=1))
    assert not (caller_dir / "messages").exists()
    assert list((res.cache_dir / "messages").glob("*.jsonld"))


@pytest.fixture
def print_cwd(tmp_path):
    script = tmp_path / "print_cwd.py"
    script.write_text("import os, time\ntime.sleep(0.3)\nprint(os.getcwd())\n")
    return [sys.executable, str(script)]


def test_shell_task_never_changes_process_cwd(caller_dir, tmp_path, print_cwd):
    """Two overlapping shell tasks leave the process cwd alone, and each runs in its
    own cache dir"""
    out, seen = {}, []

    def run(i):
        with Submitter(
            worker="debug", cache_root=tmp_path / f"cache{i}", chdir=False
        ) as sub:
            out[i] = sub(shell.define(print_cwd)()).outputs.stdout.strip()

    def watch():
        for _ in range(20):
            seen.append(Path(os.getcwd()).resolve())
            time.sleep(0.03)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    threads.append(threading.Thread(target=watch))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(seen) == {caller_dir}
    for i in range(2):
        assert Path(out[i]).resolve().is_relative_to((tmp_path / f"cache{i}").resolve())
    assert Path(os.getcwd()).resolve() == caller_dir


def test_python_cwd_while_shell_task_runs_in_another_thread(
    caller_dir, tmp_path, print_cwd
):
    marker = tmp_path / "started"

    @python.define
    def WaitThenCwd(marker: str) -> str:
        import os
        import time

        for _ in range(100):
            if os.path.exists(marker):
                break
            time.sleep(0.05)
        time.sleep(0.2)
        return os.getcwd()

    def run_shell():
        marker.touch()
        with Submitter(
            worker="debug", cache_root=tmp_path / "cache-sh", chdir=False
        ) as sub:
            sub(shell.define(print_cwd)())

    t = threading.Thread(target=run_shell)
    t.start()
    with Submitter(
        worker="debug", cache_root=tmp_path / "cache-py", chdir=False
    ) as sub:
        res = sub(WaitThenCwd(marker=str(marker)))
    t.join()
    assert Path(res.outputs.out).resolve() == caller_dir


def test_shell_relative_outputs_resolve_under_cache_dir(caller_dir, tmp_path):
    from fileformats.generic import File

    script = tmp_path / "make.py"
    script.write_text("open('made.txt', 'w').write('hi')\n")
    Cmd = shell.define(
        [sys.executable, str(script)],
        outputs=(
            [
                shell.out(name="made", type=File, callable=lambda: "made.txt"),
                shell.out(name="globbed", type=list[File], default="made.*"),
            ]
        ),
    )
    cache = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=cache, chdir=False) as sub:
        res = sub(Cmd())
    assert Path(res.outputs.made).resolve().is_relative_to(cache.resolve())
    assert Path(res.outputs.globbed[0]).resolve().is_relative_to(cache.resolve())
    assert not (caller_dir / "made.txt").exists()
    assert Path(os.getcwd()).resolve() == caller_dir


def test_python_relative_file_output_warns(caller_dir, tmp_path):
    from fileformats.generic import File

    @python.define
    def Write(x: int) -> File:
        Path("out.txt").write_text(str(x))
        return File("out.txt")

    with Submitter(worker="debug", cache_root=tmp_path / "cache", chdir=False) as sub:
        with pytest.warns(UserWarning, match="rather than its cache directory"):
            sub(Write(x=1))


def test_absolute_outputs_do_not_warn(caller_dir, tmp_path):
    """A pass-through of an absolute input and an explicit absolute output dir are
    correct, so they are not reported"""
    import warnings
    from fileformats.generic import File

    data = tmp_path / "data"
    data.mkdir()
    (data / "in.txt").write_text("a")

    @python.define
    def Through(in_file: File) -> File:
        return in_file

    @python.define
    def WriteAbs(out_dir: str) -> File:
        p = Path(out_dir) / "o.txt"
        p.write_text("b")
        return File(p)

    with Submitter(worker="debug", cache_root=tmp_path / "cache", chdir=False) as sub:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sub(Through(in_file=data / "in.txt"))
            sub(WriteAbs(out_dir=str(data)))
