"""Python tasks run in the caller's working directory when ``chdir=False``
(proposal 0001, H0)."""

import os
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
    with Submitter(worker="debug", cache_root=cache) as sub:
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


def test_shell_tasks_still_run_in_cache_dir(caller_dir, tmp_path):
    cache = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=cache, chdir=False) as sub:
        res = sub(shell.define("pwd")())
    assert Path(res.outputs.stdout.strip()).resolve().is_relative_to(cache.resolve())
    assert Path(os.getcwd()).resolve() == caller_dir


def test_workflow_of_python_tasks_keeps_cwd(caller_dir, tmp_path):
    @workflow.define
    def Wf(x: int) -> str:
        p = workflow.add(Probe(x=x))
        return p.seen

    with Submitter(worker="debug", cache_root=tmp_path / "cache", chdir=False) as sub:
        res = sub(Wf(x=2))
    assert Path(res.outputs.out).resolve() == caller_dir
    assert Path(os.getcwd()).resolve() == caller_dir


def test_audit_messages_go_to_the_job_dir(caller_dir, tmp_path):
    with Submitter(
        worker="debug",
        cache_root=tmp_path / "cache",
        chdir=False,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    ) as sub:
        res = sub(Probe(x=1))
    assert not (caller_dir / "messages").exists()
    assert list((res.cache_dir / "messages").glob("*.jsonld"))
