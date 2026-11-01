"""Slurm requeue budget and user cancellation, with sbatch/squeue/sacct faked."""

import asyncio
import os
import types
import pytest
from pydra.workers import slurm


class FakeSlurm:
    """Fake the Slurm commands at the subprocess boundary."""

    def __init__(self, states):
        self.states = list(states)  # sacct State field per submission, in order
        self.calls = []

    async def __call__(self, *cmd, hide_display=False, strip=False):
        self.calls.append(cmd)
        if cmd[0] == "sbatch":
            return 0, "Submitted batch job 42\n", ""
        if cmd[0] == "squeue":
            return 0, "", ""  # never queued: straight to sacct
        if cmd[0] == "sacct":
            state = self.states.pop(0)
            exit_code = "0:0" if state == "COMPLETED" else "1:0"
            return 0, f"42|{state}|{exit_code}\n", ""
        if cmd[0] == "scontrol":
            return 0, "", ""
        raise AssertionError(cmd)

    def requeues(self):
        return [c for c in self.calls if c[0] == "scontrol"]


def run_worker(tmp_path, monkeypatch, states, **kwargs):
    fake = FakeSlurm(states)
    monkeypatch.setattr(slurm.base, "read_and_display_async", fake)
    worker = slurm.SlurmWorker(poll_delay=0, **kwargs)
    monkeypatch.setattr(
        worker,
        "_prepare_runscripts",
        lambda job, rerun=False: (tmp_path, tmp_path / "batch.sh"),
        raising=False,
    )
    job = types.SimpleNamespace(name="t", uid="u", cache_root=tmp_path)
    try:
        return fake, asyncio.run(worker.run(job))
    except Exception as exc:
        return fake, exc


def test_requeue_then_complete(tmp_path, monkeypatch):
    fake, result = run_worker(tmp_path, monkeypatch, ["TIMEOUT", "COMPLETED"])
    assert result is True
    assert len(fake.requeues()) == 1


def test_requeue_budget_is_spent(tmp_path, monkeypatch):
    fake, result = run_worker(
        tmp_path, monkeypatch, ["PREEMPTED"] * 10, requeue_budget=2
    )
    assert isinstance(result, RuntimeError)
    assert "requeue budget" in str(result)
    assert len(fake.requeues()) == 2


def test_zero_budget_never_requeues(tmp_path, monkeypatch):
    fake, result = run_worker(tmp_path, monkeypatch, ["TIMEOUT"], requeue_budget=0)
    assert isinstance(result, RuntimeError)
    assert fake.requeues() == []


def test_user_cancelled_is_not_requeued(tmp_path, monkeypatch):
    fake, result = run_worker(
        tmp_path, monkeypatch, [f"CANCELLED by {os.getuid()}", "COMPLETED"]
    )
    assert isinstance(result, slurm.SlurmJobCancelled)
    assert fake.requeues() == []


def test_cancelled_by_another_uid_is_requeued(tmp_path, monkeypatch):
    other = os.getuid() + 1
    fake, result = run_worker(
        tmp_path, monkeypatch, [f"CANCELLED by {other}", "COMPLETED"]
    )
    assert result is True
    assert len(fake.requeues()) == 1


def test_negative_budget_rejected():
    with pytest.raises(Exception):
        slurm.SlurmWorker(requeue_budget=-1)
