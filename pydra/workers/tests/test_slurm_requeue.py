"""Slurm requeue budget and user cancellation, with sbatch/squeue/sacct faked."""

import asyncio
import types
import pytest
from pydra.workers import slurm


class FakeSlurm:
    """Fake the Slurm commands at the subprocess boundary."""

    def __init__(self, states, scontrol_rc=0):
        self.scontrol_rc = scontrol_rc
        self.states = list(states)  # sacct State field per submission, in order
        self.calls = []

    async def __call__(self, *cmd, hide_display=False, strip=False):
        self.calls.append(cmd)
        if cmd[0] == "sbatch":
            return 0, "Submitted batch job 42\n", ""
        if cmd[0] == "squeue":
            return 0, "", ""  # never queued: straight to sacct
        if cmd[0] == "sacct":
            assert "-P" in cmd and "-X" in cmd and "-j" in cmd
            state = self.states.pop(0)
            exit_code = "0:15" if state.startswith("CANCELLED") else "0:0"
            return 0, f"42|{state}|{exit_code}\n", ""
        if cmd[0] == "scontrol":
            return self.scontrol_rc, "", "requeue not permitted" * bool(self.scontrol_rc)
        raise AssertionError(cmd)

    def requeues(self):
        return [c for c in self.calls if c[0] == "scontrol"]


def run_worker(tmp_path, monkeypatch, states, scontrol_rc=0, **kwargs):
    fake = FakeSlurm(states, scontrol_rc)
    monkeypatch.setattr(slurm.base, "read_and_display_async", fake)
    monkeypatch.setattr(
        slurm.SlurmWorker,
        "_prepare_runscripts",
        lambda self, job, rerun=False: (tmp_path, tmp_path / "batch.sh"),
    )
    worker = slurm.SlurmWorker(poll_delay=0, **kwargs)
    job = types.SimpleNamespace(name="t", uid="u", cache_root=tmp_path)
    return fake, worker, job


def test_requeue_then_complete(tmp_path, monkeypatch):
    fake, worker, job = run_worker(tmp_path, monkeypatch, ["TIMEOUT", "COMPLETED"])
    assert asyncio.run(worker.run(job)) is True
    assert len(fake.requeues()) == 1


def test_requeue_budget_is_spent(tmp_path, monkeypatch):
    fake, worker, job = run_worker(
        tmp_path, monkeypatch, ["PREEMPTED"] * 10, requeue_budget=2
    )
    with pytest.raises(RuntimeError, match="requeue budget"):
        asyncio.run(worker.run(job))
    assert len(fake.requeues()) == 2


def test_zero_budget_never_requeues(tmp_path, monkeypatch):
    fake, worker, job = run_worker(tmp_path, monkeypatch, ["TIMEOUT"], requeue_budget=0)
    with pytest.raises(RuntimeError, match="requeue budget"):
        asyncio.run(worker.run(job))
    assert fake.requeues() == []


@pytest.mark.parametrize("state", ["NODE_FAIL", "BOOT_FAIL"])
def test_node_failure_is_requeued(tmp_path, monkeypatch, state):
    fake, worker, job = run_worker(tmp_path, monkeypatch, [state, "COMPLETED"])
    assert asyncio.run(worker.run(job)) is True
    assert len(fake.requeues()) == 1


def test_failed_with_empty_error_file_raises_cleanly(tmp_path, monkeypatch):
    fake, worker, job = run_worker(tmp_path, monkeypatch, ["FAILED"])
    err = tmp_path / "slurm_scripts" / "u" / "slurm-42.err"
    err.parent.mkdir(parents=True)
    err.write_text("")
    with pytest.raises(Exception, match="unknown reason") as exc:
        asyncio.run(worker.run(job))
    assert not isinstance(exc.value, IndexError)


def test_refused_requeue_is_reported_as_such(tmp_path, monkeypatch):
    fake, worker, job = run_worker(
        tmp_path, monkeypatch, ["TIMEOUT"] * 5, scontrol_rc=1
    )
    with pytest.raises(RuntimeError, match="scontrol requeue.*not permitted"):
        asyncio.run(worker.run(job))
    assert len(fake.requeues()) == 1


@pytest.mark.parametrize("state", ["CANCELLED by 1234", "CANCELLED by 0", "CANCELLED"])
def test_cancelled_is_never_requeued(tmp_path, monkeypatch, state):
    fake, worker, job = run_worker(tmp_path, monkeypatch, [state, "COMPLETED"])
    with pytest.raises(slurm.SlurmJobCancelled, match="cancelled"):
        asyncio.run(worker.run(job))
    assert fake.requeues() == []


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        slurm.SlurmWorker(requeue_budget=-1)
