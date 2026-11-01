"""Slurm requeue budget and user cancellation, with sbatch/squeue/sacct faked."""

import asyncio
import types
import pytest
from pydra.utils.resources import ComputeResources
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
            assert "-P" in cmd and "-X" in cmd and "-j" in cmd
            state = self.states.pop(0)
            exit_code = "0:15" if state.startswith("CANCELLED") else "0:0"
            return 0, f"42|{state}|{exit_code}\n", ""
        if cmd[0] == "scontrol":
            return 0, "", ""
        raise AssertionError(cmd)

    def requeues(self):
        return [c for c in self.calls if c[0] == "scontrol"]


def run_worker(tmp_path, monkeypatch, states, resources=None, **kwargs):
    fake = FakeSlurm(states)
    monkeypatch.setattr(slurm.base, "read_and_display_async", fake)
    monkeypatch.setattr(
        slurm.SlurmWorker,
        "_prepare_runscripts",
        lambda self, job, rerun=False: (tmp_path, tmp_path / "batch.sh"),
    )
    worker = slurm.SlurmWorker(poll_delay=0, **kwargs)
    job = types.SimpleNamespace(
        name="t", uid="u", cache_root=tmp_path, resources=resources
    )
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


@pytest.mark.parametrize("state", ["CANCELLED by 1234", "CANCELLED by 0", "CANCELLED"])
def test_cancelled_is_never_requeued(tmp_path, monkeypatch, state):
    fake, worker, job = run_worker(tmp_path, monkeypatch, [state, "COMPLETED"])
    with pytest.raises(slurm.SlurmJobCancelled, match="cancelled"):
        asyncio.run(worker.run(job))
    assert fake.requeues() == []


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        slurm.SlurmWorker(requeue_budget=-1)


def sbatch_args(fake):
    (call,) = [c for c in fake.calls if c[0] == "sbatch"]
    return list(call[1:])


def test_resources_become_sbatch_args(tmp_path, monkeypatch):
    res = ComputeResources(cpus=2, mem="1G", gpus=1, gpu_type="h100", time=120)
    fake, worker, job = run_worker(tmp_path, monkeypatch, ["COMPLETED"], resources=res)
    assert asyncio.run(worker.run(job)) is True
    args = sbatch_args(fake)
    for expected in (
        "--cpus-per-task=2",
        "--mem=1G",
        "--gres=gpu:h100:1",
        "--time=0-00:02:00",
    ):
        assert expected in args
    assert args[-1] == str(tmp_path / "batch.sh")


def test_node_resources_follow_worker_wide_args(tmp_path, monkeypatch):
    fake, worker, job = run_worker(
        tmp_path,
        monkeypatch,
        ["COMPLETED"],
        resources=ComputeResources(cpus=8),
        sbatch_args="--cpus-per-task=1 --partition=x",
    )
    asyncio.run(worker.run(job))
    args = sbatch_args(fake)
    assert "--partition=x" in args
    assert args.index("--cpus-per-task=8") > args.index("--cpus-per-task=1")


def test_no_resources_adds_no_resource_args(tmp_path, monkeypatch):
    fake, worker, job = run_worker(tmp_path, monkeypatch, ["COMPLETED"])
    asyncio.run(worker.run(job))
    assert not [
        a
        for a in sbatch_args(fake)
        if a.startswith(("--cpus", "--mem", "--gres", "--time"))
    ]
