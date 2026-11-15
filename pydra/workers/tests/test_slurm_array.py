"""One ``squeue`` per poll cycle and ``sbatch --array`` for split nodes, with Slurm faked."""

import asyncio
import pickle
import re
import types
import pytest
from pydra.utils.resources import ComputeResources
from pydra.workers import slurm


class FakeSlurm:
    """Fake sbatch/squeue/sacct/scontrol with a queue that persists across calls.

    ``plans`` maps an array index (0 for a plain submission) to the sacct State of each
    successive attempt; a task stays in the queue for ``queue_cycles`` squeue calls
    after each (re)submission.
    """

    def __init__(self, tmp_path, plans=None, queue_cycles=3):
        self.tmp_path = tmp_path
        self.plans = plans or {}
        self.queue_cycles = queue_cycles
        self.calls = []
        self.tasks = {}  # job id -> {"index", "attempt", "listed", "err", "out"}
        self.next_id = 100

    def of(self, name):
        return [c for c in self.calls if c[0] == name]

    def _flag(self, cmd, name):
        found = [c.split("=", 1)[1] for c in cmd if c.startswith(f"--{name}=")]
        return found[0] if found else str(self.tmp_path / f"user-%j.{name}")

    async def __call__(self, *cmd, hide_display=False, strip=False):
        self.calls.append(cmd)
        if cmd[0] == "sbatch":
            self.next_id += 1
            base_id = str(self.next_id)
            arrays = [c for c in cmd if c.startswith("--array=")]
            n = int(arrays[0].rsplit("-", 1)[1]) + 1 if arrays else None
            for i in range(n or 1):
                tid = f"{base_id}_{i}" if n else base_id
                out, err = self._flag(cmd, "output"), self._flag(cmd, "error")
                sub = lambda p: (  # noqa: E731
                    p.replace("%A", base_id).replace("%a", str(i)).replace("%j", tid)
                )
                self.tasks[tid] = dict(
                    index=i,
                    attempt=0,
                    listed=self.queue_cycles,
                    out=sub(out),
                    err=sub(err),
                )
            return 0, f"Submitted batch job {base_id}\n", ""
        if cmd[0] == "squeue":
            assert "--me" in cmd and "-j" not in cmd, cmd
            listed = [t for t, v in self.tasks.items() if v["listed"] > 0]
            for t in listed:
                self.tasks[t]["listed"] -= 1
            return 0, "".join(f"{t}\n" for t in listed), ""
        if cmd[0] == "sacct":
            assert "-P" in cmd and "-X" in cmd
            tid = cmd[cmd.index("-j") + 1]
            task = self.tasks[tid]
            plan = self.plans.get(task["index"], ["COMPLETED"])
            state = plan[min(task["attempt"], len(plan) - 1)]
            code = "0:15" if state.startswith("CANCELLED") else "0:0"
            if state == "FAILED":
                code = "1:0"
            return 0, f"{tid}|{state}|{code}\n", ""
        if cmd[0] == "scontrol":
            assert cmd[1] == "requeue"
            task = self.tasks[cmd[2]]
            task["attempt"] += 1
            task["listed"] = self.queue_cycles
            return 0, "", ""
        raise AssertionError(cmd)


def make(tmp_path, monkeypatch, n, split=True, plans=None, queue_cycles=3, **kwargs):
    fake = FakeSlurm(tmp_path, plans, queue_cycles)
    monkeypatch.setattr(slurm.base, "read_and_display_async", fake)

    def prepare(self, job, rerun=False):
        script_dir = tmp_path / "scripts" / job.uid
        script_dir.mkdir(parents=True, exist_ok=True)
        (script_dir / "_job.pklz").write_bytes(b"x")
        return script_dir, script_dir / "batch.sh"

    monkeypatch.setattr(slurm.SlurmWorker, "_prepare_runscripts", prepare)
    kwargs.setdefault("poll_delay", 0)
    worker = slurm.SlurmWorker(**kwargs)
    jobs = [
        types.SimpleNamespace(
            name="node",
            uid=f"u{i}",
            cache_root=tmp_path,
            resources=None,
            lockfile=tmp_path / f"c{i}-key.lock",
            state_index=i if split else None,
        )
        for i in range(n)
    ]
    return fake, worker, jobs


def run_all(worker, jobs, **kwargs):
    async def go():
        return await asyncio.wait_for(
            asyncio.gather(*(worker.run(j) for j in jobs), return_exceptions=True), 30
        )

    return asyncio.run(go())


def test_one_squeue_per_cycle_for_many_jobs(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 5, split=False, queue_cycles=3)
    assert run_all(worker, jobs) == [True] * 5
    # three cycles with the jobs listed, one in which they are gone: not 5 per cycle
    assert len(fake.of("squeue")) == 4
    assert len(fake.of("sbatch")) == 5


def test_split_elements_are_submitted_as_one_array(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 4)
    assert run_all(worker, jobs) == [True] * 4
    (sbatch,) = fake.of("sbatch")
    assert "--array=0-3" in sbatch
    assert len(fake.of("squeue")) == 4
    script = tmp_path / "slurm_scripts" / "array_u0_4"
    manifest = (script / "jobs.txt").read_text().split()
    assert manifest == [
        str(tmp_path / "scripts" / f"u{i}" / "_job.pklz") for i in range(4)
    ]
    assert sbatch[-1] == str(script / "batchscript_array.sh")
    body = (script / "batchscript_array.sh").read_text()
    assert "SLURM_ARRAY_TASK_ID" in body and "load_and_run" in body


def test_each_task_has_its_own_output_and_error_file(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 3)
    run_all(worker, jobs)
    (sbatch,) = fake.of("sbatch")
    assert any(re.fullmatch(r"--output=.*slurm-%A_%a\.out", a) for a in sbatch)
    assert any(re.fullmatch(r"--error=.*slurm-%A_%a\.err", a) for a in sbatch)
    errors = [worker.error[t] for t in sorted(fake.tasks)]
    assert [e.rsplit("slurm-", 1)[1] for e in errors] == [
        "101_0.err",
        "101_1.err",
        "101_2.err",
    ]
    assert len({v["out"] for v in fake.tasks.values()}) == 3


def test_failed_array_element_is_reported_for_that_job_only(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 4, plans={2: ["FAILED"]})
    (tmp_path / "slurm_scripts" / "array_u0_4").mkdir(parents=True)
    # the failing task's own error file; the others have none
    (tmp_path / "slurm_scripts" / "array_u0_4" / "slurm-101_2.err").write_text(
        "Traceback...\nValueError: element 2 broke\n"
    )
    results = run_all(worker, jobs)
    assert results[0] is True and results[1] is True and results[3] is True
    assert isinstance(results[2], Exception)
    assert "element 2 broke" in str(results[2])
    assert fake.of("scontrol") == []


def test_cancelled_array_element_is_not_requeued(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 3, plans={1: ["CANCELLED by 7"]})
    results = run_all(worker, jobs)
    assert isinstance(results[1], slurm.SlurmJobCancelled)
    assert results[0] is True and results[2] is True
    assert fake.of("scontrol") == []


def test_requeue_inside_an_array(tmp_path, monkeypatch):
    fake, worker, jobs = make(
        tmp_path, monkeypatch, 3, plans={1: ["TIMEOUT", "COMPLETED"]}
    )
    jobs[1].lockfile.write_text("")
    assert run_all(worker, jobs) == [True] * 3
    assert [c[2] for c in fake.of("scontrol")] == ["101_1"]
    assert not jobs[1].lockfile.exists()
    assert len(fake.of("sbatch")) == 1


def test_requeue_budget_is_per_array_element(tmp_path, monkeypatch):
    fake, worker, jobs = make(
        tmp_path, monkeypatch, 3, plans={0: ["PREEMPTED"] * 9}, requeue_budget=2
    )
    results = run_all(worker, jobs)
    assert isinstance(results[0], RuntimeError) and "requeue budget" in str(results[0])
    assert results[1] is True and results[2] is True
    assert len(fake.of("scontrol")) == 2


def test_a_single_split_element_uses_plain_sbatch(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 1)
    assert run_all(worker, jobs) == [True]
    (sbatch,) = fake.of("sbatch")
    assert not [a for a in sbatch if a.startswith("--array")]


def test_different_shapes_are_not_mixed_in_one_array(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 4)
    for job in jobs[2:]:
        job.resources = ComputeResources(cpus=4)
    assert run_all(worker, jobs) == [True] * 4
    arrays = sorted(a for c in fake.of("sbatch") for a in c if a.startswith("--array"))
    assert arrays == ["--array=0-1", "--array=0-1"]
    cpus = [("--cpus-per-task=4" in c) for c in fake.of("sbatch")]
    assert sorted(cpus) == [False, True]


@pytest.mark.parametrize("args", ["-o /x/o-%j.out", "--error=/x/e-%j.err"])
def test_user_output_error_or_array_args_disable_arrays(tmp_path, monkeypatch, args):
    fake, worker, jobs = make(tmp_path, monkeypatch, 3, sbatch_args=args)
    assert run_all(worker, jobs) == [True] * 3
    assert len(fake.of("sbatch")) == 3


def test_array_size_is_capped(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 5, max_array_size=2)
    assert run_all(worker, jobs) == [True] * 5
    arrays = [a for c in fake.of("sbatch") for a in c if a.startswith("--array")]
    assert arrays == ["--array=0-1", "--array=0-1"]
    assert len(fake.of("sbatch")) == 3  # two arrays and one remaining plain job


def test_worker_still_pickles_after_a_run(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 2)
    run_all(worker, jobs)
    clone = pickle.loads(pickle.dumps(worker))
    assert clone.error == {} and clone.max_array_size == worker.max_array_size


@pytest.mark.parametrize("args", ["--array=1-3", "-a 1-3", "-ofile", "--output=x"])
def test_user_array_and_output_options_make_a_job_ineligible(args):
    job = types.SimpleNamespace(state_index=0)
    assert slurm.SlurmWorker(sbatch_args=args)._array_eligible(job) is False
    assert slurm.SlurmWorker(sbatch_args="--partition=x")._array_eligible(job) is True
