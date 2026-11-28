"""One ``squeue`` per poll cycle and ``sbatch --array`` for a round of split nodes, with Slurm faked."""

import asyncio
import sys
import pickle
import re
import types
import pytest
from pydra.utils.resources import ComputeResources
from pydra.workers import slurm

USER_FLAG = slurm.SlurmWorker._squeue_cmd()[2]


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
        self.squeue_calls = 0
        self.squeue_fail_first = 0
        self.squeue_fail_rc = 1
        self.seen = set()  # ids some squeue has listed
        self.premature = []  # ids sacct was asked about before squeue listed them

    async def squeue_hook(self):
        """Overridden to suspend inside squeue."""

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
            assert USER_FLAG in cmd, cmd
            assert "--me" not in cmd and "-j" not in cmd, cmd
            self.squeue_calls += 1
            if self.squeue_calls <= self.squeue_fail_first:
                return self.squeue_fail_rc, "", "slurm_load_jobs error: timed out"
            listed = [t for t, v in self.tasks.items() if v["listed"] > 0]
            for t in listed:
                self.tasks[t]["listed"] -= 1
            self.seen.update(listed)
            await self.squeue_hook()
            return 0, "".join(f"{t}\n" for t in listed), ""
        if cmd[0] == "sacct":
            assert "-P" in cmd and "-X" in cmd
            tid = cmd[cmd.index("-j") + 1]
            if tid not in self.seen:
                self.premature.append(tid)
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
            rerun=False,
            lockfile=tmp_path / f"c{i}-key.lock",
            state_index=i if split else None,
            state_key=None,
            workflow_path=(),
        )
        for i in range(n)
    ]
    return fake, worker, jobs


def run_all(worker, jobs, reruns=None):
    for job, rerun in zip(jobs, reruns or [False] * len(jobs)):
        job.rerun = rerun

    async def go():
        return await asyncio.wait_for(worker.submit_many(jobs), 30)

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


def test_array_key_has_workflow_path(tmp_path, monkeypatch):
    """Node names repeat across nested workflows, so two jobs of equal name,
    resources, cache root and rerun flag are one array only in one workflow"""
    fake, worker, jobs = make(tmp_path, monkeypatch, 4)
    for job in jobs[:2]:
        job.workflow_path = ("a",)
    for job in jobs[2:]:
        job.workflow_path = ("b",)
    assert run_all(worker, jobs) == [True] * 4
    arrays = sorted(a for c in fake.of("sbatch") for a in c if a.startswith("--array"))
    assert arrays == ["--array=0-1", "--array=0-1"]


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
    job = types.SimpleNamespace(state_index=0, state_key=None)
    assert slurm.SlurmWorker(sbatch_args=args)._array_eligible(job) is False
    assert slurm.SlurmWorker(sbatch_args="--partition=x")._array_eligible(job) is True


def test_squeue_asks_for_this_user_not_me(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 2, split=False)
    run_all(worker, jobs)
    for call in fake.of("squeue"):
        assert USER_FLAG in call and "--me" not in call


def test_failed_squeue_cycle_is_skipped_not_read_as_empty(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 3, split=False)
    fake.squeue_fail_first = 2
    assert run_all(worker, jobs) == [True] * 3
    assert fake.premature == []  # no job went to sacct on a failed poll
    assert len(fake.of("squeue")) == 2 + 4


def test_squeue_failing_in_a_row_fails_the_waiting_jobs(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 3, split=False)
    fake.squeue_fail_first = 99
    results = run_all(worker, jobs)
    assert all(
        isinstance(r, RuntimeError) and "3 times in a row" in str(r) for r in results
    )
    assert len(fake.of("squeue")) == 3  # one per cycle, up to the limit
    assert fake.of("sacct") == []


def test_squeue_failure_limit_is_a_setting(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 1, split=False, squeue_failures=5)
    fake.squeue_fail_first = 99
    (result,) = run_all(worker, jobs)
    assert "5 times in a row" in str(result) and len(fake.of("squeue")) == 5


def test_nonzero_exit_without_a_message_is_also_a_failure(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 1, split=False)
    fake.squeue_fail_first = 1
    orig = fake.__call__

    async def silent(*cmd, **kw):
        rc, out, err = await orig(*cmd, **kw)
        return rc, out, "" if cmd[0] == "squeue" else err

    monkeypatch.setattr(slurm.base, "read_and_display_async", silent)
    assert run_all(worker, jobs) == [True]
    assert fake.premature == []


def test_hung_squeue_subprocess_is_killed_at_the_timeout(tmp_path, monkeypatch):
    """A real child that sleeps past ``squeue_timeout`` fails the job at the timeout,
    and is not waited for."""
    real = slurm.base.read_and_display_async
    fake, worker, jobs = make(
        tmp_path, monkeypatch, 1, split=False, squeue_timeout=0.3, squeue_failures=1
    )
    children = []

    async def hanging_squeue(*cmd, **kwargs):
        if cmd[0] != "squeue":
            return await fake(*cmd, **kwargs)
        original = asyncio.create_subprocess_exec

        async def spy(*args, **kw):
            proc = await original(*args, **kw)
            children.append(proc)
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
        return await real(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)

    monkeypatch.setattr(slurm.base, "read_and_display_async", hanging_squeue)
    (result,) = run_all(worker, jobs)
    assert isinstance(result, RuntimeError)
    assert "squeue_timeout=0.3" in str(result) and "squeue" in str(result)
    # the child is gone, stopped rather than waited for: one that ran out its 30 s
    # sleep would have exited 0
    assert children and children[0].returncode not in (None, 0)


def test_job_submitted_during_a_running_squeue_is_not_judged_by_it(
    tmp_path, monkeypatch
):
    """A poll in flight predates the job submitted meanwhile, so its listing must not
    be used to decide that job: it is absent from it only because it was not yet queued.
    """
    fake, worker, jobs = make(tmp_path, monkeypatch, 2, split=False, queue_cycles=2)
    started, release = asyncio.Event(), asyncio.Event()
    first = True

    async def pause():
        nonlocal first
        if first:
            first = False
            started.set()
            await release.wait()

    fake.squeue_hook = pause

    async def go():
        a = asyncio.ensure_future(worker.run(jobs[0]))
        await started.wait()  # job 0's squeue is running
        b = asyncio.ensure_future(worker.run(jobs[1]))  # submitted during it
        for _ in range(20):
            await asyncio.sleep(0)
        release.set()
        return await asyncio.wait_for(asyncio.gather(a, b), 30)

    assert asyncio.run(go()) == [True, True]
    assert fake.premature == []


def test_cancelled_poller_does_not_strand_its_waiters(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 1, split=False)
    started = asyncio.Event()

    async def hang():
        started.set()
        await asyncio.sleep(60)

    fake.squeue_hook = hang

    async def go():
        job = asyncio.ensure_future(worker.run(jobs[0]))
        await started.wait()
        worker._poller.cancel()
        return await asyncio.wait_for(asyncio.gather(job, return_exceptions=True), 5)

    (result,) = asyncio.run(go())
    assert isinstance(result, RuntimeError) and "poller stopped" in str(result)


def test_cache_root_is_part_of_the_grouping_key(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 4)
    for job in jobs[2:]:
        job.cache_root = tmp_path / "other"
    assert run_all(worker, jobs) == [True] * 4
    sbatches = fake.of("sbatch")
    assert [a for c in sbatches for a in c if a.startswith("--array")] == [
        "--array=0-1",
        "--array=0-1",
    ]
    assert len(sbatches) == 2
    manifests = sorted(p.read_text().split()[0] for p in tmp_path.rglob("jobs.txt"))
    assert len(manifests) == 2


def test_rerun_is_part_of_the_grouping_key(tmp_path, monkeypatch):
    fake, worker, jobs = make(tmp_path, monkeypatch, 4)
    assert run_all(worker, jobs, reruns=[False, False, True, True]) == [True] * 4
    sbatches = fake.of("sbatch")
    assert len(sbatches) == 2
    scripts = sorted(p.read_text() for p in tmp_path.rglob("batchscript_array.sh"))
    assert len(scripts) == 2
    assert sum("rerun=True" in t for t in scripts) == 1
    assert sum("rerun=False" in t for t in scripts) == 1


@pytest.mark.parametrize("split", [False, True])
def test_no_poller_task_is_left_pending_after_close(tmp_path, monkeypatch, split):
    fake, worker, jobs = make(
        tmp_path, monkeypatch, 2, split=split, poll_delay=30, queue_cycles=0
    )

    def pollers():
        return [
            t
            for t in asyncio.all_tasks()
            if "_poll_cycles" in repr(t.get_coro()) and not t.done()
        ]

    async def go():
        assert await asyncio.wait_for(
            asyncio.gather(*(worker.run(j) for j in jobs)), 30
        ) == [True, True]
        worker.close()
        for _ in range(5):
            await asyncio.sleep(0)
        return pollers()

    assert asyncio.run(go()) == []


def test_close_leaves_no_pending_task_in_a_stopped_loop(tmp_path, monkeypatch):
    fake, worker, jobs = make(
        tmp_path, monkeypatch, 1, split=False, poll_delay=30, queue_cycles=0
    )
    loop = asyncio.new_event_loop()
    loop.run_until_complete(worker.run(jobs[0]))
    worker.close()
    assert not [t for t in asyncio.all_tasks(loop) if not t.done()]
    loop.close()
