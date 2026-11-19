"""``submit_many``: a round of jobs becomes arrays, with Slurm and SGE faked"""

import asyncio
import time
import types
import pytest
from pydra.engine.job import Job
from pydra.compose import python, workflow
from pydra.engine.executor import WorkerExecutor
from pydra.engine.submitter import Submitter
from pydra.utils.resources import ComputeResources
from pydra.workers import debug, sge, slurm


class FakeSlurm:
    """sbatch, squeue and sacct: every task is listed once, then completes"""

    def __init__(self):
        self.calls = []
        self.next_id = 100
        self.tasks = {}

    def of(self, name):
        return [c for c in self.calls if c[0] == name]

    async def __call__(self, *cmd, hide_display=False, strip=False):
        self.calls.append(cmd)
        if cmd[0] == "sbatch":
            self.next_id += 1
            arrays = [c for c in cmd if c.startswith("--array=")]
            n = int(arrays[0].rsplit("-", 1)[1]) + 1 if arrays else None
            for i in range(n or 1):
                self.tasks[f"{self.next_id}_{i}" if n else str(self.next_id)] = 1
            return 0, f"Submitted batch job {self.next_id}\n", ""
        if cmd[0] == "squeue":
            listed = [t for t, left in self.tasks.items() if left > 0]
            for t in listed:
                self.tasks[t] -= 1
            return 0, "".join(f"{t}\n" for t in listed), ""
        if cmd[0] == "sacct":
            return 0, f"{cmd[cmd.index('-j') + 1]}|COMPLETED|0:0\n", ""
        raise AssertionError(cmd)


def slurm_round(tmp_path, monkeypatch, n, **kwargs):
    fake = FakeSlurm()
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
            lockfile=tmp_path / f"c{i}.lock",
            state_index=i,
        )
        for i in range(n)
    ]
    return fake, worker, jobs


def submit(worker, jobs):
    return asyncio.run(asyncio.wait_for(worker.submit_many(jobs), 30))


def arrays(fake):
    return [a for c in fake.of("sbatch") for a in c if a.startswith("--array")]


def test_round_of_one_node_is_one_array(tmp_path, monkeypatch):
    fake, worker, jobs = slurm_round(tmp_path, monkeypatch, 5)
    assert submit(worker, jobs) == [True] * 5
    assert arrays(fake) == ["--array=0-4"] and len(fake.of("sbatch")) == 1


def test_two_resources_two_arrays(tmp_path, monkeypatch):
    fake, worker, jobs = slurm_round(tmp_path, monkeypatch, 4)
    for job in jobs[2:]:
        job.resources = ComputeResources(cpus=4)
    assert submit(worker, jobs) == [True] * 4
    assert arrays(fake) == ["--array=0-1", "--array=0-1"]
    assert sorted("--cpus-per-task=4" in c for c in fake.of("sbatch")) == [False, True]


def test_max_array_size_splits(tmp_path, monkeypatch):
    fake, worker, jobs = slurm_round(tmp_path, monkeypatch, 5, max_array_size=2)
    assert submit(worker, jobs) == [True] * 5
    assert arrays(fake) == ["--array=0-1", "--array=0-1"]
    assert len(fake.of("sbatch")) == 3


def test_done_is_called_as_each_job_ends(tmp_path, monkeypatch):
    fake, worker, jobs = slurm_round(tmp_path, monkeypatch, 3)
    ended = []
    asyncio.run(worker.submit_many(jobs, done=lambda j, o: ended.append((j.uid, o))))
    assert sorted(ended) == [("u0", True), ("u1", True), ("u2", True)]


def test_worker_without_submit_many_per_job(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        executor = WorkerExecutor(debug.Worker())
        assert executor.batches is False
    assert WorkerExecutor(slurm.SlurmWorker()).batches is True
    assert WorkerExecutor(sge.SgeWorker()).batches is True


class Loaded:
    """A job loaded from its pickle"""


class FakeSge:
    """qsub, qstat and qacct; the result files of the tasks already exist"""

    def __init__(self):
        self.calls = []
        self.first_qsub = None

    async def __call__(self, *cmd, hide_display=False, strip=False):
        self.calls.append(cmd)
        if cmd[0] == "qsub":
            self.first_qsub = self.first_qsub or time.monotonic()
            return (
                0,
                f'Your job-array {len(self.calls)}.1-2:1 ("x") has been submitted',
                "",
            )
        if cmd[0] == "qacct":
            return 0, "failed 0\n", ""
        raise AssertionError(cmd)


@pytest.fixture
def sge_round(tmp_path, monkeypatch):
    fake = FakeSge()
    monkeypatch.setattr(sge.base, "read_and_display_async", fake)
    monkeypatch.setattr(
        sge,
        "save",
        lambda script_dir, job: (script_dir / "_job.pklz").write_bytes(b"x"),
    )

    def load_job(job_pkl, ind=None):
        cache_dir = tmp_path / "results" / job_pkl.split("/")[-2]
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "_result.pklz").write_bytes(b"x")
        loaded = Loaded()
        loaded.cache_dir = cache_dir
        return loaded

    monkeypatch.setattr(sge, "load_job", load_job)
    monkeypatch.setattr(
        Job, "cache_dir", property(lambda self: tmp_path), raising=False
    )
    monkeypatch.setattr(Job, "name", "node", raising=False)

    def make(n, **kwargs):
        kwargs.setdefault("poll_delay", 0)
        worker = sge.SgeWorker(**kwargs)
        jobs = []
        for i in range(n):
            job = object.__new__(Job)
            job._uid = f"u{i}"
            job._cache_root = tmp_path
            job.resources = None
            job.rerun = False
            jobs.append(job)
        return worker, jobs

    return fake, make


def test_sge_round_is_one_array(sge_round):
    fake, make = sge_round
    worker, jobs = make(3)
    assert asyncio.run(asyncio.wait_for(worker.submit_many(jobs), 30)) == [True] * 3
    (qsub,) = [c for c in fake.calls if c[0] == "qsub"]
    assert qsub[1:3] == ("-t", "1-3")


def test_sge_no_collect_sleep(sge_round):
    fake, make = sge_round
    worker, jobs = make(2)
    start = time.monotonic()
    assert asyncio.run(asyncio.wait_for(worker.submit_many(jobs), 30)) == [True] * 2
    assert fake.first_qsub - start < 0.2


def test_collect_jobs_delay_warns():
    with pytest.warns(DeprecationWarning, match="collect_jobs_delay"):
        worker = sge.SgeWorker(collect_jobs_delay=5)
    assert worker.collect_jobs_delay == 5


@python.define
def AddOne(x: int) -> int:
    return x + 1


@workflow.define
def Split(xs: list[int]) -> list[int]:
    n = workflow.add(AddOne().split(x=xs), name="N")
    return n.out


def test_split_round_one_array(any_worker, tmp_path):
    xs = list(range(4))
    with Submitter(worker=any_worker, cache_root=tmp_path) as sub:
        result = sub(Split(xs=xs))
        assert list(result.outputs.out) == [x + 1 for x in xs]
        if any_worker == "slurm":
            # the four elements of the one round are the tasks of one array
            ids = [i for i in sub.worker.error if "_" in i]
            assert len(ids) == 4 and len({i.split("_")[0] for i in ids}) == 1


class Batching(debug.Worker):
    """A worker that takes a whole round and runs it where the store sees the results"""

    _plugin_name = "debug"

    def __init__(self):
        super().__init__()
        self.rounds = []

    async def run(self, job, rerun=False):
        raise AssertionError("a batching worker is given rounds")

    async def submit_many(self, jobs, done=None):
        self.rounds.append(len(jobs))
        for job in jobs:
            job.run(rerun=job.rerun)
            if done is not None:
                done(job, True)
        return [True] * len(jobs)


def test_executor_hands_the_round_to_submit_many(tmp_path):
    from pydra.engine.drivers import drive
    from pydra.engine.runner import GraphRunner

    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = GraphRunner(
            Split(xs=[0, 1, 2, 3]), sub.store, on_error="raise_at_end", submitter=sub
        )
        worker = Batching()
        worker.loop = sub.loop
        executor = WorkerExecutor(worker)
        sub.loop.run_until_complete(
            drive(
                runner,
                {"default": executor},
                lambda job: "default",
                max_concurrent=float("inf"),
                on_error="raise_at_end",
                audit=sub.audit,
            )
        )
        assert runner.done and not runner.failed
        assert worker.rounds == [4]
