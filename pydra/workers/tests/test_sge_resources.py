"""SGE worker: resources become qsub arguments, with the job-file machinery faked."""

import asyncio
import types
import pytest
from pydra.engine.job import Job
from pydra.utils.resources import ComputeResources
from pydra.workers import sge


@pytest.fixture
def make_job(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sge,
        "save",
        lambda script_dir, job: (script_dir / "_job.pklz").write_bytes(b"x"),
    )
    monkeypatch.setattr(
        Job, "cache_dir", property(lambda self: tmp_path), raising=False
    )

    def make(resources):
        job = object.__new__(Job)
        job._uid = "u"
        job._cache_root = tmp_path
        job.resources = resources
        return job

    return make


def qsub_args(worker, job):
    return worker._prepare_runscripts(job)[5]


def test_resources_become_qsub_args(make_job):
    res = ComputeResources(cpus=2, mem="4G", time=90)
    args = qsub_args(sge.SgeWorker(), make_job(res))
    assert args == "-pe smp 2 -l mem_free=4G -l h_rt=00:01:30"


def test_resources_follow_worker_default_args(make_job):
    worker = sge.SgeWorker(default_qsub_args="-q all.q")
    args = qsub_args(worker, make_job(ComputeResources(cpus=2)))
    assert args == "-q all.q -pe smp 2"


def test_no_resources_leaves_args_unchanged(make_job):
    worker = sge.SgeWorker(default_qsub_args="-q all.q")
    assert qsub_args(worker, make_job(None)) == "-q all.q"


def test_jobs_with_equal_resources_share_an_array(make_job):
    worker = sge.SgeWorker()
    res = ComputeResources(cpus=2)
    qsub_args(worker, make_job(res))
    qsub_args(worker, make_job(res))
    ((key, queued),) = worker.tasks_to_run_by_threads_requested.items()
    assert key == "-pe smp 2" and len(queued) == 2


def test_node_resources_replace_worker_wide_args(make_job):
    worker = sge.SgeWorker(
        default_qsub_args="-q all.q -pe smp 1 -l mem_free=1G,h_rt=00:10:00"
    )
    args = qsub_args(worker, make_job(ComputeResources(cpus=4, mem="4G")))
    assert args == "-q all.q -l h_rt=00:10:00 -pe smp 4 -l mem_free=4G"


def test_accounting_follows_the_node(make_job):
    worker = sge.SgeWorker(default_qsub_args="-pe smp 1 -l mem_free=1G")
    args = qsub_args(worker, make_job(ComputeResources(cpus=4, mem="8G")))
    threads, mem_free, mem_free_cmd = worker._requested(args)
    assert (threads, mem_free, mem_free_cmd) == (4, 8, "mem_free=8G")


def test_site_names_are_configurable(make_job):
    worker = sge.SgeWorker(
        pe_name="threaded",
        mem_resource="h_vmem",
        gpu_resource="ngpus",
        gpu_type_resource=None,
    )
    res = ComputeResources(cpus=2, mem="4G", gpus=1, gpu_type="a100")
    args = qsub_args(worker, make_job(res))
    assert args == "-pe threaded 2 -l h_vmem=4G -l ngpus=1"
    assert worker._requested(args) == (2, 4, "h_vmem=4G")


def test_jobs_with_different_resources_do_not_share_an_array(make_job):
    worker = sge.SgeWorker()
    qsub_args(worker, make_job(ComputeResources(cpus=2)))
    qsub_args(worker, make_job(ComputeResources(cpus=4)))
    assert set(worker.tasks_to_run_by_threads_requested) == {"-pe smp 2", "-pe smp 4"}


@pytest.mark.parametrize(
    "cmd, gigabytes",
    [
        ("mem_free=8G", 8),
        ("mem_free=512M", 0.5),
        ("mem_free=2T", 2048),
        ("h_vmem=1024M", 1),
    ],
)
def test_memory_is_normalised_to_gigabytes(cmd, gigabytes):
    assert sge.SgeWorker._gigabytes(cmd) == gigabytes


def test_array_length_counts_megabytes_as_megabytes(make_job):
    import asyncio

    worker = sge.SgeWorker(max_mem_free=4, max_job_array_length=100)
    args = qsub_args(worker, make_job(ComputeResources(mem=512)))
    _, _, cmd = worker._requested(args)
    queued = worker.tasks_to_run_by_threads_requested[args]
    queued.extend([("p", i, False) for i in range(20)])
    # 4 GB across 512 MB jobs is 8 per array, not 4 / 512 -> 1
    assert len(asyncio.run(worker.get_tasks_to_run(args, cmd))) == 8


def test_rerun_removes_the_evicted_jobs_lockfiles(tmp_path, monkeypatch):
    tasks = {}
    for name in ("a", "b"):
        cache_dir = tmp_path / name
        cache_dir.mkdir()
        lockfile = tmp_path / f"{name}-key.lock"
        lockfile.write_text("")
        tasks[f"{name}.pklz"] = types.SimpleNamespace(
            uid=name, cache_dir=cache_dir, lockfile=lockfile
        )
    monkeypatch.setattr(sge, "load_job", lambda job_pkl, ind=None: tasks[job_pkl])
    worker = sge.SgeWorker()
    worker.result_files_by_jobid["1"] = "results"

    async def resubmit(self, sargs, tasks_to_run, error_file):
        assert not any(t.lockfile.exists() for t in tasks.values())
        return "2"

    monkeypatch.setattr(sge.SgeWorker, "submit_array_job", resubmit)
    tasks_to_run = [(job_pkl, 0, False) for job_pkl in tasks]
    jobid = asyncio.run(
        worker._rerun_job_array(tmp_path, "u", [], tasks_to_run, None, "1")
    )
    assert jobid == "2"
    assert set(worker.job_pkls_rerun) == set(tasks)
    assert worker.result_files_by_jobid["2"] == "results"
