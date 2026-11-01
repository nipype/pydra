"""SGE worker: resources become qsub arguments, with the job-file machinery faked."""

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
