import os
import shutil
from pathlib import Path
import random
import platform
import typing as ty
import pytest
import cloudpickle as cp
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job
from pydra.compose import workflow
from fileformats.generic import Directory, File
from pydra.engine.tests.utils import Multiply, RaiseXeq1
from pydra.utils.general import position_sort
from pydra.compose.shell.templating import parse_format_string
from pydra.engine.job import save, load_and_run
from pydra.workers.cf import get_available_cpus
from pydra.utils.hash import hash_function


def test_get_available_cpus():
    assert get_available_cpus() > 0
    try:
        import psutil

        has_psutil = True
    except ImportError:
        has_psutil = False

    if hasattr(os, "sched_getaffinity"):
        assert get_available_cpus() == len(os.sched_getaffinity(0))

    if has_psutil and platform.system().lower() != "darwin":
        assert get_available_cpus() == len(psutil.Process().cpu_affinity())

    if platform.system().lower() == "darwin":
        assert get_available_cpus() == os.cpu_count()


def test_cf_pool_never_forks_the_submitting_process():
    import pickle
    from pydra.workers.cf import ConcurrentFuturesWorker

    worker = ConcurrentFuturesWorker(n_procs=1)
    restored = pickle.loads(pickle.dumps(worker))
    for pool in (worker.pool, restored.pool):
        assert pool._mp_context.get_start_method() != "fork"
        pool.shutdown()


def _hash_cache_location():
    from pydra.utils.hash import PersistentCache

    return str(PersistentCache.location_default())


def test_cf_workers_follow_the_cache_environment(tmp_path, monkeypatch):
    """Each pool's workers see the cache root set when the pool was created, not the
    one the forkserver started with"""
    from pydra.workers.cf import ConcurrentFuturesWorker

    for name in ("a", "b"):
        monkeypatch.setenv("PYDRA_CACHE_ROOT", str(tmp_path / name))
        worker = ConcurrentFuturesWorker(n_procs=1)
        try:
            location = worker.pool.submit(_hash_cache_location).result()
        finally:
            worker.pool.shutdown()
        assert location == str(tmp_path / name / "hashes")
