"""The overhead of a warm pool on trivial jobs (RFC 0005 sections 4.4, 9 and 11).

Runs one node of ``--jobs`` trivial jobs (``x -> x``) through a `WarmPool` on a `LocalSubstrate`,
driven as an external driver drives it, once with `PrivatePickle` payloads and once with a JSON codec
that carries each job as an import reference and its inputs inline in the ticket. Reports, per codec:

- ``wall_per_job_ms``: the run's wall time over its jobs, from the first ``submit`` to the last report
  (worker start-up included);
- ``write_per_job_ms``: the coordinator's time writing payloads and tickets (follow generations,
  slots), over its jobs;
- ``call_per_job_ms``: the workers' median per-job wall, from their samples.

Usage: ``python benchmarks/bench_pool_overhead.py [--jobs 10000] [--workers 4] [--quantum 64]``.
The figures are measurements of one host and storage; nothing in pydra is tuned to them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

MODULE = '''
import json
from pydra.compose import python, workflow
from pydra.workers.pool.codec import JSON_MEDIA


@python.define
def Echo(x: int) -> int:
    return x


@workflow.define
def Many(xs: list) -> list:
    return workflow.add(Echo().split(x=xs), name="E").out


_SUBMITTERS = {}


class JsonCodec:
    """A job as an import reference and its inputs, carried in the ticket as JSON (no pickle)."""

    name = "bench-json@1"
    media_type = JSON_MEDIA

    def encode(self, job):
        record = {
            "task": f"{__name__}:Echo",
            "inputs": {"x": job.task.x},
            "name": job.name,
            "store": str(job.store.path),
        }
        return json.dumps(record).encode(), JSON_MEDIA

    def decode(self, data, store):
        import importlib
        from pydra.engine.job import Job
        from pydra.engine.submitter import Submitter

        record = json.loads(data)
        module, _, qualname = record["task"].partition(":")
        task = getattr(importlib.import_module(module), qualname)(**record["inputs"])
        sub = _SUBMITTERS.get(record["store"])
        if sub is None:
            sub = _SUBMITTERS[record["store"]] = Submitter(
                cache_root=record["store"], worker="debug"
            )
        return Job(task=task, submitter=sub, name=record["name"])
'''


def run(codec_name: str, jobs: int, workers: int, quantum: int, base: Path) -> dict:
    from pydra.engine.drivers import drive
    from pydra.engine.lease import LeaseSettings
    from pydra.engine.runner import GraphRunner
    from pydra.engine.submitter import Submitter
    from pydra.workers.pool.classes import WorkerClass
    from pydra.workers.pool.codec import PrivatePickle
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.settings import PoolSettings
    from pydra.workers.pool.substrate import LocalSubstrate

    import _pydra_bench_pool as tasks

    work = base / codec_name
    sub = Submitter(cache_root=work / "store", worker="debug")
    codec = PrivatePickle() if codec_name == "private-pickle" else tasks.JsonCodec()
    lease = LeaseSettings(
        stale_after=30.0, renew_fraction=0.2, attribute_delay=0.0, timeout=60.0
    )
    settings = PoolSettings(
        lease=lease,
        poll_s=0.05,
        idle_polls=40,
        unmeasured_workers=workers,
        max_queued_jobs=jobs,
    )
    cls = WorkerClass(
        name="cpu", device="cpu", max_workers=workers, quantum_jobs=quantum
    )
    pool = WarmPool(
        work / "pool",
        [cls],
        LocalSubstrate(),
        store=sub.store,
        settings=settings,
        codec=codec,
        probe=False,
    )
    spent = [0.0]
    write = pool._write

    def timed():
        began = time.perf_counter()
        try:
            write()
        finally:
            spent[0] += time.perf_counter() - began

    pool._write = timed
    runner = GraphRunner(
        tasks.Many(xs=list(range(jobs))),
        sub.store,
        on_error="raise_at_end",
        submitter=sub,
    )
    began = time.perf_counter()
    asyncio.run(
        drive(
            runner,
            {"pool": pool},
            lambda job: "pool",
            on_error="raise_at_end",
        )
    )
    wall = time.perf_counter() - began
    if runner.errors or runner.not_run or not runner.done:
        problems = runner.errors or runner.not_run
        raise RuntimeError(
            f"the run did not complete: {len(problems)} jobs, {problems[:1]}"
        )
    calls = []
    for samples in (work / "pool" / "workers").glob("*/samples.jsonl"):
        calls += [
            json.loads(line)["wall_s"] for line in samples.read_text().splitlines()
        ]
    return {
        "codec": codec_name,
        "jobs": jobs,
        "workers": workers,
        "quantum_jobs": quantum,
        "wall_s": round(wall, 3),
        "wall_per_job_ms": round(1000 * wall / jobs, 3),
        "write_per_job_ms": round(1000 * spent[0] / jobs, 3),
        "call_per_job_ms": round(1000 * statistics.median(calls), 3) if calls else None,
        "samples": len(calls),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quantum", type=int, default=64)
    parser.add_argument("--codec", choices=["private-pickle", "json"], action="append")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="pydra-bench-pool-") as tmp:
        base = Path(tmp)
        (base / "_pydra_bench_pool.py").write_text(MODULE)
        sys.path.insert(0, str(base))
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(base), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
        for codec in args.codec or ["private-pickle", "json"]:
            record = run(codec, args.jobs, args.workers, args.quantum, base)
            print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
