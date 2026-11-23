"""Per-item wall time of batched calls against one call per job, on a fake model (RFC 0005 W10).

Runs one node of ``--jobs`` jobs of a task declared ``batched=`` through a `WarmPool` with one worker
on a `LocalSubstrate`, once per batch size in ``--batch`` (1 runs the task's function, one job per
call), set by a fit that returns it under the task's declared ceiling ``batch=1024``. The fake
model costs ``--overhead-ms`` per call, whatever it holds, plus ``--item-ms`` per item, as a forward
pass on a device does. Reports, per batch size:

- ``wall_per_job_ms``: the run's wall time over its jobs, from the first ``submit`` to the last report
  (worker start-up included);
- ``call_per_item_ms``: the median over calls of the call's wall over its jobs, from the samples;
- ``calls``: the calls the worker made.

Usage: ``python benchmarks/bench_pool_batched.py [--jobs 64] [--batch 1 --batch 8 --batch 32]
[--overhead-ms 50] [--item-ms 2]``. A fake-model measurement of one host: real GPU figures come from a
model on its device.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

MODULE = """
import os
import time
from pydra.compose import python, workflow

OVERHEAD = float(os.environ.get("BENCH_OVERHEAD_S", "0.05"))
ITEM = float(os.environ.get("BENCH_ITEM_S", "0.002"))


def forward_many(x):
    time.sleep(OVERHEAD + ITEM * len(x))
    return [v * 2 for v in x]


@python.define(batched=forward_many, batch=1024)
def Forward(x: int) -> int:
    time.sleep(OVERHEAD + ITEM)
    return x * 2


@workflow.define
def Many(xs: list) -> list:
    return workflow.add(Forward().split(x=xs), name="F").out
"""


def batch_of(size: int):
    """A `DeclaredFit` that batches ``size`` jobs per call and holds them all in memory"""
    from pydra.workers.pool.placement import DeclaredFit

    class BatchOf(DeclaredFit):
        def batch(self, request, cls, units):
            return size

        def memory_bound(self, request, cls, units):
            return size

    return BatchOf()


def run(jobs: int, batch: int, base: Path) -> dict:
    from pydra.engine.lease import LeaseSettings
    from pydra.engine.submitter import Submitter
    from pydra.workers.pool.classes import WorkerClass
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.settings import PoolSettings
    from pydra.workers.pool.substrate import LocalSubstrate

    import _pydra_bench_batched as tasks

    work = base / f"batch-{batch}"
    store = Submitter(cache_root=work / "store", worker="debug").store
    lease = LeaseSettings(
        stale_after=30.0, renew_fraction=0.2, attribute_delay=0.0, timeout=60.0
    )
    settings = PoolSettings(
        lease=lease,
        poll_s=0.05,
        idle_polls=40,
        unmeasured_workers=1,
        max_queued_jobs=jobs,
    )
    cls = WorkerClass(name="cpu", device="cpu", max_workers=1, quantum_jobs=jobs)
    pool = WarmPool(
        work / "pool",
        [cls],
        LocalSubstrate(),
        store=store,
        settings=settings,
        fit=batch_of(batch),
        probe=False,
    )
    try:
        with Submitter(store=store, worker=pool) as sub:
            began = time.perf_counter()
            result = sub(tasks.Many(xs=list(range(jobs))))
            wall = time.perf_counter() - began
    finally:
        pool.close()
    if result.errored or result.outputs.out != [2 * x for x in range(jobs)]:
        raise RuntimeError(f"the run did not complete: {result}")
    per_item, calls = [], set()
    for samples in (work / "pool" / "workers").glob("*/samples.jsonl"):
        for line in samples.read_text().splitlines():
            record = json.loads(line)
            per_item.append(record["wall_s"] / record["batch_size"])
            calls.add((samples.parent.name, record["ended"]))
    return {
        "jobs": jobs,
        "batch": batch,
        "wall_s": round(wall, 3),
        "wall_per_job_ms": round(1000 * wall / jobs, 3),
        "call_per_item_ms": round(1000 * statistics.median(per_item), 3),
        "calls": len(calls),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument("--overhead-ms", type=float, default=50.0)
    parser.add_argument("--item-ms", type=float, default=2.0)
    args = parser.parse_args(argv)
    os.environ["BENCH_OVERHEAD_S"] = str(args.overhead_ms / 1000)
    os.environ["BENCH_ITEM_S"] = str(args.item_ms / 1000)
    with tempfile.TemporaryDirectory(prefix="pydra-bench-batched-") as tmp:
        base = Path(tmp)
        (base / "_pydra_bench_batched.py").write_text(MODULE)
        sys.path.insert(0, str(base))
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(base), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
        for batch in args.batch or [1, 8, 32]:
            record = run(args.jobs, batch, base)
            record.update(overhead_ms=args.overhead_ms, item_ms=args.item_ms)
            print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
