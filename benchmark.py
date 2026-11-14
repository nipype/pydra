"""Benchmarks for the execution extensions of proposal 0001.

Run ``python benchmark.py <command>``:

``toy`` (the default)
    Cost of a sync against an async function call.
``overhead [kind:worker:n ...]``
    Per-item overhead of a trivial python task split over ``n`` ints (``kind`` is
    ``split``) or of a 2-node workflow split over ``n`` ints (``kind`` is ``chain``), on
    ``worker`` (``debug`` or ``cf``). Prints cold and cached time per item, the extension
    of 10% more items, and the filesystem entries per item. With no spec it runs
    ``split:debug:200 split:cf:200 chain:debug:200 chain:cf:200``.
``partial-failure``
    Whether one failing element of an 8-way split blocks the downstream node for the
    healthy elements.

``BENCH_TMP`` names the directory the caches are made in (default: the system temp
directory). Timings are only meaningful on an idle machine.
"""

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_OVERHEAD = (
    "split:debug:200",
    "split:cf:200",
    "chain:debug:200",
    "chain:cf:200",
)


def sync_function(x):
    return x * 2


async def async_function(x):
    return x * 2


def benchmark_sync():
    start_time = time.time()
    for _ in range(1000000):
        sync_function(10)
    end_time = time.time()
    return end_time - start_time


async def benchmark_async():
    start_time = time.time()
    for _ in range(1000000):
        await async_function(10)
    end_time = time.time()
    return end_time - start_time


def toy():
    sync_time = benchmark_sync()
    print(f"Sync function time: {sync_time:.6f} seconds")

    async_time = asyncio.run(benchmark_async())
    print(f"Async function time: {async_time:.6f} seconds")


def _count_files(root):
    return sum(len(f) + len(d) for _, d, f in os.walk(root))


def overhead(kind, n, worker, **worker_kwargs):
    """Time ``n`` items cold, cached, and with 10% more items, and print the result."""
    from pydra.compose import python, workflow
    from pydra.engine.submitter import Submitter

    @python.define
    def Double(x: int) -> int:
        return x * 2

    @python.define
    def AddOne(x: int) -> int:
        return x + 1

    @workflow.define
    def Chain(x: int) -> int:
        a = workflow.add(Double(x=x))
        b = workflow.add(AddOne(x=a.out))
        return b.out

    root = Path(
        tempfile.mkdtemp(
            prefix=f"pydra-{kind}-{worker}-{n}-", dir=os.environ.get("BENCH_TMP")
        )
    )

    def make(m):
        return (Double() if kind == "split" else Chain()).split(x=list(range(m)))

    out = {}
    try:
        for label, m in (("cold", n), ("warm", n), ("extend10pct", n + n // 10)):
            task = make(m)
            t0 = time.perf_counter()
            with Submitter(worker=worker, cache_root=root, **worker_kwargs) as sub:
                res = sub(task)
            out[label] = time.perf_counter() - t0
            assert not res.errored, res
        files = _count_files(root)
    finally:
        shutil.rmtree(root)
    cold_ms = 1e3 * out["cold"] / n
    warm_ms = 1e3 * out["warm"] / n
    print(
        f"{kind:6s} worker={worker:5s} n={n:5d} cold={out['cold']:.2f}s ({cold_ms:.1f} ms/item) "
        f"warm={out['warm']:.2f}s ({warm_ms:.2f} ms/item) +10%={out['extend10pct']:.2f}s "
        f"fs_entries={files} ({files / n:.1f}/item)",
        flush=True,
    )


def partial_failure():
    """Does one failing split element block downstream work for the other elements?"""
    from pydra.compose import python, workflow
    from pydra.engine.submitter import Submitter

    @python.define
    def MaybeFail(x: int) -> int:
        if x == 3:
            raise ValueError("bad record")
        return x

    @python.define
    def Next(x: int) -> int:
        return x + 100

    @workflow.define
    def Wf(xs: list[int]) -> list[int]:
        a = workflow.add(MaybeFail().split(x=xs))
        b = workflow.add(Next(x=a.out))
        return b.out

    root = Path(tempfile.mkdtemp(dir=os.environ.get("BENCH_TMP")))
    try:
        try:
            with Submitter(worker="cf", n_procs=2, cache_root=root) as sub:
                res = sub(Wf(xs=list(range(8))))
            print("errored:", res.errored, "outputs:", getattr(res, "outputs", None))
        except Exception as e:
            print("raised:", type(e).__name__, str(e)[:300].replace("\n", " "))
        done = sorted(
            str(f.relative_to(root)).split("-")[0] for f in root.rglob("_result.pklz")
        )
        print("result files:", len(done))
        print("by node:", {n: done.count(n) for n in sorted(set(done))})
    finally:
        shutil.rmtree(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("toy", help="sync against async function calls")
    p = sub.add_parser("overhead", help="per-item overhead, cold and cached")
    p.add_argument(
        "specs",
        nargs="*",
        metavar="kind:worker:n",
        help=f"default: {' '.join(DEFAULT_OVERHEAD)}",
    )
    sub.add_parser("partial-failure", help="one failed element of a split")
    args = parser.parse_args(argv)
    if args.command in (None, "toy"):
        toy()
    elif args.command == "overhead":
        for spec in args.specs or DEFAULT_OVERHEAD:
            kind, worker, n = spec.split(":")
            overhead(kind, int(n), worker, **({"n_procs": 4} if worker == "cf" else {}))
    else:
        partial_failure()


if __name__ == "__main__":
    sys.exit(main())
