"""Cold against warm environments (RFC 0005 sections 3.1, 9 and 11).

For each runtime asked for, runs ``--commands`` shell commands (``echo <i>``) twice:
cold, through the environment's ``execute`` (``docker run``, ``apptainer exec``, a
``module`` purge and load per command), and warm, through a `WarmSession` (one
``docker run -d`` and a ``docker exec`` per command, one Apptainer instance and an
``exec`` in it per command, one Lmod resolution). Prints one JSON line per runtime:

- ``cold_ms`` and ``warm_ms``: median and p95 wall per command;
- ``cold_total_s`` and ``warm_total_s``: the whole series, the warm start included;
- ``start_s``: the warm start (the first warm command's wall less the warm median);
- ``equal``: every warm command's output equals the cold one's.

Runtimes: ``native`` (cold only: ``Native.warm`` is ``None``), ``docker`` (``--image``,
default ``busybox``), ``apptainer`` (``--sif-image``, default ``docker://alpine:3.20``,
resolved to its digest and run from the local SIF), ``lmod`` (``--modules``; without
it a synthetic one-module tree is built and put on ``MODULEPATH``).

Usage::

    python benchmarks/bench_warm_environment.py --runtime native docker
    python benchmarks/bench_warm_environment.py --runtime apptainer lmod --commands 200

The figures are measurements of one host; nothing in pydra is tuned to them.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from pydra.compose import shell
from pydra.engine.job import Job
from pydra.engine.submitter import Submitter
from pydra.environments import docker, lmod, native, singularity
from pydra.environments.base import WarmSession


def _jobs(env, cache: Path, n: int) -> list[Job]:
    sub = Submitter(
        worker="debug",
        store="pickle",
        environment=env,
        cache_root=cache,
        telemetry=False,
    )
    jobs = []
    for i in range(n):
        task = shell.define("echo")(executable="echo", append_args=[str(i)])
        job = Job(task=task, name=f"echo{i}", submitter=sub)
        if job.resolves_reference:
            job.resolved  # the submitting process resolves, once per run
        jobs.append(job)
    return jobs


def _time(run, jobs) -> tuple[list[float], list[str]]:
    walls, outs = [], []
    for job in jobs:
        start = time.perf_counter()
        out = run(job)
        walls.append(time.perf_counter() - start)
        outs.append(out["stdout"])
    return walls, outs


def _summary(walls: list[float]) -> dict:
    ordered = sorted(walls)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "median": round(statistics.median(ordered) * 1000, 2),
        "p95": round(p95 * 1000, 2),
    }


def _environments(runtime: str, args, tmp: Path):
    if runtime == "native":
        return native.Native(), native.Native(warm=True)
    if runtime == "docker":
        return docker.Docker(image=args.image), docker.Docker(
            image=args.image, warm=True
        )
    if runtime == "apptainer":
        ref = singularity.Singularity(image=args.sif_image).resolve()
        return singularity.Singularity(image=ref.run), singularity.Singularity(
            image=ref.run, warm=True
        )
    if runtime == "lmod":
        modules = args.modules
        if not modules:
            tree = tmp / "modules" / "tool"
            tree.mkdir(parents=True)
            (tree / "2.0.lua").write_text('setenv("TOOL_VERSION", "2.0")\n')
            os.environ["MODULEPATH"] = str(tmp / "modules")
            modules = ["tool"]
        return lmod.Lmod(modules=modules), lmod.Lmod(modules=modules, warm=True)
    raise SystemExit(f"unknown runtime {runtime!r}")


def bench(runtime: str, args) -> dict:
    with tempfile.TemporaryDirectory(prefix="bench-warm-") as tmp:
        tmp = Path(tmp)
        cold_env, warm_env = _environments(runtime, args, tmp)
        cold_walls, cold_outs = _time(
            lambda job: job.environment.execute(job),
            _jobs(cold_env, tmp / "cold", args.commands),
        )
        record = {
            "runtime": runtime,
            "commands": args.commands,
            "cold_ms": _summary(cold_walls),
            "cold_total_s": round(sum(cold_walls), 3),
        }
        session = WarmSession(warm_env)
        try:
            warm_walls, warm_outs = _time(
                session.execute, _jobs(warm_env, tmp / "warm", args.commands)
            )
        finally:
            session.close()
        if session.starts == 0:
            record["warm"] = None  # not warmable: every command ran cold
            return record
        median = statistics.median(warm_walls)
        record.update(
            {
                "warm_ms": _summary(warm_walls),
                "warm_total_s": round(sum(warm_walls), 3),
                "start_s": round(warm_walls[0] - median, 3),
                "starts": session.starts,
                "cold_runs": session.cold_runs,
                "speedup_total": round(sum(cold_walls) / sum(warm_walls), 2),
                "equal": warm_outs == cold_outs,
            }
        )
        return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runtime",
        nargs="+",
        default=["native"],
        choices=["native", "docker", "apptainer", "lmod"],
    )
    parser.add_argument("--commands", type=int, default=200)
    parser.add_argument("--image", default="busybox")
    parser.add_argument("--sif-image", default="docker://alpine:3.20")
    parser.add_argument("--modules", nargs="*", default=None)
    args = parser.parse_args(argv)
    failed = False
    for runtime in args.runtime:
        record = bench(runtime, args)
        print(json.dumps(record), flush=True)
        failed |= record.get("equal") is False
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
