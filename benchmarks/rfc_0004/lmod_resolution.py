"""Lmod resolution on a cluster (RFC 0004 sections 5.3, 7 and 10).

Builds a hierarchical module tree with a sticky module, then checks on the real Lmod:

* a compiler loaded in the calling shell does not select a build (``openmpi`` alone is
  unresolved once the shell's ``gcc`` is purged);
* two compiler builds give two keys;
* a ``depends_on`` dependency and the sticky module are recorded, not keyed;
* the full names resolved equal ``module -t list`` after ``module purge; module load``;
* a job runs the build its key names, whatever compiler the worker's shell has loaded.

Then it times ``Lmod(...).resolve()`` (no memo) for module lists of 1, 3 and 6 modules of
the synthetic tree, and for any site lists given with ``--site "a b|c d e"``, and prints
the median and p95 per list. Exits non-zero if a check fails.

Usage (on a node with Lmod; ``MODULESHOME`` set)::

    python benchmarks/rfc_0004/lmod_resolution.py --runs 20 [--site "gcc/12.2.0 python/3.11"]
"""

import argparse
import os
import re
import statistics
import subprocess as sp
import sys
import tempfile
import time
from pathlib import Path

from pydra.compose import shell
from pydra.engine.submitter import Submitter
from pydra.environments.lmod import Lmod

TOOLS = [f"tool{i}" for i in range(1, 7)]


def build_tree(root: Path) -> Path:
    core = root / "Core"

    def module(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    module(core / "stdenv" / "1.0.lua", 'add_property("lmod", "sticky")\n')
    for version in ("11.2", "12.2"):
        compiler = root / "Compiler" / "gcc" / version
        module(
            core / "gcc" / f"{version}.lua",
            f'family("compiler")\nprepend_path("MODULEPATH", "{compiler}")\n',
        )
        module(
            compiler / "openmpi" / "4.1.5.lua",
            f'setenv("MPI_BUILD", "gcc-{version}")\n',
        )
    module(core / "fsl" / "6.0.7.9.lua", 'depends_on("gcc/12.2")\n')
    for tool in TOOLS:
        module(core / tool / "1.0.lua", f'setenv("{tool.upper()}", "1")\n')
    return core


def apply_load(*names: str) -> None:
    """Load ``names`` into this process's environment, as a shell would"""
    output = Lmod.run_lmod_cmd("python", "load", *names)
    os.environ.update(
        re.findall(r"""os\.environ\[['"](.*?)['"]\]\s*=\s*['"](.*?)['"]""", output)
    )


def module_list(names: list[str]) -> list[str]:
    script = (
        f'source "{os.environ["MODULESHOME"]}/init/bash" && module purge && '
        f"module load {' '.join(names)} && module -t list 2>&1"
    )
    out = sp.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    return [line.strip() for line in out.stdout.splitlines() if "/" in line]


def timings(names: list[str], runs: int) -> tuple[float, float]:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        ref = Lmod(modules=names).resolve()
        times.append((time.perf_counter() - start) * 1000)
        if ref.unresolved:
            raise SystemExit(f"{names} did not resolve: {ref.key}")
    times.sort()
    return statistics.median(times), times[max(0, round(0.95 * len(times)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--site", default="", help="site module lists, '|'-separated")
    args = parser.parse_args()
    if "MODULESHOME" not in os.environ:
        print("FAIL: MODULESHOME is not set (no Lmod here)")
        return 1
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}")
        if not ok:
            failures.append(name)

    site_path = os.environ.get("MODULEPATH", "")
    with tempfile.TemporaryDirectory() as tmp:
        core = build_tree(Path(tmp))
        os.environ["MODULEPATH"] = str(core)
        apply_load("stdenv/1.0")
        apply_load("gcc/12.2")  # the calling shell's compiler

        alone = Lmod(modules=["openmpi"]).resolve()
        check("shell compiler does not leak", alone.unresolved, alone.key)
        a = Lmod(modules=["gcc/12.2", "openmpi"]).resolve()
        b = Lmod(modules=["gcc/11.2", "openmpi"]).resolve()
        check("compiler builds key apart", a.key != b.key, f"{a.key} | {b.key}")
        fsl = Lmod(modules=["fsl"]).resolve()
        deps = fsl.recorded.get("dependencies", [])
        check(
            "dependency and sticky recorded, not keyed",
            fsl.key == "lmod@1:fsl/6.0.7.9"
            and "gcc/12.2" in deps
            and "stdenv/1.0" in deps,
            f"{fsl.key} deps={deps}",
        )
        for names in (["gcc/11.2", "openmpi"], ["fsl"], TOOLS[:3]):
            ref = Lmod(modules=names).resolve()
            resolved = sorted(
                set(ref.recorded["modules"].values())
                | set(ref.recorded["dependencies"])
            )
            listed = sorted(module_list(names))
            check(
                f"names equal module -t list for {names}",
                resolved == listed,
                f"resolved={resolved} listed={listed}",
            )

        apply_load("gcc/11.2")  # the worker's shell now has the other compiler
        with Submitter(
            environment=Lmod(modules=["gcc/12.2", "openmpi"]),
            cache_root=Path(tmp) / "cache",
            telemetry=False,
            resolve_images="require",
        ) as sub:
            result = sub(shell.define("printenv MPI_BUILD")())
        check(
            "job runs the keyed build",
            result.outputs.stdout.strip() == "gcc-12.2",
            result.outputs.stdout.strip(),
        )

        rows = [
            (names, timings(names, args.runs))
            for names in (TOOLS[:1], TOOLS[:3], TOOLS)
        ]
        os.environ["MODULEPATH"] = site_path
        for spec in filter(None, args.site.split("|")):
            rows.append((spec.split(), timings(spec.split(), args.runs)))

    print(f"\n| modules | n | median ms | p95 ms |  (runs={args.runs})")
    print("|---|---|---|---|")
    for names, (median, p95) in rows:
        print(f"| {' '.join(names)} | {len(names)} | {median:.1f} | {p95:.1f} |")
    print(f"\n{'FAIL' if failures else 'PASS'}: {len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
