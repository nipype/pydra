import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).parents[3] / "benchmarks" / "bench_pool_batched.py"


@pytest.mark.skipif(not BENCHMARK.exists(), reason="run from a source checkout")
def test_the_batched_calls_benchmark_runs(tmp_path):
    """The benchmark of batched calls runs against the current fit protocol: 4 jobs at a
    batch of 2 take 2 calls"""
    out = subprocess.run(
        [sys.executable, str(BENCHMARK), "--jobs", "4", "--batch", "2"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr
    (row,) = [
        json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")
    ]
    assert (row["jobs"], row["batch"], row["calls"]) == (4, 2, 2)
