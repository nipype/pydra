"""Per-task overhead of pydra 1.0a12: trivial python task split over N ints, and a 2-node workflow."""
import os, sys, time, shutil, tempfile
from pathlib import Path
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

def count_files(root):
    return sum(len(f) + len(d) for _, d, f in os.walk(root))

def run(kind, n, worker, **wkw):
    root = Path(tempfile.mkdtemp(prefix=f"pydra-{kind}-{worker}-{n}-", dir=os.environ.get("BENCH_TMP")))
    def make(m):
        return (Double() if kind == "split" else Chain()).split(x=list(range(m)))
    out = {}
    for label, m in (("cold", n), ("warm", n), ("extend10pct", n + n // 10)):
        task = make(m)
        t0 = time.perf_counter()
        with Submitter(worker=worker, cache_root=root, **wkw) as sub:
            res = sub(task)
        dt = time.perf_counter() - t0
        assert not res.errored, res
        out[label] = dt
    files = count_files(root)
    print(f"{kind:6s} worker={worker:5s} n={n:5d} cold={out['cold']:.2f}s ({1e3*out['cold']/n:.1f} ms/item) "
          f"warm={out['warm']:.2f}s ({1e3*out['warm']/n:.2f} ms/item) +10%={out['extend10pct']:.2f}s fs_entries={files} ({files/n:.1f}/item)", flush=True)
    shutil.rmtree(root)

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        kind, worker, n = spec.split(":")
        kw = {"n_procs": 4} if worker == "cf" else {}
        run(kind, int(n), worker, **kw)
