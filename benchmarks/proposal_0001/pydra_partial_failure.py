"""Does one failing split element block downstream work for the other elements? pydra 1.0a12."""
import os, tempfile
from pathlib import Path
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

root = Path(tempfile.mkdtemp(dir=os.environ["BENCH_TMP"]))
try:
    with Submitter(worker="cf", n_procs=2, cache_root=root) as sub:
        res = sub(Wf(xs=list(range(8))))
    print("errored:", res.errored, "outputs:", getattr(res, "outputs", None))
except Exception as e:
    print("raised:", type(e).__name__, str(e)[:300].replace("\n", " "))
n_next = sum(1 for d in root.iterdir() if d.is_dir() and (d / "_result.pklz").exists())
print("result dirs:", n_next)
import pickle
ran_next = []
for d in root.iterdir():
    f = d / "_result.pklz"
    if d.is_dir() and f.exists() and "MaybeFail" not in d.name:
        pass
print(sorted(d.name.split("-")[0] for d in root.iterdir() if d.is_dir()))
