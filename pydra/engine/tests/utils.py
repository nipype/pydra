# Tasks for testing
import time
import sys
import shutil
import typing as ty
from pathlib import Path
import functools
import operator
import subprocess as sp
import pytest
from fileformats.generic import File
from pydra.engine.helpers import list_fields
from pydra.engine.specs import ShellDef
from ..submitter import Submitter
from pydra.design import workflow, python

if ty.TYPE_CHECKING:
    from pydra.engine.environments import Environment


need_docker = pytest.mark.skipif(
    shutil.which("docker") is None or sp.call(["docker", "info"]),
    reason="no docker within the container",
)
need_singularity = pytest.mark.skipif(
    shutil.which("singularity") is None, reason="no singularity available"
)
no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker command not adjusted for windows docker",
)
need_slurm = pytest.mark.skipif(
    not (bool(shutil.which("sbatch")) and bool(shutil.which("sacct"))),
    reason="slurm not available",
)
need_sge = pytest.mark.skipif(
    not (bool(shutil.which("qsub")) and bool(shutil.which("qacct"))),
    reason="sge not available",
)


def get_output_names(task):
    return sorted(f.name for f in list_fields(task.Outputs))


def run_no_submitter(
    shell_def: ShellDef,
    cache_dir: Path | None = None,
    plugin: str | None = None,
    environment: "Environment | None" = None,
):
    """helper function to return result when running without submitter"""
    return shell_def(worker=plugin, cache_dir=cache_dir, environment=environment)


def run_submitter(
    shell_def: ShellDef,
    cache_dir: Path | None = None,
    plugin: str | None = None,
    environment: "Environment | None" = None,
):
    """helper function to return result when running with submitter
    with specific plugin
    """
    with Submitter(worker=plugin, cache_dir=cache_dir, environment=environment) as sub:
        results = sub(shell_def)
    if results.errored:
        raise RuntimeError(f"task {shell_def} failed:\n" + "\n".join(results.errors))
    return results.outputs


dot_check = sp.run(["which", "dot"], stdout=sp.PIPE, stderr=sp.PIPE)
if dot_check.stdout:
    DOT_FLAG = True
else:
    DOT_FLAG = False


@python.define
def Op4Var(a, b, c, d) -> str:
    return f"{a} {b} {c} {d}"


@python.define
def FunAddTwo(a: int) -> int:
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@python.define
def FunAddTwoNoType(a):
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@python.define
def FunAddTwoWithThreadCount(a: int, sgeThreads: int = 1) -> int:
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@python.define
def FunAddVar(a: ty.Union[int, float], b: ty.Union[int, float]) -> ty.Union[int, float]:
    return a + b


@python.define
def FunAddVarNoType(a, b):
    return a + b


@python.define(outputs={"sum": float, "sub": float})
def FunAddSubVar(a: float, b: float):
    return a + b, a - b


@python.define
def FunAddVarNone(a: int, b: ty.Optional[int]) -> int:
    if b is None:
        return a
    else:
        return a + b


@python.define
def FunAddVarDefault(a: int, b: int = 1) -> int:
    return a + b


@python.define
def FunAddVarDefaultNoType(a, b=1):
    return a + b


@python.define
def FunAddVar3(a: int, b: int, c: int) -> int:
    return a + b + c


@python.define
def FunAddVar4(a: int, b: int, c: int, d: int) -> int:
    return a + b + c + d


@python.define
def Moment(lst: ty.List[float], n: float) -> float:
    return sum([i**n for i in lst]) / len(lst)


@python.define
def FunDiv(a: ty.Union[int, float], b: ty.Union[int, float]) -> float:
    return a / b


@python.define
def Multiply(x: int, y: int) -> int:
    return x * y


@python.define
def Divide(x: int, y: int) -> int:
    return x // y


@python.define
def MultiplyList(x: list, y: int) -> list:
    return x * y


@python.define
def MultiplyMixed(x: list, y: int) -> list:
    return x * y


@python.define
def Add2(x: int) -> int:
    if x == 1 or x == 12:
        time.sleep(1)
    return x + 2


@python.define
def FileOrIntIdentity(in_file: ty.Union[File, int]) -> File:
    return in_file


@python.define
def FileAndIntIdentity(in_file: File, in_int: int) -> File:
    return in_file, in_int


@python.define
def ListOfListOfFileOrIntIdentity(
    in_file: ty.List[ty.List[ty.Union[int, File]]],
) -> ty.List[ty.List[ty.Union[int, File]]]:
    return in_file


@python.define
def ListOfDictOfFileOrIntIdentity(
    in_file: ty.List[ty.Dict[ty.Any, ty.Union[File, int]]],
) -> ty.List[ty.Dict[ty.Any, ty.Union[File, int]]]:
    return in_file


@python.define
def RaiseXeq1(x: int) -> int:
    if x == 1:
        raise Exception("x is 1, so i'm raising an exception!")
    return x


@python.define(outputs={"out_add": float, "out_sub": float})
def Add2Sub2Res(res):
    """function that takes entire output as an input"""
    return res["out"] + 2, res["out"] - 2


@python.define(outputs={"out_add": ty.List[float], "out_sub": ty.List[float]})
def Add2Sub2ResList(res):
    """function that takes entire output as an input"""
    return [r["out"] + 2 for r in res], [r["out"] - 2 for r in res]


@python.define
def Power(a: int, b: int) -> int:
    return a**b


@python.define
def Identity(x):
    return x


@python.define(outputs={"out1": ty.Any, "out2": ty.Any})
def Identity2Flds(x1, x2):
    return x1, x2


@python.define
def Ten(x) -> int:
    return 10


@python.define
def Add2Wait(x: int) -> int:
    time.sleep(2)
    return x + 2


@python.define
def ListOutput(x: int) -> ty.List[int]:
    return [x, 2 * x, 3 * x]


@python.define
def ListSum(x: ty.Sequence[ty.Union[int, float]]) -> ty.Union[int, float]:
    return sum(x)


@python.define
def FunDict(d: dict) -> str:
    kv_list = [f"{k}:{v}" for (k, v) in d.items()]
    return "_".join(kv_list)


@python.define
def FunWriteFile(filename: Path, text="hello") -> File:
    with open(filename, "w") as f:
        f.write(text)
    return File(filename)


@python.define
def FunWriteFileList(
    filename_list: ty.List[ty.Union[str, File, Path]], text="hi"
) -> ty.List[File]:
    for ii, filename in enumerate(filename_list):
        with open(filename, "w") as f:
            f.write(f"from file {ii}: {text}")
    filename_list = [Path(filename).absolute() for filename in filename_list]
    return filename_list


@python.define
def FunWriteFileList2Dict(
    filename_list: ty.List[ty.Union[str, File, Path]], text="hi"
) -> ty.Dict[str, ty.Union[File, int]]:
    filename_dict = {}
    for ii, filename in enumerate(filename_list):
        with open(filename, "w") as f:
            f.write(f"from file {ii}: {text}")
        filename_dict[f"file_{ii}"] = Path(filename).absolute()
    # adding an additional field with int
    filename_dict["random_int"] = 20
    return filename_dict


@python.define
def FunFile(filename: File):
    with open(filename) as f:
        txt = f.read()
    return txt


@python.define
def FunFileList(filename_list: ty.List[File]):
    txt_list = []
    for filename in filename_list:
        with open(filename) as f:
            txt_list.append(f.read())
    return " ".join(txt_list)


@workflow.define(outputs=["out"])
def BasicWorkflow(x):
    task1 = workflow.add(FunAddTwo(a=x), name="A")
    task2 = workflow.add(FunAddVar(a=task1.out, b=2), name="B")
    return task2.out


@workflow.define(outputs=["out"])
def BasicWorkflowWithThreadCount(x):
    task1 = workflow.add(FunAddTwoWithThreadCount(a=x, sgeThreads=4))
    task2 = workflow.add(FunAddVar(a=task1.out, b=2))
    return task2.out


@workflow.define(outputs=["out1", "out2"])
def BasicWorkflowWithThreadCountConcurrent(x):
    task1_1 = workflow.add(FunAddTwoWithThreadCount(a=x, sgeThreads=4))
    task1_2 = workflow.add(FunAddTwoWithThreadCount(a=x, sgeThreads=2))
    task2 = workflow.add(FunAddVar(a=task1_1.out, b=2))
    return task2.out, task1_2.out

    # return Workflow(x=5)


@python.define(outputs={"sum": int, "products": ty.List[int]})
def ListMultSum(scalar: int, in_list: ty.List[int]) -> ty.Tuple[int, ty.List[int]]:
    products = [scalar * x for x in in_list]
    return functools.reduce(operator.add, products, 0), products


@python.define(outputs={"x": str, "y": int, "z": float})
def Foo(a: str, b: int, c: float) -> ty.Tuple[str, int, float]:
    return a, b, c
