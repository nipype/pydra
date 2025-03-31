import typing as ty
import sys
import pytest
import cloudpickle as cp
from pathlib import Path
import glob as glob
from pydra.compose import python, workflow
from pydra.engine.submitter import Submitter


no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker/singularity command not adjusted for windows",
)


def test_exception_func():
    @python.define
    def raise_exception(c, d):
        raise Exception()

    bad_funk = raise_exception(c=17, d=3.2)
    assert pytest.raises(Exception, bad_funk)


def test_result_none_1():
    """checking if None is properly returned as the result"""

    @python.define
    def FunNone(x):
        return None

    task = FunNone(x=3)
    outputs = task()
    assert outputs.out is None


def test_result_none_2():
    """checking if None is properly set for all outputs"""

    @python.define(outputs=["out1", "out2"])
    def FunNone(x) -> tuple[ty.Any, ty.Any]:
        return None  # Do we actually want this behaviour?

    task = FunNone(x=3)
    outputs = task()
    assert outputs.out1 is None
    assert outputs.out2 is None


def test_traceback(tmpdir):
    """checking if the error raised in a function is properly returned;
    checking if there is an error filename in the error message that contains
    full traceback including the line in the python function
    """

    @python.define
    def FunError(x):
        raise Exception("Error from the function")

    with pytest.raises(Exception, match="Error from the function") as exinfo:
        with Submitter(worker="cf", cache_root=tmpdir) as sub:
            sub(FunError(x=3), raise_errors=True)

    # getting error file from the error message
    error_file_match = (
        str(exinfo.value.__notes__[0]).split("here: ")[-1].split("_error.pklz")[0]
    )
    error_file = Path(error_file_match) / "_error.pklz"
    # checking if the file exists
    assert error_file.exists()
    # reading error message from the pickle file
    error_tb = cp.loads(error_file.read_bytes())["error message"]
    # the error traceback should be a list and should point to a specific line in the function
    assert isinstance(error_tb, list)
    assert "in FunError" in error_tb[-2]


def test_traceback_wf(tmp_path: Path):
    """checking if the error raised in a function is properly returned by a workflow;
    checking if there is an error filename in the error message that contains
    full traceback including the line in the python function
    """

    @python.define
    def FunError(x):
        raise Exception("Error from the function")

    @workflow.define
    def Workflow(x_list):
        fun_error = workflow.add(FunError().split(x=x_list), name="fun_error")
        return fun_error.out

    wf = Workflow(x_list=[3, 4])
    with pytest.raises(RuntimeError, match="Job 'fun_error.*, errored") as exinfo:
        with Submitter(worker="cf", cache_root=tmp_path) as sub:
            sub(wf, raise_errors=True)

    # getting error file from the error message
    output_dir_match = Path(
        str(exinfo.value).split("See output directory for details: ")[-1].strip()
    )
    assert output_dir_match.exists()
    error_file = output_dir_match / "_error.pklz"
    # checking if the file exists
    assert error_file.exists()
    assert "in FunError" in str(exinfo.value)


@pytest.mark.flaky(reruns=3)
def test_rerun_errored(tmp_path, capfd):
    """Test rerunning a task containing errors.
    Only the errored tasks should be rerun"""

    @python.define
    def PassOdds(x):
        if x % 2 == 0:
            print(f"x={x} -> x%2 = {bool(x % 2)} (even error)\n")
            raise Exception("even error")
        else:
            print(f"x={x} -> x%2 = {bool(x % 2)}\n")
            return x

    pass_odds = PassOdds().split("x", x=[1, 2, 3, 4, 5])

    with pytest.raises(Exception):
        pass_odds(cache_root=tmp_path, worker="cf", n_procs=3)
    with pytest.raises(Exception):
        pass_odds(cache_root=tmp_path, worker="cf", n_procs=3)

    out, err = capfd.readouterr()
    stdout_lines = out.splitlines()

    tasks_run = 0
    errors_found = 0

    for line in stdout_lines:
        if "-> x%2" in line:
            tasks_run += 1
        if "(even error)" in line:
            errors_found += 1

    # There should have been 5 messages of the form "x%2 = XXX" after calling task() the first time
    # and another 2 messagers after calling the second time
    assert tasks_run == 7
    assert errors_found == 4
