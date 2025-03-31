from pathlib import Path
import pytest
import cloudpickle as cp
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.compose import workflow
from pydra.engine.tests.utils import Multiply, RaiseXeq1
from pydra.engine.job import save, load_and_run


def test_save(tmpdir):
    outdir = Path(tmpdir)
    with pytest.raises(ValueError):
        save(tmpdir)
    foo = Job(name="mult", task=Multiply(x=1, y=2), submitter=Submitter())
    # save job
    save(outdir, job=foo)
    del foo
    # load saved job
    job_pkl = outdir / "_job.pklz"
    foo: Job = cp.loads(job_pkl.read_bytes())
    assert foo.name == "mult"
    assert foo.inputs["x"] == 1 and foo.inputs["y"] == 2
    # execute job and save result
    res: Result = foo.run()
    assert res.outputs.out == 2
    save(outdir, result=res)
    del res
    # load saved result
    res_pkl = outdir / "_result.pklz"
    res: Result = cp.loads(res_pkl.read_bytes())
    assert res.outputs.out == 2


def test_load_and_run(tmpdir):
    """testing load_and_run for pickled job"""
    job_pkl = Path(tmpdir.join("task_main.pkl"))
    # Note that tasks now don't have state arrays and indices, just a single resolved
    # set of parameters that are ready to run
    job = Job(name="mult", task=Multiply(x=2, y=10), submitter=Submitter())
    with job_pkl.open("wb") as fp:
        cp.dump(job, fp)
    resultfile = load_and_run(job_pkl=job_pkl)
    # checking the result files
    result = cp.loads(resultfile.read_bytes())
    assert result.outputs.out == 20


def test_load_and_run_exception_run(tmpdir):
    """testing raising exception and saving info in crashfile when when load_and_run"""
    job_pkl = Path(tmpdir.join("task_main.pkl"))
    cache_root = Path(tmpdir.join("cache"))
    cache_root.mkdir()

    job = Job(
        task=RaiseXeq1(x=1),
        name="raise",
        submitter=Submitter(worker="cf", cache_root=cache_root),
    )

    with job_pkl.open("wb") as fp:
        cp.dump(job, fp)

    with pytest.raises(Exception) as excinfo:
        load_and_run(job_pkl=job_pkl)
    exc_msg = excinfo.value.args[0]
    assert "i'm raising an exception!" in exc_msg
    # checking if the crashfile has been created
    assert "crash" in excinfo.value.__notes__[0]
    errorfile = Path(excinfo.value.__notes__[0].split("here: ")[1])
    assert errorfile.exists()

    resultfile = errorfile.parent / "_result.pklz"
    assert resultfile.exists()
    # checking the content
    result_exception = cp.loads(resultfile.read_bytes())
    assert result_exception.errored is True

    job = Job(task=RaiseXeq1(x=2), name="wont_raise", submitter=Submitter())

    with job_pkl.open("wb") as fp:
        cp.dump(job, fp)

    # the second job should be fine
    resultfile = load_and_run(job_pkl=job_pkl)
    result_1 = cp.loads(resultfile.read_bytes())
    assert result_1.outputs.out == 2


def test_load_and_run_wf(tmpdir, worker):
    """testing load_and_run for pickled job"""
    wf_pkl = Path(tmpdir.join("wf_main.pkl"))

    @workflow.define
    def Workflow(x, y=10):
        multiply = workflow.add(Multiply(x=x, y=y))
        return multiply.out

    job = Job(
        name="mult",
        task=Workflow(x=2),
        submitter=Submitter(cache_root=tmpdir, worker=worker),
    )

    with wf_pkl.open("wb") as fp:
        cp.dump(job, fp)

    resultfile = load_and_run(job_pkl=wf_pkl)
    # checking the result files
    result = cp.loads(resultfile.read_bytes())
    assert result.outputs.out == 20
