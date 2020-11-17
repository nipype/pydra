from dateutil import parser
import re
import shutil
import subprocess as sp
import time

import pytest

from .utils import gen_basic_wf
from ..core import Workflow
from ..task import ShellCommandTask
from ..submitter import Submitter
from ... import mark

slurm_available = bool(shutil.which("sbatch"))


@mark.task
def sleep_add_one(x):
    time.sleep(1)
    return x + 1


def test_callable_wf(plugin, tmpdir):
    wf = gen_basic_wf()

    with pytest.raises(NotImplementedError):
        wf()

    res = wf(plugin="cf")
    assert res.output.out == 9
    del wf, res

    wf = gen_basic_wf()
    wf.cache_dir = tmpdir

    sub = Submitter(plugin)
    res = wf(submitter=sub)
    assert res.output.out == 9


def test_concurrent_wf(plugin, tmpdir):
    # concurrent workflow
    # A --> C
    # B --> D
    wf = Workflow("new_wf", input_spec=["x", "y"])
    wf.inputs.x = 5
    wf.inputs.y = 10
    wf.add(sleep_add_one(name="taska", x=wf.lzin.x))
    wf.add(sleep_add_one(name="taskb", x=wf.lzin.y))
    wf.add(sleep_add_one(name="taskc", x=wf.taska.lzout.out))
    wf.add(sleep_add_one(name="taskd", x=wf.taskb.lzout.out))
    wf.set_output([("out1", wf.taskc.lzout.out), ("out2", wf.taskd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out1 == 7
    assert res.output.out2 == 12


def test_concurrent_wf_nprocs(tmpdir):
    # concurrent workflow
    # setting n_procs in Submitter that is passed to the worker
    # A --> C
    # B --> D
    wf = Workflow("new_wf", input_spec=["x", "y"])
    wf.inputs.x = 5
    wf.inputs.y = 10
    wf.add(sleep_add_one(name="taska", x=wf.lzin.x))
    wf.add(sleep_add_one(name="taskb", x=wf.lzin.y))
    wf.add(sleep_add_one(name="taskc", x=wf.taska.lzout.out))
    wf.add(sleep_add_one(name="taskd", x=wf.taskb.lzout.out))
    wf.set_output([("out1", wf.taskc.lzout.out), ("out2", wf.taskd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter("cf", n_procs=2) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out1 == 7
    assert res.output.out2 == 12


def test_wf_in_wf(plugin, tmpdir):
    """WF(A --> SUBWF(A --> B) --> B)"""
    wf = Workflow(name="wf_in_wf", input_spec=["x"])
    wf.inputs.x = 3
    wf.add(sleep_add_one(name="wf_a", x=wf.lzin.x))

    # workflow task
    subwf = Workflow(name="sub_wf", input_spec=["x"])
    subwf.add(sleep_add_one(name="sub_a", x=subwf.lzin.x))
    subwf.add(sleep_add_one(name="sub_b", x=subwf.sub_a.lzout.out))
    subwf.set_output([("out", subwf.sub_b.lzout.out)])
    # connect, then add
    subwf.inputs.x = wf.wf_a.lzout.out
    subwf.cache_dir = tmpdir

    wf.add(subwf)
    wf.add(sleep_add_one(name="wf_b", x=wf.sub_wf.lzout.out))
    wf.set_output([("out", wf.wf_b.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 7


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf2(plugin_dask_opt, tmpdir):
    """ workflow as a node
        workflow-node with one task and no splitter
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(sleep_add_one(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wfnd.inputs.x = 2

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 3


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf_with_state(plugin_dask_opt, tmpdir):
    wf = Workflow(name="wf_with_state", input_spec=["x"])
    wf.add(sleep_add_one(name="taska", x=wf.lzin.x))
    wf.add(sleep_add_one(name="taskb", x=wf.taska.lzout.out))

    wf.inputs.x = [1, 2, 3]
    wf.split("x")
    wf.set_output([("out", wf.taskb.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(wf)

    res = wf.result()

    assert res[0].output.out == 3
    assert res[1].output.out == 4
    assert res[2].output.out == 5


@pytest.mark.skipif(not slurm_available, reason="slurm not installed")
def test_slurm_wf(tmpdir):
    wf = gen_basic_wf()
    wf.cache_dir = tmpdir
    # submit workflow and every task as slurm job
    with Submitter("slurm") as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 9
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()
    # ensure each task was executed with slurm
    assert len([sd for sd in script_dir.listdir() if sd.isdir()]) == 2


@pytest.mark.skipif(not slurm_available, reason="slurm not installed")
def test_slurm_wf_cf(tmpdir):
    # submit entire workflow as single job executing with cf worker
    wf = gen_basic_wf()
    wf.cache_dir = tmpdir
    wf.plugin = "cf"
    with Submitter("slurm") as sub:
        sub(wf)
    res = wf.result()
    assert res.output.out == 9
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()
    # ensure only workflow was executed with slurm
    sdirs = [sd for sd in script_dir.listdir() if sd.isdir()]
    assert len(sdirs) == 1
    assert sdirs[0].basename == wf.checksum


@pytest.mark.skipif(not slurm_available, reason="slurm not installed")
def test_slurm_wf_state(tmpdir):
    wf = gen_basic_wf()
    wf.split("x")
    wf.inputs.x = [5, 6]
    wf.cache_dir = tmpdir
    with Submitter("slurm") as sub:
        sub(wf)
    res = wf.result()
    assert res[0].output.out == 9
    assert res[1].output.out == 10
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()
    sdirs = [sd for sd in script_dir.listdir() if sd.isdir()]
    assert len(sdirs) == 2 * len(wf.inputs.x)


@pytest.mark.skipif(not slurm_available, reason="slurm not installed")
@pytest.mark.flaky(reruns=3)
def test_slurm_max_jobs(tmpdir):
    wf = Workflow("new_wf", input_spec=["x", "y"], cache_dir=tmpdir)
    wf.inputs.x = 5
    wf.inputs.y = 10
    wf.add(sleep_add_one(name="taska", x=wf.lzin.x))
    wf.add(sleep_add_one(name="taskb", x=wf.lzin.y))
    wf.add(sleep_add_one(name="taskc", x=wf.taska.lzout.out))
    wf.add(sleep_add_one(name="taskd", x=wf.taskb.lzout.out))
    wf.set_output([("out1", wf.taskc.lzout.out), ("out2", wf.taskd.lzout.out)])
    with Submitter("slurm", max_jobs=1) as sub:
        sub(wf)

    jobids = []
    time.sleep(0.5)  # allow time for sacct to collect itself
    for fl in (tmpdir / "SlurmWorker_scripts").visit("slurm-*.out"):
        jid = re.search(r"(?<=slurm-)\d+", fl.strpath)
        assert jid.group()
        jobids.append(jid.group())
        time.sleep(0.2)
        del jid

    # query sacct for job eligibility timings
    queued = []
    for jid in sorted(jobids):
        out = sp.run(["sacct", "-Xnj", jid, "-o", "Eligible"], capture_output=True)
        et = out.stdout.decode().strip()
        queued.append(parser.parse(et))
        del out, et

    # compare timing between queued jobs
    prev = None
    for et in sorted(queued, reverse=True):
        if prev is None:
            prev = et
            continue
        assert (prev - et).seconds >= 2


@pytest.mark.skipif(not slurm_available, reason="slurm not installed")
def test_slurm_args_1(tmpdir):
    """ testing sbatch_args provided to the submitter"""
    task = sleep_add_one(x=1)
    task.cache_dir = tmpdir
    # submit workflow and every task as slurm job
    with Submitter("slurm", sbatch_args="-N1") as sub:
        sub(task)

    res = task.result()
    assert res.output.out == 2
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()


@pytest.mark.skipif(not slurm_available, reason="slurm not installed")
def test_slurm_args_2(tmpdir):
    """ testing sbatch_args provided to the submitter
        exception should be raised for invalid options
    """
    task = sleep_add_one(x=1)
    task.cache_dir = tmpdir
    # submit workflow and every task as slurm job
    with pytest.raises(RuntimeError, match="Error returned from sbatch:"):
        with Submitter("slurm", sbatch_args="-N1 --invalid") as sub:
            sub(task)


@mark.task
def sleep(x, job_name_part):
    time.sleep(x)
    import subprocess as sp

    # getting the job_id of the first job that sleeps
    job_id = 999
    while job_id != "":
        time.sleep(3)
        id_p1 = sp.Popen(["squeue"], stdout=sp.PIPE)
        id_p2 = sp.Popen(["grep", job_name_part], stdin=id_p1.stdout, stdout=sp.PIPE)
        id_p3 = sp.Popen(["awk", "{print $1}"], stdin=id_p2.stdout, stdout=sp.PIPE)
        job_id = id_p3.communicate()[0].decode("utf-8").strip()

    return x


@mark.task
def cancel(job_name_part):
    import subprocess as sp

    # getting the job_id of the first job that sleeps
    job_id = ""
    while job_id == "":
        time.sleep(1)
        id_p1 = sp.Popen(["squeue"], stdout=sp.PIPE)
        id_p2 = sp.Popen(["grep", job_name_part], stdin=id_p1.stdout, stdout=sp.PIPE)
        id_p3 = sp.Popen(["awk", "{print $1}"], stdin=id_p2.stdout, stdout=sp.PIPE)
        job_id = id_p3.communicate()[0].decode("utf-8").strip()

    # # canceling the job
    proc = sp.run(["scancel", job_id, "--verbose"], stdout=sp.PIPE, stderr=sp.PIPE)
    # cancelling the job returns message in the sterr
    return proc.stderr.decode("utf-8").strip()


@pytest.mark.flaky(reruns=1)
@pytest.mark.skipif(not slurm_available, reason="slurm not installed")
def test_slurm_cancel_rerun_1(tmpdir):
    """ testing that tasks run with slurm is re-queue
        Running wf with 2 tasks, one sleeps and the other trying to get
        job_id of the first task and cancel it.
        The first job should be re-queue and finish without problem.
        (possibly has to be improved, in theory cancel job might finish before cancel)
    """
    wf = Workflow(
        name="wf",
        input_spec=["x", "job_name_cancel", "job_name_resqueue"],
        cache_dir=tmpdir,
    )
    wf.add(sleep(name="sleep1", x=wf.lzin.x, job_name_part=wf.lzin.job_name_cancel))
    wf.add(cancel(name="cancel1", job_name_part=wf.lzin.job_name_resqueue))
    wf.inputs.x = 10
    wf.inputs.job_name_resqueue = "sleep1"
    wf.inputs.job_name_cancel = "cancel1"

    wf.set_output([("out", wf.sleep1.lzout.out), ("canc_out", wf.cancel1.lzout.out)])
    with Submitter("slurm") as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 10
    # checking if indeed the sleep-task job was cancelled by cancel-task
    assert "Terminating" in res.output.canc_out
    assert "Invalid" not in res.output.canc_out
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()


@pytest.mark.flaky(reruns=1)
@pytest.mark.skipif(not slurm_available, reason="slurm not installed")
def test_slurm_cancel_rerun_2(tmpdir):
    """ testing that tasks run with slurm that has --no-requeue
        Running wf with 2 tasks, one sleeps and the other gets
        job_id of the first task and cancel it.
        The first job is not able t be rescheduled and the error is returned.
    """
    wf = Workflow(name="wf", input_spec=["x", "job_name"], cache_dir=tmpdir)
    wf.add(sleep(name="sleep2", x=wf.lzin.x))
    wf.add(cancel(name="cancel2", job_name_part=wf.lzin.job_name))

    wf.inputs.x = 10
    wf.inputs.job_name = "sleep2"

    wf.set_output([("out", wf.sleep2.lzout.out), ("canc_out", wf.cancel2.lzout.out)])
    with pytest.raises(Exception):
        with Submitter("slurm", sbatch_args="--no-requeue") as sub:
            sub(wf)
