from dateutil import parser
import secrets
import re
import subprocess as sp
import time
import attrs
import typing as ty
from random import randint
import os
from unittest.mock import patch
import pytest
from fileformats.generic import Directory
from .utils import (
    need_sge,
    need_slurm,
    gen_basic_wf,
    gen_basic_wf_with_threadcount,
    gen_basic_wf_with_threadcount_concurrent,
)
from ..core import Workflow, TaskBase
from ..submitter import Submitter
from ..workers import SerialWorker
from ... import mark
from pathlib import Path
from datetime import datetime


@mark.task
def sleep_add_one(x):
    time.sleep(1)
    return x + 1


def test_callable_wf(plugin, tmpdir):
    wf = gen_basic_wf()
    res = wf()
    assert res.output.out == 9
    del wf, res

    # providing plugin
    wf = gen_basic_wf()
    res = wf(plugin="cf")
    assert res.output.out == 9
    del wf, res

    # providing plugin_kwargs
    wf = gen_basic_wf()
    res = wf(plugin="cf", plugin_kwargs={"n_procs": 2})
    assert res.output.out == 9
    del wf, res

    # providing wrong plugin_kwargs
    wf = gen_basic_wf()
    with pytest.raises(TypeError, match="an unexpected keyword argument"):
        wf(plugin="cf", plugin_kwargs={"sbatch_args": "-N2"})

    # providing submitter
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
    """workflow as a node
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

    wf.split("x", x=[1, 2, 3])
    wf.set_output([("out", wf.taskb.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(wf)

    res = wf.result()

    assert res[0].output.out == 3
    assert res[1].output.out == 4
    assert res[2].output.out == 5


def test_serial_wf():
    # Use serial plugin to execute workflow instead of CF
    wf = gen_basic_wf()
    res = wf(plugin="serial")
    assert res.output.out == 9


@need_slurm
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


@need_slurm
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
    # slurm scripts should be in the dirs that are using uid in the name
    assert sdirs[0].basename == wf.uid


@need_slurm
def test_slurm_wf_state(tmpdir):
    wf = gen_basic_wf()
    wf.split("x", x=[5, 6])
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


@need_slurm
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


@need_slurm
def test_slurm_args_1(tmpdir):
    """testing sbatch_args provided to the submitter"""
    task = sleep_add_one(x=1)
    task.cache_dir = tmpdir
    # submit workflow and every task as slurm job
    with Submitter("slurm", sbatch_args="-N1") as sub:
        sub(task)

    res = task.result()
    assert res.output.out == 2
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()


@need_slurm
def test_slurm_args_2(tmpdir):
    """testing sbatch_args provided to the submitter
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
@need_slurm
def test_slurm_cancel_rerun_1(tmpdir):
    """testing that tasks run with slurm is re-queue
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
@need_slurm
def test_slurm_cancel_rerun_2(tmpdir):
    """testing that tasks run with slurm that has --no-requeue
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


@need_sge
def test_sge_wf(tmpdir):
    """testing that a basic workflow can be run with the SGEWorker"""
    wf = gen_basic_wf()
    wf.cache_dir = tmpdir
    # submit workflow and every task as sge job
    with Submitter(
        "sge",
    ) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 9
    script_dir = tmpdir / "SGEWorker_scripts"
    assert script_dir.exists()
    # ensure each task was executed with sge
    assert len([sd for sd in script_dir.listdir() if sd.isdir()]) == 2


@need_sge
def test_sge_wf_cf(tmpdir):
    """testing the SGEWorker can submit SGE tasks while the workflow
    uses the concurrent futures plugin"""
    # submit entire workflow as single job executing with cf worker
    wf = gen_basic_wf()
    wf.cache_dir = tmpdir
    wf.plugin = "cf"
    with Submitter("sge") as sub:
        sub(wf)
    res = wf.result()
    assert res.output.out == 9
    script_dir = tmpdir / "SGEWorker_scripts"
    assert script_dir.exists()
    # ensure only workflow was executed with slurm
    sdirs = [sd for sd in script_dir.listdir() if sd.isdir()]
    assert len(sdirs) == 1
    # sge scripts should be in the dirs that are using uid in the name
    assert Path(sdirs[0]).name == wf.uid


@need_sge
def test_sge_wf_state(tmpdir):
    """testing the SGEWorker can be used with a workflow with state"""
    wf = gen_basic_wf()
    wf.split("x")
    wf.inputs.x = [5, 6]
    wf.cache_dir = tmpdir
    with Submitter("sge") as sub:
        sub(wf)
    res = wf.result()
    assert res[0].output.out == 9
    assert res[1].output.out == 10
    script_dir = tmpdir / "SGEWorker_scripts"
    assert script_dir.exists()
    sdirs = [sd for sd in script_dir.listdir() if sd.isdir()]
    assert len(sdirs) == 2 * len(wf.inputs.x)


def qacct_output_to_dict(qacct_output):
    stdout_dict = {}
    for line in qacct_output.splitlines():
        key_value = line.split(None, 1)
        if key_value[0] not in stdout_dict:
            stdout_dict[key_value[0]] = []
        if len(key_value) > 1:
            stdout_dict[key_value[0]].append(key_value[1])
        else:
            stdout_dict[key_value[0]].append(None)

    print(stdout_dict)
    return stdout_dict


@need_sge
def test_sge_set_threadcount(tmpdir):
    """testing the number of threads for an SGEWorker task can be set
    using the input_spec variable sgeThreads"""
    wf = gen_basic_wf_with_threadcount()
    wf.inputs.x = 5
    wf.cache_dir = tmpdir

    jobids = []
    with Submitter("sge") as sub:
        sub(wf)
        jobids = list(sub.worker.jobid_by_task_uid.values())
        jobids.sort()

    print(f"jobids: {jobids}")

    out_job0 = (
        sp.run(["qacct", "-j", jobids[0]], capture_output=True).stdout.decode().strip()
    )
    out_job1 = (
        sp.run(["qacct", "-j", jobids[1]], capture_output=True).stdout.decode().strip()
    )

    out_job0_dict = qacct_output_to_dict(out_job0)
    out_job1_dict = qacct_output_to_dict(out_job1)

    assert int(out_job0_dict["slots"][0]) == 4
    assert int(out_job1_dict["slots"][0]) == 1


@need_sge
def test_sge_limit_maxthreads(tmpdir):
    """testing the ability to limit the number of threads used by the SGE
    at one time with the max_threads argument to SGEWorker"""
    wf = gen_basic_wf_with_threadcount_concurrent()
    wf.inputs.x = [5, 6]
    wf.split("x")
    wf.cache_dir = tmpdir

    jobids = []
    with Submitter("sge", max_threads=8) as sub:
        sub(wf)
        jobids = list(sub.worker.jobid_by_task_uid.values())
        jobids.sort()

    out_job0 = (
        sp.run(["qacct", "-j", jobids[0]], capture_output=True).stdout.decode().strip()
    )
    out_job1 = (
        sp.run(["qacct", "-j", jobids[1]], capture_output=True).stdout.decode().strip()
    )
    out_job2 = (
        sp.run(["qacct", "-j", jobids[2]], capture_output=True).stdout.decode().strip()
    )
    out_job3 = (
        sp.run(["qacct", "-j", jobids[3]], capture_output=True).stdout.decode().strip()
    )

    qacct_output_to_dict(out_job0)
    out_job1_dict = qacct_output_to_dict(out_job1)
    out_job2_dict = qacct_output_to_dict(out_job2)
    qacct_output_to_dict(out_job3)

    job_1_endtime = datetime.strptime(
        out_job1_dict["end_time"][0], "%a %b %d %H:%M:%S %Y"
    )
    # Running both task_1_1 and task_1_2 at once would exceed max_threads,
    # so task_1_2 waits for task_1_1 to complete
    job_2_starttime = datetime.strptime(
        out_job2_dict["start_time"][0], "%a %b %d %H:%M:%S %Y"
    )
    assert job_1_endtime < job_2_starttime


@need_sge
def test_sge_no_limit_maxthreads(tmpdir):
    """testing unlimited threads can be used at once by SGE
    when max_threads is not set"""
    wf = gen_basic_wf_with_threadcount_concurrent()
    wf.inputs.x = [5, 6]
    wf.split("x")
    wf.cache_dir = tmpdir

    jobids = []
    with Submitter("sge", max_threads=None) as sub:
        sub(wf)
        jobids = list(sub.worker.jobid_by_task_uid.values())
        jobids.sort()

    out_job0 = (
        sp.run(["qacct", "-j", jobids[0]], capture_output=True).stdout.decode().strip()
    )
    out_job1 = (
        sp.run(["qacct", "-j", jobids[1]], capture_output=True).stdout.decode().strip()
    )
    out_job2 = (
        sp.run(["qacct", "-j", jobids[2]], capture_output=True).stdout.decode().strip()
    )

    qacct_output_to_dict(out_job0)
    out_job1_dict = qacct_output_to_dict(out_job1)
    out_job2_dict = qacct_output_to_dict(out_job2)

    job_1_endtime = datetime.strptime(
        out_job1_dict["end_time"][0], "%a %b %d %H:%M:%S %Y"
    )
    # Running both task_1_1 and task_1_2 at once would not exceed max_threads,
    # so task_1_2 does not wait for task_1_1 to complete
    job_2_starttime = datetime.strptime(
        out_job2_dict["start_time"][0], "%a %b %d %H:%M:%S %Y"
    )
    assert job_1_endtime > job_2_starttime


def test_hash_changes_in_task_inputs_file(tmp_path):
    @mark.task
    def output_dir_as_input(out_dir: Directory) -> Directory:
        (out_dir.fspath / "new-file.txt").touch()
        return out_dir

    task = output_dir_as_input(out_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Input field hashes have changed"):
        task()


def test_hash_changes_in_task_inputs_unstable(tmp_path):
    @attrs.define
    class Unstable:
        value: int  # type: ignore

        def __bytes_repr__(self, cache) -> ty.Iterator[bytes]:
            """Random 128-bit bytestring"""
            yield secrets.token_bytes(16)

    @mark.task
    def unstable_input(unstable: Unstable) -> int:
        return unstable.value

    task = unstable_input(unstable=Unstable(1))
    with pytest.raises(RuntimeError, match="Input field hashes have changed"):
        task()


def test_hash_changes_in_workflow_inputs(tmp_path):
    @mark.task
    def output_dir_as_output(out_dir: Path) -> Directory:
        (out_dir / "new-file.txt").touch()
        return out_dir

    wf = Workflow(
        name="test_hash_change", input_spec={"in_dir": Directory}, in_dir=tmp_path
    )
    wf.add(output_dir_as_output(out_dir=wf.lzin.in_dir, name="task"))
    wf.set_output(("out_dir", wf.task.lzout.out))
    with pytest.raises(RuntimeError, match="Input field hashes have changed.*Workflow"):
        wf()


def test_hash_changes_in_workflow_graph(tmpdir):
    class X:
        """Dummy class with unstable hash (i.e. which isn't altered in a node in which
        it is an input)"""

        value = 1

        def __bytes_repr__(self, cache):
            """Bytes representation from class attribute, which will be changed be
            'alter_x" node.

            NB: this is a contrived example where the bytes_repr implementation returns
            a bytes representation of a class attribute in order to trigger the exception,
            hopefully cases like this will be very rare"""
            yield bytes(self.value)

    @mark.task
    @mark.annotate({"return": {"x": X, "y": int}})
    def identity(x: X) -> ty.Tuple[X, int]:
        return x, 99

    @mark.task
    def alter_x(y):
        X.value = 2
        return y

    @mark.task
    def to_tuple(x, y):
        return (x, y)

    wf = Workflow(name="wf_with_blocked_tasks", input_spec=["x", "y"])
    wf.add(identity(name="taska", x=wf.lzin.x))
    wf.add(alter_x(name="taskb", y=wf.taska.lzout.y))
    wf.add(to_tuple(name="taskc", x=wf.taska.lzout.x, y=wf.taskb.lzout.out))
    wf.set_output([("out", wf.taskc.lzout.out)])

    wf.inputs.x = X()

    wf.cache_dir = tmpdir

    with pytest.raises(
        RuntimeError, match="Graph of 'wf_with_blocked_tasks' workflow is not empty"
    ):
        with Submitter("cf") as sub:
            result = sub(wf)


@mark.task
def to_tuple(x, y):
    return (x, y)


class BYOAddVarWorker(SerialWorker):
    """A dummy worker that adds 1 to the output of the task"""

    plugin_name = "byo_add_env_var"

    def __init__(self, add_var, **kwargs):
        super().__init__(**kwargs)
        self.add_var = add_var

    async def exec_serial(self, runnable, rerun=False, environment=None):
        if isinstance(runnable, TaskBase):
            with patch.dict(os.environ, {"BYO_ADD_VAR": str(self.add_var)}):
                result = runnable._run(rerun, environment=environment)
            return result
        else:  # it could be tuple that includes pickle files with tasks and inputs
            return super().exec_serial(runnable, rerun, environment)


@mark.task
def add_env_var_task(x: int) -> int:
    return x + int(os.environ.get("BYO_ADD_VAR", 0))


def test_byo_worker():

    task1 = add_env_var_task(x=1)

    with Submitter(plugin=BYOAddVarWorker, add_var=10) as sub:
        assert sub.plugin == "byo_add_env_var"
        result = task1(submitter=sub)

    assert result.output.out == 11

    task2 = add_env_var_task(x=2)

    with Submitter(plugin="serial") as sub:
        result = task2(submitter=sub)

    assert result.output.out == 2


def test_bad_builtin_worker():

    with pytest.raises(NotImplementedError, match="No worker for 'bad-worker' plugin"):
        Submitter(plugin="bad-worker")


def test_bad_byo_worker():

    class BadWorker:
        pass

    with pytest.raises(
        ValueError, match="Worker class must have a 'plugin_name' str attribute"
    ):
        Submitter(plugin=BadWorker)
