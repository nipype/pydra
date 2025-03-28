from dateutil import parser
import secrets
import re
import subprocess as sp
import time
import attrs
import typing as ty
import os
from unittest.mock import patch
import pytest
from pydra.compose import workflow, shell
from fileformats.generic import Directory
from pydra.engine.job import Job
from pydra.engine.submitter import Submitter
from pydra.workers import debug
from pydra.environments import singularity
from pydra.compose import python
from pathlib import Path
from datetime import datetime
from pydra.engine.result import Result
from .utils import (
    need_sge,
    need_slurm,
    need_singularity,
    BasicWorkflow,
    BasicWorkflowWithThreadCount,
    BasicWorkflowWithThreadCountConcurrent,
)
import logging

logger = logging.getLogger("pydra.worker")


@python.define
def SleepAddOne(x):
    time.sleep(1)
    return x + 1


def test_callable_wf(any_worker, tmpdir):
    wf = BasicWorkflow(x=5)
    outputs = wf(cache_dir=tmpdir)
    assert outputs.out == 9
    del wf, outputs

    # providing any_worker
    wf = BasicWorkflow(x=5)
    outputs = wf(worker="cf")
    assert outputs.out == 9
    del wf, outputs

    # providing plugin_kwargs
    wf = BasicWorkflow(x=5)
    outputs = wf(worker="cf", n_procs=2)
    assert outputs.out == 9
    del wf, outputs

    # providing wrong plugin_kwargs
    wf = BasicWorkflow(x=5)
    with pytest.raises(TypeError, match="an unexpected keyword argument"):
        wf(worker="cf", sbatch_args="-N2")

    # providing submitter
    wf = BasicWorkflow(x=5)

    with Submitter(worker=any_worker, cache_dir=tmpdir) as sub:
        res = sub(wf)
    assert res.outputs.out == 9


def test_concurrent_wf(any_worker, tmpdir):
    # concurrent workflow
    # A --> C
    # B --> D
    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x, y):
        taska = workflow.add(SleepAddOne(x=x), name="taska")
        taskb = workflow.add(SleepAddOne(x=y), name="taskb")
        taskc = workflow.add(SleepAddOne(x=taska.out), name="taskc")
        taskd = workflow.add(SleepAddOne(x=taskb.out), name="taskd")
        return taskc.out, taskd.out

    wf = Workflow(x=5, y=10)

    with Submitter(worker=any_worker, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, " ".join(results.errors["error message"])
    outputs = results.outputs
    assert outputs.out1 == 7
    assert outputs.out2 == 12


def test_concurrent_wf_nprocs(tmpdir):
    # concurrent workflow
    # setting n_procs in Submitter that is passed to the worker
    # A --> C
    # B --> D
    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x, y):
        taska = workflow.add(SleepAddOne(x=x), name="taska")
        taskb = workflow.add(SleepAddOne(x=y), name="taskb")
        taskc = workflow.add(SleepAddOne(x=taska.out), name="taskc")
        taskd = workflow.add(SleepAddOne(x=taskb.out), name="taskd")
        return taskc.out, taskd.out

    wf = Workflow(x=5, y=10)
    with Submitter(worker="cf", n_procs=2, cache_dir=tmpdir) as sub:
        res = sub(wf)

    assert not res.errored, " ".join(res.errors["error message"])
    outputs = res.outputs
    assert outputs.out1 == 7
    assert outputs.out2 == 12


def test_wf_in_wf(any_worker, tmpdir):
    """WF(A --> SUBWF(A --> B) --> B)"""

    # workflow task
    @workflow.define
    def SubWf(x):
        sub_a = workflow.add(SleepAddOne(x=x), name="sub_a")
        sub_b = workflow.add(SleepAddOne(x=sub_a.out), name="sub_b")
        return sub_b.out

    @workflow.define
    def WfInWf(x):
        a = workflow.add(SleepAddOne(x=x), name="a")
        subwf = workflow.add(SubWf(x=a.out), name="subwf")
        b = workflow.add(SleepAddOne(x=subwf.out), name="b")
        return b.out

    wf = WfInWf(x=3)

    with Submitter(worker=any_worker, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, " ".join(results.errors["error message"])
    outputs = results.outputs
    assert outputs.out == 7


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf2(any_worker, tmpdir):
    """workflow as a node
    workflow-node with one task and no splitter
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(SleepAddOne(x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    wf = Workflow(x=2)

    with Submitter(worker=any_worker, cache_dir=tmpdir) as sub:
        res = sub(wf)

    assert res.outputs.out == 3


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf_with_state(any_worker, tmpdir):
    @workflow.define
    def Workflow(x):
        taska = workflow.add(SleepAddOne(x=x), name="taska")
        taskb = workflow.add(SleepAddOne(x=taska.out), name="taskb")
        return taskb.out

    wf = Workflow().split(x=[1, 2, 3])

    with Submitter(cache_dir=tmpdir, worker=any_worker) as sub:
        res = sub(wf)

    assert res.outputs.out[0] == 3
    assert res.outputs.out[1] == 4
    assert res.outputs.out[2] == 5


def test_debug_wf():
    # Use serial any_worker to execute workflow instead of CF
    wf = BasicWorkflow(x=5)
    outputs = wf(worker="debug")
    assert outputs.out == 9


@need_slurm
def test_slurm_wf(tmpdir):
    wf = BasicWorkflow(x=1)
    # submit workflow and every task as slurm job
    with Submitter(worker="slurm", cache_dir=tmpdir) as sub:
        res = sub(wf)

    outputs = res.outputs
    assert outputs.out == 5
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()
    # ensure each task was executed with slurm
    assert len([sd for sd in script_dir.listdir() if sd.isdir()]) == 2


@pytest.mark.skip(
    reason=(
        "There currently isn't a way to specify a worker to run a whole workflow within "
        "a single SLURM job"
    )
)
@need_slurm
def test_slurm_wf_cf(tmpdir):
    # submit entire workflow as single job executing with cf worker
    wf = BasicWorkflow(x=1)
    with Submitter(worker="slurm", cache_dir=tmpdir) as sub:
        res = sub(wf)
    outputs = res.outputs
    assert outputs.out == 5
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()
    # ensure only workflow was executed with slurm
    sdirs = [sd for sd in script_dir.listdir() if sd.isdir()]
    assert len(sdirs) == 1
    # slurm scripts should be in the dirs that are using uid in the name
    assert sdirs[0].basename == wf.uid


@need_slurm
def test_slurm_wf_state(tmpdir):
    wf = BasicWorkflow().split(x=[5, 6])
    with Submitter(worker="slurm", cache_dir=tmpdir) as sub:
        res = sub(wf)

    assert res.outputs.out == [9, 10]
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()
    sdirs = [sd for sd in script_dir.listdir() if sd.isdir()]
    assert len(sdirs) == 2 * len(wf.x)


@need_slurm
@pytest.mark.flaky(reruns=3)
def test_slurm_max_jobs(tmp_path):
    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x, y):
        taska = workflow.add(SleepAddOne(x=x), name="taska")
        taskb = workflow.add(SleepAddOne(x=y), name="taskb")
        taskc = workflow.add(SleepAddOne(x=taska.out), name="taskc")
        taskd = workflow.add(SleepAddOne(x=taskb.out), name="taskd")
        return taskc.out, taskd.out

    wf = Workflow(x=5, y=10)

    with Submitter(worker="slurm", cache_dir=tmp_path, max_concurrent=1) as sub:
        res = sub(wf)

    assert not res.errored, " ".join(res.errors["error message"])

    jobids = []
    time.sleep(0.5)  # allow time for sacct to collect itself
    for fl in (tmp_path / "SlurmWorker_scripts").glob("**/slurm-*.out"):
        jid = re.search(r"(?<=slurm-)\d+", str(fl))
        assert jid.group()
        jobids.append(jid.group())
        time.sleep(0.2)
        del jid
        with open(fl, "r") as f:
            print(f.read())

    assert jobids

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
    task = SleepAddOne(x=1)
    # submit workflow and every task as slurm job
    with Submitter(worker="slurm", cache_dir=tmpdir, sbatch_args="-N1") as sub:
        res = sub(task)

    assert res.outputs.out == 2
    script_dir = tmpdir / "SlurmWorker_scripts"
    assert script_dir.exists()


@need_slurm
def test_slurm_args_2(tmpdir):
    """testing sbatch_args provided to the submitter
    exception should be raised for invalid options
    """
    task = SleepAddOne(x=1)
    # submit workflow and every task as slurm job
    with pytest.raises(RuntimeError, match="Error returned from sbatch:"):
        with Submitter(
            worker="slurm", cache_dir=tmpdir, sbatch_args="-N1 --invalid"
        ) as sub:
            sub(task)


@need_singularity
@need_slurm
@pytest.mark.skip(reason="TODO, xfail incorrect")
@pytest.mark.xfail(
    reason="slurm can complain if the number of submitted jobs exceeds the limit"
)
@pytest.mark.parametrize("n", [10, 50, 100])
def test_singularity_st_2(tmp_path, n):
    """splitter over args (checking bigger splitters if slurm available)"""
    args_n = list(range(n))
    image = "docker://alpine"
    Singu = shell.define("echo")
    singu = Singu().split("args", args=args_n)
    with Submitter(
        worker="slurm",
        environment=singularity.Environment(image=image),
        cache_dir=tmp_path,
    ) as sub:
        res = sub(singu)

    assert "1" in res.outputs.stdout[1]
    assert str(n - 1) in res.outputs.stdout[-1]
    assert res.outputs.return_code[0] == res.outputs.return_code[1] == 0


@python.define
def Sleep(x, job_name_part):
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


@python.define
def Cancel(job_name_part):
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


@pytest.mark.skip(reason="this test is hanging, need to work out why")
@pytest.mark.flaky(reruns=1)
@need_slurm
def test_slurm_cancel_rerun_1(tmpdir):
    """testing that tasks run with slurm is re-queue
    Running wf with 2 tasks, one sleeps and the other trying to get
    job_id of the first task and cancel it.
    The first job should be re-queue and finish without problem.
    (possibly has to be improved, in theory cancel job might finish before cancel)
    """

    @workflow.define(outputs=["out", "canc_out"])
    def Workflow(x, job_name_cancel, job_name_resqueue):
        sleep1 = workflow.add(Sleep(x=x, job_name_part=job_name_cancel), name="sleep1")
        cancel1 = workflow.add(Cancel(job_name_part=job_name_resqueue), name="cancel1")
        return sleep1.out, cancel1.out

    wf = Workflow(x=10, job_name_resqueue="sleep1", job_name_cancel="cancel1")

    with Submitter(worker="slurm", cache_dir=tmpdir) as sub:
        res = sub(wf)

    outputs = res.outputs
    assert outputs.out == 10
    # checking if indeed the sleep-task job was cancelled by cancel-task
    assert "Terminating" in outputs.canc_out
    assert "Invalid" not in outputs.canc_out
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

    @workflow.define(outputs=["out", "canc_out"])
    def Workflow(x, job_name):
        sleep2 = workflow.add(Sleep(x=x, job_name_part=job_name), name="sleep2")
        cancel2 = workflow.add(Cancel(job_name_part=job_name), name="cancel2")
        return sleep2.out, cancel2.out

    wf = Workflow(x=10, job_name="sleep2")

    with pytest.raises(Exception):
        with Submitter(
            worker="slurm", cache_dir=tmpdir, sbatch_args="--no-requeue"
        ) as sub:
            sub(wf, raise_errors=True)


@need_sge
def test_sge_wf(tmpdir):
    """testing that a basic workflow can be run with the SGEWorker"""
    wf = BasicWorkflow(x=1)
    # submit workflow and every task as sge job
    with Submitter(worker="sge", cache_dir=tmpdir) as sub:
        res = sub(wf)

    outputs = res.outputs
    assert outputs.out == 9
    script_dir = tmpdir / "SGEWorker_scripts"
    assert script_dir.exists()
    # ensure each task was executed with sge
    assert len([sd for sd in script_dir.listdir() if sd.isdir()]) == 2


@need_sge
def test_sge_wf_cf(tmp_path):
    """testing the SGEWorker can submit SGE tasks while the workflow
    uses the concurrent futures any_worker"""
    # submit entire workflow as single job executing with cf worker
    wf = BasicWorkflow(x=1)
    with Submitter(worker="sge", cache_dir=tmp_path) as sub:
        res = sub(wf)
    outputs = res.outputs
    assert outputs.out == 9
    script_dir = tmp_path / "SGEWorker_scripts"
    assert script_dir.exists()
    # ensure only workflow was executed with slurm
    sdirs = [sd for sd in script_dir.listdir() if sd.isdir()]
    assert len(sdirs) == 1
    # sge scripts should be in the dirs that are using uid in the name
    assert Path(sdirs[0]).name == wf.uid


@need_sge
def test_sge_wf_state(tmpdir):
    """testing the SGEWorker can be used with a workflow with state"""
    wf = BasicWorkflow().split(x=[5, 6])
    with Submitter(worker="sge", cache_dir=tmpdir) as sub:
        res = sub(wf)
    assert res.output.out[0] == 9
    assert res.output.out[1] == 10
    script_dir = tmpdir / "SGEWorker_scripts"
    assert script_dir.exists()
    sdirs = [sd for sd in script_dir.listdir() if sd.isdir()]
    assert len(sdirs) == 2 * len(wf.x)


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
    wf = BasicWorkflowWithThreadCount(x=5)

    jobids = []
    with Submitter(worker="sge", cache_dir=tmpdir) as sub:
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
    wf = BasicWorkflowWithThreadCountConcurrent().split(x=[5, 6])

    jobids = []
    with Submitter(worker="sge", max_threads=8, cache_dir=tmpdir) as sub:
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
    wf = BasicWorkflowWithThreadCountConcurrent().split(x=[5, 6])

    jobids = []
    with Submitter(worker="sge", max_threads=None, cache_dir=tmpdir) as sub:
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
    @python.define
    def output_dir_as_input(out_dir: Directory) -> Directory:
        (out_dir.fspath / "new-file.txt").touch()
        return out_dir

    task = output_dir_as_input(out_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Input field hashes have changed"):
        task(cache_dir=tmp_path)


def test_hash_changes_in_task_inputs_unstable(tmp_path):
    @attrs.define
    class Unstable:
        value: int  # type: ignore

        def __bytes_repr__(self, cache) -> ty.Iterator[bytes]:
            """Random 128-bit bytestring"""
            yield secrets.token_bytes(16)

    @python.define
    def unstable_input(unstable: Unstable) -> int:
        return unstable.value

    task = unstable_input(unstable=Unstable(1))
    with pytest.raises(RuntimeError, match="Input field hashes have changed"):
        task(cache_dir=tmp_path)


def test_hash_changes_in_workflow_inputs(tmp_path):
    @python.define
    def OutputDirAsOutput(out_dir: Path) -> Directory:
        (out_dir / "new-file.txt").touch()
        return out_dir

    @workflow.define(outputs=["out_dir"])
    def Workflow(in_dir: Directory):
        task = workflow.add(OutputDirAsOutput(out_dir=in_dir), name="task")
        return task.out

    in_dir = tmp_path / "in_dir"
    in_dir.mkdir()
    cache_dir = tmp_path / "cache_dir"
    cache_dir.mkdir()

    wf = Workflow(in_dir=in_dir)
    with pytest.raises(RuntimeError, match="Input field hashes have changed.*"):
        wf(cache_dir=cache_dir)


@python.define
def to_tuple(x, y):
    return (x, y)


class BYOAddVarWorker(debug.Worker):
    """A dummy worker that adds 1 to the output of the task"""

    _plugin_name = "byo_add_env_var"

    def __init__(self, add_var, **kwargs):
        super().__init__(**kwargs)
        self.add_var = add_var

    def run(
        self,
        task: "Job",
        rerun: bool = False,
    ) -> "Result":
        with patch.dict(os.environ, {"BYO_ADD_VAR": str(self.add_var)}):
            return super().run(task, rerun)


@python.define
def AddEnvVarTask(x: int) -> int:
    return x + int(os.environ.get("BYO_ADD_VAR", 0))


def test_byo_worker(tmp_path):

    task1 = AddEnvVarTask(x=1)

    with Submitter(worker=BYOAddVarWorker, add_var=10, cache_dir=tmp_path) as sub:
        assert sub.worker.plugin_name() == "byo_add_env_var"
        result = sub(task1)

    assert result.outputs.out == 11

    task2 = AddEnvVarTask(x=2)

    new_cache_dir = tmp_path / "new"

    with Submitter(worker="debug", cache_dir=new_cache_dir) as sub:
        result = sub(task2)

    assert result.outputs.out == 2


def test_bad_builtin_worker():

    with pytest.raises(ValueError, match="No worker matches 'bad-worker'"):
        Submitter(worker="bad-worker")


def test_bad_byo_worker1():

    from pydra.workers import base

    class BadWorker(base.Worker):

        def run(self, task: Job, rerun: bool = False) -> Result:
            pass

    with pytest.raises(ValueError, match="Cannot infer plugin name of Worker "):
        Submitter(worker=BadWorker)


def test_bad_byo_worker2():

    class BadWorker:
        pass

    with pytest.raises(
        TypeError,
        match="Worker must be a Worker object, name of a worker or a Worker class",
    ):
        Submitter(worker=BadWorker)
