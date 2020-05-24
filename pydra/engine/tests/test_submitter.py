from dateutil import parser
import re
import shutil
import subprocess as sp
import time

import pytest

from .utils import gen_basic_wf
from ..core import Workflow
from ..submitter import Submitter
from ... import mark

slurm_available = bool(shutil.which("sbatch"))


@mark.task
def sleep_add_one(x):
    time.sleep(1)
    return x + 1


def test_callable_wf(plugin):
    wf = gen_basic_wf()
    with pytest.raises(NotImplementedError):
        wf()

    res = wf(plugin="cf")
    assert res.output.out == 9
    del wf, res

    wf = gen_basic_wf()
    sub = Submitter(plugin)
    res = wf(submitter=sub)
    assert res.output.out == 9


def test_concurrent_wf(plugin):
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
    with Submitter(plugin) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out1 == 7
    assert res.output.out2 == 12


def test_concurrent_wf_nprocs():
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
    # wf.plugin = 'cf'
    # res = wf.run()
    with Submitter("cf", n_procs=2) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out1 == 7
    assert res.output.out2 == 12


def test_wf_in_wf(plugin):
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
    wf.add(subwf)

    wf.add(sleep_add_one(name="wf_b", x=wf.sub_wf.lzout.out))
    wf.set_output([("out", wf.wf_b.lzout.out)])

    with Submitter(plugin) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 7


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf2(plugin_dask_opt):
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

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 3


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf_with_state(plugin_dask_opt):
    wf = Workflow(name="wf_with_state", input_spec=["x"])
    wf.add(sleep_add_one(name="taska", x=wf.lzin.x))
    wf.add(sleep_add_one(name="taskb", x=wf.taska.lzout.out))

    wf.inputs.x = [1, 2, 3]
    wf.split("x")
    wf.set_output([("out", wf.taskb.lzout.out)])

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
