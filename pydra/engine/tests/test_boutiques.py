import os, shutil
import subprocess as sp
from pathlib import Path
import attr
import pytest

from ..core import Workflow
from ..task import ShellCommandTask
from ..submitter import Submitter
from ..boutiques import BoshTask
from .utils import result_no_submitter, result_submitter, no_win
from ...engine.specs import File

need_bosh_docker = pytest.mark.skipif(
    shutil.which("docker") is None
    or sp.call(["docker", "info"])
    or sp.call(["which", "bosh"]),
    reason="requires docker and bosh",
)

Infile = Path(__file__).resolve().parent / "data_tests" / "test.nii.gz"

pytestmark = pytest.mark.skip()


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)  # need for travis
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_boutiques_1(maskfile, plugin, results_function, tmpdir):
    """simple task to run fsl.bet using BoshTask"""
    btask = BoshTask(name="NA", zenodo_id="1482743")
    btask.inputs.infile = Infile
    btask.inputs.maskfile = maskfile
    btask.cache_dir = tmpdir
    res = results_function(btask, plugin)

    assert res.output.return_code == 0

    # checking if the outfile exists and if it has a proper name
    assert res.output.outfile.name == "test_brain.nii.gz"
    assert res.output.outfile.exists()
    # files that do not exist were set to NOTHING
    assert res.output.out_outskin_off == attr.NOTHING


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_spec_1():
    """testing spec: providing input/output fields names"""
    btask = BoshTask(
        name="NA",
        zenodo_id="1482743",
        infile=Infile,
        maskfile="test_brain.nii.gz",
        input_spec_names=["infile", "maskfile"],
        output_spec_names=["outfile", "out_outskin_off"],
    )

    assert len(btask.input_spec.fields) == 2
    assert btask.input_spec.fields[0][0] == "infile"
    assert btask.input_spec.fields[1][0] == "maskfile"
    assert hasattr(btask.inputs, "infile")
    assert hasattr(btask.inputs, "maskfile")

    assert len(btask.output_spec.fields) == 2
    assert btask.output_spec.fields[0][0] == "outfile"
    assert btask.output_spec.fields[1][0] == "out_outskin_off"


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_spec_2():
    """testing spec: providing partial input/output fields names"""
    btask = BoshTask(
        name="NA",
        zenodo_id="1482743",
        infile=Infile,
        maskfile="test_brain.nii.gz",
        input_spec_names=["infile"],
        output_spec_names=[],
    )

    assert len(btask.input_spec.fields) == 1
    assert btask.input_spec.fields[0][0] == "infile"
    assert hasattr(btask.inputs, "infile")
    # input doesn't see maskfile
    assert not hasattr(btask.inputs, "maskfile")

    assert len(btask.output_spec.fields) == 0


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
def test_boutiques_wf_1(maskfile, plugin, tmpdir):
    """wf with one task that runs fsl.bet using BoshTask"""
    wf = Workflow(name="wf", input_spec=["maskfile", "infile"])
    wf.inputs.maskfile = maskfile
    wf.inputs.infile = Infile
    wf.cache_dir = tmpdir

    wf.add(
        BoshTask(
            name="bet",
            zenodo_id="1482743",
            infile=wf.lzin.infile,
            maskfile=wf.lzin.maskfile,
        )
    )

    wf.set_output([("outfile", wf.bet.lzout.outfile)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.outfile.name == "test_brain.nii.gz"
    assert res.output.outfile.exists()


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
@pytest.mark.xfail(reason="issues with bosh for 4472771")
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
def test_boutiques_wf_2(maskfile, plugin, tmpdir):
    """wf with two BoshTasks (fsl.bet and fsl.stats) and one ShellTask"""
    wf = Workflow(name="wf", input_spec=["maskfile", "infile"])
    wf.inputs.maskfile = maskfile
    wf.inputs.infile = Infile
    wf.cache_dir = tmpdir

    wf.add(
        BoshTask(
            name="bet",
            zenodo_id="1482743",
            infile=wf.lzin.infile,
            maskfile=wf.lzin.maskfile,
        )
    )
    # used to be "3240521", but can't access anymore
    wf.add(
        BoshTask(
            name="stat", zenodo_id="4472771", input_file=wf.bet.lzout.outfile, v=True
        )
    )
    wf.add(ShellCommandTask(name="cat", executable="cat", args=wf.stat.lzout.output))

    wf.set_output(
        [
            ("outfile_bet", wf.bet.lzout.outfile),
            ("out_stat", wf.stat.lzout.output),
            ("out", wf.cat.lzout.stdout),
        ]
    )

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.outfile_bet.name == "test_brain.nii.gz"
    assert res.output.outfile_bet.exists()

    assert res.output.out_stat.name == "output.txt"
    assert res.output.out_stat.exists()

    assert int(res.output.out.rstrip().split()[0]) == 11534336
    assert float(res.output.out.rstrip().split()[1]) == 11534336.0
