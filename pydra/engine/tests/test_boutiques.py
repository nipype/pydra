import os, shutil
import subprocess as sp
from pathlib import Path
import pytest

from ..core import Workflow
from ..task import ShellCommandTask
from ..submitter import Submitter
from ..boutiques import BoshTask
from .utils import result_no_submitter, result_submitter, no_win

need_bosh_docker = pytest.mark.skipif(
    shutil.which("docker") is None
    or sp.call(["docker", "info"] or sp.call(["bosh", "version"])),
    reason="requires docker and bosh",
)

if bool(shutil.which("sbatch")):
    Plugins = ["cf", "slurm"]
else:
    Plugins = ["cf"]

Infile = Path(__file__).resolve().parent / "data_tests" / "test.nii.gz"


@no_win
@need_bosh_docker
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
@pytest.mark.parametrize("plugin", Plugins)
def test_boutiques_1(maskfile, plugin, results_function):
    """ simple task to run fsl.bet using BoshTask"""
    btask = BoshTask(name="NA", zenodo_id="1482743")
    btask.inputs.infile = Infile
    btask.inputs.maskfile = maskfile
    res = results_function(btask, plugin)

    assert res.output.return_code == 0

    # checking if the outfile exists and if it has proper name
    assert res.output.outfile.name == "test_brain.nii.gz"
    assert res.output.outfile.exists()
    # other files should also have proper names, but they do not exist
    assert res.output.out_outskin_off.name == "test_brain_outskin_mesh.off"
    assert not res.output.out_outskin_off.exists()


@no_win
@need_bosh_docker
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
@pytest.mark.parametrize("plugin", Plugins)
def test_boutiques_wf_1(maskfile, plugin):
    """ wf with one task that runs fsl.bet using BoshTask"""
    wf = Workflow(name="wf", input_spec=["maskfile", "infile"])
    wf.inputs.maskfile = maskfile
    wf.inputs.infile = Infile

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
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
@pytest.mark.parametrize("plugin", Plugins)
def test_boutiques_wf_2(maskfile, plugin):
    """ wf with two BoshTasks (fsl.bet and fsl.stats) and one ShellTask"""
    wf = Workflow(name="wf", input_spec=["maskfile", "infile"])
    wf.inputs.maskfile = maskfile
    wf.inputs.infile = Infile

    wf.add(
        BoshTask(
            name="bet",
            zenodo_id="1482743",
            infile=wf.lzin.infile,
            maskfile=wf.lzin.maskfile,
        )
    )
    wf.add(
        BoshTask(
            name="stat", zenodo_id="3240521", input_file=wf.bet.lzout.outfile, v=True
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
