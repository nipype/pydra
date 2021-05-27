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
def test_boutiques_prefix(plugin, tmpdir):
    """simple task to run fsl.bet using BoshTask, but using the 'zenodo' prefix for the zenodo_id"""
    btask = BoshTask(name="NA", zenodo_id="zenodo.1482743")
    btask.inputs.infile = Infile
    btask.inputs.maskfile = "test_brain.nii.gz"
    btask.cache_dir = tmpdir
    res = result_submitter(btask, plugin)

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


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_load_bosh_file():
    cpath = Path(__file__).parent.absolute()
    test_file = cpath / "aux_files/boutiques_specs_test_1.json"
    btask = BoshTask(name="touch", bosh_file=str(test_file))
    assert btask.bosh_file == test_file


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_load_bosh_file_path():
    cpath = Path(__file__).parent.absolute()
    test_file = cpath / "aux_files/boutiques_specs_test_1.json"
    btask = BoshTask(name="touch", bosh_file=test_file)
    assert btask.bosh_file == test_file


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_specs_1(plugin, tmpdir):
    """Tests if mandatory metadata gets set and if default value is used"""
    cpath = Path(__file__).parent.absolute()
    test_file = cpath / "aux_files/boutiques_specs_test_1.json"
    btask = BoshTask(name="touch", bosh_file=test_file)
    assert btask.bosh_file == Path(test_file)

    btask()
    assert (
        btask.input_spec.fields[0][3]["mandatory"] == False
    )  # if default value exists it must be false
    assert btask.input_spec.fields[0][1] == str
    assert btask.output_spec.fields[0][1].metadata["mandatory"] == True
    assert btask.result().output.created_file.name == "outy.txt"


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_specs_absolute(plugin, tmpdir):
    cpath = Path(__file__).parent.absolute()
    test_file = cpath / "aux_files/boutiques_specs_test_1_absolute.json"
    out_file = Path(tmpdir) / "outy.txt"
    btask = BoshTask(
        name="touch",
        bosh_file=test_file,
        bindings=[[str(tmpdir), str(tmpdir)]],
        file=str(out_file),
    )
    assert btask.bosh_file == Path(test_file)

    btask()
    # No default value in this test so metadata is the 3rd element.
    assert btask.input_spec.fields[0][2]["mandatory"] == True
    assert btask.input_spec.fields[0][1] == str
    assert btask.output_spec.fields[0][1].metadata["mandatory"] == True
    assert btask.output_spec.fields[0][1].metadata["absolute_path"] == True
    assert btask.result().output.created_file == out_file


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_specs_extension(plugin, tmpdir):
    """Tests if mandatory metadata gets set and if default value is used"""
    cpath = Path(__file__).parent.absolute()
    test_file = cpath / "aux_files/boutiques_specs_test_1_extension.json"
    btask = BoshTask(name="touch", bosh_file=test_file)
    assert btask.bosh_file == Path(test_file)

    btask()
    assert (
        btask.input_spec.fields[0][3]["mandatory"] == False
    )  # if default value exists it must be false
    assert btask.input_spec.fields[0][1] == str
    assert btask.output_spec.fields[0][1].metadata["mandatory"] == False
    assert btask.result().output.created_file.name == "outy.tt"


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_specs_extension_wf(plugin, tmpdir):
    """Tests if mandatory metadata gets set and if default value is used"""
    cpath = Path(__file__).parent.absolute()
    test_file = cpath / "aux_files/boutiques_specs_test_1_extension.json"
    wf = Workflow(name="wf", input_spec=["name1", "name2"])
    wf.inputs.name2 = "outy.tt"
    wf.split("name1", name1=["outy.tt.txt.gz.nii.txt", "outy.txt.tt.gz"])
    btask = BoshTask(
        name="btask", bosh_file=test_file, file=wf.lzin.name1, file_hidden=wf.lzin.name2
    )
    wf.add(btask)
    wf.set_output([("out", wf.btask.lzout.created_file)])
    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)
    res = wf.result()
    for r in res:
        assert r.output.out.name == "outy.tt"


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_specs_2(plugin, tmpdir):
    """also tests for if no output-files are provided"""
    cpath = Path(__file__).parent.absolute()
    test_file = cpath / "aux_files/boutiques_specs_test_2.json"
    btask = BoshTask(name="touch", bosh_file=test_file, file="test")
    assert btask.bosh_file == Path(test_file)

    btask()
    assert btask.result().output.stdout.find("echo -ftttest") != -1


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_specs_2(plugin, tmpdir):
    """also tests for if no output-files are provided"""
    cpath = Path(__file__).parent.absolute()
    test_file = cpath / "aux_files/boutiques_specs_test_int.json"
    btask = BoshTask(name="touch", bosh_file=test_file, file=1)
    assert btask.bosh_file == Path(test_file)
    assert btask.input_spec.fields[0][1] == int
    btask()
