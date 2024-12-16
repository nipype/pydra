import shutil
import subprocess as sp
import attr
import pytest
from pydra.engine.helpers import attrs_values
from .utils import result_no_submitter, result_submitter, no_win
from pydra.design import workflow, boutiques, shell

need_bosh_docker = pytest.mark.skipif(
    shutil.which("docker") is None
    or sp.call(["docker", "info"])
    or sp.call(["which", "bosh"]),
    reason="requires docker and bosh",
)

pytestmark = pytest.mark.skip()


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)  # need for travis
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_boutiques_1(maskfile, plugin, results_function, tmpdir, data_tests_dir):
    """simple task to run fsl.bet using BoshTask"""
    btask = boutiques.define(zenodo_id="1482743")
    btask.infile = data_tests_dir / "test.nii.gz"
    btask.maskfile = maskfile
    res = btask(plugin, cache_dir=tmpdir)

    assert res.output.return_code == 0

    # checking if the outfile exists and if it has a proper name
    assert res.output.outfile.name == "test_brain.nii.gz"
    assert res.output.outfile.exists()
    # files that do not exist were set to NOTHING
    assert res.output.out_outskin_off == attr.NOTHING


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_spec_1(data_tests_dir):
    """testing spec: providing input/output fields names"""
    btask = boutiques.define(
        zenodo_id="1482743",
        input_spec_names=["infile", "maskfile"],
        output_spec_names=["outfile", "out_outskin_off"],
    )(
        infile=data_tests_dir / "test.nii.gz",
        maskfile="test_brain.nii.gz",
    )

    assert len(btask.input_spec.fields) == 2
    assert btask.input_spec.fields[0][0] == "infile"
    assert btask.input_spec.fields[1][0] == "maskfile"
    assert hasattr(btask.spec, "infile")
    assert hasattr(btask.spec, "maskfile")

    assert len(btask.output_spec.fields) == 2
    assert btask.output_spec.fields[0][0] == "outfile"
    assert btask.output_spec.fields[1][0] == "out_outskin_off"


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
def test_boutiques_spec_2(data_tests_dir):
    """testing spec: providing partial input/output fields names"""
    btask = boutiques.define(
        zenodo_id="1482743", input_spec_names=["infile"], output_spec_names=[]
    )(
        infile=data_tests_dir / "test.nii.gz",
        maskfile="test_brain.nii.gz",
    )

    fields = attrs_values(btask)
    assert len(fields) == 1
    assert fields[0][0] == "infile"
    assert hasattr(btask.spec, "infile")
    # input doesn't see maskfile
    assert not hasattr(btask.spec, "maskfile")

    assert len(btask.output_spec.fields) == 0


@no_win
@need_bosh_docker
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
def test_boutiques_wf_1(maskfile, plugin, tmpdir, infile):
    """wf with one task that runs fsl.bet using BoshTask"""

    def Workflow(maskfile, infile):
        bet = workflow.add(
            boutiques.define(zenodo_id="1482743")(
                infile=infile,
                maskfile=maskfile,
            )
        )

        return bet.outfile

    wf = Workflow(maskfile=maskfile, infile=infile)
    wf(plugin=plugin, cache_dir=tmpdir)

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
def test_boutiques_wf_2(maskfile, plugin, tmpdir, infile):
    """wf with two BoshTasks (fsl.bet and fsl.stats) and one ShellTask"""

    @workflow.define(outputs=["outfile_bet", "out_stat", "out"])
    def Workflow(maskfile, infile):

        bet = workflow.add(
            boutiques.define(zenodo_id="1482743")(
                infile=infile,
                maskfile=maskfile,
            )
        )
        # used to be "3240521", but can't access anymore
        stat = workflow.add(
            boutiques.define(zenodo_id="4472771")(
                input_file=bet.outfile,
                v=True,
            )
        )
        cat = workflow.add(shell.define("cat <file>")(file=stat.output))
        return bet.outfile, stat.output, cat.stdout

    res = Workflow(maskfile=maskfile, infile=infile)(plugin=plugin, cache_dir=tmpdir)
    assert res.output.outfile_bet.name == "test_brain.nii.gz"
    assert res.output.outfile_bet.exists()

    assert res.output.out_stat.name == "output.txt"
    assert res.output.out_stat.exists()

    assert int(res.output.out.rstrip().split()[0]) == 11534336
    assert float(res.output.out.rstrip().split()[1]) == 11534336.0
