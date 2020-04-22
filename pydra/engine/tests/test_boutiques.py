import os, shutil
import subprocess as sp
from pathlib import Path
import pytest

from ..boutiques import BoshTask

need_bosh_docker = pytest.mark.skipif(
    shutil.which("docker") is None
    or sp.call(["docker", "info"] or sp.call(["bosh", "version"])),
    reason="requires docker and bosh",
)


@need_bosh_docker
@pytest.mark.parametrize(
    "maskfile", ["test_brain.nii.gz", "test_brain", "test_brain.nii"]
)
def test_boutiques_1(maskfile):
    btask = BoshTask(name="NA", zenodo="zenodo.1482743")
    btask.inputs.infile = Path(__file__).resolve().parent / "data_tests" / "test.nii.gz"
    btask.inputs.maskfile = maskfile
    res = btask()

    assert res.output.return_code == 0

    # checking if the outfile exists and if it has proper name
    assert res.output.outfile.name == "test_brain.nii.gz"
    assert res.output.outfile.exists()
    # other files should also have proper names, but they do not exist
    assert res.output.out_outskin_off.name == "test_brain_outskin_mesh.off"
    assert not res.output.out_outskin_off.exists()
