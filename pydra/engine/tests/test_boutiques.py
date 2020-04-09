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
def test_boutiques():
    btask = BoshTask(name="NA", zenodo="zenodo.1482743")
    btask.inputs.infile = Path(__file__).resolve().parent / "data_tests" / "test.nii.gz"
    btask.inputs.maskfile = "test_brain.nii.gz"
    btask()
