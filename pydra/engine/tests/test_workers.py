import shutil

import pytest

from .utils import add2, fun_div, gen_basic_wf
from ..submitter import Submitter

# list of (plugin, available)
plugins = {"slurm": bool(shutil.which("sbatch"))}


@pytest.mark.skipif(not plugins["slurm"], reason="slurm not installed")
def test_slurm_wf(tmpdir):
    wf = gen_basic_wf()
    wf.cache_dir = tmpdir
    # submit workflow and every task as slurm job
    wf(plugin="slurm")
    assert wf.result()


@pytest.mark.skipif(not plugins["slurm"], reason="slurm not installed")
def test_slurm_wf_cf(tmpdir):
    # submit entire workflow as single job executing with cf worker
    wf2 = gen_basic_wf()
    wf2.plugin = "cf"
    with Submitter("slurm") as sub:
        sub(wf2)
    assert wf2.result()
