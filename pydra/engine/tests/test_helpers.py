from pathlib import Path

import pytest
import cloudpickle as cp

from .utils import multiply
from .. import helpers


def test_save(tmpdir):
    outdir = Path(tmpdir)
    with pytest.raises(ValueError):
        helpers.save(tmpdir)
    foo = multiply(name="mult", x=1, y=2)
    # save task
    helpers.save(outdir, task=foo)
    del foo
    # load saved task
    task_pkl = outdir / "_task.pklz"
    foo = cp.loads(task_pkl.read_bytes())
    assert foo.name == "mult"
    assert foo.inputs.x == 1 and foo.inputs.y == 2
    # execute task and save result
    res = foo()
    assert res.output.out == 2
    helpers.save(outdir, result=res)
    del res
    # load saved result
    res_pkl = outdir / "_result.pklz"
    res = cp.loads(res_pkl.read_bytes())
    assert res.output.out == 2


def test_create_pyscript(tmpdir):
    outdir = Path(tmpdir)
    with pytest.raises(Exception):
        helpers.create_pyscript(outdir, "checksum")
    foo = multiply(name="mult", x=1, y=2)
    helpers.save(outdir, task=foo)
    pyscript = helpers.create_pyscript(outdir, foo.checksum)
    assert pyscript.exists()
