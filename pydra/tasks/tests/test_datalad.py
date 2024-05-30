import typing as ty
from pathlib import Path
import os, sys
import attr
import pytest


from ...tasks.datalad import DataladInterface
from ...engine.core import Workflow
from ...engine.submitter import Submitter
from ...engine.helpers import hash_value
from ...engine.tests.utils import need_gitannex


@need_gitannex
def test_datalad_interface(tmpdir):
    """
    Testing datalad interface
    """
    import datalad.api as dl

    # change PosixPath to str
    tmpdir = str(tmpdir)
    # creating a dataset
    ds = dl.Dataset(tmpdir).create()
    ds.save()
    ds_path = ds.pathobj

    # creating a file to download
    file_path = ds_path / "file.txt"
    file_path.write_text("test")
    ds.save()

    tmpdir = Path(tmpdir)

    # install the dataset to a new location
    ds2 = dl.install(source=tmpdir, path=tmpdir / "ds2")
    ds2_path = ds2.pathobj

    # use datalad interface to download the file
    dl_interface = DataladInterface(
        name="dl_interface", in_file="file.txt", dataset_path=ds2_path
    )
    # running the task
    res = dl_interface()

    assert os.path.exists(res.output.out_file)
    assert os.path.basename(res.output.out_file) == "file.txt"


# Path: pydra/tasks/tests/test_datalad.py
# Compare this snippet from pydra/tasks/datalad.py:
