import typing as ty
import os, sys
import attr
import pytest


"""
Functions to test datalad interface
"""


def test_datalad_interface(tmpdir):
    """
    Testing datalad interface
    """
    import datalad.api as dl
    from pydra.tasks.datalad import DataladInterface
    from pydra.engine.core import Workflow
    from pydra.engine.submitter import Submitter
    from pydra.engine.helpers import hash_value

    # change PosixPath to str
    tmpdir = str(tmpdir)
    # creating a dataset
    ds = dl.create(tmpdir)
    ds.save()
    ds_path = ds.pathobj

    # creating a file to download
    file_path = ds_path / "file.txt"
    file_path.write_text("test")
    ds.save()

    # creating a workflow
    wf = Workflow(name="wf", input_spec=["dataset_path", "dataset_url", "in_file"])
    wf.inputs.dataset_path = ds_path
    wf.inputs.dataset_url = ""
    wf.inputs.in_file = "file.txt"

    # adding datalad task
    wf.add(
        DataladInterface(
            name="dl",
            in_file=wf.lzin.in_file,
            dataset_path=wf.lzin.dataset_path,
            dataset_url=wf.lzin.dataset_url,
        )
    )

    # running the workflow
    with Submitter(plugin="cf") as sub:
        sub(wf)

    # checking if the file was downloaded
    assert wf.result().output.out_file.exists()


# Path: pydra/tasks/tests/test_datalad.py
# Compare this snippet from pydra/tasks/datalad.py:
