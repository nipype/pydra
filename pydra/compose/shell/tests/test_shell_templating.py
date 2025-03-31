import typing as ty
from pydra.compose import shell
from pydra.compose.shell.templating import argstr_formatting
from pathlib import Path
from unittest.mock import Mock
from pydra.compose.shell.templating import template_update_single
import os
import shutil
from pathlib import Path
import random
import platform
import typing as ty
import pytest
import cloudpickle as cp
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job
from pydra.compose import workflow
from fileformats.generic import Directory, File
from pydra.engine.tests.utils import Multiply, RaiseXeq1
from pydra.utils.general import position_sort
from pydra.compose.shell.templating import parse_format_string
from pydra.engine.job import save, load_and_run
from pydra.workers.cf import get_available_cpus
from pydra.utils.hash import hash_function


@pytest.mark.parametrize(
    "pos_args",
    [
        [(2, "b"), (1, "a"), (3, "c")],
        [(-2, "b"), (1, "a"), (-1, "c")],
        [(None, "b"), (1, "a"), (-1, "c")],
        [(-3, "b"), (None, "a"), (-1, "c")],
        [(None, "b"), (1, "a"), (None, "c")],
    ],
)
def test_position_sort(pos_args):
    final_args = position_sort(pos_args)
    assert final_args == ["a", "b", "c"]


def test_parse_format_string1():
    assert parse_format_string("{a}") == {"a"}


def test_parse_format_string2():
    assert parse_format_string("{abc}") == {"abc"}


def test_parse_format_string3():
    assert parse_format_string("{a:{b}}") == {"a", "b"}


def test_parse_format_string4():
    assert parse_format_string("{a:{b[2]}}") == {"a", "b"}


def test_parse_format_string5():
    assert parse_format_string("{a.xyz[somekey].abc:{b[a][b].d[0]}}") == {"a", "b"}


def test_parse_format_string6():
    assert parse_format_string("{a:05{b[a 2][b].e}}") == {"a", "b"}


def test_parse_format_string7():
    assert parse_format_string(
        "{a1_field} {b2_field:02f} -test {c3_field[c]} -me {d4_field[0]}"
    ) == {"a1_field", "b2_field", "c3_field", "d4_field"}


def test_argstr_formatting():
    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        a1_field: str
        b2_field: float
        c3_field: ty.Dict[str, str]
        d4_field: ty.List[str] = shell.arg(sep=" ")
        executable = "dummy"

        class Outputs(shell.Outputs):
            pass

    values = dict(a1_field="1", b2_field=2.0, c3_field={"c": "3"}, d4_field=["4"])
    assert (
        argstr_formatting(
            "{a1_field} {b2_field:02f} -test {c3_field[c]} -me {d4_field[0]}",
            values,
        )
        == "1 2.000000 -test 3 -me 4"
    )


def test_template_formatting(tmp_path: Path):
    field = Mock()
    field.name = "grad"
    field.argstr = "--grad"
    field.path_template = ("{in_file}.bvec", "{in_file}.bval")
    field.keep_extension = False
    task = Mock()
    values = {"in_file": Path("/a/b/c/file.txt"), "grad": True}

    assert template_update_single(
        field,
        task,
        values=values,
        output_dir=tmp_path,
        spec_type="input",
    ) == [tmp_path / "file.bvec", tmp_path / "file.bval"]
