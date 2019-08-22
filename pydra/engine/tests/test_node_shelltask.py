# -*- coding: utf-8 -*-

import typing as ty
import os
import pytest

from ..task import ShellCommandTask
from ..submitter import Submitter


@pytest.fixture(scope="module")
def change_dir(request):
    orig_dir = os.getcwd()
    test_dir = os.path.join(orig_dir, "test_outputs")
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    def move2orig():
        os.chdir(orig_dir)

    request.addfinalizer(move2orig)


def test_shell_cmd_1(tmpdir):
    """ simple command, no arguments"""
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly()
    assert res.output.stdout == str(shelly.output_dir) + "\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_1a(tmpdir):
    """ simple command, no arguments"""
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    with Submitter(plugin="cf") as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == str(shelly.output_dir) + "\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_2(tmpdir):
    """ a command with arguments, cmd and args given as executable"""
    cmd = ["echo", "hail", "pydra"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly()
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_2a(tmpdir):
    """ a command with arguments, using executable and args"""
    cmd = ["echo", "hail", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(executable=cmd[0], args=cmd[1:])
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == " ".join(cmd)
    res = shelly()
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.xfail(reason="submitterdoesn't work for shelly")
def test_shell_cmd_3(tmpdir):
    cmd = [["pwd"], ["pwd"]]

    # all args given as executable
    shelly = ShellCommandTask(name="shelly", executable=cmd).split("executable")
    #    assert shelly.cmdline == " ".join(cmd)
    res = shelly()
    # breakpoint()
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"
