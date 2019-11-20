# -*- coding: utf-8 -*-

import os, shutil
import pytest
from pathlib import Path
import dataclasses as dc

from ..task import SingularityTask, DockerTask
from ..submitter import Submitter
from ..core import Workflow
from ..specs import ShellOutSpec, SpecInfo, File, SingularitySpec

if bool(shutil.which("sbatch")):
    Plugins = ["cf", "slurm"]
else:
    Plugins = ["cf"]


need_docker = pytest.mark.skipif(
    shutil.which("docker") is None, reason="no docker available"
)
need_singularity = pytest.mark.skipif(
    shutil.which("singularity") is None, reason="no singularity available"
)


@need_singularity
def test_singularity_1_nosubm():
    """ simple command in a container, a default bindings and working directory is added
        no submitter
    """
    cmd = "pwd"
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image)
    assert singu.inputs.image == "library://sylabsed/linux/alpine"
    assert singu.inputs.container == "singularity"
    assert (
        singu.cmdline
        == f"singularity exec -B {singu.output_dir}:/output_pydra:rw {image} {cmd}"
    )

    res = singu()
    assert "SingularityTask" in res.output.stdout
    assert res.output.return_code == 0


@need_singularity
def test_singularity_2_nosubm():
    """ a command with arguments, cmd and args given as executable
        no submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image)
    assert (
        singu.cmdline
        == f"singularity exec -B {singu.output_dir}:/output_pydra:rw {image} {' '.join(cmd)}"
    )

    res = singu()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0


@need_singularity
@pytest.mark.parametrize("plugin", Plugins)
def test_singularity_2(plugin):
    """ a command with arguments, cmd and args given as executable
        using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image)
    assert (
        singu.cmdline
        == f"singularity exec -B {singu.output_dir}:/output_pydra:rw {image} {' '.join(cmd)}"
    )

    with Submitter(plugin=plugin) as sub:
        singu(submitter=sub)
    res = singu.result()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0


@need_singularity
@pytest.mark.parametrize("plugin", Plugins)
def test_singularity_2a(plugin):
    """ a command with arguments, using executable and args
        using submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(
        name="singu", executable=cmd_exec, args=cmd_args, image=image
    )
    assert (
        singu.cmdline
        == f"singularity exec -B {singu.output_dir}:/output_pydra:rw {image} {cmd_exec} {' '.join(cmd_args)}"
    )

    with Submitter(plugin=plugin) as sub:
        singu(submitter=sub)
    res = singu.result()
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0


@need_singularity
@pytest.mark.parametrize("plugin", Plugins)
def test_singularity_3(plugin, tmpdir):
    """ a simple command in container with bindings,
        creating directory in tmp dir and checking if it is in the container
    """
    # creating a new directory
    tmpdir.mkdir("new_dir")
    cmd = ["ls", "/tmp_dir"]
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image)
    # binding tmp directory to the container
    singu.inputs.bindings = [(str(tmpdir), "/tmp_dir", "ro")]

    with Submitter(plugin=plugin) as sub:
        singu(submitter=sub)

    res = singu.result()
    assert res.output.stdout == "new_dir\n"
    assert res.output.return_code == 0


# tests with State


@need_singularity
@pytest.mark.parametrize("plugin", Plugins)
def test_singularity_st_1(plugin):
    """ commands without arguments in container
        splitter = executable
    """
    cmd = ["pwd", "ls"]
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image).split(
        "executable"
    )
    assert singu.state.splitter == "singu.executable"

    res = singu(plugin=plugin)
    assert "SingularityTask" in res[0].output.stdout
    assert res[1].output.stdout == ""
    assert res[0].output.return_code == res[1].output.return_code == 0


@need_singularity
@pytest.mark.parametrize("plugin", Plugins)
def test_singularity_st_2(plugin):
    """ command with arguments in docker, checking the distribution
        splitter = image
    """
    cmd = ["cat", "/etc/issue"]
    image = ["library://sylabsed/linux/alpine", "library://sylabsed/examples/lolcow"]
    singu = SingularityTask(name="singu", executable=cmd, image=image).split("image")
    assert singu.state.splitter == "singu.image"

    res = singu(plugin=plugin)
    assert "Alpine" in res[0].output.stdout
    assert "Ubuntu" in res[1].output.stdout
    assert res[0].output.return_code == res[1].output.return_code == 0


@need_singularity
@pytest.mark.parametrize("plugin", Plugins)
def test_singularity_st_3(plugin):
    """ outer splitter image and executable
    """
    cmd = ["pwd", ["cat", "/etc/issue"]]
    image = ["library://sylabsed/linux/alpine", "library://sylabsed/examples/lolcow"]
    singu = SingularityTask(name="singu", executable=cmd, image=image).split(
        ["image", "executable"]
    )
    assert singu.state.splitter == ["singu.image", "singu.executable"]
    res = singu(plugin=plugin)

    assert "SingularityTask" in res[0].output.stdout
    assert "Alpine" in res[1].output.stdout
    assert "SingularityTask" in res[2].output.stdout
    assert "Ubuntu" in res[3].output.stdout


@need_singularity
@pytest.mark.parametrize("plugin", Plugins)
def test_wf_singularity_1(plugin, tmpdir):
    """ a workflow with two connected task
        the first one read the file that is bounded to the container,
        the second uses echo
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")

    image = "library://sylabsed/linux/alpine"
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2"])
    wf.inputs.cmd1 = ["cat", "/tmp_dir/file_pydra.txt"]
    wf.inputs.cmd2 = ["echo", "message from the previous task:"]
    wf.add(
        SingularityTask(
            name="singu_cat",
            image=image,
            executable=wf.lzin.cmd1,
            bindings=[(str(tmpdir), "/tmp_dir", "ro")],
            strip=True,
        )
    )
    wf.add(
        SingularityTask(
            name="singu_echo",
            image=image,
            executable=wf.lzin.cmd2,
            args=wf.singu_cat.lzout.stdout,
            strip=True,
        )
    )
    wf.set_output([("out", wf.singu_echo.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == "message from the previous task: hello from pydra"


@need_docker
@need_singularity
@pytest.mark.parametrize("plugin", Plugins)
def test_wf_singularity_1a(plugin, tmpdir):
    """ a workflow with two connected task - using both containers: Docker and Singul.
        the first one read the file that is bounded to the container,
        the second uses echo
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")

    image_sing = "library://sylabsed/linux/alpine"
    image_doc = "ubuntu"
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2"])
    wf.inputs.cmd1 = ["cat", "/tmp_dir/file_pydra.txt"]
    wf.inputs.cmd2 = ["echo", "message from the previous task:"]
    wf.add(
        SingularityTask(
            name="singu_cat",
            image=image_sing,
            executable=wf.lzin.cmd1,
            bindings=[(str(tmpdir), "/tmp_dir", "ro")],
            strip=True,
        )
    )
    wf.add(
        DockerTask(
            name="docky_echo",
            image=image_doc,
            executable=wf.lzin.cmd2,
            args=wf.singu_cat.lzout.stdout,
            strip=True,
        )
    )
    wf.set_output([("out", wf.docky_echo.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == "message from the previous task: hello from pydra"
