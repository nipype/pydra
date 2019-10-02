# -*- coding: utf-8 -*-

import os, shutil
import pytest


from ..task import DockerTask
from ..submitter import Submitter
from ..core import Workflow
from ..specs import ShellOutSpec, SpecInfo, File

if bool(shutil.which("sbatch")):
    Plugins = ["cf", "slurm"]
else:
    Plugins = ["cf"]

need_docker = pytest.mark.skipif(
    shutil.which("docker") is None, reason="no docker within the container"
)


@need_docker
def test_docker_1_nosubm():
    """ simple command in a container, no arguments
        no submitter
    """
    cmd = "whoami"
    docky = DockerTask(name="docky", executable=cmd, image="busybox")
    assert docky.inputs.image == "busybox"
    assert docky.inputs.container == "docker"
    assert docky.cmdline == f"docker run {docky.inputs.image} {cmd}"

    res = docky()
    assert res.output.stdout == "root\n"
    assert res.output.return_code == 0
    if res.output.stderr:
        assert "Unable to find image" in res.output.stderr


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_1(plugin):
    """ simple command in a container, no arguments
        using submitter
    """
    cmd = "whoami"
    docky = DockerTask(name="docky", executable=cmd, image="busybox")

    with Submitter(plugin=plugin) as sub:
        docky(submitter=sub)

    res = docky.result()
    assert res.output.stdout == "root\n"
    assert res.output.return_code == 0
    if res.output.stderr:
        assert "Unable to find image" in res.output.stderr


@need_docker
def test_docker_2_nosubm():
    """ a command with arguments, cmd and args given as executable
        no submitter
    """
    cmd = ["echo", "hail", "pydra"]
    docky = DockerTask(name="docky", executable=cmd, image="busybox")
    assert docky.cmdline == f"docker run {docky.inputs.image} {' '.join(cmd)}"

    res = docky()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0
    if res.output.stderr:
        assert "Unable to find image" in res.output.stderr


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_2(plugin):
    """ a command with arguments, cmd and args given as executable
        using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    docky = DockerTask(name="docky", executable=cmd, image="busybox")
    assert docky.cmdline == f"docker run {docky.inputs.image} {' '.join(cmd)}"

    with Submitter(plugin=plugin) as sub:
        docky(submitter=sub)
    res = docky.result()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0
    if res.output.stderr:
        assert "Unable to find image" in res.output.stderr


@need_docker
def test_docker_2a_nosubm():
    """ a command with arguments, using executable and args
        no submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    docky = DockerTask(
        name="docky", executable=cmd_exec, args=cmd_args, image="busybox"
    )
    assert docky.inputs.executable == "echo"
    assert (
        docky.cmdline
        == f"docker run {docky.inputs.image} {cmd_exec} {' '.join(cmd_args)}"
    )

    res = docky()
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0
    if res.output.stderr:
        assert "Unable to find image" in res.output.stderr


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_2a(plugin):
    """ a command with arguments, using executable and args
        using submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    docky = DockerTask(
        name="docky", executable=cmd_exec, args=cmd_args, image="busybox"
    )
    assert docky.inputs.executable == "echo"
    assert (
        docky.cmdline
        == f"docker run {docky.inputs.image} {cmd_exec} {' '.join(cmd_args)}"
    )

    with Submitter(plugin=plugin) as sub:
        docky(submitter=sub)
    res = docky.result()
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0
    if res.output.stderr:
        assert "Unable to find image" in res.output.stderr


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_3(plugin, tmpdir):
    """ a simple command in container with bindings,
        creating directory in tmp dir and checking if it is in the container
    """
    # creating a new directory
    tmpdir.mkdir("new_dir")
    cmd = ["ls", "/tmp_dir"]
    docky = DockerTask(name="docky", executable=cmd, image="busybox")
    # binding tmp directory to the container
    docky.inputs.bindings = [(str(tmpdir), "/tmp_dir", "ro")]

    with Submitter(plugin=plugin) as sub:
        docky(submitter=sub)

    res = docky.result()
    assert res.output.stdout == "new_dir\n"
    assert res.output.return_code == 0
    if res.output.stderr:
        assert "Unable to find image" in res.output.stderr


# tests with State


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_st_1(plugin):
    """ commands without arguments in container
        splitter = executable
    """
    cmd = ["pwd", "whoami"]
    docky = DockerTask(name="docky", executable=cmd, image="busybox").split(
        "executable"
    )
    assert docky.state.splitter == "docky.executable"

    res = docky(plugin=plugin)
    assert res[0].output.stdout == "/\n"
    assert res[1].output.stdout == "root\n"
    assert res[0].output.return_code == res[1].output.return_code == 0


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_st_2(plugin):
    """ command with arguments in docker, checking the distribution
        splitter = image
    """
    cmd = ["cat", "/etc/issue"]
    docky = DockerTask(name="docky", executable=cmd, image=["debian", "ubuntu"]).split(
        "image"
    )
    assert docky.state.splitter == "docky.image"

    res = docky(plugin=plugin)
    assert "Debian" in res[0].output.stdout
    assert "Ubuntu" in res[1].output.stdout
    assert res[0].output.return_code == res[1].output.return_code == 0


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_st_3(plugin):
    """ outer splitter image and executable
    """
    cmd = ["whoami", ["cat", "/etc/issue"]]
    docky = DockerTask(name="docky", executable=cmd, image=["debian", "ubuntu"]).split(
        ["image", "executable"]
    )
    assert docky.state.splitter == ["docky.image", "docky.executable"]
    res = docky(plugin=plugin)

    assert res[0].output.stdout == "root\n"
    assert "Debian" in res[1].output.stdout
    assert res[2].output.stdout == "root\n"
    assert "Ubuntu" in res[3].output.stdout


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_st_4(plugin):
    """ outer splitter image and executable, combining with images
    """
    cmd = ["whoami", ["cat", "/etc/issue"]]
    docky = (
        DockerTask(name="docky", executable=cmd, image=["debian", "ubuntu"])
        .split(["image", "executable"])
        .combine("image")
    )
    assert docky.state.splitter == ["docky.image", "docky.executable"]
    assert docky.state.combiner == ["docky.image"]
    assert docky.state.splitter_final == "docky.executable"

    res = docky(plugin=plugin)

    # checking the first command
    res_cmd1 = res[0]
    assert res_cmd1[0].output.stdout == "root\n"
    assert res_cmd1[1].output.stdout == "root\n"

    # checking the second command
    res_cmd2 = res[1]
    assert "Debian" in res_cmd2[0].output.stdout
    assert "Ubuntu" in res_cmd2[1].output.stdout


# tests with workflows


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_wf_docker_1(plugin, tmpdir):
    """ a workflow with two connected task
        the first one read the file that is bounded to the container,
        the second uses echo
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")

    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2"])
    wf.inputs.cmd1 = ["cat", "/tmp_dir/file_pydra.txt"]
    wf.inputs.cmd2 = ["echo", "message from the previous task:"]
    wf.add(
        DockerTask(
            name="docky_cat",
            image="busybox",
            executable=wf.lzin.cmd1,
            bindings=[(str(tmpdir), "/tmp_dir", "ro")],
            strip=True,
        )
    )
    wf.add(
        DockerTask(
            name="docky_echo",
            image="ubuntu",
            executable=wf.lzin.cmd2,
            args=wf.docky_cat.lzout.stdout,
            strip=True,
        )
    )
    wf.set_output([("out", wf.docky_echo.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == "message from the previous task: hello from pydra"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_wf_docker_2(plugin, tmpdir):
    """ a workflow with two connected task that run python scripts
        the first one creates a text file and the second one reads the file
    """

    scripts_dir = os.path.join(os.path.dirname(__file__), "data_tests")

    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2"])
    wf.inputs.cmd1 = ["python", "/scripts/saving.py", "-f", "/outputs/tmp.txt"]
    wf.inputs.cmd2 = ["python", "/scripts/loading.py", "-f"]
    wf.add(
        DockerTask(
            name="save",
            image="python:3.7-alpine",
            executable=wf.lzin.cmd1,
            bindings=[(str(tmpdir), "/outputs", None), (scripts_dir, "/scripts", "ro")],
            strip=True,
        )
    )
    wf.add(
        DockerTask(
            name="load",
            image="python:3.7-alpine",
            executable=wf.lzin.cmd2,
            args=wf.save.lzout.stdout,
            bindings=[(str(tmpdir), "/outputs", None), (scripts_dir, "/scripts", "ro")],
            strip=True,
        )
    )
    wf.set_output([("out", wf.load.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == "Hello!"


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_1(plugin, tmpdir):
    """
        customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "/outputs/newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp.txt")],
        bases=(ShellOutSpec,),
    )
    docky = DockerTask(
        name="docky",
        image="ubuntu",
        bindings=[(".", "/outputs", None)],
        executable=cmd,
        output_spec=my_output_spec,
    )

    with Submitter(plugin=plugin) as sub:
        docky(submitter=sub)

    res = docky.result()
    # assert res.output.stdout == ""
    assert res.output.newfile.exists()
