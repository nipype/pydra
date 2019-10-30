# -*- coding: utf-8 -*-

import os, shutil
import pytest
from pathlib import Path
import dataclasses as dc

from ..task import DockerTask
from ..submitter import Submitter
from ..core import Workflow
from ..specs import ShellOutSpec, SpecInfo, File, DockerSpec

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
            bindings=[(str(tmpdir), "/outputs"), (scripts_dir, "/scripts", "ro")],
            strip=True,
        )
    )
    wf.add(
        DockerTask(
            name="load",
            image="python:3.7-alpine",
            executable=wf.lzin.cmd2,
            args=wf.save.lzout.stdout,
            bindings=[(str(tmpdir), "/outputs"), (scripts_dir, "/scripts", "ro")],
            strip=True,
        )
    )
    wf.set_output([("out", wf.load.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == "Hello!"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_outputspec_1(plugin, tmpdir):
    """
        customised output_spec, adding files to the output, providing specific pathname
        path in bindings is a relative path to output_dir
        (len(bind_tuple)==3, so True is added as default to the fourth place)
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
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_outputspec_1a(plugin, tmpdir):
    """
        customised output_spec, adding files to the output, providing specific pathname
        path in bindings is a relative path to output_dir (True in a bind_tuple[3])
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
        bindings=[(".", "/outputs", None, True)],
        executable=cmd,
        output_spec=my_output_spec,
    )

    with Submitter(plugin=plugin) as sub:
        docky(submitter=sub)

    res = docky.result()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_outputspec_2(plugin, tmpdir):
    """
        customised output_spec, adding files to the output, providing specific pathname
        path in bindings is an absolut path (bind_tuple[3]==False)
    """
    cmd = ["touch", "/outputs/newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, Path(tmpdir) / "newfile_tmp.txt")],
        bases=(ShellOutSpec,),
    )
    docky = DockerTask(
        name="docky",
        image="ubuntu",
        bindings=[(tmpdir, "/outputs", None, False)],
        executable=cmd,
        output_spec=my_output_spec,
    )

    with Submitter(plugin=plugin) as sub:
        docky(submitter=sub)

    res = docky.result()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


# tests with customised input_spec


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_1(plugin, tmpdir):
    """ a simple customized input spec for docker task """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    filename = "/tmp_dir/file_pydra.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                File,
                dc.field(
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    }
                ),
            )
        ],
        bases=(DockerSpec,),
    )

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=cmd,
        file=filename,
        bindings=[(str(tmpdir), "/tmp_dir", "ro")],
        input_spec=my_input_spec,
        strip=True,
    )

    res = docky()

    assert res.output.stdout == "hello from pydra"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_1a(plugin, tmpdir):
    """ a simple customized input spec for docker task
        a default value is used
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    filename = "/tmp_dir/file_pydra.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                File,
                dc.field(
                    default=filename,
                    metadata={"position": 1, "help_string": "input file"},
                ),
            )
        ],
        bases=(DockerSpec,),
    )

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=cmd,
        bindings=[(str(tmpdir), "/tmp_dir", "ro")],
        input_spec=my_input_spec,
        strip=True,
    )

    res = docky()

    assert res.output.stdout == "hello from pydra"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_2(plugin, tmpdir):
    """ a customized input spec with two fields for docker task """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra\n")

    with open(tmpdir.join("file_nice.txt"), "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename_1 = "/tmp_dir/file_pydra.txt"
    filename_2 = "/tmp_dir/file_nice.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                File,
                dc.field(metadata={"position": 1, "help_string": "input file 1"}),
            ),
            (
                "file2",
                File,
                dc.field(
                    default=filename_2,
                    metadata={"position": 2, "help_string": "input file 2"},
                ),
            ),
        ],
        bases=(DockerSpec,),
    )

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=cmd,
        file1=filename_1,
        bindings=[(str(tmpdir), "/tmp_dir", "ro")],
        input_spec=my_input_spec,
        strip=True,
    )

    res = docky()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_2a_except(plugin, tmpdir):
    """ a customized input spec with two fields
        first one uses a default, and second doesn't - raises a dataclass exception
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra\n")

    with open(tmpdir.join("file_nice.txt"), "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename_1 = "/tmp_dir/file_pydra.txt"
    filename_2 = "/tmp_dir/file_nice.txt"

    # the field with default value can't be before value without default
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                File,
                dc.field(
                    default=filename_1,
                    metadata={"position": 1, "help_string": "input file 1"},
                ),
            ),
            (
                "file2",
                File,
                dc.field(metadata={"position": 2, "help_string": "input file 2"}),
            ),
        ],
        bases=(DockerSpec,),
    )

    with pytest.raises(TypeError) as excinfo:
        docky = DockerTask(
            name="docky",
            image="busybox",
            executable=cmd,
            file2=filename_2,
            bindings=[(str(tmpdir), "/tmp_dir", "ro")],
            input_spec=my_input_spec,
            strip=True,
        )
    assert "non-default argument 'file2' follows default argument" == str(excinfo.value)


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_2a(plugin, tmpdir):
    """ a customized input spec with two fields
        first one uses a default by using metadata['default_value'],
        this is fine even if the second field is not using any defaults
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra\n")

    with open(tmpdir.join("file_nice.txt"), "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename_1 = "/tmp_dir/file_pydra.txt"
    filename_2 = "/tmp_dir/file_nice.txt"

    # if you want set default in the first field you can use default_value in metadata
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                File,
                dc.field(
                    metadata={
                        "default_value": filename_1,
                        "position": 1,
                        "help_string": "input file 1",
                    }
                ),
            ),
            (
                "file2",
                File,
                dc.field(metadata={"position": 2, "help_string": "input file 2"}),
            ),
        ],
        bases=(DockerSpec,),
    )

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=cmd,
        file2=filename_2,
        bindings=[(str(tmpdir), "/tmp_dir", "ro")],
        input_spec=my_input_spec,
        strip=True,
    )

    res = docky()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@pytest.mark.xfail(
    reason="the file hash is not calculated, "
    "since the files can't be found (container path provided)"
    "so at the end both states give the same cheksum"
    "and the second item is not run..."
)
@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_state_1(plugin, tmpdir):
    """ a customised input spec for a docker file with a splitter,
        splitter is on files - causes issues in a docker task (see xfail reason) TODO
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")
    with open(tmpdir.join("file_nice.txt"), "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = ["/tmp_dir/file_pydra.txt", "/tmp_dir/file_nice.txt"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                File,
                dc.field(
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    }
                ),
            )
        ],
        bases=(DockerSpec,),
    )

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=cmd,
        file=filename,
        bindings=[(str(tmpdir), "/tmp_dir", "ro")],
        input_spec=my_input_spec,
        strip=True,
    ).split("file")

    res = docky()
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_state_1a_tmp(plugin, tmpdir):
    """ a customised input spec for a docker file with a splitter,
        files from the input spec represented as string to avoid problems...
        this is probably just a temporary solution
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")
    with open(tmpdir.join("file_nice.txt"), "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = ["/tmp_dir/file_pydra.txt", "/tmp_dir/file_nice.txt"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                str,
                dc.field(
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    }
                ),
            )
        ],
        bases=(DockerSpec,),
    )

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=cmd,
        file=filename,
        bindings=[(str(tmpdir), "/tmp_dir", "ro")],
        input_spec=my_input_spec,
        strip=True,
    ).split("file")

    res = docky()
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_state_1b(plugin, tmpdir):
    """ a customised input spec for a docker file with a splitter,
        files from the input spec have the same path in the local os and the container,
        so hash is calculated and the test works fine
    """
    file_1 = tmpdir.join("file_pydra.txt")
    file_2 = tmpdir.join("file_nice.txt")
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = [str(file_1), str(file_2)]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                File,
                dc.field(
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    }
                ),
            )
        ],
        bases=(DockerSpec,),
    )

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=cmd,
        file=filename,
        # recreates the same directory
        bindings=[(str(tmpdir), str(tmpdir), "ro")],
        input_spec=my_input_spec,
        strip=True,
    ).split("file")

    res = docky()
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"
