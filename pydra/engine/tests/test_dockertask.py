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
    """ simple command in a container, a default bindings and working directory is added
        no submitter
    """
    cmd = "whoami"
    docky = DockerTask(name="docky", executable=cmd, image="busybox")
    assert docky.inputs.image == "busybox"
    assert docky.inputs.container == "docker"
    assert (
        docky.cmdline
        == f"docker run -v {docky.output_dir}:/output_pydra:rw -w /output_pydra {docky.inputs.image} {cmd}"
    )

    res = docky()
    assert res.output.stdout == "root\n"
    assert res.output.return_code == 0
    if res.output.stderr:
        assert "Unable to find image" in res.output.stderr


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_1(plugin):
    """ simple command in a container, a default bindings and working directory is added
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
    assert (
        docky.cmdline
        == f"docker run -v {docky.output_dir}:/output_pydra:rw -w /output_pydra {docky.inputs.image} {' '.join(cmd)}"
    )

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
    assert (
        docky.cmdline
        == f"docker run -v {docky.output_dir}:/output_pydra:rw -w /output_pydra {docky.inputs.image} {' '.join(cmd)}"
    )

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
        == f"docker run -v {docky.output_dir}:/output_pydra:rw -w /output_pydra {docky.inputs.image} {cmd_exec} {' '.join(cmd_args)}"
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
        == f"docker run -v {docky.output_dir}:/output_pydra:rw -w /output_pydra {docky.inputs.image} {cmd_exec} {' '.join(cmd_args)}"
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
    assert res[0].output.stdout == "/output_pydra\n"
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
        output_path is automatically added to the bindings
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp.txt")],
        bases=(ShellOutSpec,),
    )
    docky = DockerTask(
        name="docky", image="ubuntu", executable=cmd, output_spec=my_output_spec
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
    filename = str(tmpdir.join("file_pydra.txt"))
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"

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
    filename = str(tmpdir.join("file_pydra.txt"))
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"

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
        input_spec=my_input_spec,
        strip=True,
    )

    res = docky()
    assert res.output.stdout == "hello from pydra"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_2(plugin, tmpdir):
    """ a customized input spec with two fields for docker task """
    filename_1 = tmpdir.join("file_pydra.txt")
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")

    filename_2 = tmpdir.join("file_nice.txt")
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

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
    filename_1 = tmpdir.join("file_pydra.txt")
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmpdir.join("file_nice.txt")
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

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
    filename_1 = tmpdir.join("file_pydra.txt")
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmpdir.join("file_nice.txt")
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

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
        input_spec=my_input_spec,
        strip=True,
    )

    res = docky()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_3(plugin, tmpdir):
    """ input file is in the container, so metadata["container_path"]: True,
        the input will be treated as a str """
    filename = "/proc/1/cgroup"

    cmd = "cat"

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
                        "container_path": True,
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
        input_spec=my_input_spec,
        strip=True,
    )

    res = docky()
    assert "docker" in res.output.stdout


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_3a(plugin, tmpdir):
    """ input file does not exist in the local file system,
        but metadata["container_path"] is not used,
        so exception is raised
    """
    filename = "/_proc/1/cgroup"

    cmd = "cat"

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
        input_spec=my_input_spec,
        strip=True,
    )

    with pytest.raises(Exception) as excinfo:
        res = docky()
    assert "use field.metadata['container_path']=True" in str(excinfo.value)


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_cmd_inputspec_copyfile_1(plugin, tmpdir):
    """ shelltask changes a file in place,
        adding copyfile=True to the file-input from input_spec
        hardlink or copy in the output_dir should be created
    """
    file = tmpdir.join("file_pydra.txt")
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                File,
                dc.field(
                    metadata={
                        "position": 1,
                        "help_string": "orig file",
                        "mandatory": True,
                        "copyfile": True,
                    }
                ),
            ),
            (
                "out_file",
                str,
                dc.field(
                    metadata={
                        "output_file_template": "{orig_file}",
                        "help_string": "output file",
                    }
                ),
            ),
        ],
        bases=(DockerSpec,),
    )

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=cmd,
        input_spec=my_input_spec,
        orig_file=str(file),
    )

    res = docky()
    assert res.output.stdout == ""
    assert res.output.out_file.exists()
    # the file is  copied, and than it is changed in place
    assert res.output.out_file.parent == docky.output_dir
    with open(res.output.out_file, "r") as f:
        assert "hi from pydra\n" == f.read()
    # the original file is unchanged
    with open(file, "r") as f:
        assert "hello from pydra\n" == f.read()


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_inputspec_state_1(plugin, tmpdir):
    """ a customised input spec for a docker file with a splitter,
        splitter is on files
    """
    filename_1 = tmpdir.join("file_pydra.txt")
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmpdir.join("file_nice.txt")
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = [str(filename_1), str(filename_2)]

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
        input_spec=my_input_spec,
        strip=True,
    ).split("file")

    res = docky()
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_wf_inputspec_1(plugin, tmpdir):
    """ a customized input spec for workflow with docker tasks """
    filename = tmpdir.join("file_pydra.txt")
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"

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

    wf = Workflow(name="wf", input_spec=["cmd", "file"])
    wf.inputs.cmd = cmd
    wf.inputs.file = filename

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=wf.lzin.cmd,
        file=wf.lzin.file,
        input_spec=my_input_spec,
        strip=True,
    )
    wf.add(docky)

    wf.set_output([("out", wf.docky.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == "hello from pydra"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_wf_state_inputspec_1(plugin, tmpdir):
    """ a customized input spec for workflow with docker tasks that has a state"""
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

    wf = Workflow(name="wf", input_spec=["cmd", "file"])
    wf.inputs.cmd = cmd
    wf.inputs.file = filename

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=wf.lzin.cmd,
        file=wf.lzin.file,
        input_spec=my_input_spec,
        strip=True,
    )
    wf.add(docky)
    wf.split("file")

    wf.set_output([("out", wf.docky.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res[0].output.out == "hello from pydra"
    assert res[1].output.out == "have a nice one"


@need_docker
@pytest.mark.parametrize("plugin", Plugins)
def test_docker_wf_ndst_inputspec_1(plugin, tmpdir):
    """ a customized input spec for workflow with docker tasks with states"""
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

    wf = Workflow(name="wf", input_spec=["cmd", "file"])
    wf.inputs.cmd = cmd
    wf.inputs.file = filename

    docky = DockerTask(
        name="docky",
        image="busybox",
        executable=wf.lzin.cmd,
        file=wf.lzin.file,
        input_spec=my_input_spec,
        strip=True,
    ).split("file")
    wf.add(docky)

    wf.set_output([("out", wf.docky.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == ["hello from pydra", "have a nice one"]
