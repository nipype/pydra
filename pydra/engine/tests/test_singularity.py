# -*- coding: utf-8 -*-

import os, shutil
import subprocess as sp
import pytest
import attr

from ..task import SingularityTask, DockerTask, ShellCommandTask
from ..submitter import Submitter
from ..core import Workflow
from ..specs import ShellOutSpec, SpecInfo, File, SingularitySpec


need_docker = pytest.mark.skipif(
    shutil.which("docker") is None or sp.call(["docker", "info"]),
    reason="no docker available",
)
need_singularity = pytest.mark.skipif(
    shutil.which("singularity") is None, reason="no singularity available"
)


@need_singularity
def test_singularity_1_nosubm(tmpdir):
    """ simple command in a container, a default bindings and working directory is added
        no submitter
    """
    cmd = "pwd"
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image, cache_dir=tmpdir)
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
def test_singularity_2_nosubm(tmpdir):
    """ a command with arguments, cmd and args given as executable
        no submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image, cache_dir=tmpdir)
    assert (
        singu.cmdline
        == f"singularity exec -B {singu.output_dir}:/output_pydra:rw {image} {' '.join(cmd)}"
    )

    res = singu()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0


@need_singularity
def test_singularity_2(plugin, tmpdir):
    """ a command with arguments, cmd and args given as executable
        using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image, cache_dir=tmpdir)
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
def test_singularity_2_singuflag(plugin, tmpdir):
    """ a command with arguments, cmd and args given as executable
        using ShellComandTask with container_info=("singularity", image)
    """
    cmd = ["echo", "hail", "pydra"]
    image = "library://sylabsed/linux/alpine"
    shingu = ShellCommandTask(
        name="shingu",
        executable=cmd,
        container_info=("singularity", image),
        cache_dir=tmpdir,
    )
    assert (
        shingu.cmdline
        == f"singularity exec -B {shingu.output_dir}:/output_pydra:rw {image} {' '.join(cmd)}"
    )

    with Submitter(plugin=plugin) as sub:
        shingu(submitter=sub)
    res = shingu.result()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0


@need_singularity
def test_singularity_2a(plugin, tmpdir):
    """ a command with arguments, using executable and args
        using submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(
        name="singu", executable=cmd_exec, args=cmd_args, image=image, cache_dir=tmpdir
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
def test_singularity_3(plugin, tmpdir):
    """ a simple command in container with bindings,
        creating directory in tmp dir and checking if it is in the container
    """
    # creating a new directory
    tmpdir.mkdir("new_dir")
    cmd = ["ls", "/tmp_dir"]
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singu", executable=cmd, image=image, cache_dir=tmpdir)
    # binding tmp directory to the container
    singu.inputs.bindings = [(str(tmpdir), "/tmp_dir", "ro")]

    with Submitter(plugin=plugin) as sub:
        singu(submitter=sub)

    res = singu.result()
    assert "new_dir\n" in res.output.stdout
    assert res.output.return_code == 0


@need_singularity
def test_singularity_3_singuflag(plugin, tmpdir):
    """ a simple command in container with bindings,
        creating directory in tmp dir and checking if it is in the container
        using ShellComandTask with container_info=("singularity", image)
    """
    # creating a new directory
    tmpdir.mkdir("new_dir")
    cmd = ["ls", "/tmp_dir"]
    image = "library://sylabsed/linux/alpine"
    shingu = SingularityTask(
        name="singu",
        executable=cmd,
        container_info=("singularity", image),
        cache_dir=tmpdir,
    )
    # binding tmp directory to the container
    shingu.inputs.bindings = [(str(tmpdir), "/tmp_dir", "ro")]

    with Submitter(plugin=plugin) as sub:
        shingu(submitter=sub)

    res = shingu.result()
    assert "new_dir\n" in res.output.stdout
    assert res.output.return_code == 0


@need_singularity
def test_singularity_3_singuflagbind(plugin, tmpdir):
    """ a simple command in container with bindings,
        creating directory in tmp dir and checking if it is in the container
        using ShellComandTask with container_info=("singularity", image, bindings)
    """
    # creating a new directory
    tmpdir.mkdir("new_dir")
    cmd = ["ls", "/tmp_dir"]
    image = "library://sylabsed/linux/alpine"
    shingu = SingularityTask(
        name="singu",
        executable=cmd,
        container_info=("singularity", image, [(str(tmpdir), "/tmp_dir", "ro")]),
        cache_dir=tmpdir,
    )

    with Submitter(plugin=plugin) as sub:
        shingu(submitter=sub)

    res = shingu.result()
    assert "new_dir\n" in res.output.stdout
    assert res.output.return_code == 0


# tests with State


@need_singularity
def test_singularity_st_1(plugin, tmpdir):
    """ commands without arguments in container
        splitter = executable
    """
    cmd = ["pwd", "ls"]
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(
        name="singu", executable=cmd, image=image, cache_dir=tmpdir
    ).split("executable")
    assert singu.state.splitter == "singu.executable"

    res = singu(plugin=plugin)
    assert "SingularityTask" in res[0].output.stdout
    assert res[1].output.stdout == ""
    assert res[0].output.return_code == res[1].output.return_code == 0


@need_singularity
def test_singularity_st_2(plugin, tmpdir):
    """ command with arguments in docker, checking the distribution
        splitter = image
    """
    cmd = ["cat", "/etc/issue"]
    image = ["library://sylabsed/linux/alpine", "library://sylabsed/examples/lolcow"]
    singu = SingularityTask(
        name="singu", executable=cmd, image=image, cache_dir=tmpdir
    ).split("image")
    assert singu.state.splitter == "singu.image"

    res = singu(plugin=plugin)
    assert "Alpine" in res[0].output.stdout
    assert "Ubuntu" in res[1].output.stdout
    assert res[0].output.return_code == res[1].output.return_code == 0


@need_singularity
def test_singularity_st_3(plugin, tmpdir):
    """ outer splitter image and executable
    """
    cmd = ["pwd", ["cat", "/etc/issue"]]
    image = ["library://sylabsed/linux/alpine", "library://sylabsed/examples/lolcow"]
    singu = SingularityTask(
        name="singu", executable=cmd, image=image, cache_dir=tmpdir
    ).split(["image", "executable"])
    assert singu.state.splitter == ["singu.image", "singu.executable"]
    res = singu(plugin=plugin)

    assert "SingularityTask" in res[0].output.stdout
    assert "Alpine" in res[1].output.stdout
    assert "SingularityTask" in res[2].output.stdout
    assert "Ubuntu" in res[3].output.stdout


@need_singularity
def test_wf_singularity_1(plugin, tmpdir):
    """ a workflow with two connected task
        the first one read the file that is bounded to the container,
        the second uses echo
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")

    image = "library://sylabsed/linux/alpine"
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2"], cache_dir=tmpdir)
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
def test_wf_singularity_1a(plugin, tmpdir):
    """ a workflow with two connected task - using both containers: Docker and Singul.
        the first one read the file that is bounded to the container,
        the second uses echo
    """
    with open(tmpdir.join("file_pydra.txt"), "w") as f:
        f.write("hello from pydra")

    image_sing = "library://sylabsed/linux/alpine"
    image_doc = "ubuntu"
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2"], cache_dir=tmpdir)
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
            name="singu_echo",
            image=image_doc,
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


# tests with customized output_spec


@need_singularity
def test_singularity_outputspec_1(plugin, tmpdir):
    """
        customised output_spec, adding files to the output, providing specific pathname
        output_path is automatically added to the bindings
    """
    cmd = ["touch", "newfile_tmp.txt"]
    image = "library://sylabsed/linux/alpine"

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp.txt")],
        bases=(ShellOutSpec,),
    )
    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        output_spec=my_output_spec,
        cache_dir=tmpdir,
    )

    with Submitter(plugin=plugin) as sub:
        singu(submitter=sub)

    res = singu.result()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


# tests with customised input_spec


@need_singularity
def test_singularity_inputspec_1(plugin, tmpdir):
    """ a simple customized input spec for singularity task """
    filename = str(tmpdir.join("file_pydra.txt"))
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    },
                ),
            )
        ],
        bases=(SingularitySpec,),
    )

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        file=filename,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmpdir,
    )

    res = singu()
    assert res.output.stdout == "hello from pydra"


@need_singularity
def test_singularity_inputspec_1a(plugin, tmpdir):
    """ a simple customized input spec for singularity task
        a default value is used
    """
    filename = str(tmpdir.join("file_pydra.txt"))
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    default=filename,
                    metadata={"position": 1, "help_string": "input file"},
                ),
            )
        ],
        bases=(SingularitySpec,),
    )

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmpdir,
    )

    res = singu()
    assert res.output.stdout == "hello from pydra"


@need_singularity
def test_singularity_inputspec_2(plugin, tmpdir):
    """ a customized input spec with two fields for singularity task """
    filename_1 = tmpdir.join("file_pydra.txt")
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")

    filename_2 = tmpdir.join("file_nice.txt")
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                attr.ib(
                    type=File, metadata={"position": 1, "help_string": "input file 1"}
                ),
            ),
            (
                "file2",
                attr.ib(
                    type=File,
                    default=filename_2,
                    metadata={"position": 2, "help_string": "input file 2"},
                ),
            ),
        ],
        bases=(SingularitySpec,),
    )

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        file1=filename_1,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmpdir,
    )

    res = singu()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_inputspec_2a_except(plugin, tmpdir):
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
    image = "library://sylabsed/linux/alpine"

    # the field with default value can't be before value without default
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                attr.ib(
                    type=File,
                    default=filename_1,
                    metadata={"position": 1, "help_string": "input file 1"},
                ),
            ),
            (
                "file2",
                attr.ib(
                    type=File, metadata={"position": 2, "help_string": "input file 2"}
                ),
            ),
        ],
        bases=(SingularitySpec,),
    )

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        file2=filename_2,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmpdir,
    )
    res = singu()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_inputspec_2a(plugin, tmpdir):
    """ a customized input spec with two fields
        first one uses a default value,
        this is fine even if the second field is not using any defaults
    """
    filename_1 = tmpdir.join("file_pydra.txt")
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmpdir.join("file_nice.txt")
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    image = "library://sylabsed/linux/alpine"

    # if you want set default in the first field you can use default_value in metadata
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                attr.ib(
                    type=File,
                    default=filename_1,
                    metadata={"position": 1, "help_string": "input file 1"},
                ),
            ),
            (
                "file2",
                attr.ib(
                    type=File, metadata={"position": 2, "help_string": "input file 2"}
                ),
            ),
        ],
        bases=(SingularitySpec,),
    )

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        file2=filename_2,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmpdir,
    )

    res = singu()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_cmd_inputspec_copyfile_1(plugin, tmpdir):
    """ shelltask changes a file in place,
        adding copyfile=True to the file-input from input_spec
        hardlink or copy in the output_dir should be created
    """
    file = tmpdir.join("file_pydra.txt")
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "help_string": "orig file",
                        "mandatory": True,
                        "copyfile": True,
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{orig_file}",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(SingularitySpec,),
    )

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        input_spec=my_input_spec,
        orig_file=str(file),
        cache_dir=tmpdir,
    )

    res = singu()
    assert res.output.stdout == ""
    assert res.output.out_file.exists()
    # the file is  copied, and than it is changed in place
    assert res.output.out_file.parent == singu.output_dir
    with open(res.output.out_file, "r") as f:
        assert "hi from pydra\n" == f.read()
    # the original file is unchanged
    with open(file, "r") as f:
        assert "hello from pydra\n" == f.read()


@need_singularity
def test_singularity_inputspec_state_1(plugin, tmpdir):
    """ a customised input spec for a singularity file with a splitter,
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
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    },
                ),
            )
        ],
        bases=(SingularitySpec,),
    )

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        file=filename,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmpdir,
    ).split("file")

    res = singu()
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@need_singularity
def test_singularity_inputspec_state_1b(plugin, tmpdir):
    """ a customised input spec for a singularity file with a splitter,
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
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    },
                ),
            )
        ],
        bases=(SingularitySpec,),
    )

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=cmd,
        file=filename,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmpdir,
    ).split("file")

    res = singu()
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@need_singularity
def test_singularity_wf_inputspec_1(plugin, tmpdir):
    """ a customized input spec for workflow with singularity tasks """
    filename = tmpdir.join("file_pydra.txt")
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    },
                ),
            )
        ],
        bases=(SingularitySpec,),
    )

    wf = Workflow(name="wf", input_spec=["cmd", "file"], cache_dir=tmpdir)
    wf.inputs.cmd = cmd
    wf.inputs.file = filename

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=wf.lzin.cmd,
        file=wf.lzin.file,
        input_spec=my_input_spec,
        strip=True,
    )
    wf.add(singu)

    wf.set_output([("out", wf.singu.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == "hello from pydra"


@need_singularity
def test_singularity_wf_state_inputspec_1(plugin, tmpdir):
    """ a customized input spec for workflow with singularity tasks that has a state"""
    file_1 = tmpdir.join("file_pydra.txt")
    file_2 = tmpdir.join("file_nice.txt")
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = [str(file_1), str(file_2)]
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    },
                ),
            )
        ],
        bases=(SingularitySpec,),
    )

    wf = Workflow(name="wf", input_spec=["cmd", "file"], cache_dir=tmpdir)
    wf.inputs.cmd = cmd
    wf.inputs.file = filename

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=wf.lzin.cmd,
        file=wf.lzin.file,
        input_spec=my_input_spec,
        strip=True,
    )
    wf.add(singu)
    wf.split("file")

    wf.set_output([("out", wf.singu.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res[0].output.out == "hello from pydra"
    assert res[1].output.out == "have a nice one"


@need_singularity
def test_singularity_wf_ndst_inputspec_1(plugin, tmpdir):
    """ a customized input spec for workflow with singularity tasks with states"""
    file_1 = tmpdir.join("file_pydra.txt")
    file_2 = tmpdir.join("file_nice.txt")
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = [str(file_1), str(file_2)]
    image = "library://sylabsed/linux/alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "help_string": "input file",
                    },
                ),
            )
        ],
        bases=(SingularitySpec,),
    )

    wf = Workflow(name="wf", input_spec=["cmd", "file"], cache_dir=tmpdir)
    wf.inputs.cmd = cmd
    wf.inputs.file = filename

    singu = SingularityTask(
        name="singu",
        image=image,
        executable=wf.lzin.cmd,
        file=wf.lzin.file,
        input_spec=my_input_spec,
        strip=True,
    ).split("file")
    wf.add(singu)

    wf.set_output([("out", wf.singu.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == ["hello from pydra", "have a nice one"]
