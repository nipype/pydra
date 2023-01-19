from ..environments import Native, Docker
from ..task import ShellCommandTask
from ..submitter import Submitter
from ..specs import (
    ShellSpec,
    SpecInfo,
    File,
)

import attr


def test_native_1(tmpdir):
    """simple command, no arguments"""
    cmd = ["whoami"]
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=tmpdir.mkdir("shelly")
    )
    assert shelly.cmdline == " ".join(cmd)

    env_res = Native().execute(shelly)
    shelly()
    assert env_res == shelly.output_

    shelly_call = ShellCommandTask(
        name="shelly_call", executable=cmd, cache_dir=tmpdir.mkdir("shelly_call")
    )
    shelly_call(environment=Native())
    assert env_res == shelly_call.output_

    shelly_subm = ShellCommandTask(
        name="shelly_subm", executable=cmd, cache_dir=tmpdir.mkdir("shelly_subm")
    )
    with Submitter(plugin="cf") as sub:
        shelly_subm(submitter=sub, environment=Native())
    assert env_res == shelly_subm.result().output.__dict__

    # TODO: should be removed at the end
    shelly_old = ShellCommandTask(
        name="shelly_old", executable=cmd, cache_dir=tmpdir, environment="old"
    )
    shelly_old()
    assert env_res == shelly_old.result().output.__dict__


def test_docker_1(tmpdir):
    """docker env: simple command, no arguments"""
    cmd = ["whoami"]
    docker = Docker(image="busybox")
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=tmpdir.mkdir("shelly")
    )
    assert shelly.cmdline == " ".join(cmd)
    env_res = docker.execute(shelly)

    shelly_env = ShellCommandTask(
        name="shelly",
        executable=cmd,
        cache_dir=tmpdir.mkdir("shelly_env"),
        environment=docker,
    )
    shelly_env()
    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=tmpdir.mkdir("shelly_call")
    )
    shelly_call(environment=docker)
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


def test_docker_1_subm(tmpdir, plugin):
    """docker env with submitter: simple command, no arguments"""
    cmd = ["whoami"]
    docker = Docker(image="busybox")
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=tmpdir.mkdir("shelly")
    )
    assert shelly.cmdline == " ".join(cmd)
    env_res = docker.execute(shelly)

    shelly_env = ShellCommandTask(
        name="shelly",
        executable=cmd,
        cache_dir=tmpdir.mkdir("shelly_env"),
        environment=docker,
    )
    with Submitter(plugin=plugin) as sub:
        shelly_env(submitter=sub)
    assert env_res == shelly_env.result().output.__dict__

    shelly_call = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=tmpdir.mkdir("shelly_call")
    )
    with Submitter(plugin=plugin) as sub:
        shelly_call(submitter=sub, environment=docker)
    assert env_res == shelly_call.result().output.__dict__


def create_shelly_inputfile(tempdir, filename, name, executable):
    """creating a task with a simple input_spec"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "help_string": "files",
                        "mandatory": True,
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name=name,
        executable=executable,
        cache_dir=tempdir.mkdir(name),
        input_spec=my_input_spec,
        file=filename,
    )
    return shelly


def test_shell_fileinp(tmpdir):
    """task with a file in the command/input"""
    input_dir = tmpdir.mkdir("inputs")
    filename = input_dir.join("file.txt")
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly", executable=["cat"]
    )
    env_res = Native().execute(shelly)

    shelly_env = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = Native()
    shelly_env()
    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_call", executable=["cat"]
    )
    shelly_call(environment=Native())
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


def test_shell_fileinp_st(tmpdir):
    """task (with a splitter) with a file in the command/input"""
    input_dir = tmpdir.mkdir("inputs")
    filename_1 = input_dir.join("file_1.txt")
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir.join("file_2.txt")
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly_env = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = Native()
    shelly_env.split("file")
    shelly_env()
    assert shelly_env.result()[0].output.stdout.strip() == "hello"
    assert shelly_env.result()[1].output.stdout.strip() == "hi"

    shelly_call = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_call", executable=["cat"]
    )
    shelly_call.split("file")
    shelly_call(environment=Native())
    assert shelly_call.result()[0].output.stdout.strip() == "hello"
    assert shelly_call.result()[1].output.stdout.strip() == "hi"


def test_docker_fileinp(tmpdir):
    """docker env: task with a file in the command/input"""
    docker = Docker(image="busybox")

    input_dir = tmpdir.mkdir("inputs")
    filename = input_dir.join("file.txt")
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly", executable=["cat"]
    )
    env_res = docker.execute(shelly)

    shelly_env = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = docker
    shelly_env()

    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_call", executable=["cat"]
    )
    shelly_call(environment=docker)
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


def test_docker_fileinp_subm(tmpdir, plugin):
    """docker env with a submitter: task with a file in the command/input"""
    docker = Docker(image="busybox")

    input_dir = tmpdir.mkdir("inputs")
    filename = input_dir.join("file.txt")
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly", executable=["cat"]
    )
    env_res = docker.execute(shelly)

    shelly_env = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = docker
    with Submitter(plugin=plugin) as sub:
        shelly_env(submitter=sub)
    assert env_res == shelly_env.result().output.__dict__

    shelly_call = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_call", executable=["cat"]
    )
    with Submitter(plugin=plugin) as sub:
        shelly_call(submitter=sub, environment=docker)
    assert env_res == shelly_call.result().output.__dict__


def test_docker_fileinp_st(tmpdir):
    """docker env: task (with a splitter) with a file in the command/input"""
    docker = Docker(image="busybox")

    input_dir = tmpdir.mkdir("inputs")
    filename_1 = input_dir.join("file_1.txt")
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir.join("file_2.txt")
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly_env = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = docker
    shelly_env.split("file")
    shelly_env()
    assert shelly_env.result()[0].output.stdout.strip() == "hello"
    assert shelly_env.result()[1].output.stdout.strip() == "hi"

    shelly_call = create_shelly_inputfile(
        tempdir=tmpdir, filename=filename, name="shelly_call", executable=["cat"]
    )
    shelly_call.split("file")
    shelly_call(environment=docker)
    assert shelly_call.result()[0].output.stdout.strip() == "hello"
    assert shelly_call.result()[1].output.stdout.strip() == "hi"


def create_shelly_outputfile(tempdir, filename, name, executable="cp"):
    """creating a task with an input_spec that contains a template"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file_orig",
                attr.ib(
                    type=File,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "file_copy",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{file_orig}_copy",
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name=name,
        executable=executable,
        cache_dir=tempdir.mkdir(name),
        input_spec=my_input_spec,
        file_orig=filename,
    )
    return shelly


def test_shell_fileout(tmpdir):
    """task with a file in the output"""
    input_dir = tmpdir.mkdir("inputs")
    filename = input_dir.join("file.txt")
    with open(filename, "w") as f:
        f.write("hello ")

    # execute does not create the cashedir, so this part will fail,
    # but I guess we don't want to use it this way anyway
    # shelly = create_shelly_outputfile(tempdir=tmpdir, filename=filename, name="shelly")
    # env_res = Native().execute(shelly)

    shelly_env = create_shelly_outputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env"
    )
    shelly_env.environment = Native()
    shelly_env()
    assert (
        shelly_env.result().output.file_copy == shelly_env.output_dir / "file_copy.txt"
    )

    shelly_call = create_shelly_outputfile(
        tempdir=tmpdir, filename=filename, name="shelly_call"
    )
    shelly_call(environment=Native())
    assert (
        shelly_call.result().output.file_copy
        == shelly_call.output_dir / "file_copy.txt"
    )


def test_shell_fileout_st(tmpdir):
    """task (with a splitter) with a file in the output"""
    input_dir = tmpdir.mkdir("inputs")
    filename_1 = input_dir.join("file_1.txt")
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir.join("file_2.txt")
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly_env = create_shelly_outputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env"
    )
    shelly_env.environment = Native()
    shelly_env.split("file_orig")
    shelly_env()
    assert (
        shelly_env.result()[0].output.file_copy
        == shelly_env.output_dir[0] / "file_1_copy.txt"
    )
    assert (
        shelly_env.result()[1].output.file_copy
        == shelly_env.output_dir[1] / "file_2_copy.txt"
    )

    shelly_call = create_shelly_outputfile(
        tempdir=tmpdir, filename=filename, name="shelly_call"
    )
    shelly_call.split("file_orig")
    shelly_call(environment=Native())
    assert (
        shelly_call.result()[0].output.file_copy
        == shelly_call.output_dir[0] / "file_1_copy.txt"
    )
    assert (
        shelly_call.result()[1].output.file_copy
        == shelly_call.output_dir[1] / "file_2_copy.txt"
    )


def test_docker_fileout(tmpdir):
    """docker env: task with a file in the output"""
    docker_env = Docker(image="busybox")

    input_dir = tmpdir.mkdir("inputs")
    filename = input_dir.join("file.txt")
    with open(filename, "w") as f:
        f.write("hello ")

    shelly_env = create_shelly_outputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env"
    )
    shelly_env.environment = docker_env
    shelly_env()
    assert (
        shelly_env.result().output.file_copy == shelly_env.output_dir / "file_copy.txt"
    )


def test_docker_fileout_st(tmpdir):
    """docker env: task (with a splitter) with a file in the output"""
    docker_env = Docker(image="busybox")

    input_dir = tmpdir.mkdir("inputs")
    filename_1 = input_dir.join("file_1.txt")
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir.join("file_2.txt")
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly_env = create_shelly_outputfile(
        tempdir=tmpdir, filename=filename, name="shelly_env"
    )
    shelly_env.environment = docker_env
    shelly_env.split("file_orig")
    shelly_env()
    assert (
        shelly_env.result()[0].output.file_copy
        == shelly_env.output_dir[0] / "file_1_copy.txt"
    )
    assert (
        shelly_env.result()[1].output.file_copy
        == shelly_env.output_dir[1] / "file_2_copy.txt"
    )
