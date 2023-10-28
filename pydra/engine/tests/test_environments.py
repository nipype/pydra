from pathlib import Path

from ..environments import Native, Docker, Singularity
from ..task import ShellCommandTask
from ..submitter import Submitter
from ..specs import (
    ShellSpec,
    SpecInfo,
    File,
)
from .utils import no_win, need_docker, need_singularity

import attr


def makedir(path, name):
    newdir = path / name
    newdir.mkdir()
    return newdir


def test_native_1(tmp_path):
    """simple command, no arguments"""
    newcache = lambda x: makedir(tmp_path, x)

    cmd = ["whoami"]
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly")
    )
    assert shelly.cmdline == " ".join(cmd)

    env_res = Native().execute(shelly)
    shelly()
    assert env_res == shelly.output_

    shelly_call = ShellCommandTask(
        name="shelly_call", executable=cmd, cache_dir=newcache("shelly_call")
    )
    shelly_call(environment=Native())
    assert env_res == shelly_call.output_

    shelly_subm = ShellCommandTask(
        name="shelly_subm", executable=cmd, cache_dir=newcache("shelly_subm")
    )
    with Submitter(plugin="cf") as sub:
        shelly_subm(submitter=sub, environment=Native())
    assert env_res == shelly_subm.result().output.__dict__


@no_win
@need_docker
def test_docker_1(tmp_path):
    """docker env: simple command, no arguments"""
    newcache = lambda x: makedir(tmp_path, x)

    cmd = ["whoami"]
    docker = Docker(image="busybox")
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly")
    )
    assert shelly.cmdline == " ".join(cmd)
    env_res = docker.execute(shelly)

    shelly_env = ShellCommandTask(
        name="shelly",
        executable=cmd,
        cache_dir=newcache("shelly_env"),
        environment=docker,
    )
    shelly_env()
    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly_call")
    )
    shelly_call(environment=docker)
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


@no_win
@need_docker
def test_docker_1_subm(tmp_path):
    """docker env with submitter: simple command, no arguments"""
    newcache = lambda x: makedir(tmp_path, x)

    cmd = ["whoami"]
    docker = Docker(image="busybox")
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly")
    )
    assert shelly.cmdline == " ".join(cmd)
    env_res = docker.execute(shelly)

    shelly_env = ShellCommandTask(
        name="shelly",
        executable=cmd,
        cache_dir=newcache("shelly_env"),
        environment=docker,
    )
    with Submitter(plugin="cf") as sub:
        shelly_env(submitter=sub)
    assert env_res == shelly_env.result().output.__dict__

    shelly_call = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly_call")
    )
    with Submitter(plugin="cf") as sub:
        shelly_call(submitter=sub, environment=docker)
    assert env_res == shelly_call.result().output.__dict__


@no_win
@need_singularity
def test_singularity_1(tmp_path):
    """singularity env: simple command, no arguments"""
    newcache = lambda x: makedir(tmp_path, x)

    cmd = ["whoami"]
    sing = Singularity(image="docker://alpine")
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly")
    )
    assert shelly.cmdline == " ".join(cmd)
    env_res = sing.execute(shelly)

    shelly_env = ShellCommandTask(
        name="shelly",
        executable=cmd,
        cache_dir=newcache("shelly_env"),
        environment=sing,
    )
    shelly_env()
    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly_call")
    )
    shelly_call(environment=sing)
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


@no_win
@need_singularity
def test_singularity_1_subm(tmp_path, plugin):
    """docker env with submitter: simple command, no arguments"""
    newcache = lambda x: makedir(tmp_path, x)

    cmd = ["whoami"]
    sing = Singularity(image="docker://alpine")
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly")
    )
    assert shelly.cmdline == " ".join(cmd)
    env_res = sing.execute(shelly)

    shelly_env = ShellCommandTask(
        name="shelly",
        executable=cmd,
        cache_dir=newcache("shelly_env"),
        environment=sing,
    )
    with Submitter(plugin=plugin) as sub:
        shelly_env(submitter=sub)
    assert env_res == shelly_env.result().output.__dict__

    shelly_call = ShellCommandTask(
        name="shelly", executable=cmd, cache_dir=newcache("shelly_call")
    )
    with Submitter(plugin=plugin) as sub:
        shelly_call(submitter=sub, environment=sing)
    for key in [
        "stdout",
        "return_code",
    ]:  # singularity gives info about cashed image in stderr
        assert env_res[key] == shelly_call.result().output.__dict__[key]


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

    kwargs = {} if filename is None else {"file": filename}
    shelly = ShellCommandTask(
        name=name,
        executable=executable,
        cache_dir=makedir(tempdir, name),
        input_spec=my_input_spec,
        **kwargs,
    )
    return shelly


def test_shell_fileinp(tmp_path):
    """task with a file in the command/input"""
    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly", executable=["cat"]
    )
    env_res = Native().execute(shelly)

    shelly_env = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = Native()
    shelly_env()
    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly_call", executable=["cat"]
    )
    shelly_call(environment=Native())
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


def test_shell_fileinp_st(tmp_path):
    """task (with a splitter) with a file in the command/input"""
    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly_env = create_shelly_inputfile(
        tempdir=tmp_path, filename=None, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = Native()
    shelly_env.split(file=filename)
    shelly_env()
    assert shelly_env.result()[0].output.stdout.strip() == "hello"
    assert shelly_env.result()[1].output.stdout.strip() == "hi"

    shelly_call = create_shelly_inputfile(
        tempdir=tmp_path, filename=None, name="shelly_call", executable=["cat"]
    )
    shelly_call.split(file=filename)
    shelly_call(environment=Native())
    assert shelly_call.result()[0].output.stdout.strip() == "hello"
    assert shelly_call.result()[1].output.stdout.strip() == "hi"


@no_win
@need_docker
def test_docker_fileinp(tmp_path):
    """docker env: task with a file in the command/input"""
    docker = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly", executable=["cat"]
    )
    env_res = docker.execute(shelly)

    shelly_env = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = docker
    shelly_env()

    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly_call", executable=["cat"]
    )
    shelly_call(environment=docker)
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


@no_win
@need_docker
def test_docker_fileinp_subm(tmp_path, plugin):
    """docker env with a submitter: task with a file in the command/input"""
    docker = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly", executable=["cat"]
    )
    env_res = docker.execute(shelly)

    shelly_env = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = docker
    with Submitter(plugin=plugin) as sub:
        shelly_env(submitter=sub)
    assert env_res == shelly_env.result().output.__dict__

    shelly_call = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly_call", executable=["cat"]
    )
    with Submitter(plugin=plugin) as sub:
        shelly_call(submitter=sub, environment=docker)
    assert env_res == shelly_call.result().output.__dict__


@no_win
@need_docker
def test_docker_fileinp_st(tmp_path):
    """docker env: task (with a splitter) with a file in the command/input"""
    docker = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly_env = create_shelly_inputfile(
        tempdir=tmp_path, filename=None, name="shelly_env", executable=["cat"]
    )
    shelly_env.environment = docker
    shelly_env.split(file=filename)
    shelly_env()
    assert shelly_env.result()[0].output.stdout.strip() == "hello"
    assert shelly_env.result()[1].output.stdout.strip() == "hi"

    shelly_call = create_shelly_inputfile(
        tempdir=tmp_path, filename=None, name="shelly_call", executable=["cat"]
    )
    shelly_call.split(file=filename)
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

    kwargs = {} if filename is None else {"file_orig": filename}
    shelly = ShellCommandTask(
        name=name,
        executable=executable,
        cache_dir=makedir(tempdir, name),
        input_spec=my_input_spec,
        **kwargs,
    )
    return shelly


def test_shell_fileout(tmp_path):
    """task with a file in the output"""
    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    # execute does not create the cashedir, so this part will fail,
    # but I guess we don't want to use it this way anyway
    # shelly = create_shelly_outputfile(tempdir=tmp_path, filename=filename, name="shelly")
    # env_res = Native().execute(shelly)

    shelly_env = create_shelly_outputfile(
        tempdir=tmp_path, filename=filename, name="shelly_env"
    )
    shelly_env.environment = Native()
    shelly_env()
    assert (
        Path(shelly_env.result().output.file_copy)
        == shelly_env.output_dir / "file_copy.txt"
    )

    shelly_call = create_shelly_outputfile(
        tempdir=tmp_path, filename=filename, name="shelly_call"
    )
    shelly_call(environment=Native())
    assert (
        Path(shelly_call.result().output.file_copy)
        == shelly_call.output_dir / "file_copy.txt"
    )


def test_shell_fileout_st(tmp_path):
    """task (with a splitter) with a file in the output"""
    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly_env = create_shelly_outputfile(
        tempdir=tmp_path, filename=None, name="shelly_env"
    )
    shelly_env.environment = Native()
    shelly_env.split(file_orig=filename)
    shelly_env()
    assert (
        Path(shelly_env.result()[0].output.file_copy)
        == shelly_env.output_dir[0] / "file_1_copy.txt"
    )
    assert (
        Path(shelly_env.result()[1].output.file_copy)
        == shelly_env.output_dir[1] / "file_2_copy.txt"
    )

    shelly_call = create_shelly_outputfile(
        tempdir=tmp_path, filename=None, name="shelly_call"
    )
    shelly_call.split(file_orig=filename)
    shelly_call(environment=Native())
    assert (
        Path(shelly_call.result()[0].output.file_copy)
        == shelly_call.output_dir[0] / "file_1_copy.txt"
    )
    assert (
        Path(shelly_call.result()[1].output.file_copy)
        == shelly_call.output_dir[1] / "file_2_copy.txt"
    )


@no_win
@need_docker
def test_docker_fileout(tmp_path):
    """docker env: task with a file in the output"""
    docker_env = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly_env = create_shelly_outputfile(
        tempdir=tmp_path, filename=filename, name="shelly_env"
    )
    shelly_env.environment = docker_env
    shelly_env()
    assert (
        Path(shelly_env.result().output.file_copy)
        == shelly_env.output_dir / "file_copy.txt"
    )


@no_win
@need_docker
def test_docker_fileout_st(tmp_path):
    """docker env: task (with a splitter) with a file in the output"""
    docker_env = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly_env = create_shelly_outputfile(
        tempdir=tmp_path, filename=None, name="shelly_env"
    )
    shelly_env.environment = docker_env
    shelly_env.split(file_orig=filename)
    shelly_env()
    assert (
        Path(shelly_env.result()[0].output.file_copy)
        == shelly_env.output_dir[0] / "file_1_copy.txt"
    )
    assert (
        Path(shelly_env.result()[1].output.file_copy)
        == shelly_env.output_dir[1] / "file_2_copy.txt"
    )
