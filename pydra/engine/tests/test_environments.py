from pathlib import Path
import typing as ty
from ..environments import Native, Docker, Singularity
from ..task import ShellDef
from ..submitter import Submitter
from fileformats.generic import File
from pydra.design import shell
from pydra.engine.core import Task
from pydra.engine.helpers import attrs_values
from .utils import no_win, need_docker, need_singularity
import pytest


def makedir(path, name):
    newdir = path / name
    newdir.mkdir()
    return newdir


def drop_stderr(dct: dict[str, ty.Any]):
    return {k: v for k, v in dct.items() if k != "stderror"}


def test_native_1(tmp_path):
    """simple command, no arguments"""

    def newcache(x):
        return makedir(tmp_path, x)

    cmd = "whoami"
    Shelly = shell.define(cmd)
    shelly = Shelly()
    assert shelly.cmdline == cmd

    shelly_job = Task(
        definition=shelly,
        submitter=Submitter(cache_dir=newcache("shelly-task")),
        name="shelly",
    )
    env_outputs = Native().execute(shelly_job)

    outputs = shelly(cache_dir=newcache("shelly-exec"))
    assert drop_stderr(env_outputs) == drop_stderr(attrs_values(outputs))

    outputs = shelly(environment=Native())
    assert env_outputs == attrs_values(outputs)

    with Submitter(cache_dir=newcache("shelly-submitter"), environment=Native()) as sub:
        result = sub(shelly)
    assert drop_stderr(env_outputs) == drop_stderr(attrs_values(result.outputs))


@no_win
@need_docker
def test_docker_1(tmp_path):
    """docker env: simple command, no arguments"""

    def newcache(x):
        makedir(tmp_path, x)

    cmd = ["whoami"]
    docker = Docker(image="busybox")
    Shelly = shell.define(cmd)
    shelly = Shelly()
    assert shelly.cmdline == " ".join(cmd)

    shelly_job = Task(
        definition=shelly,
        submitter=Submitter(cache_dir=newcache("shelly")),
        name="shelly",
    )
    env_res = docker.execute(shelly_job)

    shelly_env = ShellDef(
        name="shelly",
        executable=cmd,
        cache_dir=newcache("shelly_env"),
        environment=docker,
    )
    shelly_env()
    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = ShellDef(
        name="shelly", executable=cmd, cache_dir=newcache("shelly_call")
    )
    shelly_call(environment=docker)
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


@no_win
@need_docker
@pytest.mark.parametrize(
    "docker",
    [
        Docker(image="busybox"),
        Docker(image="busybox", tag="latest", xargs="--rm"),
        Docker(image="busybox", xargs=["--rm"]),
    ],
)
def test_docker_1_subm(tmp_path, docker):
    """docker env with submitter: simple command, no arguments"""

    def newcache(x):
        makedir(tmp_path, x)

    cmd = "whoami"
    docker = Docker(image="busybox")
    shelly = shell.define(cmd)()
    shelly_job = Task(
        definition=shelly,
        submitter=Submitter(cache_dir=newcache("shelly")),
        name="shelly",
    )
    assert shelly.cmdline == cmd
    env_res = docker.execute(shelly_job)

    with Submitter(
        worker="cf", cache_dir=newcache("shelly_env"), environment=docker
    ) as sub:
        result = sub(shelly)
    assert env_res == attrs_values(result.outputs)

    shelly_call = ShellDef(
        name="shelly", executable=cmd, cache_dir=newcache("shelly_call")
    )
    with Submitter(worker="cf") as sub:
        shelly_call(submitter=sub, environment=docker)
    assert env_res == shelly_call.result().output.__dict__


@no_win
@need_singularity
def test_singularity_1(tmp_path):
    """singularity env: simple command, no arguments"""

    def newcache(x):
        makedir(tmp_path, x)

    cmd = ["whoami"]
    sing = Singularity(image="docker://alpine")
    shell_def = shell.define(cmd)
    shelly = Task(
        definition=shell_def,
        submitter=Submitter(cache_dir=newcache("shelly")),
        name="shelly",
    )
    assert shell_def.cmdline == " ".join(cmd)
    env_res = sing.execute(shelly)

    shelly_env = ShellDef(
        name="shelly",
        executable=cmd,
        cache_dir=newcache("shelly_env"),
        environment=sing,
    )
    shelly_env()
    assert env_res == shelly_env.output_ == shelly_env.result().output.__dict__

    shelly_call = ShellDef(
        name="shelly", executable=cmd, cache_dir=newcache("shelly_call")
    )
    shelly_call(environment=sing)
    assert env_res == shelly_call.output_ == shelly_call.result().output.__dict__


@no_win
@need_singularity
def test_singularity_1_subm(tmp_path, plugin):
    """docker env with submitter: simple command, no arguments"""

    def newcache(x):
        makedir(tmp_path, x)

    cmd = ["whoami"]
    sing = Singularity(image="docker://alpine")
    shell_def = shell.define(cmd)
    shelly = Task(
        definition=shell_def,
        submitter=Submitter(cache_dir=newcache("shelly")),
        name="shelly",
    )
    assert shell_def.cmdline == " ".join(cmd)
    env_res = sing.execute(shelly)

    shelly_env = ShellDef(
        name="shelly",
        executable=cmd,
        cache_dir=newcache("shelly_env"),
        environment=sing,
    )
    with Submitter(worker=plugin) as sub:
        shelly_env(submitter=sub)
    assert env_res == shelly_env.result().output.__dict__

    shelly_call = ShellDef(
        name="shelly", executable=cmd, cache_dir=newcache("shelly_call")
    )
    with Submitter(worker=plugin) as sub:
        shelly_call(submitter=sub, environment=sing)
    for key in [
        "stdout",
        "return_code",
    ]:  # singularity gives info about cashed image in stderr
        assert env_res[key] == shelly_call.result().output.__dict__[key]


def create_shelly_inputfile(tempdir, filename, name, executable):
    """creating a task with a simple input_spec"""
    inputs = [
        shell.arg(
            name="file",
            type=File,
            position=1,
            help="files",
            argstr="",
        )
    ]

    kwargs = {} if filename is None else {"file": filename}
    shelly = shell.define(
        executable,
        input=inputs,
    )(**kwargs)
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
    with Submitter(worker=plugin) as sub:
        shelly_env(submitter=sub)
    assert env_res == shelly_env.result().output.__dict__

    shelly_call = create_shelly_inputfile(
        tempdir=tmp_path, filename=filename, name="shelly_call", executable=["cat"]
    )
    with Submitter(worker=plugin) as sub:
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
    my_input_spec = [
        shell.arg(
            name="file_orig",
            type=File,
            position=2,
            help="new file",
            argstr="",
        ),
        shell.arg(
            name="file_copy",
            type=str,
            output_file_template="{file_orig}_copy",
            help="output file",
            argstr="",
        ),
    ]

    kwargs = {} if filename is None else {"file_orig": filename}
    shelly = shell.define(executable)(
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
