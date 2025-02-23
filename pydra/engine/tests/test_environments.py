from pathlib import Path
import typing as ty
from ..environments import Native, Docker, Singularity
from ..submitter import Submitter
from fileformats.generic import File
from pydra.design import shell
from pydra.engine.core import Task
from pydra.engine.task import ShellDef
from pydra.engine.helpers import attrs_values
from .utils import no_win, need_docker, need_singularity
import pytest


def makedir(path, name):
    newdir = path / name
    newdir.mkdir()
    return newdir


def drop_stderr(dct: dict[str, ty.Any]):
    return {k: v for k, v in dct.items() if k != "stderr"}


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
        submitter=Submitter(cache_dir=newcache("native-task")),
        name="native",
    )
    env_outputs = Native().execute(shelly_job)

    outputs = shelly(cache_dir=newcache("native-exec"))
    assert drop_stderr(env_outputs) == drop_stderr(attrs_values(outputs))

    outputs = shelly(environment=Native(), cache_dir=newcache("native-call"))
    assert env_outputs == attrs_values(outputs)

    with Submitter(cache_dir=newcache("native-submitter"), environment=Native()) as sub:
        result = sub(shelly)
    assert drop_stderr(env_outputs) == drop_stderr(attrs_values(result.outputs))


@no_win
@need_docker
def test_docker_1(tmp_path):
    """docker env: simple command, no arguments"""

    def newcache(x):
        makedir(tmp_path, x)

    cmd = "whoami"
    docker = Docker(image="busybox")
    Shelly = shell.define(cmd)
    shelly = Shelly()
    assert shelly.cmdline == cmd

    shelly_job = Task(
        definition=shelly,
        submitter=Submitter(cache_dir=newcache("docker")),
        name="docker",
    )
    outputs_dict = docker.execute(shelly_job)

    with Submitter(cache_dir=newcache("docker_sub"), environment=docker) as sub:
        result = sub(shelly)

    outputs = shelly(environment=docker, cache_dir=newcache("docker_call"))
    # If busybox isn't found locally, then the stderr will have the download progress from
    # the Docker auto-pull in it
    for key in ["stdout", "return_code"]:
        assert (
            outputs_dict[key]
            == attrs_values(outputs)[key]
            == attrs_values(result.outputs)[key]
        )


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
        submitter=Submitter(cache_dir=newcache("docker")),
        name="docker",
    )
    assert shelly.cmdline == cmd
    outputs_dict = docker.execute(shelly_job)

    with Submitter(
        worker="cf", cache_dir=newcache("docker_sub"), environment=docker
    ) as sub:
        result = sub(shelly)
    assert outputs_dict == attrs_values(result.outputs)

    outputs = shelly(cache_dir=newcache("docker_call"), environment=docker)
    assert outputs_dict == attrs_values(outputs)


@no_win
@need_singularity
def test_singularity_1(tmp_path):
    """singularity env: simple command, no arguments"""

    def newcache(x):
        makedir(tmp_path, x)

    cmd = "whoami"
    sing = Singularity(image="docker://alpine")
    Shelly = shell.define(cmd)
    shelly = Shelly()
    shelly_job = Task(
        definition=shelly,
        submitter=Submitter(cache_dir=newcache("singu")),
        name="singu",
    )
    assert shelly.cmdline == cmd
    outputs_dict = sing.execute(shelly_job)

    with Submitter(cache_dir=newcache("singu_sub"), environment=sing) as sub:
        results = sub(shelly)
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(results.outputs))

    outputs = shelly(environment=sing, cache_dir=newcache("singu_call"))
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(outputs))


@no_win
@need_singularity
def test_singularity_1_subm(tmp_path, plugin):
    """docker env with submitter: simple command, no arguments"""

    def newcache(x):
        makedir(tmp_path, x)

    cmd = "whoami"
    sing = Singularity(image="docker://alpine")
    Shelly = shell.define(cmd)
    shelly = Shelly()
    shelly_job = Task(
        definition=shelly,
        submitter=Submitter(cache_dir=newcache("singu")),
        name="singu",
    )
    assert shelly.cmdline == cmd
    outputs_dict = sing.execute(shelly_job)

    with Submitter(
        worker=plugin, environment=sing, cache_dir=newcache("singu_sub")
    ) as sub:
        results = sub(shelly)
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(results.outputs))

    outputs = shelly(environment=sing, cache_dir=newcache("singu_call"))
    # singularity gives info about cashed image in stderr
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(outputs))


def shelly_with_input_factory(filename, executable) -> ShellDef:
    """creating a task with a simple input_spec"""
    Shelly = shell.define(
        executable,
        inputs=[
            shell.arg(
                name="file",
                type=File,
                position=1,
                help="files",
                argstr="",
            )
        ],
    )
    return Shelly(**({} if filename is None else {"file": filename}))


def make_job(task: ShellDef, tempdir: Path, name: str):
    return Task(
        definition=task,
        submitter=Submitter(cache_dir=makedir(tempdir, name)),
        name=name,
    )


def test_shell_fileinp(tmp_path):
    """task with a file in the command/input"""

    def newcache(x):
        return makedir(tmp_path, x)

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = shelly_with_input_factory(filename=filename, executable="cat")
    shelly_job = make_job(shelly, tmp_path, "native")
    outputs_dict = Native().execute(shelly_job)

    with Submitter(environment=Native(), cache_dir=newcache("native_sub")) as sub:
        results = sub(shelly)
    assert outputs_dict == attrs_values(results.outputs)

    outputs = shelly(environment=Native(), cache_dir=newcache("native_call"))
    assert outputs_dict == attrs_values(outputs)


def test_shell_fileinp_st(tmp_path):
    """task (with a splitter) with a file in the command/input"""

    def newcache(x):
        return makedir(tmp_path, x)

    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly = shelly_with_input_factory(filename=None, executable="cat")
    with Submitter(environment=Native(), cache_dir=newcache("native")) as sub:
        results = sub(shelly.split(file=filename))
    assert [s.strip() for s in results.outputs.stdout] == ["hello", "hi"]

    outputs = shelly.split(file=filename)(
        environment=Native(), cache_dir=newcache("native_call")
    )
    assert [s.strip() for s in outputs.stdout] == ["hello", "hi"]


@no_win
@need_docker
def test_docker_fileinp(tmp_path):
    """docker env: task with a file in the command/input"""

    def newcache(x):
        return makedir(tmp_path, x)

    docker = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = shelly_with_input_factory(filename=filename, executable="cat")
    outputs_dict = docker.execute(make_job(shelly, tmp_path, "docker"))

    with Submitter(environment=docker, cache_dir=newcache("shell_sub")) as sub:
        results = sub(shelly)

    assert outputs_dict == attrs_values(results.outputs)

    outputs = shelly(environment=docker, cache_dir=newcache("docker_call"))
    assert outputs_dict == attrs_values(outputs)


@no_win
@need_docker
def test_docker_fileinp_subm(tmp_path, plugin):
    """docker env with a submitter: task with a file in the command/input"""

    def newcache(x):
        return makedir(tmp_path, x)

    docker = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = shelly_with_input_factory(filename=filename, executable="cat")
    shelly_job = make_job(shelly, tmp_path, "docker_job")
    outputs_dict = docker.execute(shelly_job)

    with Submitter(
        environment=docker, cache_dir=newcache("docker_sub"), worker=plugin
    ) as sub:
        results = sub(shelly)
    with Submitter(worker=plugin) as sub:
        results = sub(shelly)
    assert outputs_dict == attrs_values(results.outputs)

    outputs = shelly(environment=docker, cache_dir=newcache("docker_call"))
    assert outputs_dict == attrs_values(outputs)


@no_win
@need_docker
def test_docker_fileinp_st(tmp_path):
    """docker env: task (with a splitter) with a file in the command/input"""

    def newcache(x):
        return makedir(tmp_path, x)

    docker = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly = shelly_with_input_factory(filename=None, executable="cat")

    with Submitter(environment=docker, cache_dir=newcache("docker_sub")) as sub:
        results = sub(shelly.split(file=filename))

    assert [s.strip() for s in results.outputs.stdout] == ["hello", "hi"]

    outputs = shelly.split(file=filename)(
        environment=docker, cache_dir=newcache("docker_call")
    )
    assert [s.strip() for s in outputs.stdout] == ["hello", "hi"]


def shelly_outputfile_factory(filename, executable="cp"):
    """creating a task with an input_spec that contains a template"""
    Shelly = shell.define(
        executable,
        inputs=[
            shell.arg(
                name="file_orig",
                type=File,
                position=1,
                help="new file",
                argstr="",
            ),
        ],
        outputs=[
            shell.outarg(
                name="file_copy",
                type=File,
                path_template="{file_orig}_copy",
                help="output file",
                argstr="",
                position=2,
                keep_extension=True,
            ),
        ],
    )

    return Shelly(**({} if filename is None else {"file_orig": filename}))


def test_shell_fileout(tmp_path):
    """task with a file in the output"""

    def newcache(x):
        return Path(makedir(tmp_path, x))

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    # execute does not create the cashedir, so this part will fail,
    # but I guess we don't want to use it this way anyway
    # shelly = create_shelly_outputfile(tempdir=tmp_path, filename=filename, name="native")
    # outputs_dict = Native().execute(shelly)

    shelly = shelly_outputfile_factory(filename=filename)

    with Submitter(environment=Native(), cache_dir=newcache("native_sub")) as sub:
        result = sub(shelly)
    assert Path(result.outputs.file_copy) == result.output_dir / "file_copy.txt"

    call_cache = newcache("native_call")

    outputs = shelly(environment=Native(), cache_dir=call_cache)
    assert Path(outputs.file_copy) == call_cache / shelly._checksum / "file_copy.txt"


def test_shell_fileout_st(tmp_path):
    """task (with a splitter) with a file in the output"""

    def newcache(x):
        return Path(makedir(tmp_path, x))

    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly = shelly_outputfile_factory(filename=None)
    with Submitter(environment=Native(), cache_dir=newcache("native")) as sub:
        results = sub(shelly.split(file_orig=filename))

    assert [f.name for f in results.outputs.file_copy] == [
        "file_1_copy.txt",
        "file_2_copy.txt",
    ]

    call_cache = newcache("native_call")

    outputs = shelly.split(file_orig=filename)(
        environment=Native(), cache_dir=call_cache
    )
    assert [f.name for f in outputs.file_copy] == ["file_1_copy.txt", "file_2_copy.txt"]


@no_win
@need_docker
def test_docker_fileout(tmp_path):
    """docker env: task with a file in the output"""

    def newcache(x):
        return Path(makedir(tmp_path, x))

    docker_env = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = shelly_outputfile_factory(filename=filename)

    with Submitter(environment=docker_env, cache_dir=newcache("docker")) as sub:
        results = sub(shelly)
    assert results.outputs.file_copy == File(results.output_dir / "file_copy.txt")


@no_win
@need_docker
def test_docker_fileout_st(tmp_path):
    """docker env: task (with a splitter) with a file in the output"""

    def newcache(x):
        return Path(makedir(tmp_path, x))

    docker_env = Docker(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly = shelly_outputfile_factory(filename=None)

    with Submitter(environment=docker_env, cache_dir=newcache("docker_sub")) as sub:
        results = sub(shelly.split(file_orig=filename))
    assert [f.name for f in results.outputs.file_copy] == [
        "file_1_copy.txt",
        "file_2_copy.txt",
    ]
