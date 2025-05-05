from pathlib import Path
import typing as ty
from pydra.environments import native, docker, singularity
from pydra.engine.submitter import Submitter
from fileformats.generic import File
from pydra.compose import shell
from pydra.engine.job import Job
from pydra.utils.general import attrs_values
from pydra.engine.tests.utils import no_win, need_docker, need_singularity
import pytest


def makedir(path: Path, name: str) -> Path:
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

    shelly_job = Job(
        task=shelly,
        submitter=Submitter(cache_root=newcache("native-task")),
        name="native",
    )
    env_outputs = native.Environment().execute(shelly_job)

    outputs = shelly(cache_root=newcache("native-exec"))
    assert drop_stderr(env_outputs) == drop_stderr(attrs_values(outputs))

    outputs = shelly(
        environment=native.Environment(), cache_root=newcache("native-call")
    )
    assert drop_stderr(env_outputs) == drop_stderr(attrs_values(outputs))

    with Submitter(
        cache_root=newcache("native-submitter"), environment=native.Environment()
    ) as sub:
        result = sub(shelly)
    assert drop_stderr(env_outputs) == drop_stderr(attrs_values(result.outputs))


@no_win
@need_docker
def test_docker_1(tmp_path):
    """docker env: simple command, no arguments"""

    def newcache(x):
        return makedir(tmp_path, x)

    cmd = "whoami"
    dock = docker.Environment(image="busybox")
    Shelly = shell.define(cmd)
    shelly = Shelly()
    assert shelly.cmdline == cmd

    shelly_job = Job(
        task=shelly,
        submitter=Submitter(cache_root=newcache("docker")),
        name="docker",
    )
    outputs_dict = dock.execute(shelly_job)

    with Submitter(cache_root=newcache("docker_sub"), environment=dock) as sub:
        result = sub(shelly)

    outputs = shelly(environment=dock, cache_root=newcache("docker_call"))
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
    "dock",
    [
        docker.Environment(image="busybox"),
        docker.Environment(image="busybox", tag="latest", xargs="--rm"),
        docker.Environment(image="busybox", xargs=["--rm"]),
    ],
)
def test_docker_1_subm(tmp_path, dock):
    """docker env with submitter: simple command, no arguments"""

    def newcache(x):
        return makedir(tmp_path, x)

    cmd = "whoami"
    shelly = shell.define(cmd)()
    shelly_job = Job(
        task=shelly,
        submitter=Submitter(cache_root=newcache("docker")),
        name="docker",
    )
    assert shelly.cmdline == cmd
    outputs_dict = dock.execute(shelly_job)

    with Submitter(
        worker="cf", cache_root=newcache("docker_sub"), environment=dock
    ) as sub:
        result = sub(shelly)
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(result.outputs))

    outputs = shelly(cache_root=newcache("docker_call"), environment=dock)
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(outputs))


@no_win
@need_singularity
def test_singularity_1(tmp_path):
    """singularity env: simple command, no arguments"""

    def newcache(x):
        return makedir(tmp_path, x)

    cmd = "whoami"
    sing = singularity.Environment(image="docker://alpine", xargs=["--fakeroot"])
    Shelly = shell.define(cmd)
    shelly = Shelly()
    shelly_job = Job(
        task=shelly,
        submitter=Submitter(cache_root=newcache("singu")),
        name="singu",
    )
    assert shelly.cmdline == cmd
    outputs_dict = sing.execute(shelly_job)

    with Submitter(cache_root=newcache("singu_sub"), environment=sing) as sub:
        results = sub(shelly)
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(results.outputs))

    outputs = shelly(environment=sing, cache_root=newcache("singu_call"))
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(outputs))


@no_win
@need_singularity
def test_singularity_1_subm(tmp_path, worker):
    """docker env with submitter: simple command, no arguments"""

    def newcache(x: str) -> Path:
        return makedir(tmp_path, x)

    cmd = "whoami"
    sing = singularity.Environment(image="docker://alpine", xargs=["--fakeroot"])
    Shelly = shell.define(cmd)
    shelly = Shelly()
    shelly_job = Job(
        task=shelly,
        submitter=Submitter(cache_root=newcache("singu")),
        name="singu",
    )
    assert shelly.cmdline == cmd
    outputs_dict = sing.execute(shelly_job)

    with Submitter(
        worker=worker, environment=sing, cache_root=newcache("singu_sub")
    ) as sub:
        results = sub(shelly)
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(results.outputs))

    outputs = shelly(environment=sing, cache_root=newcache("singu_call"))
    # singularity gives info about cashed image in stderr
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(outputs))


def shelly_with_input_factory(filename, executable) -> shell.Task:
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


def make_job(task: shell.Task, tempdir: Path, name: str):
    return Job(
        task=task,
        submitter=Submitter(cache_root=makedir(tempdir, name)),
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
    outputs_dict = native.Environment().execute(shelly_job)

    with Submitter(
        environment=native.Environment(), cache_root=newcache("native_sub")
    ) as sub:
        results = sub(shelly)
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(results.outputs))

    outputs = shelly(
        environment=native.Environment(), cache_root=newcache("native_call")
    )
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(outputs))


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
    with Submitter(
        environment=native.Environment(), cache_root=newcache("native")
    ) as sub:
        results = sub(shelly.split(file=filename))
    assert [s.strip() for s in results.outputs.stdout] == ["hello", "hi"]

    outputs = shelly.split(file=filename)(
        environment=native.Environment(), cache_root=newcache("native_call")
    )
    assert [s.strip() for s in outputs.stdout] == ["hello", "hi"]


@no_win
@need_docker
def test_docker_fileinp(tmp_path):
    """docker env: task with a file in the command/input"""

    def newcache(x):
        return makedir(tmp_path, x)

    dock = docker.Environment(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = shelly_with_input_factory(filename=filename, executable="cat")
    outputs_dict = dock.execute(make_job(shelly, tmp_path, "docker"))

    with Submitter(environment=dock, cache_root=newcache("shell_sub")) as sub:
        results = sub(shelly)

    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(results.outputs))

    outputs = shelly(environment=dock, cache_root=newcache("docker_call"))
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(outputs))


@no_win
@need_docker
def test_docker_fileinp_subm(tmp_path, worker):
    """docker env with a submitter: task with a file in the command/input"""

    def newcache(x):
        return makedir(tmp_path, x)

    dock = docker.Environment(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = shelly_with_input_factory(filename=filename, executable="cat")
    shelly_job = make_job(shelly, tmp_path, "docker_job")
    outputs_dict = dock.execute(shelly_job)

    with Submitter(
        environment=dock, cache_root=newcache("docker_sub"), worker=worker
    ) as sub:
        results = sub(shelly)
    with Submitter(worker=worker) as sub:
        results = sub(shelly)
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(results.outputs))

    outputs = shelly(environment=dock, cache_root=newcache("docker_call"))
    assert drop_stderr(outputs_dict) == drop_stderr(attrs_values(outputs))


@no_win
@need_docker
def test_docker_fileinp_st(tmp_path):
    """docker env: task (with a splitter) with a file in the command/input"""

    def newcache(x):
        return makedir(tmp_path, x)

    dock = docker.Environment(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly = shelly_with_input_factory(filename=None, executable="cat")

    with Submitter(environment=dock, cache_root=newcache("docker_sub")) as sub:
        results = sub(shelly.split(file=filename))

    assert [s.strip() for s in results.outputs.stdout] == ["hello", "hi"]

    outputs = shelly.split(file=filename)(
        environment=dock, cache_root=newcache("docker_call")
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
    # outputs_dict = native.Environment().execute(shelly)

    shelly = shelly_outputfile_factory(filename=filename)

    with Submitter(
        environment=native.Environment(), cache_root=newcache("native_sub")
    ) as sub:
        result = sub(shelly)
    assert Path(result.outputs.file_copy) == result.cache_dir / "file_copy.txt"

    call_cache = newcache("native_call")

    outputs = shelly(environment=native.Environment(), cache_root=call_cache)
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
    with Submitter(
        environment=native.Environment(), cache_root=newcache("native")
    ) as sub:
        results = sub(shelly.split(file_orig=filename))

    assert [f.name for f in results.outputs.file_copy] == [
        "file_1_copy.txt",
        "file_2_copy.txt",
    ]

    call_cache = newcache("native_call")

    outputs = shelly.split(file_orig=filename)(
        environment=native.Environment(), cache_root=call_cache
    )
    assert [f.name for f in outputs.file_copy] == ["file_1_copy.txt", "file_2_copy.txt"]


@no_win
@need_docker
def test_docker_fileout(tmp_path):
    """docker env: task with a file in the output"""

    def newcache(x):
        return Path(makedir(tmp_path, x))

    dock = docker.Environment(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename = input_dir / "file.txt"
    with open(filename, "w") as f:
        f.write("hello ")

    shelly = shelly_outputfile_factory(filename=filename)

    with Submitter(environment=dock, cache_root=newcache("docker")) as sub:
        results = sub(shelly)
    assert results.outputs.file_copy == File(results.cache_dir / "file_copy.txt")


@no_win
@need_docker
def test_docker_fileout_st(tmp_path):
    """docker env: task (with a splitter) with a file in the output"""

    def newcache(x):
        return Path(makedir(tmp_path, x))

    dock = docker.Environment(image="busybox")

    input_dir = makedir(tmp_path, "inputs")
    filename_1 = input_dir / "file_1.txt"
    with open(filename_1, "w") as f:
        f.write("hello ")

    filename_2 = input_dir / "file_2.txt"
    with open(filename_2, "w") as f:
        f.write("hi ")

    filename = [filename_1, filename_2]

    shelly = shelly_outputfile_factory(filename=None)

    with Submitter(environment=dock, cache_root=newcache("docker_sub")) as sub:
        results = sub(shelly.split(file_orig=filename))
    assert [f.name for f in results.outputs.file_copy] == [
        "file_1_copy.txt",
        "file_2_copy.txt",
    ]


@no_win
@need_docker
def test_entrypoint(tmp_path):
    """docker env: task with a file in the output"""

    import docker as docker_engine

    dc = docker_engine.from_env()

    # Create executable that runs validator then produces some mock output
    # files
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    entrypoint = build_dir / "entrypoint.sh"
    with open(entrypoint, "w") as f:
        f.write("#!/bin/sh\necho hello $1")

    IMAGE_TAG = "pydra-test-entrypoint"

    # Build mock BIDS app image
    with open(build_dir / "Dockerfile", "w") as f:
        f.write(
            """FROM busybox
ADD ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]"""
        )

    dc.images.build(path=str(build_dir), tag=IMAGE_TAG + ":latest")

    @shell.define
    class TestEntrypoint(shell.Task):
        """task with a file in the output"""

        executable = None
        persons_name: str = shell.arg(help="the name of the person to say hello to")

        class Outputs(shell.Outputs):
            pass

    test_entrypoint = TestEntrypoint(persons_name="Guido")

    outputs = test_entrypoint(environment=docker.Environment(image=IMAGE_TAG))

    assert outputs.stdout == "hello Guido\n"
