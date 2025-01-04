import pytest
from ..task import ShellTask
from ..submitter import Submitter
from fileformats.generic import File
from ..environments import Docker
from pydra.design import shell, workflow
from .utils import no_win, need_docker, result_submitter, result_no_submitter


@no_win
@need_docker
def test_docker_1_nosubm():
    """simple command in a container, a default bindings and working directory is added
    no submitter
    """
    cmd = "whoami"
    docky = shell.define(cmd)(environment=Docker(image="busybox"))
    assert docky.environment.image == "busybox"
    assert docky.environment.tag == "latest"
    assert isinstance(docky.environment, Docker)
    assert docky.cmdline == cmd

    res = docky()
    assert res.output.stdout == "root\n"
    assert res.output.return_code == 0


@no_win
@need_docker
def test_docker_1(plugin):
    """simple command in a container, a default bindings and working directory is added
    using submitter
    """
    cmd = "whoami"
    docky = shell.define(cmd)(environment=Docker(image="busybox"))

    with Submitter(plugin=plugin) as sub:
        docky(submitter=sub)

    res = docky.result()
    assert res.output.stdout == "root\n"
    assert res.output.return_code == 0


@no_win
@need_docker
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_docker_2(results_function, plugin):
    """a command with arguments, cmd and args given as executable
    with and without submitter
    """
    cmdline = "echo hail pydra"
    docky = shell.define(cmdline)(environment=Docker(image="busybox"))
    # cmdline doesn't know anything about docker
    assert docky.cmdline == cmdline
    res = results_function(docky, plugin)
    assert res.output.stdout.strip() == " ".join(cmdline.split()[1:])
    assert res.output.return_code == 0


@no_win
@need_docker
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_docker_2a(results_function, plugin):
    """a command with arguments, using executable and args
    using submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    docky = ShellTask(
        name="docky",
        executable=cmd_exec,
        args=cmd_args,
        environment=Docker(image="busybox"),
    )
    assert docky.definition.executable == "echo"
    assert docky.cmdline == f"{cmd_exec} {' '.join(cmd_args)}"

    res = results_function(docky, plugin)
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0


# tests with State


@no_win
@need_docker
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_docker_st_1(results_function, plugin):
    """commands without arguments in container
    splitter = executable
    """
    cmd = ["pwd", "whoami"]
    docky = ShellTask(name="docky", environment=Docker(image="busybox")).split(
        "executable", executable=cmd
    )
    assert docky.state.splitter == "docky.executable"

    res = results_function(docky, plugin)
    assert res[0].output.stdout == f"/mnt/pydra{docky.output_dir[0]}\n"
    assert res[1].output.stdout == "root\n"
    assert res[0].output.return_code == res[1].output.return_code == 0


# tests with customized output_spec


@no_win
@need_docker
def test_docker_outputspec_1(plugin, tmp_path):
    """
    customised output_spec, adding files to the output, providing specific pathname
    output_path is automatically added to the bindings
    """
    outputs = [shell.out(name="newfile", type=File, help="new file")]
    docky = shell.define("touch newfile_tmp.txt", outputs=outputs)(
        environment=Docker(image="ubuntu")
    )

    res = docky(plugin=plugin)
    assert res.output.stdout == ""


# tests with customised input_spec


@no_win
@need_docker
def test_docker_inputspec_1(tmp_path):
    """a simple customized input definition for docker task"""
    filename = str(tmp_path / "file_pydra.txt")
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file",
            type=File,
            position=1,
            argstr="",
            help="input file",
        )
    ]

    docky = shell.define(cmd, inputs=inputs)(
        environment=Docker(image="busybox"),
        file=filename,
        strip=True,
    )

    res = docky()
    assert res.output.stdout == "hello from pydra"


@no_win
@need_docker
def test_docker_inputspec_1a(tmp_path):
    """a simple customized input definition for docker task
    a default value is used
    """
    filename = str(tmp_path / "file_pydra.txt")
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file",
            type=File,
            default=filename,
            position=1,
            argstr="",
            help="input file",
        )
    ]

    docky = shell.define(cmd, inputs=inputs)(
        environment=Docker(image="busybox"),
        strip=True,
    )

    res = docky()
    assert res.output.stdout == "hello from pydra"


@no_win
@need_docker
def test_docker_inputspec_2(plugin, tmp_path):
    """a customized input definition with two fields for docker task"""
    filename_1 = tmp_path / "file_pydra.txt"
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")

    filename_2 = tmp_path / "file_nice.txt"
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file1",
            type=File,
            position=1,
            argstr="",
            help="input file 1",
        ),
        shell.arg(
            name="file2",
            type=File,
            default=filename_2,
            position=2,
            argstr="",
            help="input file 2",
        ),
    ]
    docky = shell.define(cmd, inputs=inputs)(
        name="docky",
        environment=Docker(image="busybox"),
        file1=filename_1,
        strip=True,
    )

    res = docky()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@no_win
@need_docker
def test_docker_inputspec_2a_except(plugin, tmp_path):
    """a customized input definition with two fields
    first one uses a default, and second doesn't - raises a dataclass exception
    """
    filename_1 = tmp_path / "file_pydra.txt"
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmp_path / "file_nice.txt"
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file1",
            type=File,
            default=filename_1,
            position=1,
            argstr="",
            help="input file 1",
        ),
        shell.arg(
            name="file2",
            type=File,
            position=2,
            argstr="",
            help="input file 2",
        ),
    ]

    docky = shell.define(cmd, inputs=inputs)(
        environment=Docker(image="busybox"),
        file2=filename_2,
        strip=True,
    )
    assert docky.definition.file2.fspath == filename_2

    res = docky()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@no_win
@need_docker
def test_docker_inputspec_2a(plugin, tmp_path):
    """a customized input definition with two fields
    first one uses a default value
    this is fine even if the second field is not using any defaults
    """
    filename_1 = tmp_path / "file_pydra.txt"
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmp_path / "file_nice.txt"
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file1",
            type=File,
            default=filename_1,
            position=1,
            argstr="",
            help="input file 1",
        ),
        shell.arg(
            name="file2",
            type=File,
            position=2,
            argstr="",
            help="input file 2",
        ),
    ]

    docky = shell.define(cmd, inputs=inputs)(
        environment=Docker(image="busybox"),
        file2=filename_2,
        strip=True,
    )

    res = docky()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@no_win
@need_docker
@pytest.mark.xfail(reason="'docker' not in /proc/1/cgroup on ubuntu; TODO")
def test_docker_inputspec_3(plugin, tmp_path):
    """input file is in the container, so metadata["container_path"]: True,
    the input will be treated as a str"""
    filename = "/proc/1/cgroup"

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file",
            type=File,
            position=1,
            argstr="",
            help="input file",
            container_path=True,
        )
    ]

    docky = shell.define(cmd, inputs=inputs)(
        environment=Docker(image="busybox"),
        file=filename,
        strip=True,
    )

    cmdline = docky.cmdline
    res = docky()
    assert "docker" in res.output.stdout
    assert cmdline == docky.cmdline


@no_win
@need_docker
def test_docker_cmd_inputspec_copyfile_1(plugin, tmp_path):
    """shelltask changes a file in place,
    adding copyfile=True to the file-input from input_spec
    hardlink or copy in the output_dir should be created
    """
    file = tmp_path / "file_pydra.txt"
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]

    inputs = [
        shell.arg(
            name="orig_file",
            type=File,
            position=1,
            argstr="",
            help="orig file",
            copyfile="copy",
        ),
        shell.arg(
            name="out_file",
            type=str,
            output_file_template="{orig_file}",
            help="output file",
        ),
    ]

    docky = shell.define(cmd, inputs=inputs)(
        environment=Docker(image="busybox"),
        orig_file=str(file),
    )

    res = docky()
    assert res.output.stdout == ""
    out_file = res.output.out_file.fspath
    assert out_file.exists()
    # the file is copied, and then it is changed in place
    assert out_file.parent == docky.output_dir
    with open(out_file) as f:
        assert "hi from pydra\n" == f.read()
    # the original file is unchanged
    with open(file) as f:
        assert "hello from pydra\n" == f.read()


@no_win
@need_docker
def test_docker_inputspec_state_1(plugin, tmp_path):
    """a customised input definition for a docker file with a splitter,
    splitter is on files
    """
    filename_1 = tmp_path / "file_pydra.txt"
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmp_path / "file_nice.txt"
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file",
            type=File,
            position=1,
            argstr="",
            help="input file",
        )
    ]

    docky = shell.define(cmd, inputs=inputs)(
        environment=Docker(image="busybox"),
        strip=True,
    )

    res = docky(split={"file": [str(filename_1), str(filename_2)]})
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@no_win
@need_docker
def test_docker_inputspec_state_1b(plugin, tmp_path):
    """a customised input definition for a docker file with a splitter,
    files from the input definition have the same path in the local os and the container,
    so hash is calculated and the test works fine
    """
    file_1 = tmp_path / "file_pydra.txt"
    file_2 = tmp_path / "file_nice.txt"
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file",
            type=File,
            position=1,
            argstr="",
            help="input file",
        )
    ]
    docky = shell.define(cmd, inputs=inputs)(
        environment=Docker(image="busybox"),
        strip=True,
    )

    res = docky(split={"file": [str(file_1), str(file_2)]})
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@no_win
@need_docker
def test_docker_wf_inputspec_1(plugin, tmp_path):
    """a customized input definition for workflow with docker tasks"""
    filename = tmp_path / "file_pydra.txt"
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file",
            type=File,
            position=1,
            argstr="",
            help="input file",
        )
    ]

    @workflow.define
    def Workflow(cmd, file):

        docky = workflow.add(
            shell.define(cmd, inputs=inputs)(
                file=file,
                environment=Docker(image="busybox"),
                strip=True,
            )
        )

        return docky.stdout

    wf = Workflow(cmd=cmd, file=filename)

    res = wf.result()
    assert res.output.out == "hello from pydra"


@no_win
@need_docker
def test_docker_wf_state_inputspec_1(plugin, tmp_path):
    """a customized input definition for workflow with docker tasks that has a state"""
    file_1 = tmp_path / "file_pydra.txt"
    file_2 = tmp_path / "file_nice.txt"
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file",
            type=File,
            position=1,
            argstr="",
            help="input file",
        )
    ]

    @workflow.define
    def Workflow(cmd, file):

        docky = workflow.add(
            shell.define(cmd, inputs=inputs)(
                environment=Docker(image="busybox"),
                file=file,
                strip=True,
            )
        )

        return docky.stdout

    wf = Workflow(cmd=cmd)

    res = wf(split={"file": [file_1, file_2]})

    assert res[0].output.out == "hello from pydra"
    assert res[1].output.out == "have a nice one"


@no_win
@need_docker
def test_docker_wf_ndst_inputspec_1(plugin, tmp_path):
    """a customized input definition for workflow with docker tasks with states"""
    file_1 = tmp_path / "file_pydra.txt"
    file_2 = tmp_path / "file_nice.txt"
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"

    inputs = [
        shell.arg(
            name="file",
            type=File,
            position=1,
            argstr="",
            help="input file",
        )
    ]

    @workflow.define
    def Workflow(cmd, file):

        docky = workflow.add(
            shell.define(cmd, inputs=inputs)(
                environment=Docker(image="busybox"),
                file=file,
                strip=True,
            )
        )

        return docky.stdout

    wf = Workflow(cmd=cmd)

    res = wf(split={"file": [str(file_1), str(file_2)]})
    assert res.output.out == ["hello from pydra", "have a nice one"]
