import pytest
from pydra.engine.submitter import Submitter
from pydra.engine.specs import ShellDef, ShellOutputs
from fileformats.generic import File
from pydra.engine.environments import Docker
from pydra.design import shell, workflow
from pydra.engine.core import Task
from .utils import no_win, need_docker, run_submitter, run_no_submitter


@no_win
@need_docker
def test_docker_1_nosubm():
    """simple command in a container, a default bindings and working directory is added
    no submitter
    """
    cmd = "whoami"
    Docky = shell.define(cmd)
    docky = Docky()
    docky_task = Task(
        definition=docky,
        name="docky",
        submitter=Submitter(environment=Docker(image="busybox")),
    )
    assert docky_task.environment.image == "busybox"
    assert docky_task.environment.tag == "latest"
    assert isinstance(docky_task.environment, Docker)
    assert docky.cmdline == cmd

    res = docky_task.run()
    assert res.outputs.stdout == "root\n"
    assert res.outputs.return_code == 0


@no_win
@need_docker
def test_docker_1(plugin):
    """simple command in a container, a default bindings and working directory is added
    using submitter
    """
    cmd = "whoami"
    Docky = shell.define(cmd)
    docky = Docky()

    with Submitter(environment=Docker(image="busybox")) as sub:
        res = sub(docky)

    assert res.outputs.stdout == "root\n"
    assert res.outputs.return_code == 0


@no_win
@need_docker
@pytest.mark.parametrize("run_function", [run_no_submitter, run_submitter])
def test_docker_2(run_function, plugin, tmp_path):
    """a command with arguments, cmd and args given as executable
    with and without submitter
    """
    cmdline = "echo hail pydra"
    Docky = shell.define(cmdline)
    docky = Docky()
    # cmdline doesn't know anything about docker
    assert docky.cmdline == cmdline
    outputs = run_function(docky, tmp_path, plugin, environment=Docker(image="busybox"))
    assert outputs.stdout.strip() == " ".join(cmdline.split()[1:])
    assert outputs.return_code == 0


@no_win
@need_docker
@pytest.mark.parametrize("run_function", [run_no_submitter, run_submitter])
def test_docker_2a(run_function, plugin, tmp_path):
    """a command with arguments, using executable and args
    using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    # separate command into exec + args
    Docky = shell.define(cmd)
    docky = Docky()
    assert docky.executable == cmd
    assert docky.cmdline == " ".join(cmd)

    outputs = run_function(docky, tmp_path, plugin, environment=Docker(image="busybox"))
    assert outputs.stdout.strip() == " ".join(cmd[1:])
    assert outputs.return_code == 0


# tests with State


@no_win
@need_docker
@pytest.mark.parametrize("run_function", [run_no_submitter, run_submitter])
def test_docker_st_1(run_function, plugin, tmp_path):
    """commands without arguments in container
    splitter = executable
    """
    cmd = ["pwd", "whoami"]
    Docky = shell.define("docky")  # cmd is just a placeholder
    docky = Docky().split(executable=cmd)

    outputs = run_function(docky, tmp_path, plugin, environment=Docker(image="busybox"))
    assert outputs.stdout[0] == f"/mnt/pydra{docky.output_dir[0]}\n"
    assert outputs.stdout[1] == "root\n"
    assert outputs.return_code[0] == outputs.return_code[1] == 0


# tests with customized output_spec


@no_win
@need_docker
def test_docker_outputspec_1(plugin, tmp_path):
    """
    customised output_spec, adding files to the output, providing specific pathname
    output_path is automatically added to the bindings
    """
    outputs = [shell.out(name="newfile", type=File, help="new file")]
    Docky = shell.define("touch newfile_tmp.txt", outputs=outputs)
    docky = Docky()

    outputs = docky(plugin=plugin, environment=Docker(image="ubuntu"))
    assert outputs.stdout == ""


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

    outputs = docky()
    assert outputs.stdout.strip() == "hello from pydra"


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

    outputs = docky()
    assert outputs.stdout.strip() == "hello from pydra"


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

    Docky = shell.define(
        cmd,
        inputs=[
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
        ],
    )
    docky = Docky(
        file1=filename_1,
    )

    outputs = docky(
        name="docky",
        environment=Docker(image="busybox"),
    )
    assert outputs.stdout.strip() == "hello from pydra\nhave a nice one"


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

    Docky = shell.define(
        cmd,
        inputs=[
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
        ],
    )

    docky = Docky(
        file2=filename_2,
    )
    assert docky.file2.fspath == filename_2

    outputs = docky(environment=Docker(image="busybox"))
    assert outputs.stdout.strip() == "hello from pydra\nhave a nice one"


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

    Docky = shell.define(
        cmd,
        inputs=[
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
        ],
    )

    docky = Docky(file2=filename_2)

    outputs = docky(environment=Docker(image="busybox"))
    assert outputs.stdout.strip() == "hello from pydra\nhave a nice one"


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
    outputs = docky()
    assert "docker" in outputs.stdout
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

    @shell.define
    class Docky(ShellDef["Docky.Outputs"]):
        executable = ["sed", "-is", "s/hello/hi/"]
        orig_file: File = shell.arg(
            position=1,
            argstr="",
            help="orig file",
            copy_mode="copy",
        )

        class Outputs(ShellOutputs):
            out_file: File = shell.outarg(
                path_template="{orig_file}.txt",
                help="output file",
            )

    docky = Docky(orig_file=str(file))

    outputs = docky(environment=Docker(image="busybox"), cache_dir=tmp_path)
    assert outputs.stdout == ""
    out_file = outputs.out_file.fspath
    assert out_file.exists()
    # the file is copied, and then it is changed in place
    assert out_file.parent.parent == tmp_path
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

    Docky = shell.define(
        cmd,
        inputs=[
            shell.arg(
                name="file",
                type=File,
                position=1,
                argstr="",
                help="input file",
            )
        ],
    )

    docky = Docky().split(file=[str(filename_1), str(filename_2)])

    outputs = docky(environment=Docker(image="busybox"))
    assert outputs.stdout[0].strip() == "hello from pydra"
    assert outputs.stdout[1].strip() == "have a nice one"


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

    Docky = shell.define(
        cmd,
        inputs=[
            shell.arg(
                name="file",
                type=File,
                position=1,
                argstr="",
                help="input file",
            )
        ],
    )
    docky = Docky().split(file=[str(file_1), str(file_2)])

    outputs = docky(environment=Docker(image="busybox"))
    assert outputs.stdout[0].strip() == "hello from pydra"
    assert outputs.stdout[1].strip() == "have a nice one"


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

    outputs = wf()
    assert outputs.ou.strip() == "hello from pydra"


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

    Docky = shell.define(
        cmd,
        inputs=[
            shell.arg(
                name="file",
                type=File,
                position=1,
                argstr="",
                help="input file",
            )
        ],
    )

    @workflow.define
    def Workflow(cmd, file):

        docky = workflow.add(
            Docky(file=file),
            environment=Docker(image="busybox"),
        )

        return docky.stdout

    wf = Workflow(cmd=cmd)

    outputs = wf(split={"file": [file_1, file_2]})

    assert outputs.out[0].strip() == "hello from pydra"
    assert outputs.out[1].strip() == "have a nice one"


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

    outputs = wf(split={"file": [str(file_1), str(file_2)]})
    assert outputs.out == ["hello from pydra", "have a nice one"]
