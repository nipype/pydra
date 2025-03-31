from pydra.engine.submitter import Submitter
from pydra.compose import shell, workflow
from fileformats.generic import File
from pydra.environments import singularity
from pydra.engine.tests.utils import need_singularity


@need_singularity
def test_singularity_1_nosubm(tmp_path):
    """simple command in a container, a default bindings and working directory is added
    no submitter
    """
    cmd = "pwd"
    image = "docker://alpine"
    Singu = shell.define(cmd)
    singu = Singu()
    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert "/mnt/pydra" in outputs.stdout
    assert outputs.return_code == 0


@need_singularity
def test_singularity_2_nosubm(tmp_path):
    """a command with arguments, cmd and args given as executable
    no submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "docker://alpine"
    Singu = shell.define(" ".join(cmd))
    singu = Singu()
    assert singu.cmdline == " ".join(cmd)

    outputs = singu(
        environment=singularity.Environment(image=image),
        cache_root=tmp_path,
    )
    assert outputs.stdout.strip() == " ".join(cmd[1:])
    assert outputs.return_code == 0


@need_singularity
def test_singularity_2(worker, tmp_path):
    """a command with arguments, cmd and args given as executable
    using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "docker://alpine"
    Singu = shell.define(" ".join(cmd))
    singu = Singu()

    assert singu.cmdline == " ".join(cmd)

    with Submitter(
        worker=worker,
        environment=singularity.Environment(image=image),
        cache_root=tmp_path,
    ) as sub:
        res = sub(singu)
    assert not res.errored, "\n".join(res.errors["error message"])
    assert res.outputs.stdout.strip() == " ".join(cmd[1:])
    assert res.outputs.return_code == 0


@need_singularity
def test_singularity_2a(worker, tmp_path):
    """a command with arguments, using executable and args
    using submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    image = "docker://alpine"
    Singu = shell.define(cmd_exec)
    singu = Singu(additional_args=cmd_args)
    assert singu.cmdline == f"{cmd_exec} {' '.join(cmd_args)}"

    with Submitter(
        worker="debug",
        environment=singularity.Environment(image=image),
        cache_root=tmp_path,
    ) as sub:
        res = sub(singu)

    assert not res.errored, "\n".join(res.errors["error message"])
    assert res.outputs.stdout.strip() == " ".join(cmd_args)
    assert res.outputs.return_code == 0


# tests with State


@need_singularity
def test_singularity_st_1(worker, tmp_path):
    """commands without arguments in container
    splitter = executable
    """
    cmd = ["whoami", "pwd", "ls"]
    image = "docker://alpine"
    Singu = shell.define("dummy")
    singu = Singu().split("executable", executable=cmd)

    outputs = singu(
        worker=worker,
        environment=singularity.Environment(image=image, xargs=["--fakeroot"]),
        cache_root=tmp_path,
    )
    assert outputs.stdout[0].strip() == "root"
    assert "/mnt/pydra" in outputs.stdout[1]
    assert outputs.stdout[2].strip() == "_job.pklz"
    assert outputs.return_code == [0, 0, 0]


# tests with customized output_spec


@need_singularity
def test_singularity_outputspec_1(worker, tmp_path):
    """
    customised output_spec, adding files to the output, providing specific pathname
    output_path is automatically added to the bindings
    """
    cmd = ["touch", "newfile_tmp.txt"]
    image = "docker://alpine"

    Singu = shell.define(
        " ".join(cmd),
        outputs=[
            shell.outarg(name="newfile", type=File, path_template="newfile_tmp.txt")
        ],
    )
    singu = Singu()

    with Submitter(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    ) as sub:
        res = sub(singu)

    assert not res.errored, "\n".join(res.errors["error message"])
    assert res.outputs.stdout == ""
    assert res.outputs.newfile.fspath.exists()


# tests with customised input_spec


@need_singularity
def test_singularity_inputspec_1(worker, tmp_path):
    """a simple customized input task for singularity task"""
    filename = str((tmp_path / "file_pydra.txt"))
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    image = "docker://alpine"

    Singu = shell.define(
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

    singu = Singu(file=filename)

    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert outputs.stdout.strip() == "hello from pydra"


@need_singularity
def test_singularity_inputspec_1a(worker, tmp_path):
    """a simple customized input task for singularity task
    a default value is used
    """
    filename = str((tmp_path / "file_pydra.txt"))
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    image = "docker://alpine"

    Singu = shell.define(
        cmd,
        inputs=[
            shell.arg(
                name="file",
                type=File,
                default=filename,
                position=1,
                argstr="",
                help="input file",
            )
        ],
    )
    singu = Singu(file=filename)

    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert outputs.stdout.strip() == "hello from pydra"


@need_singularity
def test_singularity_inputspec_2(worker, tmp_path):
    """a customized input task with two fields for singularity task"""
    filename_1 = tmp_path / "file_pydra.txt"
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")

    filename_2 = tmp_path / "file_nice.txt"
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    image = "docker://alpine"

    Singu = shell.define(
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

    singu = Singu(file1=filename_1)

    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert outputs.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_inputspec_2a_except(worker, tmp_path):
    """a customized input task with two fields
    first one uses a default, and second doesn't - raises a dataclass exception
    """
    filename_1 = tmp_path / "file_pydra.txt"
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmp_path / "file_nice.txt"
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    image = "docker://alpine"

    # the field with default value can't be before value without default
    Singu = shell.define(
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

    singu = Singu(file2=filename_2)
    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert outputs.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_inputspec_2a(worker, tmp_path):
    """a customized input task with two fields
    first one uses a default value,
    this is fine even if the second field is not using any defaults
    """
    filename_1 = tmp_path / "file_pydra.txt"
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmp_path / "file_nice.txt"
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    image = "docker://alpine"

    # if you want set default in the first field you can use default_value in metadata
    Singu = shell.define(
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

    singu = Singu(file2=filename_2)

    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert outputs.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_cmd_inputspec_copyfile_1(worker, tmp_path):
    """shelltask changes a file in place,
    adding copyfile=True to the file-input from input_spec
    hardlink or copy in the output_dir should be created
    """
    file = tmp_path / "file_pydra.txt"
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    image = "docker://alpine"

    @shell.define
    class Singu(shell.Task["Singu.Outputs"]):

        executable = ["sed", "-is", "s/hello/hi/"]

        orig_file: File = shell.arg(
            position=1,
            argstr="",
            help="orig file",
            copy_mode=File.CopyMode.copy,
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.outarg(
                path_template="{orig_file}.txt",  # FIXME: Shouldn't have to specify the extension
                help="output file",
            )

    singu = Singu(orig_file=file)

    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert outputs.stdout == ""
    assert outputs.out_file.fspath.exists()
    # the file is  copied, and than it is changed in place
    assert outputs.out_file.fspath.parent.parent == tmp_path
    with open(outputs.out_file) as f:
        assert "hi from pydra\n" == f.read()
    # the original file is unchanged
    with open(file) as f:
        assert "hello from pydra\n" == f.read()


@need_singularity
def test_singularity_inputspec_state_1(tmp_path):
    """a customised input task for a singularity file with a splitter,
    splitter is on files
    """
    filename_1 = tmp_path / "file_pydra.txt"
    with open(filename_1, "w") as f:
        f.write("hello from pydra\n")
    filename_2 = tmp_path / "file_nice.txt"
    with open(filename_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = [str(filename_1), str(filename_2)]
    image = "docker://alpine"

    Singu = shell.define(
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

    singu = Singu().split("file", file=filename)

    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert outputs.stdout[0].strip() == "hello from pydra"
    assert outputs.stdout[1].strip() == "have a nice one"


@need_singularity
def test_singularity_inputspec_state_1b(worker, tmp_path):
    """a customised input task for a singularity file with a splitter,
    files from the input task have the same path in the local os and the container,
    so hash is calculated and the test works fine
    """
    file_1 = tmp_path / "file_pydra.txt"
    file_2 = tmp_path / "file_nice.txt"
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = [str(file_1), str(file_2)]
    image = "docker://alpine"

    Singu = shell.define(
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

    singu = Singu().split("file", file=filename)

    outputs = singu(
        environment=singularity.Environment(image=image), cache_root=tmp_path
    )
    assert outputs.stdout[0].strip() == "hello from pydra"
    assert outputs.stdout[1].strip() == "have a nice one"


@need_singularity
def test_singularity_wf_inputspec_1(worker, tmp_path):
    """a customized input task for workflow with singularity tasks"""
    filename = tmp_path / "file_pydra.txt"
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    image = "docker://alpine"

    Singu = shell.define(
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
    def Workflow(cmd: str, file: File) -> str:
        singu = workflow.add(
            Singu(executable=cmd, file=file),
            environment=singularity.Environment(image=image),
        )
        return singu.stdout

    with Submitter(cache_root=tmp_path) as sub:
        res = sub(Workflow(cmd=cmd, file=filename))

    assert res.outputs.out.strip() == "hello from pydra"


@need_singularity
def test_singularity_wf_state_inputspec_1(worker, tmp_path):
    """a customized input task for workflow with singularity tasks that has a state"""
    file_1 = tmp_path / "file_pydra.txt"
    file_2 = tmp_path / "file_nice.txt"
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = [str(file_1), str(file_2)]
    image = "docker://alpine"

    Singu = shell.define(
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
    def Workflow(cmd: str, file: File) -> str:
        singu = workflow.add(
            Singu(executable=cmd, file=file),
            environment=singularity.Environment(image=image),
        )
        return singu.stdout

    wf = Workflow(cmd=cmd).split("file", file=filename)

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        res = sub(wf)

    assert [o.strip() for o in res.outputs.out] == [
        "hello from pydra",
        "have a nice one",
    ]


@need_singularity
def test_singularity_wf_ndst_inputspec_1(worker, tmp_path):
    """a customized input task for workflow with singularity tasks with states"""
    file_1 = tmp_path / "file_pydra.txt"
    file_2 = tmp_path / "file_nice.txt"
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd = "cat"
    filename = [str(file_1), str(file_2)]
    image = "docker://alpine"

    Singu = shell.define(
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
    def Workflow(cmd: str, files: list[File]) -> list[str]:
        singu = workflow.add(
            Singu(executable=cmd).split(file=files),
            environment=singularity.Environment(image=image),
        )
        return singu.stdout

    wf = Workflow(cmd=cmd, files=filename)

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        res = sub(wf)

    assert [o.strip() for o in res.outputs.out] == [
        "hello from pydra",
        "have a nice one",
    ]
