import shutil
import subprocess as sp
import pytest
from ..submitter import Submitter
from pydra.design import shell, workflow
from fileformats.generic import File
from pydra.engine.environments import Singularity


need_docker = pytest.mark.skipif(
    shutil.which("docker") is None or sp.call(["docker", "info"]),
    reason="no docker available",
)
need_singularity = pytest.mark.skipif(
    shutil.which("singularity") is None, reason="no singularity available"
)

need_slurm = pytest.mark.skipif(
    not bool(shutil.which("sbatch")), reason="no singularity available"
)


@need_singularity
def test_singularity_1_nosubm(tmp_path):
    """simple command in a container, a default bindings and working directory is added
    no submitter
    """
    cmd = "pwd"
    image = "docker://alpine"
    Singu = shell.define(cmd)
    singu = Singu()
    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
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
        environment=Singularity(image=image),
        cache_dir=tmp_path,
    )
    assert outputs.stdout.strip() == " ".join(cmd[1:])
    assert outputs.return_code == 0


@need_singularity
def test_singularity_2(plugin, tmp_path):
    """a command with arguments, cmd and args given as executable
    using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "docker://alpine"
    Singu = shell.define(" ".join(cmd))
    singu = Singu()

    assert singu.cmdline == " ".join(cmd)

    with Submitter(
        worker=plugin, environment=Singularity(image=image), cache_dir=tmp_path
    ) as sub:
        res = sub(singu)
    assert res.outputs.stdout.strip() == " ".join(cmd[1:])
    assert res.outputs.return_code == 0


@need_singularity
def test_singularity_2a(plugin, tmp_path):
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
        worker=plugin, environment=Singularity(image=image), cache_dir=tmp_path
    ) as sub:
        res = sub(singu)

    assert res.outputs.stdout.strip() == " ".join(cmd_args)
    assert res.outputs.return_code == 0


# tests with State


@need_singularity
def test_singularity_st_1(plugin, tmp_path):
    """commands without arguments in container
    splitter = executable
    """
    cmd = ["whoami", "pwd", "ls"]
    image = "docker://alpine"
    Singu = shell.define("dummy")
    singu = Singu().split("executable", executable=cmd)

    outputs = singu(
        plugin=plugin, environment=Singularity(image=image), cache_dir=tmp_path
    )
    assert outputs.stdout[0] == "root"
    assert outputs.stdout[1] == "/mnt/pydra"
    assert outputs.stdout[2] == ""
    assert outputs.return_code == [0, 0, 0]


@need_singularity
@need_slurm
@pytest.mark.skip(reason="TODO, xfail incorrect")
@pytest.mark.xfail(
    reason="slurm can complain if the number of submitted jobs exceeds the limit"
)
@pytest.mark.parametrize("n", [10, 50, 100])
def test_singularity_st_2(tmp_path, n):
    """splitter over args (checking bigger splitters if slurm available)"""
    args_n = list(range(n))
    image = "docker://alpine"
    Singu = shell.define("echo")
    singu = Singu().split("args", args=args_n)
    with Submitter(
        plugin="slurm", environment=Singularity(image=image), cache_dir=tmp_path
    ) as sub:
        res = sub(singu)

    assert "1" in res.outputs.stdout[1]
    assert str(n - 1) in res.outputs.stdout[-1]
    assert res.outputs.return_code[0] == res.outputs.return_code[1] == 0


# tests with customized output_spec


@need_singularity
def test_singularity_outputspec_1(plugin, tmp_path):
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
        worker=plugin, environment=Singularity(image=image), cache_dir=tmp_path
    ) as sub:
        res = sub(singu)

    assert res.outputs.stdout == ""
    assert res.outputs.newfile.fspath.exists()


# tests with customised input_spec


@need_singularity
def test_singularity_inputspec_1(plugin, tmp_path):
    """a simple customized input definition for singularity task"""
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

    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
    assert outputs.stdout.strip() == "hello from pydra"


@need_singularity
def test_singularity_inputspec_1a(plugin, tmp_path):
    """a simple customized input definition for singularity task
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

    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
    assert outputs.stdout.strip() == "hello from pydra"


@need_singularity
def test_singularity_inputspec_2(plugin, tmp_path):
    """a customized input definition with two fields for singularity task"""
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

    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
    assert outputs.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_inputspec_2a_except(plugin, tmp_path):
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
    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
    assert outputs.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_inputspec_2a(plugin, tmp_path):
    """a customized input definition with two fields
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

    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
    assert outputs.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_cmd_inputspec_copyfile_1(plugin, tmp_path):
    """shelltask changes a file in place,
    adding copyfile=True to the file-input from input_spec
    hardlink or copy in the output_dir should be created
    """
    file = tmp_path / "file_pydra.txt"
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = "sed -is 's/hello/hi/'"
    image = "docker://alpine"

    Singu = shell.define(
        cmd,
        inputs=[
            shell.arg(
                name="orig_file",
                type=File,
                position=1,
                argstr="",
                help="orig file",
                copy_mode=File.CopyMode.copy,
            ),
        ],
        outputs=[
            shell.outarg(
                name="out_file",
                type=File,
                path_template="{orig_file}",
                help="output file",
            ),
        ],
    )

    singu = Singu(orig_file=str(file))

    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
    assert outputs.stdout == ""
    assert outputs.out_file.fspath.exists()
    # the file is  copied, and than it is changed in place
    assert outputs.out_file.fspath.parent == singu.output_dir
    with open(outputs.out_file) as f:
        assert "hi from pydra\n" == f.read()
    # the original file is unchanged
    with open(file) as f:
        assert "hello from pydra\n" == f.read()


@need_singularity
def test_singularity_inputspec_state_1(tmp_path):
    """a customised input definition for a singularity file with a splitter,
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

    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
    assert outputs.stdout[0].strip() == "hello from pydra"
    assert outputs.stdout[1].strip() == "have a nice one"


@need_singularity
def test_singularity_inputspec_state_1b(plugin, tmp_path):
    """a customised input definition for a singularity file with a splitter,
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

    outputs = singu(environment=Singularity(image=image), cache_dir=tmp_path)
    assert outputs.stdout[0].strip() == "hello from pydra"
    assert outputs.stdout[1].strip() == "have a nice one"


@need_singularity
def test_singularity_wf_inputspec_1(plugin, tmp_path):
    """a customized input definition for workflow with singularity tasks"""
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
            Singu(executable=cmd, file=file), environment=Singularity(image=image)
        )
        return singu.stdout

    with Submitter(cache_dir=tmp_path) as sub:
        res = sub(Workflow(cmd=cmd, file=filename))

    assert res.outputs.out.strip() == "hello from pydra"


@need_singularity
def test_singularity_wf_state_inputspec_1(plugin, tmp_path):
    """a customized input definition for workflow with singularity tasks that has a state"""
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
            environment=Singularity(image=image),
        )
        return singu.stdout

    wf = Workflow(cmd=cmd).split("file", file=filename)

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        res = sub(wf)

    assert [o.strip() for o in res.outputs.out] == [
        "hello from pydra",
        "have a nice one",
    ]


@need_singularity
def test_singularity_wf_ndst_inputspec_1(plugin, tmp_path):
    """a customized input definition for workflow with singularity tasks with states"""
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
            environment=Singularity(image=image),
        )
        return singu.stdout

    wf = Workflow(cmd=cmd, files=filename)

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        res = sub(wf)

    assert [o.strip() for o in res.outputs.out] == [
        "hello from pydra",
        "have a nice one",
    ]
