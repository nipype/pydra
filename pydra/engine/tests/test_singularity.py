import shutil
import subprocess as sp
import pytest
import attr

from ..task import ShellTask
from ..submitter import Submitter
from ..specs import ShellOutputs, File, ShellDef
from ..environments import Singularity


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
    singu = ShellTask(
        name="singu",
        executable=cmd,
        environment=Singularity(image=image),
        cache_dir=tmp_path,
    )
    assert singu.environment.image == "docker://alpine"
    assert isinstance(singu.environment, Singularity)
    assert singu.cmdline == cmd

    res = singu()
    assert "/mnt/pydra" in res.output.stdout
    assert res.output.return_code == 0


@need_singularity
def test_singularity_2_nosubm(tmp_path):
    """a command with arguments, cmd and args given as executable
    no submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "docker://alpine"
    singu = ShellTask(
        name="singu",
        executable=cmd,
        environment=Singularity(image=image),
        cache_dir=tmp_path,
    )
    assert singu.cmdline == " ".join(cmd)

    res = singu()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0


@need_singularity
def test_singularity_2(plugin, tmp_path):
    """a command with arguments, cmd and args given as executable
    using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    image = "docker://alpine"

    singu = ShellTask(
        name="singu",
        executable=cmd,
        environment=Singularity(image=image),
        cache_dir=tmp_path,
    )
    assert singu.cmdline == " ".join(cmd)

    with Submitter(plugin=plugin) as sub:
        singu(submitter=sub)
    res = singu.result()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0


@need_singularity
def test_singularity_2a(plugin, tmp_path):
    """a command with arguments, using executable and args
    using submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    image = "docker://alpine"
    singu = ShellTask(
        name="singu",
        executable=cmd_exec,
        args=cmd_args,
        environment=Singularity(image=image),
        cache_dir=tmp_path,
    )
    assert singu.cmdline == f"{cmd_exec} {' '.join(cmd_args)}"

    with Submitter(plugin=plugin) as sub:
        singu(submitter=sub)
    res = singu.result()
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0


# tests with State


@need_singularity
def test_singularity_st_1(plugin, tmp_path):
    """commands without arguments in container
    splitter = executable
    """
    cmd = ["pwd", "ls"]
    image = "docker://alpine"
    singu = ShellTask(
        name="singu", environment=Singularity(image=image), cache_dir=tmp_path
    ).split("executable", executable=cmd)
    assert singu.state.splitter == "singu.executable"

    res = singu(plugin=plugin)
    assert "/mnt/pydra" in res[0].output.stdout
    assert res[1].output.stdout == ""
    assert res[0].output.return_code == res[1].output.return_code == 0


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
    singu = ShellTask(
        name="singu",
        executable="echo",
        environment=Singularity(image=image),
        cache_dir=tmp_path,
    ).split("args", args=args_n)
    assert singu.state.splitter == "singu.args"
    res = singu(plugin="slurm")
    assert "1" in res[1].output.stdout
    assert str(n - 1) in res[-1].output.stdout
    assert res[0].output.return_code == res[1].output.return_code == 0


# tests with customized output_spec


@need_singularity
def test_singularity_outputspec_1(plugin, tmp_path):
    """
    customised output_spec, adding files to the output, providing specific pathname
    output_path is automatically added to the bindings
    """
    cmd = ["touch", "newfile_tmp.txt"]
    image = "docker://alpine"

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp.txt")],
        bases=(ShellOutputs,),
    )
    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        output_spec=my_output_spec,
        cache_dir=tmp_path,
    )

    with Submitter(plugin=plugin) as sub:
        singu(submitter=sub)

    res = singu.result()
    assert res.output.stdout == ""
    assert res.output.newfile.fspath.exists()


# tests with customised input_spec


@need_singularity
def test_singularity_inputspec_1(plugin, tmp_path):
    """a simple customized input definition for singularity task"""
    filename = str((tmp_path / "file_pydra.txt"))
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    image = "docker://alpine"

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
                        "argstr": "",
                        "help": "input file",
                    },
                ),
            )
        ],
        bases=(ShellDef,),
    )

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        file=filename,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmp_path,
    )

    res = singu()
    assert res.output.stdout == "hello from pydra"


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

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    default=filename,
                    metadata={"position": 1, "argstr": "", "help": "input file"},
                ),
            )
        ],
        bases=(ShellDef,),
    )

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmp_path,
    )

    res = singu()
    assert res.output.stdout == "hello from pydra"


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

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help": "input file 1",
                    },
                ),
            ),
            (
                "file2",
                attr.ib(
                    type=File,
                    default=filename_2,
                    metadata={
                        "position": 2,
                        "argstr": "",
                        "help": "input file 2",
                    },
                ),
            ),
        ],
        bases=(ShellDef,),
    )

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        file1=filename_1,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmp_path,
    )

    res = singu()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


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
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                attr.ib(
                    type=File,
                    default=filename_1,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help": "input file 1",
                    },
                ),
            ),
            (
                "file2",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 2,
                        "argstr": "",
                        "help": "input file 2",
                    },
                ),
            ),
        ],
        bases=(ShellDef,),
    )

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        file2=filename_2,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmp_path,
    )
    res = singu()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


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
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                attr.ib(
                    type=File,
                    default=filename_1,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help": "input file 1",
                    },
                ),
            ),
            (
                "file2",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 2,
                        "argstr": "",
                        "help": "input file 2",
                    },
                ),
            ),
        ],
        bases=(ShellDef,),
    )

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        file2=filename_2,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmp_path,
    )

    res = singu()
    assert res.output.stdout == "hello from pydra\nhave a nice one"


@need_singularity
def test_singularity_cmd_inputspec_copyfile_1(plugin, tmp_path):
    """shelltask changes a file in place,
    adding copyfile=True to the file-input from input_spec
    hardlink or copy in the output_dir should be created
    """
    file = tmp_path / "file_pydra.txt"
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]
    image = "docker://alpine"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help": "orig file",
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
                        "help": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellDef,),
    )

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        input_spec=my_input_spec,
        orig_file=str(file),
        cache_dir=tmp_path,
    )

    res = singu()
    assert res.output.stdout == ""
    assert res.output.out_file.fspath.exists()
    # the file is  copied, and than it is changed in place
    assert res.output.out_file.fspath.parent == singu.output_dir
    with open(res.output.out_file) as f:
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
                        "argstr": "",
                        "help": "input file",
                    },
                ),
            )
        ],
        bases=(ShellDef,),
    )

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmp_path,
    ).split("file", file=filename)

    res = singu()
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


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
                        "argstr": "",
                        "help": "input file",
                    },
                ),
            )
        ],
        bases=(ShellDef,),
    )

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=cmd,
        input_spec=my_input_spec,
        strip=True,
        cache_dir=tmp_path,
    ).split("file", file=filename)

    res = singu()
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@need_singularity
def test_singularity_wf_inputspec_1(plugin, tmp_path):
    """a customized input definition for workflow with singularity tasks"""
    filename = tmp_path / "file_pydra.txt"
    with open(filename, "w") as f:
        f.write("hello from pydra")

    cmd = "cat"
    image = "docker://alpine"

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
                        "argstr": "",
                        "help": "input file",
                    },
                ),
            )
        ],
        bases=(ShellDef,),
    )

    wf = Workflow(name="wf", input_spec=["cmd", "file"], cache_dir=tmp_path)
    wf.inputs.cmd = cmd
    wf.inputs.file = filename

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=wf.lzin.cmd,
        file=wf.lzin.file,
        input_spec=my_input_spec,
        strip=True,
    )
    wf.add(singu)

    wf.set_output([("out", wf.singu.lzout.stdout)])

    with Submitter(plugin="serial") as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == "hello from pydra"


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
                        "argstr": "",
                        "help": "input file",
                    },
                ),
            )
        ],
        bases=(ShellDef,),
    )

    wf = Workflow(name="wf", input_spec=["cmd", "file"], cache_dir=tmp_path)
    wf.inputs.cmd = cmd

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=wf.lzin.cmd,
        file=wf.lzin.file,
        input_spec=my_input_spec,
        strip=True,
    )
    wf.add(singu)
    wf.split("file", file=filename)

    wf.set_output([("out", wf.singu.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res[0].output.out == "hello from pydra"
    assert res[1].output.out == "have a nice one"


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
                        "argstr": "",
                        "help": "input file",
                    },
                ),
            )
        ],
        bases=(ShellDef,),
    )

    wf = Workflow(name="wf", input_spec=["cmd", "file"], cache_dir=tmp_path)
    wf.inputs.cmd = cmd
    wf.inputs.file = filename

    singu = ShellTask(
        name="singu",
        environment=Singularity(image=image),
        executable=wf.lzin.cmd,
        input_spec=my_input_spec,
        strip=True,
    ).split("file", file=wf.lzin.file)
    wf.add(singu)

    wf.set_output([("out", wf.singu.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == ["hello from pydra", "have a nice one"]
