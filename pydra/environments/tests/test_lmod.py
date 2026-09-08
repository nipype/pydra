import os
from pathlib import Path
import pytest
from pydra.engine.submitter import Submitter
from pydra.compose import shell
from fileformats.text import TextFile
from pydra.environments.lmod import Lmod
from pydra.engine.job import Job
from pydra.engine.tests.utils import (
    no_win,
    need_lmod,
)


def create_command(name: str, src: str, commandspath: Path) -> Path:
    """Creates a bash command within the temp path and returns
    the parent directory"""
    cmd_dir = commandspath / name
    cmd_dir.mkdir(parents=True)
    filename = cmd_dir / name
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(src)
    os.chmod(filename, 0o755)
    return cmd_dir


def create_module(
    name: str, module_src: str, modulespath: Path, version: str = "1.0"
) -> str:
    """Creates a bash command and an associated module path"""

    module_dir = modulespath / name
    module_dir.mkdir(parents=True)

    with open(module_dir / f"{version}.lua", "w") as f:
        f.write(f"""
        -- -*- lua -*-
help([===[
----------------------------------
## {name}/{version} ##
A {name} command
----------------------------------
]===])""")
        f.write(module_src)
    return f"{name}/{version}"


@pytest.fixture()
def modulespath(tmp_path: Path) -> Path:
    modulespath = tmp_path / "modules"
    modulespath.mkdir()
    os.environ["MODULEPATH"] = str(tmp_path / "modules")
    return modulespath


@pytest.fixture()
def commandspath(tmp_path: Path) -> Path:
    commandspath = tmp_path / "commands"
    commandspath.mkdir()
    return commandspath


@pytest.fixture
def cache_root(tmp_path: Path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    return cache_root


@pytest.fixture
def foo_module(commandspath: Path, modulespath: Path) -> str:
    cmd_dir = create_command(
        "foo", "echo 'hello from pydra' >> output.txt\n", commandspath
    )
    return create_module("foo", f'prepend_path("PATH", "{str(cmd_dir)}")', modulespath)


@pytest.fixture
def bar_module(commandspath: Path, modulespath: Path) -> str:
    cmd_dir = create_command("bar", 'echo "BAR == $BAR" >> output.txt\n', commandspath)
    module_src = f"""
setenv("BAR", "BAR")
prepend_path("PATH", "{cmd_dir}")
    """
    return create_module("bar", module_src, modulespath)


@pytest.fixture
def baz_module(commandspath: Path, modulespath: Path) -> str:
    module_src = 'setenv("BAR", "BAZ")'
    return create_module("baz", module_src, modulespath)


@pytest.fixture
def always_error_module(commandspath: Path, modulespath: Path) -> str:
    cmd_dir = create_command(
        "always-error",
        """
echo 'Error message' >&2;
exit 1;
""",
        commandspath,
    )
    return create_module(
        "always-error", f'prepend_path("PATH", "{cmd_dir}")', modulespath
    )


@shell.define
class Foo(shell.Task["Foo.Outputs"]):
    executable = "foo"

    class Outputs(shell.Outputs):
        out_file: TextFile = shell.outarg(
            path_template="output.txt",
            help="output file",
        )


@shell.define
class Bar(shell.Task["Bar.Outputs"]):
    executable = "bar"

    class Outputs(shell.Outputs):
        out_file: TextFile = shell.outarg(
            path_template="output.txt",
            help="output file",
        )


@shell.define
class AlwaysError(shell.Task["AlwaysError.Outputs"]):
    executable = "always-error"

    class Outputs(shell.Outputs):
        pass


@no_win
@need_lmod
def test_module_load_unload1(foo_module: str, tmp_path: Path):
    foo = Foo()
    with pytest.raises(FileNotFoundError, match="No such file or directory: 'foo'"):
        foo(cache_root=tmp_path / "1")
    outputs = foo(environment=Lmod(foo_module), cache_root=tmp_path / "2")
    assert outputs.out_file.contents == "hello from pydra\n"
    with pytest.raises(FileNotFoundError, match="No such file or directory: 'foo'"):
        foo(cache_root=tmp_path / "3")


@no_win
@need_lmod
def test_modules_submitter(foo_module: str, cache_root: Path):
    foo = Foo()
    foo_task = Job(
        task=foo,
        name="foo",
        submitter=Submitter(environment=Lmod(foo_module), cache_root=cache_root),
    )
    res = foo_task.run()
    assert res.outputs.out_file.contents == "hello from pydra\n"


@no_win
@need_lmod
def test_double_modules(bar_module: str, baz_module: str, tmp_path: Path):
    bar = Bar()
    outputs = bar(environment=Lmod(bar_module), cache_root=tmp_path / "1")
    assert outputs.out_file.contents == "BAR == BAR\n"
    outputs = bar(environment=Lmod([bar_module, baz_module]), cache_root=tmp_path / "2")
    assert outputs.out_file.contents == "BAR == BAZ\n"


@no_win
@need_lmod
def test_lmod_module_validator_fail1() -> None:
    with pytest.raises(ValueError, match="At least one module"):
        Lmod([])


@no_win
@need_lmod
def test_lmod_module_validator_fail2() -> None:
    with pytest.raises(ValueError, match="All module names must be strings"):
        assert Lmod(1)


@no_win
@need_lmod
def test_always_error(always_error_module: str):
    always_error = AlwaysError()
    with pytest.raises(RuntimeError, match="Error running 'main'"):
        always_error(environment=Lmod(always_error_module))
