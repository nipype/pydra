import importlib
import json
from pathlib import Path
import pdb
import sys

import click
import cloudpickle as cp
from pydra.utils.hash import code_digest

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
ExistingFilePath = click.Path(exists=True, dir_okay=False, resolve_path=True)


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("crashfile", type=ExistingFilePath)
@click.option(
    "-r", "--rerun", is_flag=True, flag_value=True, help="Rerun crashed code."
)
@click.option(
    "-d",
    "--debugger",
    type=click.Choice([None, "ipython", "pdb"]),
    help="Debugger to use when rerunning",
)
def crash(crashfile, rerun, debugger=None):
    """Display a crash file and rerun if required."""
    if crashfile.endswith(("pkl", "pklz")):
        with open(crashfile, "rb") as f:
            crash_content = cp.load(f)
        print("".join(crash_content["error message"]))

        if rerun:
            jobfile = Path(crashfile).parent / "_job.pklz"
            if jobfile.exists():
                with open(jobfile, "rb") as f:
                    job_obj = cp.load(f)

                if debugger == "ipython":
                    try:
                        from IPython.core import ultratb

                        sys.excepthook = ultratb.FormattedTB(
                            mode="Verbose", theme_name="Linux", call_pdb=True
                        )
                    except ImportError:
                        raise ImportError(
                            "'Ipython' needs to be installed to use the 'ipython' debugger"
                        )

                try:
                    job_obj.run(rerun=True)
                except Exception:  # noqa: E722
                    if debugger == "pdb":
                        pdb.post_mortem()
                    elif debugger == "ipython":
                        raise
            else:
                raise FileNotFoundError(f"Job file {jobfile} not found")
    else:
        raise ValueError("Only pickled crashfiles are supported")


def _import(name: str):
    """The object at the importable dotted ``module.name``"""
    module, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def check_versions(table: dict[str, dict[str, str]]) -> tuple[list[str], dict]:
    """Compare each task of ``table`` (``{"module.Task": {"version", "digest"}}``) with
    its declared version and `pydra.utils.hash.code_digest`. Returns the problems and
    the table with the rows of tasks whose version was bumped (or that were not
    versioned before) set to their current version and digest. A digest that moved
    while the version did not is a problem that no update fixes."""
    problems, updated = [], {}
    for name, row in sorted(table.items()):
        task = _import(name)
        version = getattr(task, "_version", None)
        if version is None:
            problems.append(f"{name} declares no version")
            updated[name] = row
            continue
        digest = code_digest(task)
        current = {"version": version, "digest": digest}
        if row.get("version") != version:
            problems.append(
                f"{name}: version {version!r} is not in the table "
                f"(it has {row.get('version')!r}); update the table"
            )
            updated[name] = current
        elif row.get("digest") != digest:
            problems.append(
                f"{name}: its code changed under version {version!r} (digest "
                f"{row.get('digest')} in the table, {digest} now); bump version= or "
                "remove it"
            )
            updated[name] = row
        else:
            updated[name] = row
    return problems, updated


@cli.command("check-versions", context_settings=CONTEXT_SETTINGS)
@click.argument("table", type=ExistingFilePath)
@click.option(
    "--update",
    is_flag=True,
    help="Write the current version and digest of tasks whose version was bumped.",
)
def check_versions_command(table, update):
    """Check that the code of each task in TABLE, a JSON file of
    ``{"module.Task": {"version": ..., "digest": ...}}``, is the code its declared
    version was recorded with; exits with 1 if any is not."""
    path = Path(table)
    rows = json.loads(path.read_text())
    problems, updated = check_versions(rows)
    if update:
        path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
        problems, _ = check_versions(updated)
    for problem in problems:
        click.echo(problem, err=True)
    sys.exit(1 if problems else 0)


_MISSING = "(missing)"
_WALK_HINT = (
    " (the two sites classify code differently: check PYDRA_CODE_HASH and "
    "PYDRA_USER_PACKAGES on both)"
)


def _flatten(value, prefix=""):
    if isinstance(value, dict) and (value or not prefix):
        for key in sorted(value):
            yield from _flatten(value[key], f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, value


def explain_records(one: dict, two: dict) -> list[str]:
    """The fields where two identity records differ, one ``"<field>: <one> -> <two>"``
    line each, with the dotted path of a nested field. A difference in ``walk`` names
    the settings that cause it."""
    left, right = dict(_flatten(one)), dict(_flatten(two))
    lines = []
    for path in sorted(left.keys() | right.keys()):
        a, b = left.get(path, _MISSING), right.get(path, _MISSING)
        if a == b:
            continue
        shown_a = a if a == _MISSING else json.dumps(a)
        shown_b = b if b == _MISSING else json.dumps(b)
        lines.append(
            f"{path}: {shown_a} -> {shown_b}"
            + (_WALK_HINT if path.startswith("walk.") else "")
        )
    return lines


def load_record(path: str | Path) -> dict:
    """The identity record at ``path``: a JSON file, or the cache directory of a job whose
    ``_result.pklz`` holds it (`pydra.engine.result.Result.record`)"""
    path = Path(path)
    if path.is_dir():
        result_file = path / "_result.pklz"
        if not result_file.exists():
            raise click.ClickException(
                f"{path} has no _result.pklz: only a finished job holds its identity record"
            )
        with open(result_file, "rb") as f:
            record = getattr(cp.load(f), "record", None)
        if record is None:
            raise click.ClickException(
                f"{path} holds no identity record (its result was written before "
                "pydra-identity@2)"
            )
        return record
    return json.loads(path.read_text())


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("first", type=click.Path(exists=True, resolve_path=True))
@click.argument("second", type=click.Path(exists=True, resolve_path=True))
def explain(first, second):
    """Show the fields where the identity records of two jobs differ. FIRST and SECOND
    are each a JSON record file or the cache directory of a finished job; exits with 1
    if the records differ."""
    lines = explain_records(load_record(first), load_record(second))
    if not lines:
        click.echo("the records are identical")
    for line in lines:
        click.echo(line)
    sys.exit(1 if lines else 0)


def _task_class(spec: str):
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise click.BadParameter(f"{spec!r} is not <module:task>")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise click.ClickException(f"cannot import {spec}: {e}")


@cli.command("check-reach", context_settings=CONTEXT_SETTINGS)
@click.argument("task")
@click.argument("inputs", default="{}")
def check_reach_command(task, inputs):
    """Run the function of TASK (``module:task``) once with INPUTS (a JSON object, default
    ``{}``) and list the user functions that ran but that the code walk did not reach,
    which are where ``getattr`` or ``importlib`` hides a dependency; exits with 1 if
    there are any. A development tool: it never runs during a workflow."""
    from pydra.utils.reach import reach_report

    try:
        values = json.loads(inputs)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"INPUTS is not JSON: {e}")
    if not isinstance(values, dict):
        raise click.BadParameter("INPUTS must be a JSON object")
    try:
        missed, by_name = reach_report(_task_class(task), values)
    except Exception as e:
        raise click.ClickException(f"{task}: {type(e).__name__}: {e}")
    for line in missed:
        click.echo(line)
    if by_name:
        click.echo("keyed by name at this code_hash tier, not by code:", err=True)
        for line in by_name:
            click.echo(f"  {line}", err=True)
    if not missed:
        click.echo("the walk reached every user function that ran")
    sys.exit(1 if missed else 0)
