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
