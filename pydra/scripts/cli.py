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
    if crashfile.endswith((".json", "pkl", "pklz")):
        if crashfile.endswith(".json"):  # a work directory's _error.json, no pickle
            with open(crashfile, "r", encoding="utf-8") as f:
                crash_content = json.load(f)
        else:
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
        raise ValueError("Only _error.json and pickled crashfiles are supported")


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


def _triple(item) -> str:
    """``field[index]: cause`` of one absence triple of a key record, or its JSON when it
    is not ``[field, {"$int": index} or index, cause]``"""
    if isinstance(item, list) and len(item) == 3:
        field, index, cause = item
        if isinstance(index, dict) and set(index) == {"$int"}:
            index = index["$int"]
        return f"{field}[{index}]: {cause}"
    return json.dumps(item)


def _shown(record: dict) -> dict:
    absent = record.get("absent")
    if not isinstance(absent, list):
        return record
    return {**record, "absent": [_triple(t) for t in absent]}


def explain_records(one: dict, two: dict) -> list[str]:
    """The fields where two identity records differ, one ``"<field>: <one> -> <two>"``
    line each, with the dotted path of a nested field. A difference in ``walk`` names
    the settings that cause it. Absence triples (``absent``) are shown as
    ``field[index]: cause``."""
    left, right = dict(_flatten(_shown(one))), dict(_flatten(_shown(two)))
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
    """The identity record at ``path``: a JSON file, a portable store entry (the key of its
    ``entry.json``), or the cache directory of a job whose ``_result.pklz`` holds it
    (`pydra.engine.result.Result.record`)"""
    path = Path(path)
    if path.is_dir() and (path / "entry.json").is_file():
        record = json.loads((path / "entry.json").read_text()).get("key")
        if not isinstance(record, dict):
            raise click.ClickException(
                f"{path} is an entry with no key record (one written before "
                "pydra-identity@2)"
            )
        return record
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


@cli.group(context_settings=CONTEXT_SETTINGS)
def store():
    """Maintain a portable result store."""


def _no_work_dir(checksum: str, fingerprint: str, attempt: str) -> Path:
    raise ValueError("collect does not run jobs, so it names no work directory")


def _hook(name: str):
    """The object at the importable dotted ``name``; a class is instantiated with no
    arguments"""
    obj = _import(name)
    return obj() if isinstance(obj, type) else obj


@store.command("collect", context_settings=CONTEXT_SETTINGS)
@click.argument(
    "root", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    "--trust",
    type=click.Choice(["private", "trusted", "shared"]),
    default="private",
    show_default=True,
    help="What you assert about who can write the root (RFC 0003 section 6.4)",
)
def store_collect(root, trust):
    """Remove what the portable store at ROOT keeps only for a while: staging and
    quarantined entries, and default work directories, past their ages."""
    from pydra.engine.portable.store import PortableStore

    kw = {"work_dir": _no_work_dir} if trust == "shared" else {}
    report = PortableStore(root, trust=trust, **kw).collect()
    for path in report.removed:
        click.echo(f"removed {path}")
    for path, why in report.kept:
        click.echo(f"kept {path}: {why}")
    click.echo(f"{len(report.removed)} removed, {len(report.kept)} kept")


@store.command("adopt", context_settings=CONTEXT_SETTINGS)
@click.argument(
    "root", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    "--from",
    "old_entries",
    required=True,
    help="The name of the store's version directory, which is renamed to the header's "
    "entries",
)
@click.option(
    "--header",
    "header_spec",
    required=True,
    help="The layout header to write: a JSON file, or the JSON itself",
)
@click.option(
    "--identity",
    default="pydra.engine.portable.identity.PydraIdentityV2",
    show_default=True,
    help="The importable StoreIdentity; a class is instantiated with no arguments, so "
    "one that takes parameters is given as a module-level instance",
)
@click.option(
    "--entry-format",
    default="pydra.engine.portable.entry.PydraEntryV1",
    show_default=True,
    help="The importable EntryFormat; a class is instantiated with no arguments, so "
    "one that takes parameters is given as a module-level instance",
)
@click.option(
    "--check",
    type=click.Choice(["none", "sample", "all"]),
    default="sample",
    show_default=True,
    help="Which entries are read back through the identity and the entry format",
)
@click.option(
    "--trust",
    type=click.Choice(["private", "trusted", "shared"]),
    default="trusted",
    show_default=True,
    help="What you assert about who can write the root (RFC 0003 section 6.4)",
)
def store_adopt(root, old_entries, header_spec, identity, entry_format, check, trust):
    """Make the store at ROOT, already laid out like a portable store, a portable store
    without rewriting an entry (RFC 0003 section 8.2). Stop its writers first. Exits 1
    when entries were read back and do not verify (the root is adopted)."""
    from pydra.engine.portable.adopt import AdoptRefused, adopt
    from pydra.engine.portable.layout import LayoutHeader, StoreLayoutUnknown

    data = (
        header_spec.encode("utf-8")
        if header_spec.lstrip().startswith("{")
        else Path(header_spec).read_bytes()
    )
    try:
        header = LayoutHeader.from_json(data)
        report = adopt(
            root,
            old_entries=old_entries,
            header=header,
            identity=_hook(identity),
            entry_format=_hook(entry_format),
            check=check,
            trust=trust,
        )
    except (AdoptRefused, StoreLayoutUnknown) as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"renamed {report.renamed[0]} to {report.renamed[1]}")
    click.echo("header written" if report.header_written else "header not written")
    for rel in report.misfiled:
        click.echo(f"does not verify: {rel}")
    click.echo(f"{report.checked} checked, {len(report.misfiled)} misfiled")
    if report.misfiled:
        sys.exit(1)  # adopted; each misfiled entry is quarantined when it is read
