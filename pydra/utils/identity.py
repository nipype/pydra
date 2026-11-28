"""The identity record: the preimage of a job's checksum (RFC 0004 section 3.1).

``task_record(task)`` is a JSON document of strings and lists, and
``record_checksum(record)`` is ``"<kind>-<sha256 of its RFC 8785 form>"``. A supplied
checksum (``checksum=``) replaces the record by ``{"scheme", "supplied"}`` and is its own
checksum. ``job_record(job)`` is the record of a job: the task's, plus the fields that
belong to the job and not to the task.

The digest strings of the record are ``code@1`` (the walk of the executor), ``type@1``
(the ``Outputs`` class) and, for each input, its typed digest from ``pydra.utils.digest``
(``json@1``, ``npy@1``, ``file@1``, a registered one, or ``py@1`` for what has no digest
form).

A job's record is built once, in the submitting process, and carried with the job:
``Job.record``, ``Job.checksum`` and the store identity's ``key`` never rebuild it in a
worker. ``Task._checksum`` and ``Task._hash`` rebuild it on every call in the submitting
process. A task unpickled in a worker reads the digests the submitting process computed
(``Task._carried``), because a worker may not be able to read the source of its function,
its classes and its input values; it hashes again only the values that can change
meanwhile (``rehashed_in_worker``).
"""

from __future__ import annotations

import hashlib
import os
import typing as ty

from pydra.utils.digest import digest, jcs
from pydra.utils.general import add_exc_note
from pydra.utils.hash import (
    LIBRARY_PACKAGES,
    USER_PACKAGES_ENV_VAR,
    Cache,
    is_unversioned,
    _module_is_library,
    _package_overrides,
    check_code_hash,
    code_hash_default,
    hash_function,
    owner_of,
    type_digest,
)
from pydra.utils.general import get_fields

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.job import Job

__all__ = ["IDENTITY_SCHEME", "job_record", "record_checksum", "task_record"]

IDENTITY_SCHEME = "pydra-identity@2"

Record = dict[str, ty.Any]

# The digests of the definition of each shell field are kept beside the digests of the
# inputs under ``template:<field>``, a name no field has
TEMPLATE_PREFIX = "template:"


# The attributes of a field that shape the command line, besides its type (RFC 0004
# section 5.1). Documentation (``help``) is left out, and so is everything that does not
# decide what runs.
TEMPLATE_ATTRIBUTES = (
    "argstr",
    "position",
    "sep",
    "formatter",
    "path_template",
    "keep_extension",
    "copy_mode",
)


def walk_classification(code_hash: str | None = None) -> dict[str, ty.Any]:
    """The classification that shapes what the code walk reaches: the ``code_hash`` tier
    (``code_hash``, else ``PYDRA_CODE_HASH``, else ``"user"``) and the packages
    ``PYDRA_USER_PACKAGES`` moves to user code or to library code."""
    overrides = _package_overrides(os.environ.get(USER_PACKAGES_ENV_VAR, ""))
    return {
        "code_hash": check_code_hash(code_hash or code_hash_default()),
        "user_packages": sorted(p for p, is_library in overrides if not is_library),
        "library_packages": sorted(
            p for p, is_library in overrides if is_library and p not in LIBRARY_PACKAGES
        ),
    }


def field_digests(
    task: "Task", exclude: str | None = None, collect: bool = False
) -> dict[str, str]:
    """The digest of each hashed input of ``task`` (the executor under its field name,
    the ``Outputs`` class under ``"Outputs"``), leaving out ``exclude``. With
    ``collect`` the library modules and declared code the walk reaches are kept on
    the task (for its context)."""
    inputs = task._checksum_inputs()
    inputs.pop(exclude, None)
    cache = Cache(code_hash=task._code_hash)
    code_name = task._executor_name if _executor_is_code(task) else None
    carried = task._carried or {}
    fields = {f.name: f for f in get_fields(task)}
    digests = {}
    if task._task_type() == "shell":
        for field in fields.values():
            if field.name == "executable":
                continue
            key = TEMPLATE_PREFIX + field.name
            digests[key] = (
                carried[key][1]
                if key in carried
                else f"py@1:{hash_function(template_spec(field, cache), cache=cache)}"
            )
    for name, value in inputs.items():
        if name in carried and carried[name][0] is getattr(task, name):
            if not rehashed_in_worker(fields.get(name), carried[name][0]):
                digests[name] = carried[name][1]
                continue
        if name == code_name:
            digests[name] = f"code@1:{hash_function(value, cache=cache)}"
        else:
            digests[name] = _input_digest(name, value, cache)
    digests["Outputs"] = (
        carried["Outputs"][1]
        if "Outputs" in carried
        else type_digest(task.Outputs, cache)
    )
    if collect and not carried:
        task._code_modules = frozenset(cache.modules)
        task._code_refs = sorted(cache.refs)
        task._declared_code = {
            k: ",".join(sorted(v)) for k, v in sorted(cache.declared.items())
        }
    return digests


def template_spec(field: ty.Any, cache: Cache) -> dict[str, ty.Any]:
    """The definition of a shell field that shapes the command line, which is hashed as
    ``py@1`` (RFC 0004 section 5.1): its ``argstr``, ``position``, ``sep``, ``formatter``
    (a function, walked like any user function), ``path_template``, ``keep_extension``,
    ``copy_mode`` and its type as ``type@1``"""
    spec = {a: getattr(field, a, None) for a in TEMPLATE_ATTRIBUTES}
    spec["type"] = type_digest(field.type, cache)
    return spec


def rehashed_in_worker(field: ty.Any, value: ty.Any) -> bool:
    """Whether a worker hashes ``value`` of ``field`` again instead of using the digest
    the submitting process carried: a value whose hash depends on state outside it, which
    can change between the submission and the run (a file, however it is typed, or a
    value with ``__bytes_repr__``). Functions, classes and instances of user classes
    are never hashed again, as a worker may not be able to read their source."""
    from fileformats.core.fileset import FileSet

    from pydra.utils.typing import TypeParser

    return (
        (field is not None and TypeParser.contains_type(FileSet, field.type))
        or isinstance(value, FileSet)
        or hasattr(value, "__bytes_repr__")
    )


def _input_digest(name: str, value: ty.Any, cache: Cache) -> str:
    """The typed digest of the input ``name`` (RFC 0004 section 6.2); an error names the
    field."""
    try:
        return digest(value, cache)
    except ValueError as err:  # includes UnicodeEncodeError: no JCS form
        raise ValueError(f"cannot digest input field {name!r}: {err}") from err
    except Exception as err:
        add_exc_note(err, f"while digesting input field {name!r}")
        raise


def _executor_is_code(task: "Task") -> bool:
    """Whether the executor field holds code, and so is the record's ``task.code``: a
    function or a declared version, not a shell task's ``executable``"""
    return task._executor_name is not None and task._task_type() != "shell"


def library_values(task: "Task") -> list[str]:
    """The distributions that own the library types of the input values of ``task``, as
    sorted ``dist:{dist}=={identity}`` strings (RFC 0004 section 4.1). A type of the standard
    library or the engine, of user code, or of no single distribution (the context records it
    as unknown) is left out."""
    from pydra.engine.context import _value_modules

    modules: set[str] = set()
    for field in get_fields(task):
        if field.name != task._executor_name:
            _value_modules(getattr(task, field.name), modules)
    owners = {
        o
        for m in modules
        if isinstance(m, str)
        and m != "__main__"
        and not is_unversioned(m)
        and _module_is_library(m)
        and (o := owner_of(m)) is not None
    }
    return sorted(f"dist:{dist}=={identity}" for dist, identity in owners)


def task_record(task: "Task") -> Record:
    """The identity record of ``task`` without the fields of the job that runs it."""
    if hasattr(task, "__checksum__"):
        return {"scheme": IDENTITY_SCHEME, "supplied": task._supplied_checksum()}
    digests = dict(task._compute_hashes()[1])
    outputs = digests.pop("Outputs")
    code: dict[str, ty.Any] = {
        "refs": list(task._code_refs or ()),
        "values": library_values(task),
    }
    if _executor_is_code(task) and task._executor_name in digests:
        code["code"] = digests.pop(task._executor_name)
    if task._task_type() == "shell":
        code["template"] = {
            "executable": _written_executable(task, digests.pop("executable", None)),
            "fields": {
                k.removeprefix(TEMPLATE_PREFIX): digests.pop(k)
                for k in sorted(digests)
                if k.startswith(TEMPLATE_PREFIX)
            },
        }
        code["version"] = task._version
    return {
        "scheme": IDENTITY_SCHEME,
        "kind": task._task_type(),
        "walk": walk_classification(task._code_hash),
        "task": code,
        "outputs": outputs,
        "inputs": digests,
    }


def _written_executable(task: "Task", digest: str | None) -> ty.Any:
    """The executable as written, a string or a list of strings (RFC 0004 section 5.2);
    any other value by its digest"""
    value = task.executable
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return digest


def job_record(job: "Job") -> Record:
    """The identity record of ``job``: the task's record, plus ``job`` for a shell task:
    the environment it runs in (its resolved reference, ``Job.resolved``) and the version
    the task's ``version_command`` reported when it runs natively (RFC 0004 section 3.1).
    A python task has no ``job`` field, and a supplied checksum stands for the whole
    record. The ``nodes`` of a workflow are added by the PR that keys them."""
    from pydra.compose.shell.version import tool_version
    from pydra.environments.native import Native

    record = task_record(job.task)
    if "supplied" not in record and record["kind"] == "shell":
        environment = job.environment
        native = environment is None or isinstance(environment, Native)
        record["job"] = {
            "environment": None if environment is None else job.resolved.key,
            "tool_version": tool_version(job.task) if native else None,
        }
    return record


def record_checksum(record: ty.Mapping[str, ty.Any]) -> str:
    """The checksum of ``record``: its ``supplied`` string, or
    ``f"{kind}-{sha256(JCS(record))}"``."""
    if "supplied" in record:
        return record["supplied"]
    return f"{record['kind']}-{hashlib.sha256(jcs(record)).hexdigest()}"
