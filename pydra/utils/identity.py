"""The identity record: the preimage of a job's checksum (RFC 0004 section 3.1).

``task_record(task)`` is a JSON document of strings and lists, and
``record_checksum(record)`` is ``"<kind>-<sha256 of its RFC 8785 form>"``. A supplied
checksum (``checksum=``) replaces the record by ``{"scheme", "supplied"}`` and is its own
checksum. ``job_record(job)`` is the record of a job: the task's, plus the fields that
belong to the job and not to the task.

The digest strings of the record are ``code@1`` (the walk of the executor), ``type@1``
(the ``Outputs`` class) and ``py@1`` (an input). A job's record is built once, in the
submitting process, and carried with the job: ``Job.record``, ``Job.checksum`` and the
store identity's ``key`` never rebuild it in a worker. ``Task._checksum`` and
``Task._hash`` rebuild it on every call.
"""

from __future__ import annotations

import hashlib
import os
import typing as ty

from pydra.utils.digest import jcs
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
)
from pydra.utils.general import get_fields

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.job import Job

__all__ = ["IDENTITY_SCHEME", "job_record", "record_checksum", "task_record"]

IDENTITY_SCHEME = "pydra-identity@2"

Record = dict[str, ty.Any]


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
    digests = {}
    for name, value in inputs.items():
        tag = "code@1" if name == code_name else "py@1"
        digests[name] = f"{tag}:{hash_function(value, cache=cache)}"
    digests["Outputs"] = f"type@1:{hash_function(task.Outputs, cache=cache)}"
    if collect:
        task._code_modules = frozenset(cache.modules)
        task._code_refs = sorted(cache.refs)
        task._declared_code = {
            k: ",".join(sorted(v)) for k, v in sorted(cache.declared.items())
        }
    return digests


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
    return {
        "scheme": IDENTITY_SCHEME,
        "kind": task._task_type(),
        "walk": walk_classification(task._code_hash),
        "task": code,
        "outputs": outputs,
        "inputs": digests,
    }


def job_record(job: "Job") -> Record:
    """The identity record of ``job``: the task's record (the job's own fields, ``job`` and ``nodes``,
    join it with the code that keys them, RFC 0004 section 3.1)."""
    return task_record(job.task)


def record_checksum(record: ty.Mapping[str, ty.Any]) -> str:
    """The checksum of ``record``: its ``supplied`` string, or
    ``f"{kind}-{sha256(JCS(record))}"``."""
    if "supplied" in record:
        return record["supplied"]
    return f"{record['kind']}-{hashlib.sha256(jcs(record)).hexdigest()}"
