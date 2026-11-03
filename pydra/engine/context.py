"""The execution context recorded beside each result: the interpreter, the platform,
the environment and the versions of the installed distributions. It is never part of
the job checksum; a reuse policy (see :mod:`pydra.engine.store`) compares it."""

import functools
import hashlib
import importlib.metadata
import json
import platform
import sys
import types
import typing as ty
from pydra.utils.general import STDLIB_MODULES, get_fields
from pydra.utils.hash import _module_is_library, library_packages

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.environments.base import Environment

RECORD = 2
"""The version of the record's layout, stored under ``"record"``. Records of other
versions are never reused under ``"consistent"`` or ``"exact"``."""


def _platform() -> str:
    """The OS and machine, and the C library and its version where known"""
    name = f"{sys.platform}-{platform.machine()}"
    libc, version = platform.libc_ver()
    return f"{name}-{libc}{version}" if libc else name


def _identity(dist: importlib.metadata.Distribution) -> str | None:
    """The version of an installed distribution, with the commit or archive hash it
    was installed from if it was not installed from an index, or None if it has no
    version or was installed from a local directory (whose contents it does not
    identify)"""
    version = dist.version
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "null")
    except ValueError:
        return None
    if not version or not direct:
        return version or None
    if commit := direct.get("vcs_info", {}).get("commit_id"):
        return f"{version}@{commit}"
    archive = direct.get("archive_info", {})
    if hashes := archive.get("hashes"):
        return f"{version}@" + ",".join(f"{k}={v}" for k, v in sorted(hashes.items()))
    if digest := archive.get("hash"):
        return f"{version}@{digest}"
    return None


@functools.cache
def _distributions() -> dict[str, tuple[str, str | None]]:
    """The installed distributions by name, as their version and their identity
    (see `_identity`); the first found on ``sys.path`` wins, as it is the one
    imported"""
    found = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name and name not in found:
            found[name] = (dist.version, _identity(dist))
    return found


@functools.cache
def _package_distributions() -> ty.Mapping[str, list[str]]:
    return importlib.metadata.packages_distributions()


@functools.cache
def process_context() -> dict[str, ty.Any]:
    """The part of the record that is the same for every job of this process"""
    installed = sorted(f"{n.lower()}=={v}" for n, (v, _) in _distributions().items())
    return {
        "record": RECORD,
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": _platform(),
        "installed": hashlib.sha256("\n".join(installed).encode()).hexdigest(),
    }


def _value_modules(value: ty.Any, modules: set[str]) -> None:
    """Collect the modules of the types of `value` and of the items of the builtin
    containers it holds"""
    stack, seen = [value], set()
    while stack:
        value = stack.pop()
        modules.add(type(value).__module__)
        if isinstance(value, (list, tuple, set, frozenset)) and id(value) not in seen:
            seen.add(id(value))
            stack.extend(value)
        elif isinstance(value, dict) and id(value) not in seen:
            seen.add(id(value))
            stack.extend(value.keys())
            stack.extend(value.values())


def reached_packages(task: "Task") -> set[str] | None:
    """The top-level packages of library code the task reaches: those of its class and
    bases, of the library code its function reaches (see
    `pydra.utils.hash.library_packages`) and of the types of its input values. None if
    the function's walk fails, so its packages are unknown."""
    modules = {klass.__module__ for klass in type(task).__mro__}
    for field in get_fields(task):
        if field.name != task._executor_name:
            _value_modules(getattr(task, field.name), modules)
    packages = {
        m.partition(".")[0]
        for m in modules
        if isinstance(m, str) and m != "__main__" and _module_is_library(m)
    }
    executor = getattr(task, task._executor_name or "", None)
    if isinstance(executor, types.FunctionType):
        try:
            packages.update(library_packages(executor))
        except Exception:
            return None
    return packages - STDLIB_MODULES - {"__main__"}


def package_versions(packages: ty.Iterable[str]) -> dict[str, str | None]:
    """The identities of the distributions providing `packages` (see `_identity`), by
    distribution name; a package that no installed distribution provides is listed by
    its own name with None"""
    distributions = _distributions()
    versions = {}
    for package in packages:
        names = _package_distributions().get(package)
        if not names:
            versions[package] = None
        for name in names or ():
            versions[name] = distributions.get(name, (None, None))[1]
    return dict(sorted(versions.items()))


def job_context(task: "Task", environment: "Environment") -> dict[str, ty.Any]:
    """The record of the context a job of `task` runs in under `environment`:

    * ``record``: the layout's version (`RECORD`);
    * ``python``: the interpreter's full version, ``implementation``: its name;
    * ``platform``: the OS, machine and C library;
    * ``environment``: the environment's checksum;
    * ``packages``: the identities of the distributions providing the library code the
      task reaches (see `reached_packages`), None if that is unknown;
    * ``installed``: a digest of the names and versions of all installed distributions.
    """
    packages = reached_packages(task)
    return {
        **process_context(),
        "environment": environment.checksum(),
        "packages": None if packages is None else package_versions(packages),
    }


def clear_caches() -> None:
    """Forget the installed distributions and the process's context, so they are read
    again (after installing into the running interpreter)"""
    for cached in (_distributions, _package_distributions, process_context):
        cached.cache_clear()
