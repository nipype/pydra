"""The execution context recorded beside each result: the interpreter, the platform,
the environment and the identities of the installed distributions. It is never part
of the job checksum; a reuse policy (see :mod:`pydra.engine.store`) compares it."""

import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import types
import typing as ty
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse
import attrs
from packaging.requirements import InvalidRequirement, Requirement
from pydra.utils.general import STDLIB_MODULES, get_fields
from pydra.utils.hash import (
    _module_is_library,
    check_code_hash,
    code_hash_default,
    library_modules,
    type_module,
)

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.environments.base import Environment

logger = logging.getLogger("pydra")

RECORD = 2
"""The version of the record's layout, stored under ``"record"``. Records of other
versions are never reused under ``"consistent"`` or ``"exact"``."""

GIT_TIMEOUT = 30
"""Seconds a ``git`` call identifying an editable install may take."""


def _platform() -> str:
    """The OS and machine, and the C library and its version where known"""
    name = f"{sys.platform}-{platform.machine()}"
    libc, version = platform.libc_ver()
    return f"{name}-{libc}{version}" if libc else name


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _digest(parts: ty.Iterable[bytes]) -> str:
    hsh = hashlib.sha256()
    for part in parts:
        hsh.update(len(part).to_bytes(8, "little") + part)
    return hsh.hexdigest()


def _file_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b"<missing>"


def _git(directory: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            timeout=GIT_TIMEOUT,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.decode(errors="surrogateescape") if not done.returncode else None


def _git_identity(directory: Path) -> str | None:
    """``git:<commit>``, followed by a digest of the modified and untracked files under
    `directory` that are not ignored, if it is in a git checkout"""
    commit = _git(directory, "rev-parse", "HEAD")
    top = _git(directory, "rev-parse", "--show-toplevel")
    status = _git(
        directory, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."
    )
    if not commit or not top or status is None:
        return None
    top_dir = Path(top.strip())
    entries = status.split("\0")
    changed = []
    for i, entry in enumerate(entries):
        if len(entry) > 3 and not (i and entries[i - 1][:1] == "R"):
            changed.append(entry[3:])
    identity = f"git:{commit.strip()}"
    if changed:
        identity += "+" + _digest(
            p.encode() + b"\0" + _file_bytes(top_dir / p) for p in sorted(changed)
        )
    return identity


def _directory_digest(directory: Path) -> str:
    """A digest of the paths and contents of the files under `directory`, leaving out
    hidden directories, ``__pycache__`` and compiled byte-code"""
    files = []
    for root, dirs, names in os.walk(directory):
        dirs[:] = sorted(
            d for d in dirs if not d.startswith(".") and d != "__pycache__"
        )
        files.extend(
            Path(root, n) for n in sorted(names) if not n.endswith((".pyc", ".pyo"))
        )
    return "dir:" + _digest(
        str(f.relative_to(directory)).encode() + b"\0" + _file_bytes(f) for f in files
    )


@attrs.define
class _Installed:
    """An installed distribution, identified by its version and, if it was not
    installed from an index, by what it was installed from (see `_identify`)"""

    name: str
    version: str | None
    identity: str | None
    reason: str | None  # why the identity is unknown
    requires: list[str]
    editable_root: str | None
    dist: importlib.metadata.Distribution = attrs.field(repr=False)

    @functools.cached_property
    def files(self) -> frozenset[str]:
        return frozenset(
            os.path.abspath(self.dist.locate_file(f)) for f in self.dist.files or ()
        )


def _identify(dist) -> tuple[str | None, str | None, str | None]:
    """The identity of an installed distribution, why it is unknown, and the directory
    of an editable install. The identity is its version, followed by the commit or
    archive hash it was installed from, or for an install from a local directory by
    ``git:<commit>[+<digest of changed files>]`` for an editable install in a git
    checkout, a digest of the directory's files for another editable install, and a
    digest of the files it installed (its ``RECORD``) otherwise."""
    version = dist.version
    if not version:
        return None, "it has no version", None
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "null")
    except ValueError:
        return None, "its direct_url.json cannot be read", None
    if not direct:
        return version, None, None
    if commit := direct.get("vcs_info", {}).get("commit_id"):
        return f"{version}@{commit}", None, None
    archive = direct.get("archive_info", {})
    if hashes := archive.get("hashes"):
        return (
            f"{version}@" + ",".join(f"{k}={v}" for k, v in sorted(hashes.items())),
            None,
            None,
        )
    if digest := archive.get("hash"):
        return f"{version}@{digest}", None, None
    if direct.get("dir_info", {}).get("editable"):
        url = urlparse(direct.get("url", ""))
        directory = Path(unquote(url.path))
        if url.scheme != "file" or not directory.is_dir():
            return None, f"its editable source {direct.get('url')!r} is missing", None
        root = os.path.join(os.path.abspath(directory), "")
        identity = _git_identity(directory) or _directory_digest(directory)
        return f"{version}@{identity}", None, root
    record = dist.read_text("RECORD")
    if not record:
        return None, "it was installed from a local source without a RECORD", None
    return f"{version}@record:{_digest([record.encode()])}", None, None


@functools.cache
def _distributions() -> dict[str, _Installed]:
    """The installed distributions by normalised name; the first found on
    ``sys.path`` wins, as it is the one imported"""
    found = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name or _normalise(name) in found:
            continue
        identity, reason, root = _identify(dist)
        found[_normalise(name)] = _Installed(
            name=_normalise(name),
            version=dist.version,
            identity=identity,
            reason=reason,
            requires=list(dist.requires or ()),
            editable_root=root,
            dist=dist,
        )
    return found


@functools.cache
def _package_distributions() -> dict[str, list[str]]:
    return {
        top: [_normalise(n) for n in names]
        for top, names in importlib.metadata.packages_distributions().items()
    }


@functools.cache
def _nonce() -> str:
    return uuid.uuid4().hex


_warned: set[tuple[str, str]] = set()


def _unknown(what: str, reason: str) -> None:
    if (what, reason) not in _warned:
        _warned.add((what, reason))
        logger.warning(
            "The version of %s is unknown, because %s, so results reaching it are not "
            "reused by later processes",
            what,
            reason,
        )


@functools.cache
def process_context() -> dict[str, ty.Any]:
    """The part of the record that is the same for every job of this process"""
    installed = sorted(
        f"{d.name}=={d.identity or d.version}" for d in _distributions().values()
    )
    return {
        "record": RECORD,
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": _platform(),
        "installed": _digest(i.encode() for i in installed),
    }


def _value_modules(value: ty.Any, modules: set[str]) -> None:
    """Collect the modules of the types of `value` and of the values it holds in
    builtin containers and attrs and dataclass instances"""
    stack, seen = [value], set()
    while stack:
        value = stack.pop()
        tp = type(value)
        modules.add(type_module(tp))
        if id(value) in seen:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
        elif isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif attrs.has(tp):
            stack.extend(getattr(value, f.name, None) for f in attrs.fields(tp))
        elif dataclasses.is_dataclass(tp):
            stack.extend(getattr(value, f.name, None) for f in dataclasses.fields(tp))
        else:
            continue
        seen.add(id(value))


def reached_modules(task: "Task") -> set[str] | None:
    """The library modules the task reaches: those of its class and bases, of the
    library code its function reaches at the ``"user"`` tier (collected while its
    checksum was computed, or walked again if it was not computed at that tier) and of
    the types of its input values. None if they cannot be determined."""
    modules = {klass.__module__ for klass in type(task).__mro__}
    for field in get_fields(task):
        if field.name != task._executor_name:
            _value_modules(getattr(task, field.name), modules)
    executor = getattr(task, task._executor_name or "", None)
    if isinstance(executor, types.FunctionType):
        walked = (
            task._code_modules is not None
            and task._version is None
            and not hasattr(task, "__checksum__")
            and check_code_hash(task._code_hash or code_hash_default()) == "user"
        )
        try:
            modules.update(task._code_modules if walked else library_modules(executor))
        except Exception:
            logger.warning(
                "The packages reached by %s cannot be determined, so its results are "
                "not reused by later processes",
                type(task).__name__,
                exc_info=True,
            )
            return None
    return {
        m
        for m in modules
        if isinstance(m, str)
        and m != "__main__"
        and m.partition(".")[0] not in STDLIB_MODULES
        and _module_is_library(m)
    }


def _owners(module: str) -> list[str] | None:
    """The distributions that installed `module`: the one whose files include the
    module's file, or whose editable source holds it, or for a module without a file
    (a namespace package) those providing its top-level name; None if there is none"""
    distributions = _distributions()
    candidates = _package_distributions().get(module.partition(".")[0], [])
    file = getattr(sys.modules.get(module), "__file__", None)
    if not file:
        return [c for c in candidates if c in distributions] or None
    path = os.path.abspath(file)
    for name in candidates:
        if name in distributions and path in distributions[name].files:
            return [name]
    roots = [
        (len(d.editable_root), d.name)
        for d in distributions.values()
        if d.editable_root and path.startswith(d.editable_root)
    ]
    return [max(roots)[1]] if roots else None


@functools.cache
def _requirements(name: str) -> frozenset[str]:
    """The installed distributions `name` requires, transitively, under the markers of
    the running interpreter (requirements of extras are not followed)"""
    distributions = _distributions()
    closure, stack = set(), [name]
    while stack:
        for spec in distributions[stack.pop()].requires:
            try:
                requirement = Requirement(spec)
            except InvalidRequirement:
                continue
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dep = _normalise(requirement.name)
            if dep in distributions and dep not in closure and dep != name:
                closure.add(dep)
                stack.append(dep)
    return frozenset(closure)


def package_versions(modules: ty.Iterable[str]) -> dict[str, str | None]:
    """The identities (see `_identify`) of the distributions that installed `modules`
    and of those they require, by normalised name; a module no distribution installed
    is listed by its name with None"""
    distributions = _distributions()
    versions = {}
    for module in sorted(set(modules)):
        owners = _owners(module)
        if owners is None:
            _unknown(f"module {module!r}", "no installed distribution provides it")
            versions[module] = None
            continue
        for name in owners:
            for dep in {name} | _requirements(name):
                dist = distributions[dep]
                if dist.identity is None:
                    _unknown(f"distribution {dist.name!r}", dist.reason)
                versions[dep] = dist.identity
    return dict(sorted(versions.items()))


def job_context(task: "Task", environment: "Environment") -> dict[str, ty.Any]:
    """The record of the context a job of `task` runs in under `environment`:

    * ``record``: the layout's version (`RECORD`);
    * ``python``: the interpreter's full version, ``implementation``: its name;
    * ``platform``: the OS, machine and C library;
    * ``environment``: the environment's checksum;
    * ``packages``: the identities of the distributions that installed the library
      modules the task reaches (see `reached_modules`), and of those they require,
      None if the modules cannot be determined;
    * ``installed``: a digest of the identities of all installed distributions;
    * ``unknown``: present when any of ``packages`` is unknown, set to a value unique
      to this process, so the result is reused by no other process.
    """
    modules = reached_modules(task)
    packages = None if modules is None else package_versions(modules)
    context = {
        **process_context(),
        "environment": environment.checksum(),
        "packages": packages,
    }
    if packages is None or None in packages.values():
        context["unknown"] = _nonce()
    return context


def clear_caches() -> None:
    """Forget the installed distributions and this process's context, so they are
    read again (after installing into the running interpreter); results recorded with
    unknown versions before are then no longer reused"""
    for cached in (
        _distributions,
        _package_distributions,
        _requirements,
        _nonce,
        process_context,
    ):
        cached.cache_clear()
    _warned.clear()
