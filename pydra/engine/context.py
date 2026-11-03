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


IDENTITY_MAX_FILES = 10_000
"""Most files a local install's identity reads; above it the identity is unknown."""

IDENTITY_MAX_BYTES = 256 * 2**20
"""Most bytes a local install's identity reads; above it the identity is unknown."""


class _TooLarge(Exception):
    pass


def _capped_digest(root: Path, files: list[Path]) -> str:
    """A digest of the paths (relative to `root`) and contents of `files`, raising
    `_TooLarge` above `IDENTITY_MAX_FILES` or `IDENTITY_MAX_BYTES`"""
    if len(files) > IDENTITY_MAX_FILES:
        raise _TooLarge(f"it has more than {IDENTITY_MAX_FILES} files")
    size = 0
    for f in files:
        try:
            size += f.stat().st_size
        except OSError:
            pass
        if size > IDENTITY_MAX_BYTES:
            raise _TooLarge(f"its files hold more than {IDENTITY_MAX_BYTES} bytes")
    return _digest(
        os.path.relpath(f, root).encode() + b"\0" + _file_bytes(f) for f in files
    )


def _git_identity(root: Path, paths: list[Path]) -> str | None:
    """``git:<commit>``, followed by a digest of the modified and untracked files in
    `paths` that are not ignored, if `root` is in a git checkout"""
    commit = _git(root, "rev-parse", "HEAD")
    top = _git(root, "rev-parse", "--show-toplevel")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *(str(p) for p in paths),
    )
    if not commit or not top or status is None:
        return None
    top_dir = Path(top.strip())
    entries = status.split("\0")
    changed = [
        top_dir / entry[3:]
        for i, entry in enumerate(entries)
        if len(entry) > 3 and not (i and entries[i - 1][:1] == "R")
    ]
    identity = f"git:{commit.strip()}"
    if changed:
        identity += "+" + _capped_digest(top_dir, sorted(changed))
    return identity


def _package_files(paths: list[Path]) -> list[Path]:
    """The files of the packages and modules at `paths`, leaving out hidden
    directories, ``__pycache__`` and compiled byte-code"""
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        for root, dirs, names in os.walk(path):
            dirs[:] = sorted(
                d for d in dirs if not d.startswith(".") and d != "__pycache__"
            )
            files.extend(
                Path(root, n) for n in sorted(names) if not n.endswith((".pyc", ".pyo"))
            )
            if len(files) > IDENTITY_MAX_FILES:
                return files
    return files


@attrs.define
class _Installed:
    """An installed distribution. Its identity (see `_identify`) is only read when a
    record reaches it."""

    name: str
    version: str | None
    requires: list[str]
    direct: ty.Any  # its direct_url.json, False if it cannot be read
    dist: importlib.metadata.Distribution = attrs.field(repr=False)

    @functools.cached_property
    def files(self) -> frozenset[str]:
        return frozenset(
            os.path.abspath(self.dist.locate_file(f)) for f in self.dist.files or ()
        )

    @functools.cached_property
    def editable_root(self) -> str | None:
        """The source directory of an editable install"""
        if not isinstance(self.direct, dict):
            return None
        if not self.direct.get("dir_info", {}).get("editable"):
            return None
        url = urlparse(self.direct.get("url", ""))
        directory = unquote(url.path)
        if url.scheme != "file" or not os.path.isdir(directory):
            return None
        return os.path.join(os.path.abspath(directory), "")

    def import_paths(self, root: str) -> list[Path]:
        """The packages and modules under `root` that the distribution provides: the
        names of its ``top_level.txt``, those importlib maps to it, and those of the
        loaded modules whose files are under `root`"""
        names = set((self.dist.read_text("top_level.txt") or "").split())
        names.update(t for t, d in _package_distributions().items() if self.name in d)
        for name, module in list(sys.modules.items()):
            file = getattr(module, "__file__", None)
            if file and os.path.abspath(file).startswith(root):
                names.add(name.partition(".")[0])
        paths = set()
        for name in names:
            for base in (root, os.path.join(root, "src")):
                for candidate in (Path(base, name), Path(base, name + ".py")):
                    if candidate.exists():
                        paths.add(candidate)
        return sorted(paths)

    @functools.cached_property
    def identified(self) -> tuple[str | None, str | None]:
        return _identify(self)

    @property
    def identity(self) -> str | None:
        return self.identified[0]

    @property
    def reason(self) -> str | None:
        return self.identified[1]


def _identify(installed: _Installed) -> tuple[str | None, str | None]:
    """The identity of an installed distribution, or why it is unknown. The identity
    is its version, followed by the commit or archive hash it was installed from, or
    for an install from a local directory by ``git:<commit>[+<digest of changed
    files>]`` for an editable install in a git checkout, a digest of its packages'
    files for another editable install, and a digest of the files it installed (its
    ``RECORD``) otherwise. An editable install's packages are read up to
    `IDENTITY_MAX_FILES` and `IDENTITY_MAX_BYTES`."""
    version, direct = installed.version, installed.direct
    if not version:
        return None, "it has no version"
    if direct is False:
        return None, "its direct_url.json cannot be read"
    if not direct:
        return version, None
    if commit := direct.get("vcs_info", {}).get("commit_id"):
        return f"{version}@{commit}", None
    archive = direct.get("archive_info", {})
    if hashes := archive.get("hashes"):
        return (
            f"{version}@" + ",".join(f"{k}={v}" for k, v in sorted(hashes.items())),
            None,
        )
    if digest := archive.get("hash"):
        return f"{version}@{digest}", None
    if direct.get("dir_info", {}).get("editable"):
        root = installed.editable_root
        if root is None:
            return None, f"its editable source {direct.get('url')!r} is missing"
        paths = installed.import_paths(root)
        if not paths:
            return None, f"no package of it was found under {root!r}"
        try:
            identity = _git_identity(Path(root), paths) or (
                "dir:" + _capped_digest(Path(root), _package_files(paths))
            )
        except _TooLarge as e:
            return None, str(e)
        return f"{version}@{identity}", None
    record = installed.dist.read_text("RECORD")
    if not record:
        return None, "it was installed from a local source without a RECORD"
    return f"{version}@record:{_digest([record.encode()])}", None


def _direct_url(dist) -> ty.Any:
    try:
        return json.loads(dist.read_text("direct_url.json") or "null")
    except ValueError:
        return False


@functools.cache
def _distributions() -> dict[str, _Installed]:
    """The installed distributions by normalised name; the first found on
    ``sys.path`` wins, as it is the one imported"""
    found = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name or _normalise(name) in found:
            continue
        found[_normalise(name)] = _Installed(
            name=_normalise(name),
            version=dist.version,
            requires=list(dist.requires or ()),
            direct=_direct_url(dist),
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
    return {
        "record": RECORD,
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": _platform(),
    }


@functools.lru_cache(maxsize=256)
def _installed_digest(reached: frozenset[str]) -> str:
    """A digest of every installed distribution, by its identity if it is `reached`
    and known, otherwise by its version"""
    entries = sorted(
        f"{d.name}=={(d.name in reached and d.identity) or d.version}"
        for d in _distributions().values()
    )
    return _digest(e.encode() for e in entries)


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
    * ``installed``: a digest of all installed distributions, those in ``packages`` by
      their identities and the others by their versions;
    * ``unknown``: present when any of ``packages`` is unknown, set to a value unique
      to this process, so the result is reused by no other process.
    """
    modules = reached_modules(task)
    packages = None if modules is None else package_versions(modules)
    context = {
        **process_context(),
        "environment": environment.checksum(),
        "packages": packages,
        "installed": _installed_digest(frozenset(packages or ())),
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
        _installed_digest,
        _nonce,
        process_context,
    ):
        cached.cache_clear()
    _warned.clear()
