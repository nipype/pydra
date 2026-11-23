"""The execution context recorded beside each result: the interpreter, the platform,
the environment and the identities of the installed distributions. It is never part
of the job checksum; a reuse policy (see :mod:`pydra.engine.store`) compares it."""

import csv
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import sys
import types
import typing as ty
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
import attrs
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
from pydra.utils.general import STDLIB_MODULES, get_fields
from pydra.utils.hash import (
    _module_is_library,
    _owners,
    forget_owners,
    check_code_hash,
    code_hash_default,
    library_modules,
    type_module,
)

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.environments.base import Environment, ResolvedRef

logger = logging.getLogger("pydra")

RECORD = 3
"""The version of the record's layout, stored under ``"record"``. Records of other
versions are never reused under ``"consistent"`` or ``"exact"``."""


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


def _norm(path: str | os.PathLike) -> str:
    """`path` made absolute and, on case-insensitive platforms, lower-cased, so paths
    of one file compare equal"""
    return os.path.normcase(os.path.abspath(path))


def _file_url_path(url: str) -> str | None:
    """The local path of a ``file:`` URL (``file:///C:/x`` is ``C:\\x`` on Windows)"""
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    path = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path = f"{os.sep}{os.sep}{parsed.netloc}{path}"  # a UNC share
    return path


def _file_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b"<missing>"


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
        return frozenset(_norm(self.dist.locate_file(f)) for f in self.dist.files or ())

    @functools.cached_property
    def editable_root(self) -> str | None:
        """The source directory of an editable install"""
        if not isinstance(self.direct, dict):
            return None
        if not self.direct.get("dir_info", {}).get("editable"):
            return None
        directory = _file_url_path(self.direct.get("url", ""))
        if directory is None or not os.path.isdir(directory):
            return None
        return os.path.join(os.path.abspath(directory), "")

    def holds(self, path: str) -> bool:
        """Whether `path` is under the source directory of an editable install"""
        root = self.editable_root
        return root is not None and _norm(path).startswith(
            os.path.join(_norm(root), "")
        )

    def import_paths(self, root: str) -> list[Path]:
        """The packages and modules under `root` that the distribution provides: the
        names of its ``top_level.txt`` and those importlib maps to it, or only if
        there are none, those of the loaded modules whose files are under `root`"""
        names = set((self.dist.read_text("top_level.txt") or "").split())
        names.update(t for t, d in _package_distributions().items() if self.name in d)
        if not names:
            for name, module in list(sys.modules.items()):
                file = getattr(module, "__file__", None)
                if file and self.holds(file):
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

    @property
    def keyed(self) -> str | None:
        """The identity a checksum is keyed by: `identity`, except that an install from
        an index is keyed by its public version, leaving out the local segment
        (``+cu121``), which names a build for a platform and is recorded, not keyed
        (RFC 0004 section 4.1)"""
        identity = self.identity
        if identity is None or self.direct:
            return identity
        try:
            return Version(identity).public
        except InvalidVersion:
            return identity


def _identify(installed: _Installed) -> tuple[str | None, str | None]:
    """The identity of an installed distribution, or why it is unknown. The identity
    is its version, followed by the commit or archive hash it was installed from. An
    install from a local directory, editable or not, has no version in it: it is
    ``dir:<digest of its files>``, read from the files of its packages for an editable
    install and from its ``RECORD`` entries outside ``*.dist-info`` otherwise. An
    editable install's packages are read up to `IDENTITY_MAX_FILES` and
    `IDENTITY_MAX_BYTES`."""
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
            identity = "dir:" + _capped_digest(Path(root), _package_files(paths))
        except _TooLarge as e:
            return None, str(e)
        return identity, None
    record = installed.dist.read_text("RECORD")
    if not record:
        return None, "it was installed from a local source without a RECORD"
    return f"dir:{_record_digest(record)}", None


def _record_digest(record: str) -> str:
    """A digest of the hashed ``path,hash`` entries of a ``RECORD`` that stay inside the
    install, leaving out ``*.dist-info/`` and the scripts outside it (``../``): what was
    installed, not how or where. ``direct_url.json``, ``INSTALLER`` and an installer's
    own files differ between two installs of the same files"""
    entries = set()
    for row in csv.reader(record.splitlines()):
        if (
            len(row) >= 2
            and row[1]
            and ".dist-info/" not in row[0]
            and not row[0].startswith(("../", "/"))
        ):
            entries.add((row[0], row[1]))
    return _digest(f"{path},{hash_}".encode() for path, hash_ in sorted(entries))


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


REPOSITORY_FIELD = "environment_repository"
"""The context field naming the repository an image came from. It is not compared
between contexts and not part of a slot's name."""


def job_context(
    task: "Task", environment: "Environment", resolved: "ResolvedRef | None" = None
) -> dict[str, ty.Any]:
    """The record of the context a job of `task` runs in under `environment`:

    * ``record``: the layout's version (`RECORD`);
    * ``python``: the interpreter's full version, ``implementation``: its name;
    * ``platform``: the OS, machine and C library;
    * ``environment``: the environment's resolved key (`Job.resolved`), or its
      checksum when none is given;
    * ``environment_repository``: where a container image is pulled from, when its
      reference carries one; recorded only (`REPOSITORY_FIELD`), so mirrors share slots;
    * ``packages``: the identities of the distributions that installed the library
      modules the task reaches (see `reached_modules`), and of those they require,
      None if the modules cannot be determined;
    * ``installed``: a digest of all installed distributions, those in ``packages`` by
      their identities and the others by their versions;
    * ``code``: present when the checksum reached functions with declared versions
      (the task's own, those of a workflow's nodes, those of tasks passed as inputs),
      the `code_digest` of each by ``module.qualname==version``, which the version
      stands for (see `pydra.engine.store.check_declared_code`);
    * ``unknown``: present when any of ``packages`` is unknown, or the job's tool version
      is unresolved (`pydra.engine.job.Job.context`), set to a value unique to this
      process, so the result is reused by no other process.
    """
    if task._declared_code is None and not hasattr(task, "__checksum__"):
        task._hash  # collects the modules and the code of declared versions reached
    modules = reached_modules(task)
    packages = None if modules is None else package_versions(modules)
    context = {
        **process_context(),
        "environment": environment.checksum() if resolved is None else resolved.key,
        "packages": packages,
        "installed": _installed_digest(frozenset(packages or ())),
    }
    if resolved is not None and "repository" in resolved.carried:
        context[REPOSITORY_FIELD] = resolved.carried["repository"]
    if task._declared_code:
        context["code"] = dict(task._declared_code)
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
    forget_owners()
    _warned.clear()
