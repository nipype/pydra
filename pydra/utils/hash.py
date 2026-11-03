"""Generic object hashing dispatch"""

import builtins
import sys
import os
import ast
import dis
import functools
import site
import sysconfig
import struct
import threading
import inspect
import linecache
from datetime import datetime
import typing as ty
import types
import weakref
from pathlib import Path
from collections import OrderedDict
from collections.abc import Mapping
from functools import singledispatch
from hashlib import blake2b
from time import time_ns
import logging
from typing import (
    Dict,
    Iterator,
    NewType,
    Sequence,
    Set,
)
from filelock import SoftFileLock
import attrs.exceptions
from fileformats.core.fileset import FileSet, MockMixin
from fileformats.generic import FsObject
import fileformats.core.exceptions
from pydra.utils.general import (
    STDLIB_MODULES,
    add_exc_note,
    get_fields,
    in_stdlib,
    user_cache_root,
)

logger = logging.getLogger("pydra")

FUNCTION_SRC_CHUNK_LEN_DEFAULT = 8192

TIMESTAMP_RESOLUTION_NS = 2_000_000_000
"""Files changed within this many nanoseconds of hashing are not keyed in the persistent cache."""


class _LRUCache:
    """A thread-safe mapping holding the `size` most recently used items"""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: ty.Hashable) -> ty.Any:
        """The value of `key`, or None if it is not held"""
        with self._lock:
            try:
                self._items.move_to_end(key)
            except KeyError:
                return None
            return self._items[key]

    def put(self, key: ty.Hashable, value: ty.Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


FUNCTION_BYTES_CACHE_SIZE = 4096
SOURCE_NODES_CACHE_SIZE = 128
# Encoded code of functions, keyed by (source file, source-file mtime, weak reference
# to the code object)
_function_bytes_cache = _LRUCache(FUNCTION_BYTES_CACHE_SIZE)
# Function and lambda nodes of parsed sources by first line, keyed by (file, source
# digest)
_source_nodes_cache = _LRUCache(SOURCE_NODES_CACHE_SIZE)

AST_ENCODING = "ast@1"

CODE_HASH_TIERS = ("function", "module", "user")
CODE_HASH_ENV_VAR = "PYDRA_CODE_HASH"
USER_PACKAGES_ENV_VAR = "PYDRA_USER_PACKAGES"
# Packages that are library code wherever they are installed, unless named in
# ``PYDRA_USER_PACKAGES``
LIBRARY_PACKAGES = ("pydra",)

try:
    from typing import Protocol
except ImportError:
    from typing import Protocol  # type: ignore

try:
    from typing import runtime_checkable
except ImportError:
    from typing import runtime_checkable  # type: ignore


try:
    import numpy
except ImportError:
    HAVE_NUMPY = False
else:
    HAVE_NUMPY = True

__all__ = (
    "hash_function",
    "hash_object",
    "hash_single",
    "register_serializer",
    "Hash",
    "Cache",
    "bytes_repr_mapping_contents",
    "bytes_repr_sequence_contents",
)

Hash = NewType("Hash", bytes)
CacheKey = NewType("CacheKey", ty.Tuple[ty.Hashable, ...])


def location_converter(path: ty.Union[Path, str, None]) -> Path:
    if path is None:
        path = PersistentCache.location_default()
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise ValueError(
            f"provided path to persistent cache {path} is a file not a directory"
        ) from None
    return path


@attrs.define
class PersistentCache:
    """Persistent cache in which to store computationally expensive hashes between nodes
    and workflow/task runs. It does this in via the `get_or_calculate_hash` method, which
    takes a locally unique key (e.g. file-system path + mtime) and a function to
    calculate the hash if it isn't present in the persistent store.

    The locally unique key is hashed (cheaply) using hashlib cryptography and this
    "local hash" is use to name the entry of the (potentially expensive) hash of the
    object itself (e.g. the contents of a file). This entry is saved as a text file
    within a user-specific cache directory (see `platformdirs.user_cache_dir`), with
    the name of the file being the "local hash" of the key and the contents of the
    file being the "globally unique hash" of the object itself.

    Parameters
    ----------
    location: Path
        the directory in which to store the hashes cache
    """

    location: Path = attrs.field(converter=location_converter)  # type: ignore[misc]
    cleanup_period: int = attrs.field()
    _hashes: ty.Dict[CacheKey, Hash] = attrs.field(factory=dict)

    # Class-level guard so clean_up() runs at most once per Python session per
    # cache location. Iterating and stat()-ing every entry in the cache
    # directory is expensive when the cache is large; doing it after every task
    # invocation causes significant overhead for batch workloads.
    _session_cleanups_done: ty.ClassVar[Set[Path]] = set()

    # Set the location of the persistent hash cache
    LOCATION_ENV_VAR = "PYDRA_HASH_CACHE"
    CLEANUP_ENV_VAR = "PYDRA_HASH_CACHE_CLEANUP_PERIOD"

    @classmethod
    def location_default(cls):
        try:
            location = os.environ[cls.LOCATION_ENV_VAR]
        except KeyError:
            location = user_cache_root() / "hashes"
        return location

    # the default needs to be an instance method
    @location.default
    def _location_default(self):
        return self.location_default()

    @cleanup_period.default
    def cleanup_period_default(self):
        return int(os.environ.get(self.CLEANUP_ENV_VAR, 30))

    def get_or_calculate_hash(self, key: CacheKey, calculate_hash: ty.Callable) -> Hash:
        """Check whether key is present in the persistent cache store and return it if so.
        Otherwise use `calculate_hash` to generate the hash and save it in the persistent
        store.

        Parameters
        ----------
        key : CacheKey
            locally unique key (e.g. to the host) used to lookup the corresponding hash
            in the persistent store
        calculate_hash : ty.Callable
            function to calculate the hash if it isn't present in the persistent store

        Returns
        -------
        Hash
            the hash corresponding to the key, which is either retrieved from the persistent
            store or calculated using `calculate_hash` if not present
        """
        try:
            return self._hashes[key]
        except KeyError:
            pass
        key_path = self.location / blake2b(str(key).encode()).hexdigest()
        with SoftFileLock(key_path.with_suffix(".lock")):
            if key_path.exists():
                return Hash(key_path.read_bytes())
            hsh = calculate_hash()
            key_path.write_bytes(hsh)
            self._hashes[key] = Hash(hsh)
        return Hash(hsh)

    def clean_up(self):
        """Cleans up old hash caches that haven't been accessed in the last 30 days.

        Cleanup for a given cache location runs at most once per Python session
        to avoid the O(n) directory scan becoming a per-task overhead for large
        persistent caches.
        """
        if self.location in PersistentCache._session_cleanups_done:
            return
        PersistentCache._session_cleanups_done.add(self.location)
        now = datetime.now()
        for path in self.location.iterdir():
            if path.name.endswith(".lock"):
                continue
            days = (now - datetime.fromtimestamp(path.lstat().st_atime)).days
            if days > self.cleanup_period:
                path.unlink(missing_ok=True)

    @classmethod
    def from_path(
        cls, path: ty.Union[Path, str, "PersistentCache", None]
    ) -> "PersistentCache":
        if isinstance(path, PersistentCache):
            return path
        return PersistentCache(path)


@attrs.define
class Cache:
    """Cache for hashing objects, used to avoid infinite recursion caused by circular
    references between objects, and to store hashes of objects that have already been
    hashed to avoid recomputation.

    This concept is extended to persistent caching of hashes for certain object types,
    for which calculating the hash is a potentially expensive operation (e.g.
    File/Directory types). For these classes the `bytes_repr` override function yields a
    "locally unique cache key" (e.g. file-system path + mtime) as the first item of its
    iterator.
    """

    persistent: ty.Optional[PersistentCache] = attrs.field(
        default=None,
        converter=PersistentCache.from_path,  # type: ignore[misc]
    )
    _hashes: ty.Dict[int, Hash] = attrs.field(factory=dict)
    # the `code_hash` tier of the functions hashed with this cache, `code_hash_default`
    # when None
    code_hash: ty.Optional[str] = attrs.field(default=None)
    # the library modules reached by the functions hashed with this cache
    modules: ty.Set[str] = attrs.field(factory=set)

    def __getitem__(self, object_id: int) -> Hash:
        return self._hashes[object_id]

    def __setitem__(self, object_id: int, hsh: Hash):
        self._hashes[object_id] = hsh

    def __contains__(self, object_id):
        return object_id in self._hashes


def hash_function(obj, **kwargs):
    """Generate hash of object."""
    return hash_object(obj, **kwargs).hex()


def hash_object(
    obj: object,
    cache: ty.Optional[Cache] = None,
    persistent_cache: ty.Union[PersistentCache, Path, None] = None,
) -> Hash:
    """Hash an object

    Constructs a byte string that uniquely identifies the object,
    and returns the hash of that string.

    Base Python types are implemented, including recursive lists and
    dicts. Custom types can be registered with :func:`register_serializer`.
    """
    if cache is None:
        cache = Cache(persistent=persistent_cache)
    try:
        return hash_single(obj, cache)
    except Exception as e:
        tp = type(obj)
        add_exc_note(
            e,
            (
                f"and therefore cannot hash `{obj!r}` of type "
                f"`{tp.__module__}.{tp.__name__}`. Consider implementing a "
                "specific `bytes_repr()`(see pydra.utils.hash.register_serializer) "
                "or a `__bytes_repr__()` dunder methods for this type"
            ),
        )
        raise e


def hash_single(obj: object, cache: Cache) -> Hash:
    """Single object-scoped hash

    Uses a local cache to prevent infinite recursion. This cache is unsafe
    to reuse across multiple objects, so this function should not be used directly.
    """
    objid = id(obj)
    if objid not in cache:
        # Handle recursion by putting a dummy value in the cache
        cache[objid] = Hash(b"\x00")
        bytes_it = bytes_repr(obj, cache)
        # Pop first element from the bytes_repr iterator and check whether it is a
        # "local cache key" (e.g. file-system path + mtime tuple) or the first bytes
        # chunk

        def calc_hash(first: ty.Optional[bytes] = None) -> Hash:
            """
            Calculate the hash of the object

            Parameters
            ----------
            first : ty.Optional[bytes]
                the first bytes chunk from the bytes_repr iterator, passed if the first
                chunk wasn't a local cache key
            """
            h = blake2b(digest_size=16, person=b"pydra-hash")
            # We want to use the first chunk that was popped to check for a cache-key
            # if present
            if first is not None:
                h.update(first)
            for chunk in bytes_it:  # Note that `bytes_it` is in outer scope
                h.update(chunk)
            return Hash(h.digest())

        # Read the first item of the bytes_repr iterator and check to see whether it yields
        # a "cache-key" tuple instead of a bytes chunk for the type of the object to be cached
        # (e.g. file-system path + mtime for fileformats.core.FileSet objects). If it
        # does, use that key to check the persistent cache for a precomputed hash and
        # return it if it is, otherwise calculate the hash and store it in the persistent
        # cache with that hash of that key (not to be confused with the hash of the
        # object that is saved/retrieved).
        first = next(bytes_it)
        if isinstance(first, tuple):
            tp = type(obj)
            key = (
                tp.__module__,
                tp.__name__,
            ) + first
            hsh = cache.persistent.get_or_calculate_hash(key, calc_hash)
        else:
            # If the first item is a bytes chunk (i.e. the object type doesn't have an
            # associated 'cache-key'), then simply calculate the hash of the object,
            # passing the first chunk to the `calc_hash` function so it can be included
            # in the hash calculation
            hsh = calc_hash(first=first)
        logger.debug("Hash of %s object is %s", obj, hsh)
        cache[objid] = hsh
    return cache[objid]


@runtime_checkable
class HasBytesRepr(Protocol):
    def __bytes_repr__(self, cache: Cache) -> Iterator[bytes]:
        pass  # pragma: no cover


@singledispatch
def bytes_repr(obj: object, cache: Cache) -> Iterator[bytes]:
    """Default implementation of hashing for generic objects. Single dispatch is used
    to provide hooks for class-specific implementations

    Parameters
    ----------
    obj: object
        the object to hash
    cache : Cache
        a dictionary object used to store a cache of previously cached objects to
        handle circular object references

    Yields
    -------
    bytes
        unique representation of the object in a series of bytes
    """
    cls = obj.__class__
    yield f"{cls.__module__}.{cls.__name__}:{{".encode()
    dct: Dict[str, ty.Any]
    if attrs.has(type(obj)):
        # Drop any attributes that aren't used in comparisons by default
        dct = attrs.asdict(obj, recurse=False, filter=lambda a, _: bool(a.eq))
    elif hasattr(obj, "__slots__") and obj.__slots__ is not None:
        dct = {attr: getattr(obj, attr) for attr in obj.__slots__}
    else:

        def is_special_or_method(n: str):
            return (n.startswith("__") and n.endswith("__")) or inspect.ismethod(
                getattr(obj, n)
            )

        try:
            dct = {n: v for n, v in obj.__dict__.items() if not is_special_or_method(n)}
        except AttributeError:
            dct = {n: getattr(obj, n) for n in dir(obj) if not is_special_or_method(n)}
    yield from bytes_repr_mapping_contents(dct, cache)
    yield b"}"


register_serializer = bytes_repr.register
register_serializer.__doc__ = """Register a custom serializer for a type

The generator function should yield byte strings that will be hashed
to produce the final hash. A recommended convention is to yield a
qualified type prefix (e.g. ``f"{module}.{class}"``),
followed by a colon, followed by the serialized value.

If serializing an iterable, an open and close bracket may be yielded
to identify the start and end of the iterable.

Consider using :func:`bytes_repr_mapping_contents` and
:func:`bytes_repr_sequence_contents` to serialize the contents of a mapping
or sequence. These do not include the prefix or brackets, so they can be
reused as part of a custom serializer.

As an example, the following example is the default serializer for user-defined
classes:

.. code-block:: python

    @register_serializer
    def bytes_repr(obj: object, cache: Cache) -> Iterator[bytes]:
        cls = obj.__class__
        yield f"{cls.__module__}.{cls.__name__}:{{".encode()
        yield from bytes_repr_mapping_contents(obj.__dict__, cache)
        yield b"}"

Serializers must accept a ``cache`` argument, which is a dictionary that
permits caching of hashes for recursive objects. If the hash of sub-objects
is used to create an object serialization, the :func:`hash_single` function
should be called with the same cache object.
"""


@register_serializer
def bytes_repr_dunder(obj: HasBytesRepr, cache: Cache) -> Iterator[bytes]:
    yield from obj.__bytes_repr__(cache)


@register_serializer(type(None))
@register_serializer(type(Ellipsis))
@register_serializer(bool)
@register_serializer(range)
def bytes_repr_builtin_repr(
    obj: object,
    cache: Cache,
) -> Iterator[bytes]:
    yield repr(obj).encode()


@register_serializer
def bytes_repr_slice(obj: slice, cache: Cache) -> Iterator[bytes]:
    yield b"slice("
    yield from bytes_repr_sequence_contents((obj.start, obj.stop, obj.step), cache)
    yield b")"


@register_serializer
def bytes_repr_pathlike(obj: os.PathLike, cache: Cache) -> Iterator[bytes]:
    cls = obj.__class__
    yield f"{cls.__module__}.{cls.__name__}:{os.fspath(obj)}".encode()


@register_serializer
def bytes_repr_bytes(obj: bytes, cache: Cache) -> Iterator[bytes]:
    yield f"bytes:{len(obj)}:".encode()
    yield obj


@register_serializer
def bytes_repr_str(obj: str, cache: Cache) -> Iterator[bytes]:
    val = obj.encode()
    yield f"str:{len(val)}:".encode()
    yield val


@register_serializer
def bytes_repr_int(obj: int, cache: Cache) -> Iterator[bytes]:
    try:
        # Up to 64-bit ints
        val = struct.pack("<q", obj)
        yield b"int:"
    except struct.error:
        # Big ints (old python "long")
        val = str(obj).encode()
        yield f"long:{len(val)}:".encode()
    yield val


@register_serializer
def bytes_repr_float(obj: float, cache: Cache) -> Iterator[bytes]:
    yield b"float:"
    yield struct.pack("<d", obj)


@register_serializer
def bytes_repr_complex(obj: complex, cache: Cache) -> Iterator[bytes]:
    yield b"complex:"
    yield struct.pack("<dd", obj.real, obj.imag)


@register_serializer
def bytes_repr_dict(obj: dict, cache: Cache) -> Iterator[bytes]:
    yield b"dict:{"
    yield from bytes_repr_mapping_contents(obj, cache)
    yield b"}"


@register_serializer
def bytes_repr_module(obj: types.ModuleType, cache: Cache) -> Iterator[bytes]:
    yield b"module:("
    yield hash_single(FsObject(obj.__file__), cache=cache)
    yield b")"


@register_serializer(ty._GenericAlias)
@register_serializer(ty._SpecialForm)
@register_serializer(type)
def bytes_repr_type(klass: type, cache: Cache) -> Iterator[bytes]:
    from pydra.utils.general import get_fields

    def type_location(tp: type) -> bytes:
        """Return the module and name of the type in a ASCII byte string"""
        try:
            type_name = tp.__name__
        except AttributeError:
            type_name = tp._name
        mod_path = ".".join(
            p for p in klass.__module__.split(".") if not p.startswith("_")
        )
        return f"{mod_path}.{type_name}".encode()

    # The metadata of annotated types isn't included in the hash as it doesn't change
    # how values of the type are handled
    while ty.get_origin(klass) is ty.Annotated:
        klass = ty.get_args(klass)[0]
    yield b"type:("
    origin = ty.get_origin(klass)
    args = ty.get_args(klass)
    if origin and args:
        yield b"origin:("
        yield from bytes_repr_type(origin, cache)
        yield b"),args:("
        for arg in args:
            if isinstance(
                arg, list
            ):  # sometimes (e.g. Callable) the args of a type is a list
                yield b"list:("
                yield from (b for t in arg for b in bytes_repr_type(t, cache))
                yield b")"
            else:
                yield from bytes_repr_type(arg, cache)
        yield b")"
    else:
        if klass is Ellipsis:
            yield b"..."
        elif inspect.isclass(klass) and issubclass(klass, FileSet):
            try:
                yield b"mime-like:(" + klass.mime_like.encode() + b")"
            except fileformats.core.exceptions.FormatDefinitionError:
                yield type_location(klass)
        elif fields := get_fields(klass):
            yield b"fields:("
            yield from bytes_repr_sequence_contents(fields, cache)
            yield b")"
            if hasattr(klass, "Outputs"):
                yield b",outputs:("
                yield from bytes_repr_type(klass.Outputs, cache)
                yield b")"
        elif in_stdlib(klass):
            yield type_location(klass)
        else:
            try:
                dct = {
                    n: v for n, v in klass.__dict__.items() if not n.startswith("__")
                }
            except AttributeError:
                yield type_location(klass)
            else:
                yield b"__dict__:("
                yield from bytes_repr_mapping_contents(dct, cache)
                yield b")"
                # Include annotations
                try:
                    annotations = klass.__annotations__
                except AttributeError:
                    pass
                else:
                    yield b",annotations:("
                    yield from bytes_repr_mapping_contents(annotations, cache)
                    yield b")"
                yield b",mro:("
                yield from (
                    b for t in klass.mro()[1:-1] for b in bytes_repr_type(t, cache)
                )
                yield b")"
    yield b")"


if sys.version_info >= (3, 10):
    register_serializer(types.UnionType)(bytes_repr_type)


def _file_entry(path: str, name: str) -> tuple | None:
    """``(name, mtime, size, inode, ctime)`` of the file at ``path``, from ``fstat`` on a
    descriptor opened on it; None if it cannot be opened."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        return None
    try:
        st = os.fstat(fd)
    finally:
        os.close(fd)
    return (name, st.st_mtime_ns, st.st_size, st.st_ino, st.st_ctime_ns)


def fileset_cache_key(fspaths: ty.Sequence[Path]) -> CacheKey | None:
    """The persistent hash cache key of the files and directories at ``fspaths``.

    The key holds ``(path, mtime, size, inode, ctime)`` of each file, and of every file
    beneath each directory by its relative path, along with the relative path of every
    subdirectory. Each file's times are read with ``fstat`` on a descriptor opened on it.
    On Windows ``ctime`` is the creation time and the inode can be 0 on network shares;
    the key then rests on mtime and size.

    Returns None, so that the hash is not stored, if a file cannot be opened or any
    file's mtime or ctime is within ``TIMESTAMP_RESOLUTION_NS`` of now.
    """
    entries: list[tuple] = []
    for fspath in fspaths:
        if not fspath.is_dir():
            entries.append(_file_entry(str(fspath), repr(fspath)))
            continue
        entries.append((repr(fspath),))
        for root, dirs, files in os.walk(fspath):
            dirs.sort()
            rel_root = os.path.relpath(root, fspath)
            entries.extend((os.path.join(rel_root, d),) for d in dirs)
            entries.extend(
                _file_entry(os.path.join(root, f), os.path.join(rel_root, f))
                for f in sorted(files)
            )
    if any(entry is None for entry in entries):
        return None
    settled = time_ns() - TIMESTAMP_RESOLUTION_NS
    if any(max(entry[1], entry[4]) >= settled for entry in entries if len(entry) > 1):
        return None
    return CacheKey(tuple(entries))


@register_serializer(FileSet)
def bytes_repr_fileset(
    fileset: FileSet, cache: Cache
) -> Iterator[ty.Union[CacheKey, bytes]]:
    fspaths = sorted(fileset.fspaths)
    key = fileset_cache_key(fspaths)
    if key is not None:
        yield key
    cls = type(fileset)
    yield f"{cls.__module__}.{cls.__name__}:".encode()
    # Hidden files and directories are set explicitly to be included, so that they are
    # always part of the hash regardless of the defaults of the fileformats version
    for key, chunk_iter in fileset.byte_chunks(
        ignore_hidden_files=False, ignore_hidden_dirs=False
    ):
        yield (",'" + key + "'=").encode()
        yield from chunk_iter


# Need to disable the mtime cache key for mocked filesets. Used in doctests
@register_serializer(MockMixin)
def bytes_repr_mock_fileset(
    mock_fileset: MockMixin, cache: Cache
) -> Iterator[ty.Union[CacheKey, bytes]]:
    cls = type(mock_fileset)
    yield f"{cls.__module__}.{cls.__name__}:".encode()
    for key, _ in mock_fileset.byte_chunks(
        ignore_hidden_files=False, ignore_hidden_dirs=False
    ):
        yield (",'" + key + "'").encode()


@register_serializer(list)
@register_serializer(tuple)
def bytes_repr_seq(obj: Sequence, cache: Cache) -> Iterator[bytes]:
    yield f"{obj.__class__.__name__}:(".encode()
    yield from bytes_repr_sequence_contents(obj, cache)
    yield b")"


@register_serializer(set)
@register_serializer(frozenset)
def bytes_repr_set(obj: Set, cache: Cache) -> Iterator[bytes]:
    yield f"{obj.__class__.__name__}:{{".encode()
    yield from bytes_repr_sequence_contents(sorted(obj), cache)
    yield b"}"


@register_serializer
def bytes_repr_code(obj: types.CodeType, cache: Cache) -> Iterator[bytes]:
    yield b"code:("
    yield from bytes_repr_sequence_contents(
        (
            obj.co_argcount,
            obj.co_posonlyargcount,
            obj.co_kwonlyargcount,
            obj.co_nlocals,
            obj.co_flags,
            obj.co_code,
            obj.co_consts,
            obj.co_names,
            obj.co_varnames,
            obj.co_freevars,
            obj.co_name,
            obj.co_cellvars,
        ),
        cache,
    )
    yield b")"


_AST_EXCLUDED_FIELDS = frozenset({"ctx", "type_comment", "kind"})
_FUNCTION_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DOCUMENTED_DEFS = (*_FUNCTION_DEFS, ast.ClassDef)


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _joined_str_values(values: list[ast.expr]) -> list[ast.expr]:
    """The parts of an f-string with empty strings dropped and adjacent strings merged,
    as Python versions and patch releases split them differently"""
    merged: list[ast.expr] = []
    for part in values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            if not part.value:
                continue
            if merged and isinstance(merged[-1], ast.Constant):
                part = ast.Constant(value=merged.pop().value + part.value)
        merged.append(part)
    return merged


def encode_ast(root: ast.AST) -> str:
    """Encode a function or lambda node in the version-neutral `ast@1` form

    A node is its type name followed by its fields, sorted by name, as ``name=value;``
    in parentheses. A list is its items followed by ``,`` in brackets. Any other value
    (a constant's value, an identifier, a flag) is its type name and repr, as
    ``int:1``. Excluded from the encoding are:

    * line and column positions, ``ctx``, ``type_comment`` and a string's ``kind``;
    * fields whose value is None, empty, 0 or False, so that fields added in a later
      Python version with such a default leave the encoding unchanged (a constant's
      value is always encoded);
    * the name, decorators, signature annotations (of arguments and the return value)
      and docstring (the leading string expression of the body) of a function `root`
      itself, and the name and docstring of a class `root`; those of the functions
      and classes it defines are encoded.

    The parts of an f-string are encoded with empty strings dropped and adjacent
    strings merged. A decorator of `root` that does not wrap it but sets state its
    body reads (such as an attribute of the function) is therefore not part of the
    encoding, and is ambient state outside the contract of the checksum.
    """
    parts: list[str] = []
    root_args = set()
    if isinstance(root, _FUNCTION_DEFS):
        arguments = root.args
        root_args = {
            id(a)
            for a in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            )
            if a is not None
        }

    def encode(value: ty.Any) -> None:
        if isinstance(value, list):
            parts.append("[")
            for item in value:
                encode(item)
                parts.append(",")
            parts.append("]")
            return
        if not isinstance(value, ast.AST):
            parts.append(f"{type(value).__name__}:{value!r}")
            return
        parts.append(type(value).__name__ + "(")
        for name in sorted(value._fields):
            field = getattr(value, name, None)
            if name == "value" and isinstance(
                value, (ast.Constant, ast.MatchSingleton)
            ):
                pass
            elif (
                name in _AST_EXCLUDED_FIELDS
                or (name == "annotation" and id(value) in root_args)
                or (name == "name" and value is root)
                or (
                    name in ("decorator_list", "returns")
                    and value is root
                    and isinstance(root, _FUNCTION_DEFS)
                )
                or (not field and not isinstance(field, ast.AST))
            ):
                continue
            elif (
                name == "body"
                and value is root
                and isinstance(root, _DOCUMENTED_DEFS)
                and _is_docstring(field[0])
            ):
                field = field[1:]
            elif name == "values" and isinstance(value, ast.JoinedStr):
                field = _joined_str_values(field)
                if not field:
                    continue
            parts.append(name + "=")
            encode(field)
            parts.append(";")
        parts.append(")")

    encode(root)
    return "".join(parts)


def _definition_nodes(source_file: str, lines: list[str]) -> dict[int, list[ast.AST]]:
    """The function, lambda and class nodes of a source, by the first line of their
    definition"""
    text = "".join(lines)
    key = (source_file, blake2b(text.encode(), digest_size=16).digest())
    cached = _source_nodes_cache.get(key)
    if cached is not None:
        return cached
    nodes: dict[int, list[ast.AST]] = {}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        pass
    else:
        for node in ast.walk(tree):
            if isinstance(node, _DOCUMENTED_DEFS):
                first = node.decorator_list[0] if node.decorator_list else node
                nodes.setdefault(first.lineno, []).append(node)
            elif isinstance(node, ast.Lambda):
                nodes.setdefault(node.lineno, []).append(node)
    _source_nodes_cache.put(key, nodes)
    return nodes


def _encode_function_code(obj: types.FunctionType) -> str | None:
    """Encode the AST node that `obj` was compiled from, or None if it cannot be
    singled out in the function's source. The node is matched by the code's name and
    first line, and by the source positions of its instructions, so that lambdas
    sharing a line are told apart."""
    code = obj.__code__
    try:
        lines, _ = inspect.findsource(obj)
    except (OSError, TypeError):
        return None
    positions = [
        (line, col, end_line, end_col)
        for line, end_line, col, end_col in code.co_positions()
        if None not in (line, end_line, col, end_col)
        and (line, col) != (end_line, end_col)
    ]

    def matches(node: ast.AST) -> bool:
        if isinstance(node, ast.ClassDef):
            return False
        name = node.name if isinstance(node, _FUNCTION_DEFS) else "<lambda>"
        start = (node.lineno, node.col_offset)
        end = (node.end_lineno, node.end_col_offset)
        return name == code.co_name and all(
            start <= p[:2] and p[2:] <= end for p in positions
        )

    nodes = _definition_nodes(code.co_filename, lines).get(code.co_firstlineno, [])
    encodings = {encode_ast(n) for n in nodes if matches(n)}
    return encodings.pop() if len(encodings) == 1 else None


def check_code_hash(code_hash: str) -> str:
    """Return `code_hash` if it names a tier of `CODE_HASH_TIERS`, otherwise raise"""
    if code_hash not in CODE_HASH_TIERS:
        raise ValueError(
            f"code_hash must be one of {', '.join(map(repr, CODE_HASH_TIERS))}, "
            f"not {code_hash!r}"
        )
    return code_hash


def code_hash_default() -> str:
    """The `code_hash` tier of tasks that do not set one, read on each call: the
    ``PYDRA_CODE_HASH`` environment variable if set, otherwise ``"user"``"""
    return check_code_hash(os.environ.get(CODE_HASH_ENV_VAR, "user"))


@functools.cache
def _library_roots() -> tuple[str, ...]:
    """The directories of the standard library and installed distributions"""
    paths = {sysconfig.get_paths().get(k) for k in ("stdlib", "platstdlib")}
    paths.update(sysconfig.get_paths().get(k) for k in ("purelib", "platlib"))
    paths.update(getattr(site, "getsitepackages", list)())
    if site.ENABLE_USER_SITE:
        paths.add(site.getusersitepackages())
    return tuple(sorted(os.path.join(os.path.realpath(p), "") for p in paths if p))


@functools.lru_cache(maxsize=4096)
def _path_is_library(path: str) -> bool | None:
    """Whether a source file is library code, or None for a pseudo-file (``<...>``)
    other than a frozen module"""
    if path.startswith("<"):
        return True if path.startswith("<frozen") else None
    real = os.path.realpath(path)
    parts = Path(real).parts
    return (
        "site-packages" in parts
        or "dist-packages" in parts
        or real.startswith(_library_roots())
    )


@functools.lru_cache(maxsize=8)
def _package_overrides(setting: str) -> tuple[tuple[str, bool], ...]:
    """The packages whose classification is set, and whether each is library code,
    most specific first: `LIBRARY_PACKAGES`, then the comma-separated entries of
    `setting`, ``name`` for user code and ``-name`` for library code"""
    packages = dict.fromkeys(LIBRARY_PACKAGES, True)
    for entry in setting.split(","):
        entry = entry.strip()
        if entry.startswith("-"):
            packages[entry[1:].strip()] = True
        elif entry:
            packages[entry] = False
    return tuple(sorted(packages.items(), key=lambda p: -len(p[0])))


def _package_override(module: str) -> bool | None:
    for package, is_library in _package_overrides(
        os.environ.get(USER_PACKAGES_ENV_VAR, "")
    ):
        if module == package or module.startswith(package + "."):
            return is_library
    return None


def _module_is_library(module: str, file: str | None = None) -> bool:
    """Whether the module named `module` is library code: a package of
    ``PYDRA_USER_PACKAGES`` is user code and one prefixed by ``-`` (or pydra) library
    code; otherwise a module is library code if its file is in the standard library or
    site-packages, or, when it has no file, if its name is a standard library module's
    """
    override = _package_override(module)
    if override is not None:
        return override
    if file is None:
        file = getattr(sys.modules.get(module), "__file__", None)
    if file:
        in_library = _path_is_library(file)
        if in_library is not None:
            return in_library
    return module.partition(".")[0] in STDLIB_MODULES


def is_library(obj: object) -> bool:
    """Whether `obj` is library code (see `_module_is_library`), which is identified
    by its qualified name: a function by the file of its code, a module by its own
    file, and anything else by its ``__module__``. A function whose globals are not
    those of its module (defined by ``exec`` or in a notebook namespace) is user code
    unless its file is library code."""
    if isinstance(obj, types.ModuleType):
        return _module_is_library(obj.__name__, getattr(obj, "__file__", None))
    if isinstance(obj, type):
        module = obj.__module__
    else:
        module = inspect.getattr_static(obj, "__module__", None)
    if isinstance(obj, types.FunctionType):
        module = obj.__module__ or ""
        override = _package_override(module)
        if override is None:
            override = _path_is_library(obj.__code__.co_filename)
        if override is not None:
            return override
        if getattr(sys.modules.get(module), "__dict__", None) is not obj.__globals__:
            return False
    return isinstance(module, str) and _module_is_library(module)


def _qualified_name(obj: object) -> str:
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "")
    return f"{getattr(obj, '__module__', None)}.{name}"


def _owner_name(obj: object, holder: object = None) -> str | None:
    """``module.qualname`` of a function or class if its live module holds `obj` (or
    `holder`, such as the task class made from a function) at that name, otherwise
    None"""
    module, qualname = obj.__module__, obj.__qualname__
    target = sys.modules.get(module) if isinstance(module, str) else None
    for part in qualname.split("."):
        target = getattr(target, part, None)
    held = target is obj or (holder is not None and target is holder)
    return f"{module}.{qualname}" if held else None


def _library_instance_name(value: object, name: str) -> tuple[str, str] | None:
    """The module and name at which a library module of `value`'s ``__module__`` or
    type holds `value`, otherwise None"""
    tp = type(value)
    if tp.__module__ == "builtins":
        return None
    modules = (inspect.getattr_static(value, "__module__", None), tp.__module__)
    for module_name in dict.fromkeys(m for m in modules if isinstance(m, str)):
        module = sys.modules.get(module_name)
        if module is None or not is_library(module):
            continue
        if inspect.getattr_static(module, name, None) is value:
            return module_name, name
        attrs_dict = vars(module)
        for key in sorted(attrs_dict):
            if attrs_dict[key] is value:
                return module_name, key
    return None


def type_module(tp: type) -> str:
    """The module of a type: its ``__module__``, except for an extension type that
    reports ``builtins`` without being a builtin, which is attributed to a module that
    holds it (one of the standard library if any does)"""
    module = tp.__module__
    if module != "builtins" or getattr(builtins, tp.__name__, None) is tp:
        return module
    return _holder_module(tp) or module


@functools.lru_cache(maxsize=256)
def _holder_module(tp: type) -> str | None:
    holders = []
    for name, module in list(sys.modules.items()):
        namespace = getattr(module, "__dict__", None)
        if namespace and any(v is tp for v in list(namespace.values())):
            holders.append(name)
    stdlib = [h for h in holders if h.partition(".")[0] in STDLIB_MODULES]
    return (stdlib or holders or [None])[0]


_GLOBAL_OPS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})
_ATTR_OPS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})
_global_chains_cache = _LRUCache(FUNCTION_BYTES_CACHE_SIZE)


def _global_chains(code: types.CodeType) -> tuple[tuple[str, ...], ...]:
    """The global names read by `code` and the code it defines, each with the
    attributes read from it in turn (``("utils", "helper")`` for ``utils.helper``)"""
    cached = _global_chains_cache.get(weakref.ref(code))
    if cached is not None:
        return cached
    chains = set()
    stack = [code]
    while stack:
        co = stack.pop()
        chain: tuple[str, ...] = ()
        for instr in dis.get_instructions(co):
            if instr.opname in _GLOBAL_OPS:
                if chain:
                    chains.add(chain)
                chain = (instr.argval,)
            elif chain and instr.opname in _ATTR_OPS:
                chain += (instr.argval,)
            elif instr.opname != "EXTENDED_ARG" and chain:
                chains.add(chain)
                chain = ()
        if chain:
            chains.add(chain)
        stack.extend(c for c in co.co_consts if isinstance(c, types.CodeType))
    result = tuple(sorted(chains))
    _global_chains_cache.put(weakref.ref(code), result)
    return result


def _function_code(obj: types.FunctionType, cache: Cache) -> bytes:
    """The encoded code of a function: its `ast@1` encoding, or its byte-code tagged
    with the Python version when its source is not available"""
    source_file = obj.__code__.co_filename
    try:
        mtime_ns = os.stat(source_file).st_mtime_ns
    except OSError:
        mtime_ns = None
    cache_key = (source_file, mtime_ns, weakref.ref(obj.__code__))
    cached = _function_bytes_cache.get(cache_key)
    if cached is not None:
        return cached
    encoded = _encode_function_code(obj)
    if encoded is None:
        # not cached, because the code object's bytes depend on the cache arg
        version = f"bytecode@{sys.version_info[0]}.{sys.version_info[1]}:"
        return version.encode() + b"".join(bytes_repr(obj.__code__, cache))
    result = f"{AST_ENCODING}:{encoded}".encode()
    if mtime_ns is not None:
        _function_bytes_cache.put(cache_key, result)
    return result


def _member_functions(member: object) -> list[types.FunctionType]:
    """The functions of a class attribute: a method, the function of a static or class
    method, or the accessors of a property"""
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    if isinstance(member, property):
        return [f for f in (member.fget, member.fset, member.fdel) if f is not None]
    return [member] if isinstance(member, types.FunctionType) else []


def _class_members(klass: type) -> list[tuple[str, types.FunctionType]]:
    """The functions defined in the body of `klass`, by attribute name, leaving out
    those generated by decorators such as `attrs.define` or `dataclasses.dataclass`"""
    try:
        source_file = inspect.getsourcefile(klass)
    except (OSError, TypeError):
        source_file = None
    return [
        (name, func)
        for name, member in sorted(vars(klass).items())
        for func in _member_functions(member)
        if (
            func.__code__.co_filename == source_file
            if source_file
            else _cell_or_file(func.__code__.co_filename)
        )
    ]


def _cell_or_file(filename: str) -> bool:
    """Whether code was compiled from a file or a registered notebook cell, rather
    than generated (by `attrs` or `dataclasses`)"""
    if not filename.startswith("<"):
        return True
    return filename in linecache.cache and not filename.startswith("<attrs generated")


def _class_code(klass: type, cache: Cache) -> bytes:
    """The encoded code of a class: the `ast@1` encoding of its definition, or the
    encoded code of its functions when its source is not available"""
    try:
        lines, _ = inspect.findsource(klass)
        source_file = inspect.getsourcefile(klass) or ""
    except (OSError, TypeError):
        lines = None
    if lines is not None:
        try:
            mtime_ns = os.stat(source_file).st_mtime_ns
        except OSError:
            mtime_ns = None
        cache_key = (
            source_file,
            mtime_ns,
            klass.__qualname__,
            getattr(klass, "__firstlineno__", None),
        )
        cached = _function_bytes_cache.get(cache_key)
        if cached is not None:
            return cached
        first = getattr(klass, "__firstlineno__", None)
        encodings = {
            encode_ast(node)
            for line, nodes in _definition_nodes(source_file, lines).items()
            if first is None or line == first
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == klass.__name__
        }
        if len(encodings) == 1:
            result = f"class:{AST_ENCODING}:{encodings.pop()}".encode()
            if mtime_ns is not None:
                _function_bytes_cache.put(cache_key, result)
            return result
    return (
        b"class-members:("
        + b",".join(
            name.encode() + b"=" + _function_code(func, cache)
            for name, func in _class_members(klass)
        )
        + b")"
    )


class _Empty:
    """The value of an empty closure cell"""


_EMPTY_CELL = _Empty()


class _CodeWalk:
    """The code a function reaches at a `code_hash` tier and the values it reads.

    Each definition reached is numbered in the order it is found, starting with the
    function at 0, and is encoded once, as its code and the references it reads, so
    cycles and long call chains are walked without recursion. A reference is a
    closure cell, a global (followed through the attributes read from it while it is
    a user module), or a default of a function other than a task's own. It is
    serialized as:

    * library code (see `is_library`) by its qualified name, never its code, when its
      live module holds it at that name, otherwise (a closure made by a library
      factory, a generated function) by its definition, as user code is;
    * a function or class of user code within the tier by its definition number, and
      one outside the tier by its qualified name when its live module holds it there,
      otherwise by its definition;
    * a task class by its hash, or by the qualified name of its function if that is
      library code;
    * any other value by the ordinary value rules, and the type of a user-class
      instance as a reference.

    The modules of the library code and of the library-typed values it reaches are
    collected in `modules`, and added to the cache's, for the record kept beside a
    result.
    """

    def __init__(self, root: types.FunctionType, cache: Cache, defaults: bool):
        self.cache = cache
        self.code_hash = check_code_hash(cache.code_hash or code_hash_default())
        self.root = root
        self.defaults = defaults
        self.definitions: list[ty.Any] = [root]
        self.numbers = {id(root): 0}
        # set when a reference is serialized from values, so cannot be memoised
        self.dynamic = False
        self.modules: set[str] = set()
        self.found: set[str] = set()  # the modules of the reference being serialized

    def library(self, module: str, ref: str) -> bytes:
        self.found.add(module)
        return ref.encode()

    def in_scope(self, obj: ty.Any) -> bool:
        if self.code_hash == "user":
            return True
        return self.code_hash == "module" and obj.__module__ == self.root.__module__

    def number(self, obj: ty.Any) -> int:
        if id(obj) not in self.numbers:
            self.numbers[id(obj)] = len(self.definitions)
            self.definitions.append(obj)
        return self.numbers[id(obj)]

    def function_references(
        self, func: types.FunctionType, prefix: str = ""
    ) -> Iterator[tuple[str, ty.Any]]:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
            try:
                value = cell.cell_contents
            except ValueError:
                value = _EMPTY_CELL
            yield f"{prefix}closure:{name}", value
        namespace = func.__globals__
        found = {}
        for chain in _global_chains(func.__code__):
            if chain[0] not in namespace:
                continue  # a builtin, or not defined
            value, path = namespace[chain[0]], chain[0]
            for attr in chain[1:]:
                if not isinstance(value, types.ModuleType) or is_library(value):
                    break
                try:
                    value = getattr(value, attr)
                except AttributeError:
                    break
                path += "." + attr
            found[path] = value
        for path in sorted(found):
            yield f"{prefix}global:{path}", found[path]
        if func is not self.root or self.defaults:
            for i, value in enumerate(func.__defaults__ or ()):
                yield f"{prefix}default:{i}", value
            for name, value in sorted((func.__kwdefaults__ or {}).items()):
                yield f"{prefix}kwdefault:{name}", value

    def class_references(self, klass: type) -> Iterator[tuple[str, ty.Any]]:
        for i, base in enumerate(klass.__bases__):
            yield f"base:{i}", base
        for name, func in _class_members(klass):
            yield from self.function_references(func, prefix=f"{name}.")

    def reference(self, name: str, value: ty.Any, reader: ty.Any) -> bytes:
        try:
            return self.serialize(value, name.rpartition(":")[2].split(".")[-1])
        except Exception as e:
            kind, _, var = name.rpartition(":")
            verb = "captured" if kind.endswith("closure") else "read"
            what = "class" if isinstance(reader, type) else "function"
            add_exc_note(
                e,
                f"in {var!r}, {verb} by {what} {reader.__qualname__!r}; pass it as an "
                "input, or give the task a checksum=",
            )
            raise

    def serialize(self, value: ty.Any, name: str) -> bytes:
        if value is _EMPTY_CELL:
            return b"<empty>"
        if isinstance(value, types.ModuleType):
            if is_library(value):
                self.found.add(value.__name__)
            return f"module:{value.__name__}".encode()
        if isinstance(value, logging.Logger):
            return f"logger:{value.name}".encode()
        if isinstance(value, logging.LoggerAdapter):
            return f"logger:{value.logger.name}".encode()
        if isinstance(value, type) and (fields := get_fields(value)):
            field = fields.get(getattr(value, "_executor_name", None) or "")
            executor = getattr(field, "default", None)
            if (
                isinstance(executor, types.FunctionType)
                and is_library(executor)
                and (owner := _owner_name(executor, holder=value))
            ):
                return self.library(executor.__module__, f"lib-task:{owner}")
            self.dynamic = True
            return b"task:" + bytes(hash_single(value, self.cache))
        if isinstance(value, (types.FunctionType, type)):
            library = is_library(value)
            if library and (owner := _owner_name(value)):
                ref = self.library(value.__module__, f"lib:{owner}")
                wrapped = vars(value).get("__wrapped__")
                if isinstance(value, types.FunctionType) and wrapped is not None:
                    self.dynamic = True
                    ref += b"(wraps:" + self.serialize(wrapped, name) + b")"
                return ref
            if not library and not self.in_scope(value):
                if owner := _owner_name(value):
                    return f"name:{owner}".encode()
            return f"#{self.number(value)}".encode()
        if isinstance(value, types.MethodType):
            self.dynamic = True
            return (
                b"method:("
                + self.serialize(value.__func__, name)
                + b","
                + self.serialize(value.__self__, name)
                + b")"
            )
        if isinstance(value, (types.BuiltinFunctionType, types.MethodWrapperType)):
            owner = value.__self__
            if owner is None or isinstance(owner, types.ModuleType):
                module = getattr(value, "__module__", None) or owner.__name__
                return self.library(module, f"lib:{module}.{value.__qualname__}")
            self.dynamic = True
            return (
                f"method:({value.__name__},".encode()
                + self.serialize(owner, name)
                + b")"
            )
        if isinstance(
            value,
            (
                types.MethodDescriptorType,
                types.WrapperDescriptorType,
                types.ClassMethodDescriptorType,
                types.GetSetDescriptorType,
                types.MemberDescriptorType,
            ),
        ):
            return self.library(
                value.__objclass__.__module__,
                f"lib:{_qualified_name(value.__objclass__)}.{value.__name__}",
            )
        if isinstance(value, functools.partial):
            self.dynamic = True
            return (
                b"partial:("
                + self.serialize(value.func, name)
                + b",args:("
                + b",".join(self.serialize(a, name) for a in value.args)
                + b"),keywords:("
                + b",".join(
                    k.encode() + b"=" + self.serialize(v, name)
                    for k, v in sorted(value.keywords.items())
                )
                + b"))"
            )
        if (owner := _library_instance_name(value, name)) is not None:
            return self.library(owner[0], f"lib:{owner[0]}.{owner[1]}")
        tp = type(value)
        wrapped = getattr(value, "__dict__", {}).get("__wrapped__")
        self.dynamic = True
        if is_library(tp):
            self.found.add(type_module(tp))
        if wrapped is not None and callable(value) and is_library(tp):
            return (
                f"wrapper:{_qualified_name(tp)}(".encode()
                + self.serialize(wrapped, name)
                + b")"
            )
        if isinstance(value, (dict, list, set, bytearray)):
            logger.debug("%r is mutable and is hashed by the value it holds now", name)
        ref = b"value:" + bytes(hash_single(value, self.cache))
        if not is_library(tp):
            ref += b",type:" + self.serialize(tp, name)
        return ref

    def checked(self, obj: ty.Any) -> list[tuple[str, ty.Any]]:
        """What a memoised walk must find unchanged in `obj`: its references and its
        code (a function's code object, a class's member functions)"""
        if isinstance(obj, type):
            refs = list(self.class_references(obj))
            return refs + [(f"member:{n}", f) for n, f in _class_members(obj)]
        return list(self.function_references(obj)) + [("code", obj.__code__)]

    def __iter__(self) -> Iterator[bytes]:
        """The serialized code: the function's code alone if it reads no references
        (so equal at every tier), otherwise tagged by the tier and followed by its
        references and the definitions they reach.

        The walk is memoised per process by function, tier and ``PYDRA_USER_PACKAGES``
        (see `_WalkMemo`)."""
        key = (self.code_hash, self.defaults, os.environ.get(USER_PACKAGES_ENV_VAR, ""))
        with _walk_memo_lock:
            memo = _walk_memo.get(self.root, {}).get(key)
        entries = self.replay(memo) if memo is not None else None
        if entries is None:
            self.definitions = [self.root]
            self.numbers = {id(self.root): 0}
            entries, memo = self.walk(key)
            with _walk_memo_lock:
                _walk_memo.setdefault(self.root, {})[key] = memo
        self.cache.modules.update(self.modules)
        if len(entries) == 1 and not entries[0][1]:
            yield entries[0][0]
            return
        yield f"{self.code_hash}/".encode() + entries[0][0]
        yield b",refs:(" + entries[0][1] + b")"
        for n, (code, refs) in enumerate(entries[1:], start=1):
            yield f",#{n}=(".encode() + code + b",refs:(" + refs + b"))"

    def walk(self, key: tuple) -> tuple[list[tuple[bytes, bytes]], "_WalkMemo"]:
        memo = _WalkMemo(self.root, key)
        self.modules = set()
        entries = []
        n = 0
        while n < len(self.definitions):
            obj = self.definitions[n]
            if is_library(obj):
                memo.modules.add(obj.__module__)
            if isinstance(obj, type):
                code = _class_code(obj, self.cache)
                try:
                    source_file = inspect.getsourcefile(obj)
                except (OSError, TypeError):
                    source_file = None
            else:
                code = _function_code(obj, self.cache)
                source_file = obj.__code__.co_filename
            if source_file and source_file not in memo.files:
                memo.files[source_file] = _mtime(source_file)
            checked, refs, statics = [], [], []
            for name, value in self.checked(obj):
                if name == "code" or name.startswith("member:"):
                    checked.append((name, memo.ref(value)))
                    continue
                self.dynamic, self.found = False, set()
                ref = self.reference(name, value, obj)
                refs.append(name.encode() + b"=" + ref)
                held = None if self.dynamic else memo.ref(value)
                (self.modules if held is None else memo.modules).update(self.found)
                checked.append((name, held))
                statics.append(None if held is None else ref)
            memo.checked.append(checked)
            memo.codes.append(code)
            memo.parts.append(statics)
            entries.append((code, b",".join(refs)))
            n += 1
        memo.definitions = [memo.ref(d) for d in self.definitions]
        if None in memo.definitions:
            memo.definitions = []  # cannot be checked, so never replayed
        self.modules.update(memo.modules)
        return entries, memo

    def replay(self, memo: "_WalkMemo") -> list[tuple[bytes, bytes]] | None:
        """The entries of a memoised walk if it still holds, otherwise None"""
        definitions = [ref() for ref in memo.definitions]
        if not definitions or any(d is None for d in definitions):
            return None
        if any(_mtime(f) != mtime for f, mtime in memo.files.items()):
            return None
        self.definitions = definitions
        self.numbers = {id(d): i for i, d in enumerate(definitions)}
        self.modules = set(memo.modules)
        entries = []
        for obj, checked, code, statics in zip(
            definitions, memo.checked, memo.codes, memo.parts
        ):
            current = self.checked(obj)
            if len(current) != len(checked) or any(
                name != was or (held is not None and held() is not value)
                for (name, value), (was, held) in zip(current, checked)
            ):
                return None
            parts = []
            for (name, value), static in zip(_references(current), statics):
                if static is None:
                    self.found = set()
                    static = self.reference(name, value, obj)
                    self.modules.update(self.found)
                parts.append(name.encode() + b"=" + static)
            entries.append((code, b",".join(parts)))
        if len(self.definitions) != len(definitions):
            return None  # a reference serialized again reached another definition
        return entries


def _references(checked: list[tuple[str, ty.Any]]) -> list[tuple[str, ty.Any]]:
    return [
        (name, value)
        for name, value in checked
        if name != "code" and not name.startswith("member:")
    ]


def _mtime(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


class _WalkMemo:
    """A memoised `_CodeWalk`, holding nothing it reached alive: weak references to
    the definitions it reached, to what each read and to their code, the code, the
    serialization of each reference that is not serialized from values (None for
    those that are, or that cannot be referenced weakly), the mtimes of the source
    files reached, and the modules of the library code reached by the definitions
    and by the references it serialized. It is reused while those files keep their
    mtimes and every reference read is still the same object, and dropped when any of
    them dies."""

    def __init__(self, root: types.FunctionType, key: tuple):
        self.definitions: list[weakref.ref] = []
        self.checked: list[list[tuple[str, weakref.ref | None]]] = []
        self.codes: list[bytes] = []
        self.parts: list[list[bytes | None]] = []
        self.files: dict[str, int | None] = {}
        self.modules: set[str] = set()
        root_ref = weakref.ref(root)
        memo_ref = weakref.ref(self)

        def drop(_):
            with _walk_memo_lock:
                root = root_ref()
                memos = _walk_memo.get(root) if root is not None else None
                memo = memo_ref()
                if memos is not None and memo is not None and memos.get(key) is memo:
                    del memos[key]

        self._drop = drop

    def ref(self, obj: ty.Any) -> weakref.ref | None:
        try:
            return weakref.ref(obj, self._drop)
        except TypeError:
            return None


# Memoised walks by function, then by tier, defaults and ``PYDRA_USER_PACKAGES``
_walk_memo: "weakref.WeakKeyDictionary[types.FunctionType, dict]" = (
    weakref.WeakKeyDictionary()
)
_walk_memo_lock = threading.RLock()


@attrs.frozen
class TaskFunction:
    """The function of a task as it enters the checksum. It is hashed as the function
    is, except for its defaults, which are hashed as the task's inputs."""

    function: types.FunctionType


def _bytes_repr_code(
    obj: types.FunctionType, cache: Cache, defaults: bool
) -> Iterator[bytes]:
    yield b"function:("
    if in_stdlib(obj):
        yield f"{obj.__module__}.{obj.__name__}".encode()
    else:
        yield from _CodeWalk(obj, cache, defaults)
    yield b")"


def library_modules(func: types.FunctionType) -> frozenset[str]:
    """The modules of the library code, and of the library-typed values, that `func`
    reaches at the ``"user"`` tier, and `func`'s own if it is library code (see
    `is_library`). It walks the function again, so it is for a task whose checksum
    did not walk it at that tier."""
    if in_stdlib(func) or is_library(func):
        return frozenset({func.__module__})
    cache = Cache(code_hash="user")
    for _ in _CodeWalk(func, cache, defaults=False):
        pass
    return frozenset(cache.modules)


@register_serializer
def bytes_repr_function(obj: types.FunctionType, cache: Cache) -> Iterator[bytes]:
    """Serialize a function by the `ast@1` encoding of its source (see `encode_ast`),
    falling back to its byte-code, tagged with the Python version, when the source is
    not available, together with the code it reaches at the cache's `code_hash` tier
    and the values it reads (see `_CodeWalk`)"""
    yield from _bytes_repr_code(obj, cache, defaults=True)


@register_serializer
def bytes_repr_task_function(obj: TaskFunction, cache: Cache) -> Iterator[bytes]:
    yield from _bytes_repr_code(obj.function, cache, defaults=False)


def bytes_repr_mapping_contents(mapping: Mapping, cache: Cache) -> Iterator[bytes]:
    """Serialize the contents of a mapping

    Concatenates byte-serialized keys and hashed values.

    .. code-block:: python

        >>> from pydra.utils.hash import bytes_repr_mapping_contents, Cache
        >>> generator = bytes_repr_mapping_contents({"a": 1, "b": 2}, Cache())
        >>> b''.join(generator)
        b'str:1:a=...str:1:b=...'
    """
    for key in sorted(mapping):
        yield from bytes_repr(key, cache)
        yield b"="
        yield bytes(hash_single(mapping[key], cache))
        yield b","


def bytes_repr_sequence_contents(seq: Sequence, cache: Cache) -> Iterator[bytes]:
    """Serialize the contents of a sequence

    Concatenates hashed values.

    .. code-block:: python

        >>> from pydra.utils.hash import bytes_repr_sequence_contents, Cache
        >>> generator = bytes_repr_sequence_contents([1, 2], Cache())
        >>> list(generator)
        [b'\x6d...', b'\xa3...']
    """
    for val in seq:
        yield bytes(hash_single(val, cache))


if HAVE_NUMPY:

    @register_serializer(numpy.generic)
    @register_serializer(numpy.ndarray)
    def bytes_repr_numpy(obj: numpy.ndarray, cache: Cache) -> Iterator[bytes]:
        yield f"{obj.__class__.__module__}{obj.__class__.__name__}:{obj.size}:".encode()
        if obj.dtype == "object":
            yield from bytes_repr_sequence_contents(iter(obj.ravel()), cache)
        else:
            yield obj.tobytes(order="C")


NUMPY_CHUNK_LEN = 8192
