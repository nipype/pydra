"""Generic object hashing dispatch"""

import sys
import os
import re
import ast
import struct
import inspect
from datetime import datetime
import typing as ty
import types
from pathlib import Path
from collections.abc import Mapping
from functools import singledispatch
from hashlib import blake2b
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
from pydra.utils.general import in_stdlib, user_cache_root, add_exc_note

logger = logging.getLogger("pydra")

FUNCTION_SRC_CHUNK_LEN_DEFAULT = 8192

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

    # Set the location of the persistent hash cache
    LOCATION_ENV_VAR = "PYDRA_HASH_CACHE"
    CLEANUP_ENV_VAR = "PYDRA_HASH_CACHE_CLEANUP_PERIOD"

    @classmethod
    def location_default(cls):
        try:
            location = os.environ[cls.LOCATION_ENV_VAR]
        except KeyError:
            location = user_cache_root / "hashes"
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
        """Cleans up old hash caches that haven't been accessed in the last 30 days"""
        now = datetime.now()
        for path in self.location.iterdir():
            if path.name.endswith(".lock"):
                continue
            days = (now - datetime.fromtimestamp(path.lstat().st_atime)).days
            if days > self.cleanup_period:
                path.unlink()

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


@register_serializer(FileSet)
def bytes_repr_fileset(
    fileset: FileSet, cache: Cache
) -> Iterator[ty.Union[CacheKey, bytes]]:
    fspaths = sorted(fileset.fspaths)
    # Yield the cache key for the fileset, which is a tuple of the file-system paths
    # and their mtime. Is used to store persistent cache of the fileset hashes
    # to avoid recomputation between calls
    yield CacheKey(
        tuple(repr(p) for p in fspaths)  # type: ignore[arg-type]
        + tuple(p.lstat().st_mtime_ns for p in fspaths)
    )
    cls = type(fileset)
    yield f"{cls.__module__}.{cls.__name__}:".encode()
    for key, chunk_iter in fileset.byte_chunks():
        yield (",'" + key + "'=").encode()
        yield from chunk_iter


# Need to disable the mtime cache key for mocked filesets. Used in doctests
@register_serializer(MockMixin)
def bytes_repr_mock_fileset(
    mock_fileset: MockMixin, cache: Cache
) -> Iterator[ty.Union[CacheKey, bytes]]:
    cls = type(mock_fileset)
    yield f"{cls.__module__}.{cls.__name__}:".encode()
    for key, _ in mock_fileset.byte_chunks():
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


@register_serializer
def bytes_repr_function(obj: types.FunctionType, cache: Cache) -> Iterator[bytes]:
    """Serialize a function, attempting to use the AST of the source code if available
    otherwise falling back to the byte-code of the function."""
    yield b"function:("
    if in_stdlib(obj):
        yield f"{obj.__module__}.{obj.__name__}".encode()
    else:
        try:
            src = inspect.getsource(obj)
        except OSError:
            # Fallback to using the bytes representation of the code object
            yield from bytes_repr(obj.__code__, cache)
        else:

            def dump_ast(node: ast.AST) -> bytes:
                return ast.dump(
                    node, annotate_fields=False, include_attributes=False
                ).encode()

            def strip_annotations(node: ast.AST):
                """Remove annotations from function arguments."""
                if hasattr(node, "args"):
                    for arg in node.args.args:
                        arg.annotation = None
                    for arg in node.args.kwonlyargs:
                        arg.annotation = None
                    if node.args.vararg:
                        node.args.vararg.annotation = None
                    if node.args.kwarg:
                        node.args.kwarg.annotation = None

            indent = re.match(r"(\s*)", src).group(1)
            if indent:
                src = re.sub(f"^{indent}", "", src, flags=re.MULTILINE)
            try:
                func_ast = ast.parse(src).body[0]
                strip_annotations(func_ast)
                if hasattr(func_ast, "args"):
                    yield dump_ast(func_ast.args)
                if hasattr(func_ast, "body"):
                    for stmt in func_ast.body:
                        yield dump_ast(stmt)
            except SyntaxError:
                yield src.encode()
    yield b")"


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
