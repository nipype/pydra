"""Typed digests, RFC 8785 canonical JSON and the shared tag registry (RFC 0004 §6.2).

``digest(value)`` is one string ``"<tag>@<n>:<hex>"``. ``register_codec``,
``register_fileset_tag`` and ``register_digest`` fill one registry of types; the tags are
what RFC 0003's store reads back, and a tag is only ever looked up there, never resolved as
a module or class name. numpy is optional: without it the numpy rows are inactive.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import math
import os
import re
import typing as ty
from decimal import Decimal

import attrs
from fileformats.core.exceptions import FormatDefinitionError, FormatRecognitionError
from fileformats.core.fileset import FileSet, MockMixin

try:
    import numpy as np
    from numpy.lib import format as npformat
except ImportError:  # numpy is optional
    np = None  # type: ignore[assignment]
    npformat = None  # type: ignore[assignment]

JSON = ty.Any

__all__ = [
    "Codec",
    "canonical_npy",
    "digest",
    "digest_form",
    "fileset_digest",
    "holds_references",
    "is_canonical_array",
    "jcs",
    "lookup_tag",
    "register_codec",
    "register_digest",
    "register_fileset_tag",
    "tag_of",
]

# Integers beyond 2**53 are not exact as ECMAScript doubles (this bound is one above
# JavaScript's MAX_SAFE_INTEGER, and 2**53 itself is exact).
MAX_EXACT_INT = 2**53


# ---------------------------------------------------------------------------
# RFC 8785 JSON canonicalisation
# ---------------------------------------------------------------------------


def _es_number(x: float) -> str:
    """ECMAScript ``Number::toString`` of a finite double."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("JCS cannot represent NaN or infinity")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    _, digit_tuple, exp = Decimal(float.__repr__(abs(x))).as_tuple()
    digits = "".join(map(str, digit_tuple)).rstrip("0") or "0"
    # the value is 0.<digits> * 10**n
    n = len(digit_tuple) + exp
    k = len(digits)
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mant = digits[0] + ("." + digits[1:] if k > 1 else "")
        body = f"{mant}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + body


def _check_str(s: str) -> str:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(
            "JCS strings must be well-formed Unicode (RFC 8785 3.2.2.2); found a lone surrogate"
        ) from None
    return s


def _utf16_key(s: str) -> bytes:
    return s.encode("utf-16-be")


def _jcs(obj: JSON, out: list[str]) -> None:
    if obj is None:
        out.append("null")
    elif obj is True:
        out.append("true")
    elif obj is False:
        out.append("false")
    elif isinstance(obj, str):
        out.append(json.dumps(_check_str(obj), ensure_ascii=False))
    elif isinstance(obj, int):
        if abs(obj) > MAX_EXACT_INT:
            raise ValueError(
                f"integer {obj} is not exact as a JSON number; digest forms carry it as a string"
            )
        out.append(str(obj))
    elif isinstance(obj, float):
        out.append(_es_number(obj))
    elif isinstance(obj, (list, tuple)):
        out.append("[")
        for i, item in enumerate(obj):
            if i:
                out.append(",")
            _jcs(item, out)
        out.append("]")
    elif isinstance(obj, dict):
        for key in obj:
            if not isinstance(key, str):
                raise TypeError(
                    f"JCS object keys must be strings, got {type(key).__name__}"
                )
            _check_str(key)
        out.append("{")
        for i, key in enumerate(sorted(obj, key=_utf16_key)):
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _jcs(obj[key], out)
        out.append("}")
    else:
        raise TypeError(f"{type(obj).__name__} is not JSON")


def jcs(obj: JSON) -> bytes:
    """The RFC 8785 canonical form of a JSON value, as UTF-8 bytes.

    Raises ``TypeError`` for a non-string key or a non-JSON value and ``ValueError`` for
    NaN, infinity, an integer above 2**53 or a string with a lone surrogate.
    """
    out: list[str] = []
    _jcs(obj, out)
    return "".join(out).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Tag registry
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class Codec:
    """The registry entry of one type: a ``$codec`` (``encode``/``decode``), a ``$fileset``
    type tag, and/or a registered ``digest`` function. ``tag`` is None for a digest-only
    entry."""

    type: type
    tag: str | None = None
    encode: ty.Callable | None = None
    decode: ty.Callable | None = None
    fileset: bool = False
    digest: ty.Callable[[ty.Any], str] | None = None
    # a tag pydra gave a ``fileformats`` class itself; a user registration replaces it
    default: bool = False


_BY_TAG: dict[str, Codec] = {}
_BY_TYPE: dict[type, Codec] = {}
_DIGEST_RE = re.compile(r"[a-z][a-z0-9_]*@[0-9]+:[0-9a-f]{16,}")
BUILT_IN_SCHEMES = frozenset({"json", "npy", "file", "py", "class", "code", "type"})


def _register(entry: Codec) -> None:
    assert entry.tag is not None
    existing = _BY_TAG.get(entry.tag)
    if existing is not None and (existing.type, existing.fileset) != (
        entry.type,
        entry.fileset,
    ):
        raise ValueError(
            f"tag {entry.tag!r} is already registered for {existing.type!r}"
        )
    prior = _entry(entry.type)
    if prior is not None:
        if not prior.default and prior.tag not in (None, entry.tag):
            raise ValueError(
                f"{entry.type!r} is already registered under tag {prior.tag!r}"
            )
        # a default tag is replaced by a registration; a registered digest is kept, and
        # the default tag stays readable through `lookup_fileset_tag`
        entry = attrs.evolve(entry, digest=prior.digest)
    _BY_TAG[entry.tag] = entry
    _BY_TYPE[entry.type] = entry


def register_codec(
    type_: type, tag: str, encode: ty.Callable, decode: ty.Callable
) -> None:
    """Register ``type_`` under ``$codec`` tag ``tag``; ``encode`` gives JSON-able data."""
    _register(Codec(type_, tag, encode, decode))


def register_fileset_tag(type_: type, tag: str) -> None:
    """Register a ``FileSet`` subclass under the ``$fileset`` type tag ``tag``."""
    if not (isinstance(type_, type) and issubclass(type_, FileSet)):
        raise TypeError(f"{type_!r} is not a FileSet subclass")
    _register(Codec(type_, tag, fileset=True))


def register_digest(
    type_: type, fn: ty.Callable[[ty.Any], str], *, replace: bool = False
) -> None:
    """Register ``fn(value) -> "<tag>@<n>:<hex>"`` as the digest of exactly ``type_``.

    It applies at any depth, where a digest form holds ``{"$digest": "<digest>"}``. A type
    that already has one raises ``ValueError`` unless ``replace=True``.
    """
    prior = _entry(type_)
    if prior is not None and prior.digest is not None and not replace:
        raise ValueError(f"{type_!r} already has a registered digest")
    entry = (
        attrs.evolve(prior, digest=fn) if prior is not None else Codec(type_, digest=fn)
    )
    _BY_TYPE[type_] = entry
    if entry.tag is not None and not entry.default:
        _BY_TAG[entry.tag] = entry


def _registered_digest(entry: Codec, value: ty.Any) -> str:
    assert entry.digest is not None
    result = entry.digest(value)
    if not isinstance(result, str) or not _DIGEST_RE.fullmatch(result):
        raise ValueError(
            f"digest function for {entry.type!r} returned {result!r}, "
            "not '<tag>@<n>:<hex>'"
        )
    if result.split("@", 1)[0] in BUILT_IN_SCHEMES:
        raise ValueError(
            f"digest function for {entry.type!r} returned {result!r}, whose scheme "
            f"is one of the built-in ones ({', '.join(sorted(BUILT_IN_SCHEMES))})"
        )
    return result


def _entry(type_: type) -> Codec | None:
    """The registry entry of exactly ``type_``; a ``fileformats`` class that nobody
    registered gets the default one (`_default_fileset_entry`)."""
    entry = _BY_TYPE.get(type_)
    if entry is None and isinstance(type_, type) and issubclass(type_, FileSet):
        entry = _default_fileset_entry(type_)
    return entry


def entry_of(type_: type) -> Codec | None:
    """The entry of exactly ``type_``: registered, or a default ``$fileset`` one."""
    return _entry(type_)


def lookup_tag(tag: str) -> Codec | None:
    """The entry registered under ``tag``; looks in the registry only, never imports."""
    return _BY_TAG.get(tag)


def tag_of(type_: type) -> str | None:
    """The tag registered for exactly ``type_``, or None."""
    entry = _entry(type_)
    return entry.tag if entry is not None else None


# ---------------------------------------------------------------------------
# Default file set tags
# ---------------------------------------------------------------------------


def _mime_like(type_: type) -> str | None:
    """``mime_like`` of a ``FileSet`` class, or None for one with none (defined outside
    the ``fileformats`` namespace) and for a mock, which has no files."""
    if issubclass(type_, MockMixin):
        return None
    try:
        return str(type_.mime_like)
    except FormatDefinitionError:
        return None


def _resolve_tag(tag: str) -> type | None:
    """The class ``fileformats`` resolves ``tag`` to, if its own ``mime_like`` is ``tag``.
    This imports from the ``fileformats`` namespace only. Only ``fileformats``' answer
    that no class has the tag means None; any other error, such as an installed
    ``fileformats`` plugin that fails to import, propagates."""
    from fileformats.core import from_mime

    try:
        klass = from_mime(tag)
    except FormatRecognitionError:
        return None
    if isinstance(klass, type) and issubclass(klass, FileSet):
        if _mime_like(klass) == tag:
            return klass
    return None


def _default_fileset_entry(type_: type) -> Codec | None:
    """The entry of a ``FileSet`` class nobody registered: its ``mime_like`` is the tag,
    computed here, on demand, whatever package the class comes from. A class with no
    ``mime_like``, a mock, and a class whose tag ``fileformats`` cannot resolve back to
    a class (it could not be read) have none, and are ``py@1``. Not entered in the tag
    registry: `lookup_fileset_tag` resolves a tag."""
    tag = _mime_like(type_)
    if tag is None or _resolve_tag(tag) is None:
        return None
    entry = Codec(type_, tag, fileset=True, default=True)
    _BY_TYPE[type_] = entry
    return entry


def lookup_fileset_tag(tag: str, declared: ty.Iterable[type] = ()) -> Codec | None:
    """The entry that decodes the ``$fileset`` tag ``tag``: a declared class whose own
    ``mime_like`` is ``tag`` (the reader says which class it expects), else a user
    registration of ``tag``, else the class ``fileformats`` resolves the tag to. A class
    is accepted under its ``mime_like`` whatever tag it is registered under now."""
    for klass in declared:
        if (
            isinstance(klass, type)
            and issubclass(klass, FileSet)
            and _mime_like(klass) == tag
        ):
            entry = _entry(klass) or Codec(klass, tag, fileset=True, default=True)
            return entry
    entry = _BY_TAG.get(tag)
    if entry is not None:
        return entry
    klass = _resolve_tag(tag)
    if klass is not None:
        return _entry(klass) or Codec(klass, tag, fileset=True, default=True)
    return None


# ---------------------------------------------------------------------------
# Canonical .npy
# ---------------------------------------------------------------------------


def _dtype_any(dtype: "np.dtype", pred: ty.Callable[["np.dtype"], bool]) -> bool:
    if pred(dtype):
        return True
    if dtype.fields:
        return any(_dtype_any(f[0], pred) for f in dtype.fields.values())
    return dtype.subdtype is not None and _dtype_any(dtype.subdtype[0], pred)


def holds_references(dtype: "np.dtype") -> bool:
    """True if elements of ``dtype`` are references (object, ``StringDType``, or a structured
    or sub-array dtype holding one): ``tobytes()`` of such an array is pointers, not content.
    """
    return _dtype_any(dtype, lambda d: d.hasobject or d.kind == "T")


def is_canonical_array(value: ty.Any) -> bool:
    """True for an ``ndarray`` exactly (no subclass) whose dtype holds no references: the
    arrays that have a canonical ``.npy`` form. Any other array is ``py@1`` or, in a store,
    unencodable."""
    return (
        np is not None
        and type(value) is np.ndarray
        and not holds_references(value.dtype)
    )


def canonical_npy(arr: "np.ndarray") -> bytes:
    """``.npy`` 1.0 bytes of ``arr`` as little-endian, C-ordered data.

    Raises ``TypeError`` unless ``is_canonical_array(arr)``: a subclass such as
    ``MaskedArray`` would lose its mask.
    """
    if not is_canonical_array(arr):
        raise TypeError(f"{type(arr).__name__} has no canonical .npy form")
    little = arr.dtype.newbyteorder("<")
    data = np.array(arr, dtype=little, order="C", copy=True, subok=False)
    buf = io.BytesIO()
    npformat.write_array(buf, data, version=(1, 0), allow_pickle=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Digest forms and digests
# ---------------------------------------------------------------------------


class _NoDigestForm(Exception):
    """The value has no portable digest form; ``digest`` falls back to ``py@1``."""


def fileset_digest(type_tag: str, members: ty.Sequence[tuple[str, str]]) -> str:
    """``file@1`` of a set: ``members`` are ``(key, content sha256)`` pairs."""
    files = [[k, s] for k, s in sorted(members, key=lambda m: m[0])]
    return "file@1:" + _sha256(jcs({"$fileset": type_tag, "files": files}))


def _posix_key(key: str, sep: str = os.sep) -> str:
    """A member key with ``/`` separators on every OS."""
    return key.replace(sep, "/") if sep != "/" else key


def _fileset_members(value: FileSet, cache: ty.Any = None) -> list[tuple[str, str]]:
    """The ``(key, sha256)`` members of ``value``, through the stat-to-digest caches of
    ``pydra.utils.hash`` (``cache`` is a ``hash.Cache``)."""
    from pydra.utils.hash import fileset_members

    return fileset_members(value, _read_fileset_members, cache)


def _read_fileset_members(value: FileSet) -> list[tuple[str, str]]:
    members = []
    for key, chunks in value.byte_chunks(
        ignore_hidden_files=False, ignore_hidden_dirs=False
    ):
        h = hashlib.sha256()
        for chunk in chunks:
            h.update(chunk)
        members.append((_posix_key(key), h.hexdigest()))
    return members


def _is_tagged_fileset(value: ty.Any) -> bool:
    entry = _entry(type(value))
    return isinstance(value, FileSet) and entry is not None and entry.fileset


def _npscalar(value: "np.generic") -> JSON:
    dtype = value.dtype
    # longdouble and its complex are 'g' and 'G'; float64 ('d') and complex128 ('D') stay
    # encodable on platforms where long double has the same width
    if dtype.kind not in "iufcbmM" or dtype.char in "gG":
        raise _NoDigestForm
    little = dtype.newbyteorder("<")
    kind = dtype.kind
    if kind in "iu":
        text: ty.Any = str(int(value))
    elif kind == "f":
        text = float(value).hex()
    elif kind == "c":
        text = [float(value.real).hex(), float(value.imag).hex()]
    elif kind == "b":
        text = "true" if bool(value) else "false"
    else:  # "m", "M"
        text = str(int(value.astype("int64")))
    return {"$npscalar": [little.str, text]}


_UNREGISTERED = object()


class FormWalker:
    """The walk over RFC 0003 section 6.4's table that both the digest form and the entry
    encoder take; a subclass says how numbers, strings, tuples, members, paths and values
    with no form are written."""

    tag_tuples = True

    def __init__(self, cache: ty.Any = None) -> None:
        self.cache = cache

    def walk(self, value: ty.Any) -> JSON:
        tp = type(value)
        entry = _entry(tp)
        if entry is not None:
            form = self.registered(entry, value)
            if form is not _UNREGISTERED:
                return form
        if value is None or tp is bool:
            return value
        if tp is int:
            return self.int_(value)
        if tp is float:
            return self.float_(value)
        if tp is str:
            return self.str_(value)
        if tp is complex:
            return {
                "$complex": [
                    {"$float": _float_tag(value.real)},
                    {"$float": _float_tag(value.imag)},
                ]
            }
        if tp is bytes:
            return {"$bytes": self.bytes_(value)}
        if tp is list:
            return [self.walk(v) for v in value]
        if tp is tuple:
            items = [self.walk(v) for v in value]
            return {"$tuple": items} if self.tag_tuples else items
        if tp in (set, frozenset):
            return {"$set": sorted((self.walk(v) for v in value), key=self.sort_key)}
        if tp is dict:
            if all(type(k) is str for k in value):
                return {
                    ("$" + k if k.startswith("$") else k): self.walk(v)
                    for k, v in value.items()
                }
            pairs = [[self.walk(k), self.walk(v)] for k, v in value.items()]
            return {"$dict": sorted(pairs, key=lambda p: self.sort_key(p[0]))}
        if tp in (datetime.datetime, datetime.date):
            return {"$datetime": self.datetime_(value)}
        if self.is_path(value):
            return {"$path": self.path_(value)}
        if is_canonical_array(value):
            return {"$array": self.array(value)}
        if (
            np is not None
            and tp.__module__ == "numpy"
            and isinstance(value, np.generic)
        ):
            try:
                return _npscalar(value)
            except _NoDigestForm:
                pass
        return self.fallback(value)

    def registered(self, entry: Codec, value: ty.Any) -> JSON:
        """The form of a value whose type is registered, or ``_UNREGISTERED`` to walk on"""
        if entry.digest is not None:
            return {"$digest": _registered_digest(entry, value)}
        if entry.tag is not None and not entry.fileset:
            assert entry.encode is not None
            form = {"$codec": entry.tag, "value": self.walk(entry.encode(value))}
            # a value whose type is user code carries its class (RFC 0004 section 6.2)
            from pydra.utils.hash import class_digest, is_user_class

            if is_user_class(type(value)):
                form["$class"] = class_digest(type(value))
            return form
        if entry.fileset and isinstance(value, FileSet):
            return {
                "$fileset": entry.tag,
                "files": [
                    [k, s] for k, s in sorted(_fileset_members(value, self.cache))
                ],
            }
        return _UNREGISTERED

    def int_(self, value: int) -> JSON:
        return {"$int": str(value)}

    def float_(self, value: float) -> JSON:
        return {"$float": _float_tag(value)}

    def str_(self, value: str) -> JSON:
        return value

    def bytes_(self, value: bytes) -> str:
        return _sha256(value)

    def array(self, value: ty.Any) -> str:
        return _sha256(canonical_npy(value))

    def datetime_(self, value: "datetime.date") -> str:
        return value.isoformat()

    def is_path(self, value: ty.Any) -> bool:
        return isinstance(value, os.PathLike) and not isinstance(value, FileSet)

    def path_(self, value: ty.Any) -> JSON:
        return os.fsdecode(value)

    def sort_key(self, form: JSON) -> bytes:
        return jcs(form)

    def fallback(self, value: ty.Any) -> JSON:
        raise _NoDigestForm


def _float_tag(x: float) -> str:
    return x.hex() if math.isfinite(x) else str(x)


def digest_form(value: ty.Any, cache: ty.Any = None) -> JSON:
    """RFC 0003's output encoding with typed numbers and tagged tuples (RFC 0004 §6.2).

    Raises an ``Exception`` for a value that has none, which ``digest`` turns into ``py@1``.
    ``cache`` is the ``pydra.utils.hash.Cache`` that file sets are hashed through.
    """
    return FormWalker(cache).walk(value)


def _py_digest(value: ty.Any, cache: ty.Any = None) -> str:
    from pydra.utils.hash import hash_object

    return "py@1:" + hash_object(value, cache=cache).hex()


def digest(value: ty.Any, cache: ty.Any = None) -> str:
    """The typed digest ``"<tag>@<n>:<hex>"`` of ``value``; the first §6.2 row that applies.

    ``cache`` is a ``pydra.utils.hash.Cache``, shared by the ``py@1`` fallback and the
    file sets, which are read through its persistent stat-to-digest cache."""
    entry = _entry(type(value))
    if entry is not None and entry.digest is not None:
        return _registered_digest(entry, value)
    if is_canonical_array(value):
        return "npy@1:" + _sha256(canonical_npy(value))
    if _is_tagged_fileset(value):
        entry = _entry(type(value))
        assert entry is not None
        return fileset_digest(str(entry.tag), _fileset_members(value, cache))
    try:
        form = digest_form(value, cache)
    except _NoDigestForm:
        return _py_digest(value, cache)
    return "json@1:" + _sha256(jcs(form))
