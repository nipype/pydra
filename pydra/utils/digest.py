"""Typed digests, RFC 8785 canonical JSON and the shared tag registry (RFC 0004 §6.2).

``digest(value)`` is one string ``"<tag>@<n>:<hex>"``. ``register_codec`` and
``register_fileset_tag`` fill one registry of tags that RFC 0003's store reads back; a tag
is only ever looked up there, never resolved as a module or class name.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import math
import typing as ty
from decimal import Decimal

import attrs
import numpy as np
from fileformats.core.fileset import FileSet
from numpy.lib import format as npformat
import os

JSON = ty.Any

__all__ = [
    "Codec",
    "canonical_npy",
    "digest",
    "digest_form",
    "fileset_digest",
    "jcs",
    "lookup_tag",
    "register_codec",
    "register_digest",
    "register_fileset_tag",
    "tag_of",
]

MAX_SAFE_INT = 2**53


# ---------------------------------------------------------------------------
# RFC 8785 JSON canonicalisation
# ---------------------------------------------------------------------------


def _es_number(x: float) -> str:
    """ECMAScript ``Number::toString`` of a finite double."""
    if math.isnan(x) or math.isinf(x):
        raise ValueError("JCS cannot represent NaN or infinity")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    _, digit_tuple, exp = Decimal(repr(abs(x))).as_tuple()
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


def _utf16_key(s: str) -> bytes:
    return s.encode("utf-16-be", "surrogatepass")


def _jcs(obj: JSON, out: list[str]) -> None:
    if obj is None:
        out.append("null")
    elif obj is True:
        out.append("true")
    elif obj is False:
        out.append("false")
    elif isinstance(obj, str):
        out.append(json.dumps(obj, ensure_ascii=False))
    elif isinstance(obj, int):
        if abs(obj) > MAX_SAFE_INT:
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
        out.append("{")
        for i, key in enumerate(sorted(obj, key=_utf16_key)):
            if not isinstance(key, str):
                raise TypeError(
                    f"JCS object keys must be strings, got {type(key).__name__}"
                )
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _jcs(obj[key], out)
        out.append("}")
    else:
        raise TypeError(f"{type(obj).__name__} is not JSON")


def jcs(obj: JSON) -> bytes:
    """The RFC 8785 canonical form of a JSON value, as UTF-8 bytes."""
    out: list[str] = []
    _jcs(obj, out)
    return "".join(out).encode("utf-8", "surrogatepass")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Tag registry
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class Codec:
    """A registered tag: a ``$codec`` (with encode/decode) or a ``$fileset`` type tag."""

    type: type
    tag: str
    encode: ty.Callable | None = None
    decode: ty.Callable | None = None
    fileset: bool = False


_BY_TAG: dict[str, Codec] = {}
_BY_TYPE: dict[type, Codec] = {}
_DIGESTS: dict[type, ty.Callable[[ty.Any], str]] = {}


def _register(entry: Codec) -> None:
    existing = _BY_TAG.get(entry.tag)
    if existing is not None and (existing.type, existing.fileset) != (
        entry.type,
        entry.fileset,
    ):
        raise ValueError(
            f"tag {entry.tag!r} is already registered for {existing.type!r}"
        )
    prior = _BY_TYPE.get(entry.type)
    if prior is not None and prior.tag != entry.tag:
        raise ValueError(
            f"{entry.type!r} is already registered under tag {prior.tag!r}"
        )
    _BY_TAG[entry.tag] = entry
    _BY_TYPE[entry.type] = entry


def register_codec(
    type_: type, tag: str, encode: ty.Callable, decode: ty.Callable
) -> None:
    """Register ``type_`` under ``$codec`` tag ``tag``; ``encode`` gives JSON-able data."""
    _register(Codec(type_, tag, encode, decode))


def register_fileset_tag(type_: type, tag: str) -> None:
    """Register a ``FileSet`` subclass under the ``$fileset`` type tag ``tag``."""
    _register(Codec(type_, tag, fileset=True))


def register_digest(type_: type, fn: ty.Callable[[ty.Any], str]) -> None:
    """Register ``fn(value) -> "<tag>@<n>:<hex>"`` as the digest of exactly ``type_``."""
    _DIGESTS[type_] = fn


def lookup_tag(tag: str) -> Codec | None:
    """The entry registered under ``tag``; looks in the registry only, never imports."""
    return _BY_TAG.get(tag)


def tag_of(type_: type) -> str | None:
    """The tag registered for exactly ``type_``, or None."""
    entry = _BY_TYPE.get(type_)
    return entry.tag if entry is not None else None


# ---------------------------------------------------------------------------
# Canonical .npy
# ---------------------------------------------------------------------------


def _holds_references(dtype: np.dtype) -> bool:
    if dtype.hasobject or dtype.kind == "T":
        return True
    if dtype.fields:
        return any(_holds_references(f[0]) for f in dtype.fields.values())
    if dtype.subdtype is not None:
        return _holds_references(dtype.subdtype[0])
    return False


def _is_plain_array(value: ty.Any) -> bool:
    return type(value) is np.ndarray and not _holds_references(value.dtype)


def canonical_npy(arr: np.ndarray) -> bytes:
    """``.npy`` 1.0 bytes of ``arr`` as little-endian, C-ordered data."""
    little = arr.dtype.newbyteorder("<")
    data = np.array(arr, dtype=little, order="C", copy=True, subok=False)
    buf = io.BytesIO()
    npformat.write_array(buf, data, version=(1, 0), allow_pickle=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Digest forms and digests
# ---------------------------------------------------------------------------


class _NoDigestForm(Exception):
    """The value has no portable digest form (it falls back to ``py@1``)."""


def fileset_digest(type_tag: str, members: ty.Sequence[tuple[str, str]]) -> str:
    """``file@1`` of a set: ``members`` are ``(key, content sha256)`` pairs."""
    files = [[k, s] for k, s in sorted(members, key=lambda m: m[0])]
    return "file@1:" + _sha256(jcs({"$fileset": type_tag, "files": files}))


def _fileset_members(value: FileSet) -> list[tuple[str, str]]:
    members = []
    for key, chunks in value.byte_chunks(
        ignore_hidden_files=False, ignore_hidden_dirs=False
    ):
        h = hashlib.sha256()
        for chunk in chunks:
            h.update(chunk)
        members.append((key, h.hexdigest()))
    return members


def _file_digest(value: FileSet) -> str:
    return fileset_digest(_BY_TYPE[type(value)].tag, _fileset_members(value))


def _is_tagged_fileset(value: ty.Any) -> bool:
    entry = _BY_TYPE.get(type(value))
    return isinstance(value, FileSet) and entry is not None and entry.fileset


def _npscalar(value: np.generic) -> JSON:
    dtype = value.dtype
    if dtype.kind not in "iufcbmM" or dtype in (
        np.dtype(np.longdouble),
        np.dtype(np.clongdouble),
    ):
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
    elif kind in "mM":
        text = str(int(value.astype("int64")))
    else:
        raise _NoDigestForm
    return {"$npscalar": [little.str, text]}


def digest_form(value: ty.Any) -> JSON:
    """RFC 0003's output encoding with typed numbers and tagged tuples (RFC 0004 §6.2).

    Raises ``_NoDigestForm`` (an ``Exception``) for a value that has none.
    """
    if value is None or isinstance(value, bool):
        return value
    entry = _BY_TYPE.get(type(value))
    if entry is not None and not entry.fileset:
        return {"$codec": entry.tag, "value": digest_form(entry.encode(value))}
    if entry is not None and isinstance(value, FileSet):
        return {
            "$fileset": entry.tag,
            "files": [[k, s] for k, s in sorted(_fileset_members(value))],
        }
    tp = type(value)
    if tp is int:
        return {"$int": str(value)}
    if tp is float:
        return {"$float": value.hex() if math.isfinite(value) else str(value)}
    if tp is str:
        return value
    if tp is complex:
        return {"$complex": [float(value.real).hex(), float(value.imag).hex()]}
    if tp is bytes:
        return {"$bytes": _sha256(value)}
    if tp is list:
        return [digest_form(v) for v in value]
    if tp is tuple:
        return {"$tuple": [digest_form(v) for v in value]}
    if tp in (set, frozenset):
        forms = [digest_form(v) for v in value]
        return {"$set": sorted(forms, key=jcs)}
    if tp is dict:
        if all(type(k) is str for k in value):
            return {
                ("$" + k if k.startswith("$") else k): digest_form(v)
                for k, v in value.items()
            }
        pairs = [[digest_form(k), digest_form(v)] for k, v in value.items()]
        return {"$dict": sorted(pairs, key=lambda p: jcs(p[0]))}
    if tp in (datetime.datetime, datetime.date):
        return {"$datetime": value.isoformat()}
    if isinstance(value, os.PathLike) and not isinstance(value, FileSet):
        return {"$path": os.fspath(value)}
    if _is_plain_array(value):
        return {"$array": _sha256(canonical_npy(value))}
    if tp.__module__ == "numpy" and isinstance(value, np.generic):
        return _npscalar(value)
    raise _NoDigestForm


def _py_digest(value: ty.Any) -> str:
    from pydra.utils.hash import hash_object

    return "py@1:" + hash_object(value).hex()


def digest(value: ty.Any) -> str:
    """The typed digest ``"<tag>@<n>:<hex>"`` of ``value``; the first §6.2 row that applies."""
    fn = _DIGESTS.get(type(value))
    if fn is not None:
        return fn(value)
    if _is_plain_array(value):
        return "npy@1:" + _sha256(canonical_npy(value))
    if _is_tagged_fileset(value):
        return _file_digest(value)
    try:
        form = digest_form(value)
    except _NoDigestForm:
        return _py_digest(value)
    return "json@1:" + _sha256(jcs(form))
