"""Output codecs of ``pydra-entry@1``: the encoding table of RFC 0003 section 6.4.

`Encoder` turns an output value into JSON plus members written to a staging directory, and
`Decoder` turns that JSON and the verified members back into a value. Tags (``$codec`` and
``$fileset``) live in the one registry of `pydra.utils.digest`, which RFC 0004's digests fill
too; a tag is only ever looked up there, never resolved as a module or class name.
"""

from __future__ import annotations

import collections.abc
import datetime
import hashlib
import inspect
import math
import os
import pickle
import re
import sys
import types
import typing as ty
import uuid
from pathlib import Path

from fileformats.core.fileset import FileSet

from pydra.engine.portable.errors import (
    CorruptEntry,
    EntryFormatUnknown,
    EntryNotTrusted,
    UnencodableOutput,
)
from pydra.utils.digest import (
    _UNREGISTERED,
    Codec,
    FormWalker,
    _posix_key,
    canonical_npy,
    fileset_digest,
    is_canonical_array,
    jcs,
    lookup_tag,
    np,
    npformat,
    tag_of,
)
from pydra.engine.portable.layout import FINGERPRINT
from pydra.utils.storagedir import StorageDir, WrongKindError, copy_hashed

__all__ = ["Encoder", "Decoder", "output_digest", "typed", "type_is_encodable"]

JSON = ty.Any
_NPDTYPE = re.compile(r"[<|][biufcmM][0-9]+(\[[A-Za-z0-9]+\])?")
_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def typed(encoded: JSON) -> JSON:
    """``encoded`` with every number a typed string (``{"$int": ...}`` and ``{"$float":
    hex}``), as RFC 0004 section 6.2 does, so that JCS neither conflates ``1`` and ``1.0`` nor
    loses a large integer."""
    if encoded is None or isinstance(encoded, (bool, str)):
        return encoded
    if isinstance(encoded, int):
        return {"$int": str(encoded)}
    if isinstance(encoded, float):
        return {"$float": encoded.hex()}
    if isinstance(encoded, list):
        return [typed(v) for v in encoded]
    if isinstance(encoded, dict):
        return {k: typed(v) for k, v in encoded.items()}
    raise TypeError(f"{type(encoded).__name__} is not JSON")


def output_digest(encoded: ty.Mapping[str, JSON]) -> str:
    """The output digest of an entry: the sha256, in hex, of the RFC 8785 canonical form of the
    typed encoded outputs. Arrays, bytes and file sets enter by the digests that name their
    members."""
    return hashlib.sha256(jcs(typed(dict(encoded)))).hexdigest()


def _canon(encoded: JSON) -> bytes:
    return jcs(typed(encoded))


# -- encoding ----------------------------------------------------------------------------


class Encoder(FormWalker):
    """Encodes the outputs of one entry into ``staging``.

    The walk is `FormWalker`'s with plain numbers and untagged tuples. Members (arrays,
    bytes, pickles, file sets) are written to ``staging`` as they are met,
    given the policy's file mode and, with ``durability="member"``, ``fsync``ed. ``members``
    records each one's sha256 and size.

    Parameters
    ----------
    staging : StorageDir
        The writer's own staging directory.
    can_pickle : bool
        Whether a value with no codec may be cloudpickled into a ``.pkl`` member.
    allow_naive : bool
        Whether a naive datetime may be written (not in a ``shared`` root).
    """

    tag_tuples = False

    def __init__(
        self, staging: StorageDir, *, can_pickle: bool, allow_naive: bool
    ) -> None:
        self.staging = staging
        self.policy = staging.policy
        self.can_pickle = can_pickle
        self.allow_naive = allow_naive
        self.members: dict[str, dict[str, ty.Any]] = {}

    # members

    def write(self, name: str, data: bytes) -> None:
        """Write the member ``name`` once; a second write of the same name is skipped, because
        member names that are digests of their content cannot differ."""
        if name in self.members:
            return
        with self.staging.open_new(name, self.policy.file_mode) as f:
            f.write(data)
            f.flush()
            if self.policy.durability == "member":
                os.fsync(f.fileno())
        self.members[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }

    def _blob(self, data: bytes, suffix: str) -> str:
        sha = hashlib.sha256(data).hexdigest()
        self.write(f"{sha}{suffix}", data)
        return sha

    # values

    def encode_outputs(self, outputs: ty.Mapping[str, ty.Any]) -> dict[str, JSON]:
        return {name: self.encode(value, name) for name, value in outputs.items()}

    def encode(self, value: ty.Any, field: str) -> JSON:
        """The JSON form of ``value``.

        Raises
        ------
        UnencodableOutput
            ``value`` has no codec and cannot be pickled here.
        """
        self._field = field
        return self.walk(value)

    def registered(self, entry: Codec, value: ty.Any) -> JSON:
        if entry.tag is None:
            return _UNREGISTERED
        if entry.fileset:
            return self._fileset(value, entry.tag, self._field)
        assert entry.encode is not None
        return {"$codec": entry.tag, "value": self.walk(entry.encode(value))}

    def int_(self, value: int) -> JSON:
        return value

    def float_(self, value: float) -> JSON:
        return value if math.isfinite(value) else {"$float": str(value)}

    def str_(self, value: str) -> JSON:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise UnencodableOutput(
                self._field, str, "a string with a lone surrogate"
            ) from None
        return value

    def bytes_(self, value: bytes) -> str:
        return self._blob(value, ".bin")

    def array(self, value: ty.Any) -> str:
        return self._blob(canonical_npy(value), ".npy")

    def datetime_(self, value: datetime.date) -> str:
        if (
            type(value) is datetime.datetime
            and value.utcoffset() is None
            and not self.allow_naive
        ):
            raise UnencodableOutput(
                self._field,
                type(value),
                "a naive datetime is not written to a shared root; give it a time zone",
            )
        return value.isoformat()

    def is_path(self, value: ty.Any) -> bool:
        return isinstance(value, Path) and type(value) is type(Path())

    def sort_key(self, form: JSON) -> bytes:
        return _canon(form)

    def fallback(self, value: ty.Any) -> JSON:
        return self._pickle(value, self._field)

    def _pickle(self, value: ty.Any, field: str) -> JSON:
        tp = type(value)
        if not self.can_pickle:
            raise UnencodableOutput(field, tp)
        import cloudpickle

        try:
            data = cloudpickle.dumps(value)
        except Exception as err:
            raise UnencodableOutput(field, tp, f"it cannot be pickled: {err}") from err
        return {
            "$pickle": self._blob(data, ".pkl"),
            "cloudpickle": cloudpickle.__version__,
            "python": _PYTHON,
        }

    # file sets

    def _fileset(self, value: ty.Any, tag: str, field: str) -> JSON:
        tp = type(value)
        if not isinstance(value, FileSet):
            raise UnencodableOutput(field, tp, f"tag {tag!r} is not for a FileSet")
        fspaths = sorted(value.fspaths)
        names = [p.name for p in fspaths]
        if len(set(names)) != len(names):
            raise UnencodableOutput(field, tp, "two of its paths have the same name")
        paths = value.all_file_paths(
            ignore_hidden_files=False, ignore_hidden_dirs=False
        )
        keys = [
            _posix_key(k)
            for k, _ in value.byte_chunks(
                ignore_hidden_files=False, ignore_hidden_dirs=False
            )
        ]
        if not paths:
            raise UnencodableOutput(field, tp, "it holds no files")
        dirs = [p for p in fspaths if p.is_dir()]
        for top in dirs:
            for here, subdirs, _files in os.walk(top):
                for sub in subdirs:
                    if os.path.islink(os.path.join(here, sub)):
                        raise UnencodableOutput(
                            field,
                            tp,
                            f"{os.path.join(here, sub)} is a symlink to a directory, which "
                            "is not copied",
                        )
        planned: list[tuple[str, Path, str]] = []
        for key, src in zip(keys, paths):
            owner = next((d for d in dirs if src.is_relative_to(d)), None)
            rel = (
                src.name
                if owner is None
                else f"{owner.name}/{src.relative_to(owner).as_posix()}"
            )
            planned.append((key, src, rel))
        files = self.staging.child("files", create=True, mode=self.policy.entry_mode)
        scratch = f".{uuid.uuid4().hex}"
        tree = files.child(scratch, create=True, mode=self.policy.entry_mode)
        made: list[StorageDir] = [tree]
        try:
            copied: list[tuple[str, str, str, int]] = []
            for key, src, rel in planned:
                *dir_parts, leaf = rel.split("/")
                dest = tree
                for part in dir_parts:
                    dest = dest.child(part, create=True, mode=self.policy.entry_mode)
                    made.append(dest)
                try:
                    sha, size = copy_hashed(src, dest, leaf, self.policy.file_mode)
                except WrongKindError as err:
                    raise UnencodableOutput(
                        field, tp, f"{src} is not a regular file"
                    ) from err
                copied.append((key, rel, sha, size))
            set_hex = fileset_digest(tag, [(k, sha) for k, _, sha, _ in copied]).split(
                ":", 1
            )[1]
            if self.policy.durability == "member":
                for d in made:
                    d.sync()
        finally:
            for d in made:
                d.close()
        _merge_set(files, scratch, set_hex)
        if self.policy.durability == "member":
            files.sync()
        files.close()
        for key, rel, sha, size in copied:
            self.members[f"files/{set_hex}/{rel}"] = {"sha256": sha, "size": size}
        listing = sorted(
            ([k, f"files/{set_hex}/{rel}"] for k, rel, _, _ in copied),
            key=lambda p: p[0],
        )
        return {"$fileset": tag, "files": listing}


def _remove_tree(parent: StorageDir, name: str) -> None:
    with parent.child(name) as d:
        for entry in list(d.scandir()):
            if entry.is_dir():
                _remove_tree(d, entry.name)
            else:
                d.unlink(entry.name)
    parent.rmdir(name)


def _merge_set(files: StorageDir, scratch: str, set_hex: str) -> None:
    """Make ``files/<scratch>`` the set directory ``files/<set_hex>``. When an equal set is
    already there (the name is the digest of keys and contents, not of the file names), each
    top-level name of the new tree that the set does not hold yet is moved into it, and the
    rest of the scratch tree is dropped."""
    try:
        files.lstat(set_hex)
    except FileNotFoundError:
        files.rename(scratch, files, set_hex)
        return
    with files.child(scratch) as tree, files.child(set_hex) as target:
        have = {e.name for e in target.scandir()}
        for e in list(tree.scandir()):
            if e.name not in have:
                tree.rename(e.name, target, e.name)
        if target.policy.durability == "member":
            target.sync()
    _remove_tree(files, scratch)


# -- decoding ----------------------------------------------------------------------------

_SINGLE = {
    "$float",
    "$complex",
    "$bytes",
    "$set",
    "$dict",
    "$datetime",
    "$path",
    "$array",
    "$npscalar",
}
_PAIRS = {
    "$codec": {"$codec", "value"},
    "$fileset": {"$fileset", "files"},
    "$pickle": {"$pickle", "cloudpickle", "python"},
}


def _unescape(key: str) -> str:
    return key[1:] if key.startswith("$$") else key


def _tag_keys(doc: dict) -> list[str]:
    return [k for k in doc if k.startswith("$") and not k.startswith("$$")]


class Decoder:
    """Decodes the JSON of an entry's outputs.

    Parameters
    ----------
    fetch : callable
        ``fetch(member)`` gives the verified bytes of a ``.npy``, ``.bin`` or ``.pkl`` member;
        it raises `CorruptEntry` for one that is missing or unlisted and `EntryNotTrusted` for
        a pickle the root's trust does not admit.
    members : mapping
        ``entry.json``'s ``members``: each name's ``sha256`` and ``size``.
    entry_path : str
        Where the entry is, for the paths of file sets.
    allow_pickle : bool
        Whether the root's trust admits pickle members.
    """

    def __init__(
        self,
        fetch: ty.Callable[[str], bytes],
        members: ty.Mapping[str, ty.Mapping[str, ty.Any]],
        entry_path: str,
        *,
        allow_pickle: bool,
    ) -> None:
        self.fetch = fetch
        self.members = members
        self.entry_path = entry_path
        self.allow_pickle = allow_pickle
        self.referenced: set[str] = set()
        self.trees: list[tuple[str, list[str]]] = []
        """The file sets decoded: each one's set name and its top-level names."""

    def decode_outputs(self, outputs: ty.Mapping[str, JSON]) -> dict[str, ty.Any]:
        return {name: self.decode(doc) for name, doc in outputs.items()}

    def decode(self, doc: JSON, hashable: bool = False) -> ty.Any:
        """The value of ``doc``. ``hashable`` is set for a set element or a dict key, where a
        JSON array is a tuple and a ``$set`` a frozenset."""
        if doc is None or isinstance(doc, (bool, int, float, str)):
            return doc
        if isinstance(doc, list):
            items = [self.decode(v, hashable) for v in doc]
            return tuple(items) if hashable else items
        if not isinstance(doc, dict):
            raise CorruptEntry(f"{type(doc).__name__} is not JSON")
        tags = _tag_keys(doc)
        if not tags:
            return {_unescape(k): self.decode(v) for k, v in doc.items()}
        if hashable and "$dict" in doc:
            raise CorruptEntry("a dict cannot be a set element or a dict key")
        return self._tagged(doc, tags, hashable)

    def _tagged(self, doc: dict, tags: list[str], hashable: bool) -> ty.Any:
        if len(tags) != 1:
            raise CorruptEntry(f"an object with more than one tag: {sorted(tags)}")
        tag = tags[0]
        if tag == "$digest":
            raise CorruptEntry(
                "'$digest' is reserved for input digests and is never an output"
            )
        if tag in _SINGLE:
            if set(doc) != {tag}:
                raise CorruptEntry(f"{tag} carries other keys: {sorted(doc)}")
        elif tag in _PAIRS:
            if set(doc) != _PAIRS[tag]:
                raise CorruptEntry(f"{tag} has keys {sorted(doc)}")
        else:
            raise EntryFormatUnknown(f"output tag {tag!r} is not known to this version")
        body = doc[tag]
        try:
            return getattr(self, "_" + tag[1:])(doc, body, hashable)
        except (CorruptEntry, EntryFormatUnknown, EntryNotTrusted):
            raise
        except (KeyError, TypeError, ValueError, OverflowError, IndexError) as err:
            raise CorruptEntry(f"{tag} is malformed: {err}") from err

    # tagged forms

    def _float(self, doc: dict, body: JSON, hashable: bool) -> float:
        if not isinstance(body, str):
            raise ValueError("not a string")
        if body in _NONFINITE:
            return _NONFINITE[body]
        value = float.fromhex(body)
        if value.hex() != body:  # only what float.hex writes
            raise ValueError(f"{body!r} is not a float.hex form")
        return value

    def _complex(self, doc: dict, body: JSON, hashable: bool) -> complex:
        re_, im = body
        if not (
            isinstance(re_, dict)
            and isinstance(im, dict)
            and set(re_) == {"$float"} == set(im)
        ):
            raise ValueError("a complex is two $float parts")
        return complex(
            self._float(re_, re_["$float"], False), self._float(im, im["$float"], False)
        )

    def _member(self, name: str, sha: JSON) -> bytes:
        if not (isinstance(sha, str) and FINGERPRINT.fullmatch(sha)):
            raise ValueError("a member is named by its sha256")
        self.referenced.add(name)
        return self.fetch(name)

    def _bytes(self, doc: dict, body: JSON, hashable: bool) -> bytes:
        return self._member(f"{body}.bin", body)

    def _array(self, doc: dict, body: JSON, hashable: bool) -> ty.Any:
        data = self._member(f"{body}.npy", body)
        if np is None:
            raise EntryFormatUnknown(
                "this entry holds an array and numpy is not installed"
            )
        import io

        try:
            array = npformat.read_array(io.BytesIO(data), allow_pickle=False)
        except Exception as err:
            raise CorruptEntry(f"{body}.npy is not a readable array: {err}") from err
        if not is_canonical_array(array) or canonical_npy(array) != data:
            raise CorruptEntry(f"{body}.npy is not a canonical .npy")
        return array

    def _pickle(self, doc: dict, body: JSON, hashable: bool) -> ty.Any:
        name = f"{body}.pkl"
        if doc["python"] != _PYTHON:
            raise EntryFormatUnknown(
                f"a pickle written by Python {doc['python']} is not read by Python {_PYTHON}"
            )
        if not self.allow_pickle:
            raise EntryNotTrusted(
                f"{name} is a pickle and this root does not admit them"
            )
        data = self._member(name, body)
        try:
            return pickle.loads(data)
        except Exception as err:
            raise EntryFormatUnknown(f"{name} cannot be unpickled here: {err}") from err

    def _set(self, doc: dict, body: JSON, hashable: bool) -> ty.Any:
        if not isinstance(body, list):
            raise ValueError("a $set holds an array")
        items = [self.decode(v, True) for v in body]
        return frozenset(items) if hashable else set(items)

    def _dict(self, doc: dict, body: JSON, hashable: bool) -> dict:
        out = {}
        for pair in body:
            k, v = pair
            out[self.decode(k, True)] = self.decode(v)
        return out

    def _datetime(self, doc: dict, body: JSON, hashable: bool) -> ty.Any:
        if not isinstance(body, str):
            raise ValueError("not a string")
        if len(body) == 10:
            return datetime.date.fromisoformat(body)
        return datetime.datetime.fromisoformat(body)

    def _path(self, doc: dict, body: JSON, hashable: bool) -> Path:
        if not isinstance(body, str):
            raise ValueError("not a string")
        return Path(body)

    def _npscalar(self, doc: dict, body: JSON, hashable: bool) -> ty.Any:
        dtype_str, text = body
        if np is None:
            raise EntryFormatUnknown(
                "this entry holds a numpy scalar and numpy is not installed"
            )
        if not (isinstance(dtype_str, str) and _NPDTYPE.fullmatch(dtype_str)):
            raise ValueError(f"unknown dtype {dtype_str!r}")
        dtype = np.dtype(dtype_str)
        kind = dtype.kind
        if dtype.char in "gG":
            raise ValueError("longdouble is not encodable")
        if kind in "iu":
            return dtype.type(int(text))
        if kind == "f":
            return dtype.type(float.fromhex(text))
        if kind == "c":
            return dtype.type(complex(float.fromhex(text[0]), float.fromhex(text[1])))
        if kind == "b":
            if text not in ("true", "false"):
                raise ValueError("a bool is 'true' or 'false'")
            return dtype.type(text == "true")
        return np.array([int(text)], dtype="<i8").view(dtype)[0]

    def _codec(self, doc: dict, body: JSON, hashable: bool) -> ty.Any:
        codec = lookup_tag(body) if isinstance(body, str) else None
        if codec is None:
            raise EntryFormatUnknown(
                f"codec tag {body!r} is not registered in this process"
            )
        if codec.fileset or codec.decode is None:
            raise CorruptEntry(f"{body!r} is not a $codec tag")
        value = self.decode(doc["value"], hashable)
        try:
            return codec.decode(value)
        except Exception as err:
            raise EntryFormatUnknown(
                f"codec {body!r} cannot decode this value: {err}"
            ) from err

    def _fileset(self, doc: dict, body: JSON, hashable: bool) -> ty.Any:
        codec = lookup_tag(body) if isinstance(body, str) else None
        if codec is None or not codec.fileset:
            raise EntryFormatUnknown(
                f"file set tag {body!r} is not registered in this process"
            )
        listing = doc["files"]
        if not isinstance(listing, list) or not listing:
            raise ValueError("a file set lists its files")
        keys: list[str] = []
        paths: list[str] = []
        for item in listing:
            key, path = item
            if not (isinstance(key, str) and isinstance(path, str)):
                raise ValueError("a file is a [key, path] pair of strings")
            keys.append(key)
            paths.append(path)
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise CorruptEntry("the keys of a file set are unique and sorted")
        split = [p.split("/") for p in paths]
        set_hex = split[0][1] if len(split[0]) > 1 else ""
        for parts in split:
            if (
                len(parts) < 3
                or parts[0] != "files"
                or parts[1] != set_hex
                or not FINGERPRINT.fullmatch(set_hex)
                or any(
                    p in ("", ".", "..") or "\\" in p or "\0" in p or ":" in p
                    for p in parts
                )
            ):
                raise CorruptEntry(
                    f"{paths!r} are not under one files/<set>/ directory"
                )
        shas = []
        for path in paths:
            member = self.members.get(path)
            if member is None:
                raise CorruptEntry(f"{path} is not a member of the entry")
            shas.append(member["sha256"])
        expected = fileset_digest(body, list(zip(keys, shas))).split(":", 1)[1]
        if expected != set_hex:
            raise CorruptEntry(
                f"files/{set_hex} is not the file@1 of its tag, keys and digests"
            )
        self.referenced.update(paths)
        tops = sorted({parts[2] for parts in split})
        self.trees.append((set_hex, tops))
        base = os.path.join(self.entry_path, "files", set_hex)
        try:
            value = codec.type([os.path.join(base, t) for t in tops])
            found = sorted(
                _posix_key(k)
                for k, _ in value.byte_chunks(
                    ignore_hidden_files=False, ignore_hidden_dirs=False
                )
            )
        except Exception as err:
            raise CorruptEntry(
                f"files/{set_hex} does not form a {codec.type.__name__}: {err}"
            ) from err
        if found != keys:
            raise CorruptEntry(
                f"files/{set_hex} has keys {found}, the entry records {keys}"
            )
        return value


# -- encodability of a declared type ------------------------------------------------------

_SIMPLE = {
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    list,
    tuple,
    set,
    frozenset,
    dict,
    datetime.datetime,
    datetime.date,
    Path,
    os.PathLike,
}
_CONTAINERS = {
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    collections.abc.Iterable,
    collections.abc.Collection,
    collections.abc.Set,
    collections.abc.MutableSet,
    collections.abc.Mapping,
    collections.abc.MutableMapping,
}
_UNION_ORIGINS = (ty.Union, types.UnionType)


def type_is_encodable(tp: ty.Any) -> bool:
    """Whether values of the declared type ``tp`` have a codec: a built-in row of RFC 0003
    section 6.4 or a type registered in the tag registry. A type that says nothing about its
    values (``Any``, ``object``, a type variable) is accepted, because its values can only be
    judged at publish."""
    from pydra.utils.typing import MultiOutputType, strip_annotated

    tp = strip_annotated(tp)
    if (
        tp is inspect.Parameter.empty
        or tp is ty.Any
        or tp is object
        or isinstance(tp, ty.TypeVar)
    ):
        return True
    origin = ty.get_origin(tp)
    args = ty.get_args(tp)
    if origin in _UNION_ORIGINS:
        members = [a for a in args if a is not MultiOutputType]
        if any(a is ty.Any or a is object for a in members):
            return True
        return all(type_is_encodable(a) for a in members)
    if np is not None and origin is not None:  # NDArray[...] and ndarray[...]
        from numpy.typing import NDArray

        if origin is np.ndarray or origin is NDArray:
            return True
    if origin is ty.Literal:
        return all(type(a) in (type(None), bool, int, str, bytes) for a in args)
    if origin is not None:
        if isinstance(origin, type) and tag_of(origin) is not None:
            return True
        if origin in (list, set, frozenset, tuple, dict) or origin in _CONTAINERS:
            return all(a is Ellipsis or type_is_encodable(a) for a in args)
        return False
    if not isinstance(tp, type):
        return False
    if tag_of(tp) is not None:
        return True
    if tp in _SIMPLE or tp in _CONTAINERS:
        return True
    if np is not None:
        if tp is np.ndarray:
            return True
        if issubclass(tp, np.generic):
            if tp in (np.longdouble, np.clongdouble) or issubclass(
                tp, (np.flexible, np.object_)
            ):
                return False
            return True
    return False
