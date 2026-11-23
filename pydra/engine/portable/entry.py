"""The entry format: what is inside one committed entry (RFC 0003 section 6).

The store owns the layout, staging, rename, modes, quarantine and collection; an `EntryFormat`
owns what an entry holds. `PydraEntryV1` is ``pydra-entry@1`` in ``results@1/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import typing as ty

import attrs

from pydra.engine.portable.codecs import (
    Decoder,
    Encoder,
    output_digest,
    type_is_encodable,
)
from pydra.engine.portable.errors import (
    CorruptEntry,
    EntryFormatUnknown,
    EntryNotTrusted,
    MisfiledEntry,
    UnencodableOutput,
)
from pydra.engine.portable.identity import (
    Context,
    KeyRecord,
    PydraIdentityV1,
    StoreIdentity,
)
from pydra.engine.portable.layout import FINGERPRINT, cdir_for
from pydra.engine.portable.trust import TRUSTS, Trust, check_private
from pydra.utils.storagedir import StorageDir, StoreUnavailable, WrongKindError

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task

__all__ = [
    "CorruptEntry",
    "EntryFormat",
    "EntryFormatUnknown",
    "EntryNotTrusted",
    "EntryRef",
    "MisfiledEntry",
    "PydraEntryV1",
    "StoredEntry",
    "UnencodableOutput",
    "check_encodable",
    "check_private",
]

logger = logging.getLogger("pydra")

FORMAT = "pydra-entry@1"
ENTRIES = "results@1"
_FORMAT_RE = re.compile(r"pydra-entry@([0-9]+)")
_MEMBERS = re.compile(r"entry\.json|context\.json|[0-9a-f]{64}\.(?:npy|bin|pkl)|files")
_NEVER = re.compile(r"(?!)")
JSON_CAP = 1 << 30
"""Most bytes of an entry's ``entry.json`` or ``context.json`` that are read. It bounds a
hostile file only: at about 170 bytes per member record, 1 GiB admits millions of members."""
READ_BLOCK = 1 << 20
"""Bytes read per call while hashing a member; any size gives the same digest."""


@attrs.frozen
class EntryRef:
    """Where an entry is: which root (RFC 0003 section 6.4), its ``<cdir>`` and its
    fingerprint."""

    root: str
    cdir: str
    fingerprint: str


@attrs.frozen(eq=False)
class StoredEntry:
    """What `EntryFormat.read` returns: the outputs, the context, the key record, the output
    digest and the entry's `EntryRef`."""

    outputs: dict[str, ty.Any]
    context: Context
    key: KeyRecord
    digest: str
    ref: EntryRef


class EntryFormat(ty.Protocol):
    """The hook a `PortableStore` calls for everything inside an entry (RFC 0003 section 6.1).

    Attributes
    ----------
    name : str
        The header's ``entry_format``, e.g. ``"pydra-entry@1"``.
    entries : str
        The version directory, e.g. ``"results@1"``.
    """

    name: str
    entries: str

    def bind(
        self, identity: StoreIdentity, pickle: bool, trust: Trust
    ) -> "EntryFormat":
        """This format as a store with ``identity``, header ``pickle`` and ``trust`` writes it."""
        ...

    def members(self) -> re.Pattern[str]:
        """The names an entry may hold."""
        ...

    def aside_names(self) -> re.Pattern[str]:
        """The names of entries this format set aside in place."""
        ...

    def stage(
        self,
        staging: StorageDir,
        checksum: str,
        key: KeyRecord,
        context: Context,
        outputs: ty.Mapping[str, ty.Any],
        task: "Task | None",
        files_from: StorageDir | None,
        writer: str,
    ) -> str:
        """Write the members of an entry into ``staging`` and return the output digest."""
        ...

    def context(self, entry: StorageDir) -> tuple[Context, bytes]:
        """Read ``context.json``: the context and its bytes."""
        ...

    def identify(self, entry: StorageDir) -> tuple[str, str]:
        """The checksum and the writer token that the entry records, read without verifying
        the entry (the store reads a writer's token to tell a rename whose reply was lost, and
        an entry's checksum where its directory name is a hash of it)."""
        ...

    def member_records(self, entry: StorageDir) -> dict[str, dict]:
        """The ``sha256`` and ``size`` the entry records for each member, checked for form
        but not against the members."""
        ...

    def read(
        self,
        entry: StorageDir,
        checksum: str,
        fingerprint: str,
        task: "Task | None",
        trust: Trust,
        *,
        root: StorageDir,
    ) -> StoredEntry:
        """Read and verify an entry (RFC 0003 section 6.2). ``root`` is the root the entry is
        under, which gives its `EntryRef` and the start of the ownership walk; ``entry`` may
        be at any depth below it (a quarantined copy is).

        Raises
        ------
        CorruptEntry, MisfiledEntry, EntryFormatUnknown, EntryNotTrusted
        """
        ...


def _json(data: bytes, what: str) -> ty.Any:
    def refuse(token: str) -> ty.NoReturn:
        raise ValueError(f"{token} is not JSON")

    try:
        return json.loads(data.decode("utf-8"), parse_constant=refuse)
    except (UnicodeDecodeError, ValueError, RecursionError) as err:
        raise CorruptEntry(f"{what} is not readable JSON: {err}") from None


def _dumps(doc: ty.Any, *, sort_keys: bool = False) -> bytes:
    return json.dumps(
        doc,
        sort_keys=sort_keys,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@attrs.frozen
class PydraEntryV1:
    """``pydra-entry@1``: ``entry.json`` and ``context.json``, and one member per array, bytes
    value or pickle (named by its sha256) and a ``files/<set>/`` directory per file output.

    Parameters
    ----------
    identity : StoreIdentity
        Gives the fingerprint an entry records and checks, and accepts its key record.
    pickle : bool
        The root's header ``pickle`` flag: whether this writer may write pickle members.
    trust : {"shared", "private", "trusted"}
        The trust declared for the root this writer writes. A ``shared`` writer never writes a
        pickle, whatever the header says, and refuses a naive datetime.
    """

    identity: StoreIdentity = attrs.field(factory=PydraIdentityV1)
    pickle: bool = False
    trust: Trust = attrs.field(default="shared")
    name: ty.ClassVar[str] = FORMAT
    entries: ty.ClassVar[str] = ENTRIES

    @trust.validator
    def _check_trust(self, _attribute: ty.Any, value: str) -> None:
        if value not in TRUSTS:
            raise ValueError(f"trust must be one of {TRUSTS}, got {value!r}")

    def bind(
        self, identity: StoreIdentity, pickle: bool, trust: Trust
    ) -> "PydraEntryV1":
        return attrs.evolve(self, identity=identity, pickle=pickle, trust=trust)

    def members(self) -> re.Pattern[str]:
        return _MEMBERS

    def aside_names(self) -> re.Pattern[str]:
        return _NEVER

    # -- writing -----------------------------------------------------------------------

    def stage(
        self,
        staging: StorageDir,
        checksum: str,
        key: KeyRecord,
        context: Context,
        outputs: ty.Mapping[str, ty.Any],
        task: "Task | None",
        files_from: StorageDir | None,
        writer: str,
    ) -> str:
        """Write the entry into ``staging`` and return the output digest.

        Every file output is copied (a reflink where the filesystem has one, never a link), and
        each member's sha256 and size go into ``entry.json``. ``files_from`` does not change
        that: a file inside it is copied as one outside it is.

        Raises
        ------
        UnencodableOutput
            An output has no codec and this writer may not pickle it.
        """
        from pydra.utils.digest import jcs

        jcs(context)  # a context is JSON the fingerprint can take, as written
        fingerprint = self.identity.fingerprint(context)
        encoder = Encoder(
            staging,
            can_pickle=self.pickle and self.trust != "shared",
            allow_naive=self.trust != "shared",
        )
        encoded = encoder.encode_outputs(outputs)
        encoder.write(
            "context.json",
            _dumps({"format": FORMAT, "context": dict(context)}, sort_keys=True),
        )
        digest = output_digest(encoded)
        doc = {
            "format": FORMAT,
            "checksum": checksum,
            "key": None if key is None else dict(key),
            "fingerprint": fingerprint,
            "writer": writer,
            "digest": digest,
            "outputs": encoded,
            "members": encoder.members,
        }
        with staging.open_new("entry.json", staging.policy.file_mode) as f:
            f.write(_dumps(doc))
            f.flush()
            if staging.policy.durability == "member":
                os.fsync(f.fileno())
        if staging.policy.durability == "member":
            staging.sync()
        return digest

    # -- reading -----------------------------------------------------------------------

    def member_records(self, entry: StorageDir) -> dict[str, dict]:
        doc = _json(_read_file(entry, "entry.json"), "entry.json")
        if not isinstance(doc, dict) or not isinstance(doc.get("members"), dict):
            raise CorruptEntry("entry.json lacks members")
        return _checked_members(doc["members"])

    def identify(self, entry: StorageDir) -> tuple[str, str]:
        """The ``checksum`` and ``writer`` that ``entry.json`` records.

        Raises
        ------
        CorruptEntry
            ``entry.json`` is missing, not a regular file, or lacks either.
        """
        doc = _json(_read_file(entry, "entry.json"), "entry.json")
        if (
            not isinstance(doc, dict)
            or not isinstance(doc.get("checksum"), str)
            or not isinstance(doc.get("writer"), str)
        ):
            raise CorruptEntry("entry.json records no checksum and writer")
        return doc["checksum"], doc["writer"]

    def context(self, entry: StorageDir) -> tuple[Context, bytes]:
        """Read ``context.json``: the context and the file's bytes.

        Raises
        ------
        CorruptEntry
            ``context.json`` is missing, not a regular file, or malformed.
        """
        return self._context_of(_read_file(entry, "context.json"))

    def read(
        self,
        entry: StorageDir,
        checksum: str,
        fingerprint: str,
        task: "Task | None",
        trust: Trust,
        *,
        root: StorageDir,
    ) -> StoredEntry:
        """Read ``entry`` and bind it to its address and to its members (RFC 0003 section 6.2).

        ``entry`` is opened once by the caller and every read is made through it. The
        arguments are the address the entry was found at: its checksum and fingerprint must be
        the ones it records, its context must give that fingerprint, and the identity must
        accept its key record. Every member is opened without following a symlink, must be a
        regular file and is checked against ``entry.json`` (sizes only under the policy's
        ``verify_files="size"``); a pickle member is opened only if ``trust`` admits it.

        Raises
        ------
        CorruptEntry
            A check failed; the store quarantines the entry.
        MisfiledEntry
            The entry is at an address that is not its own.
        EntryFormatUnknown
            The entry holds another version's member, or a tag this process has not
            registered; it is left in place.
        EntryNotTrusted
            A pickle member and a root whose trust does not admit it.
        StoreUnavailable
            Any other ``OSError``.
        """
        if trust not in TRUSTS:
            raise ValueError(f"trust must be one of {TRUSTS}, got {trust!r}")
        try:
            return self._read(entry, checksum, fingerprint, task, trust, root=root)
        except RecursionError:
            raise CorruptEntry("entry.json is nested too deeply") from None
        except StoreUnavailable:
            raise
        except OSError as err:
            raise StoreUnavailable(f"{entry.path}: {err}") from err
        except (ValueError, KeyError, IndexError) as err:
            # data errors; TypeError and AttributeError are defects and propagate
            raise CorruptEntry(f"{entry.path} is malformed: {err!r}") from err

    def _read(
        self,
        entry: StorageDir,
        checksum: str,
        fingerprint: str,
        task: "Task | None",
        trust: Trust,
        *,
        root: StorageDir,
    ) -> StoredEntry:
        names = {e.name: e for e in entry.scandir()}
        foreign = sorted(n for n in names if not _MEMBERS.fullmatch(n))
        if foreign:
            raise EntryFormatUnknown(
                f"{entry.path} holds {foreign[0]!r}, which is not a member of {FORMAT}"
            )
        doc = _json(_read_file(entry, "entry.json"), "entry.json")
        if not isinstance(doc, dict):
            raise CorruptEntry("entry.json does not hold an object")
        _check_format(doc.get("format"), "entry.json")
        for field, kind in (
            ("checksum", str),
            ("fingerprint", str),
            ("writer", str),
            ("digest", str),
            ("outputs", dict),
            ("members", dict),
        ):
            if not isinstance(doc.get(field), kind):
                raise CorruptEntry(f"entry.json lacks {field}")
        if doc.get("key") is not None and not isinstance(doc["key"], dict):
            raise CorruptEntry("entry.json has a malformed key")
        if doc["checksum"] != checksum:
            raise MisfiledEntry(
                f"the entry records checksum {doc['checksum']!r}, found at {checksum!r}"
            )
        if doc["fingerprint"] != fingerprint:
            raise MisfiledEntry(
                f"the entry records fingerprint {doc['fingerprint']!r}, found at {fingerprint!r}"
            )
        members = _checked_members(doc["members"])
        top = {n for n in members if "/" not in n}
        unlisted = sorted(
            n for n in names if n not in ("entry.json", "files") and n not in top
        )
        if unlisted:
            raise CorruptEntry(f"{unlisted[0]} is in the entry but not in entry.json")
        if "context.json" not in members:
            raise CorruptEntry("entry.json does not list context.json")
        pickles = sorted(n for n in members if n.endswith(".pkl"))
        if pickles:
            _admit_pickles(entry, root, pickles, trust)
        verify = entry.policy.verify_files
        blobs: dict[str, bytes] = {}
        for name, rec in members.items():
            if "/" in name:
                _verify_file(entry, name, rec, verify)
            else:
                blobs[name] = _verify_blob(entry, name, rec, verify)
        context, _raw = self._context_of(blobs["context.json"])
        if self.identity.fingerprint(context) != fingerprint:
            raise MisfiledEntry(
                "the entry's context does not give the fingerprint it is found at"
            )
        if not self.identity.check_key(checksum, doc.get("key")):
            raise MisfiledEntry(
                "the entry's key record does not digest to its checksum"
            )

        def fetch(name: str) -> bytes:
            if name not in blobs:
                raise CorruptEntry(f"{name} is not a member of the entry")
            return blobs[name]

        decoder = Decoder(
            fetch, members, entry.path, allow_pickle=trust in ("private", "trusted")
        )
        outputs = decoder.decode_outputs(doc["outputs"])
        _check_trees(entry, decoder.trees, members)
        stray = sorted(set(members) - decoder.referenced - {"context.json"})
        if stray:
            raise CorruptEntry(
                f"{stray[0]} is in entry.json but no output refers to it"
            )
        digest = output_digest(doc["outputs"])
        if digest != doc["digest"]:
            raise CorruptEntry("the outputs do not give the recorded output digest")
        return StoredEntry(
            outputs=_declared(task, outputs, entry.path),
            context=context,
            key=doc.get("key"),
            digest=digest,
            ref=EntryRef(str(root.path), cdir_for(checksum), fingerprint),
        )

    @staticmethod
    def _context_of(raw: bytes) -> tuple[Context, bytes]:
        doc = _json(raw, "context.json")
        if not isinstance(doc, dict) or not isinstance(doc.get("context"), dict):
            raise CorruptEntry("context.json holds no context")
        _check_format(doc.get("format"), "context.json")
        return doc["context"], raw


def _check_format(fmt: ty.Any, what: str) -> None:
    found = _FORMAT_RE.fullmatch(fmt) if isinstance(fmt, str) else None
    if found is None:
        raise CorruptEntry(f"{what} has format {fmt!r}")
    if fmt != FORMAT:
        raise EntryFormatUnknown(
            f"{what} has format {fmt!r}; this version reads {FORMAT}"
        )


def _check_member_name(name: str) -> None:
    parts = name.split("/")
    if len(parts) == 1:
        if name in ("entry.json", "files") or not _MEMBERS.fullmatch(name):
            raise CorruptEntry(f"{name!r} is not a member name of {FORMAT}")
        return
    if (
        parts[0] != "files"
        or len(parts) < 3
        or not FINGERPRINT.fullmatch(parts[1])
        or any(
            p in ("", ".", "..") or "\\" in p or "\0" in p or ":" in p for p in parts
        )
    ):
        raise CorruptEntry(f"{name!r} is not under files/<set>/")


# -- member access ---------------------------------------------------------------------


def _open_member(entry: StorageDir, name: str) -> ty.BinaryIO:
    """Open the member ``name`` (a path below ``entry``) for reading, through descriptors and
    without following a symlink. A member that is missing or of the wrong kind is corrupt.
    """
    *dirs, leaf = name.split("/")
    opened: list[StorageDir] = []
    cur = entry
    try:
        for part in dirs:
            cur = cur.child(part)
            opened.append(cur)
        return cur.open_read(leaf)
    except (FileNotFoundError, NotADirectoryError, WrongKindError) as err:
        raise CorruptEntry(f"member {name} cannot be read: {err}") from err
    finally:
        for d in opened:
            d.close()


def read_capped(f: ty.BinaryIO, name: str, cap: int | None = JSON_CAP) -> bytes:
    """The bytes of the open file ``f`` (named ``name`` in errors), read in blocks of
    `READ_BLOCK`; `CorruptEntry` past ``cap`` bytes (None: no cap)."""
    chunks: list[bytes] = []
    total = 0
    while block := f.read(READ_BLOCK):
        chunks.append(block)
        total += len(block)
        if cap is not None and total > cap:
            raise CorruptEntry(f"{name} is larger than {cap} bytes")
    return b"".join(chunks)


def _checked_members(members: dict) -> dict[str, dict]:
    for name, rec in members.items():
        if not (
            isinstance(name, str)
            and isinstance(rec, dict)
            and isinstance(rec.get("size"), int)
            and not isinstance(rec.get("size"), bool)
            and isinstance(rec.get("sha256"), str)
            and FINGERPRINT.fullmatch(rec["sha256"])
        ):
            raise CorruptEntry(f"entry.json has a malformed member {name!r}")
        _check_member_name(name)
    return members


def _read_file(entry: StorageDir, name: str) -> bytes:
    with _open_member(entry, name) as f:
        return read_capped(f, name)


def _verify_file(entry: StorageDir, name: str, rec: dict, verify: str) -> None:
    with _open_member(entry, name) as f:
        size = os.fstat(f.fileno()).st_size
        if size != rec["size"]:
            raise CorruptEntry(
                f"{name} has size {size}, the entry records {rec['size']}"
            )
        if verify == "size":
            return
        h = hashlib.sha256()
        for block in iter(lambda: f.read(READ_BLOCK), b""):
            h.update(block)
    if h.hexdigest() != rec["sha256"]:
        raise CorruptEntry(f"{name} does not have the sha256 the entry records")


def _verify_blob(entry: StorageDir, name: str, rec: dict, verify: str) -> bytes:
    with _open_member(entry, name) as f:
        size = os.fstat(f.fileno()).st_size
        data = f.read(min(size, rec["size"]) + 1)
    if len(data) != rec["size"]:
        raise CorruptEntry(
            f"{name} has size {len(data)}, the entry records {rec['size']}"
        )
    if verify != "size" and hashlib.sha256(data).hexdigest() != rec["sha256"]:
        raise CorruptEntry(f"{name} does not have the sha256 the entry records")
    stem = name.rsplit(".", 1)[0]
    if FINGERPRINT.fullmatch(stem) and stem != rec["sha256"]:
        raise CorruptEntry(f"{name} is not named by the sha256 it records")
    return data


def _admit_pickles(
    entry: StorageDir, root: StorageDir, pickles: list[str], trust: Trust
) -> None:
    """Gate, before any member is opened, an entry that holds pickles.

    Under ``private``, every directory from ``root`` to the entry, ``entry.json`` (whose records
    bind the pickles) and every pickle member must belong to the current user and not be
    writable by anyone else."""
    if trust == "shared":
        raise EntryNotTrusted(
            f"{entry.path} holds a pickle member ({pickles[0]}) and this root is shared: a "
            "shared root never opens one"
        )
    if trust == "trusted":
        return
    rel = os.path.relpath(entry.path, root.path)
    if rel == os.curdir or rel.split(os.sep)[0] == os.pardir:
        raise StoreUnavailable(f"{entry.path} is not below the root {root.path}")
    names = rel.split(os.sep)
    chain: list[StorageDir] = []
    cur = root
    try:
        for n in names:
            try:
                cur = cur.child(n)
            except WrongKindError as err:
                raise EntryNotTrusted(f"{entry.path}: {err}") from err
            chain.append(cur)
        here, there = cur.fstat(), entry.fstat()
        if (here.st_dev, here.st_ino) != (there.st_dev, there.st_ino):
            raise EntryNotTrusted(f"{entry.path} is not where its path says")
    finally:
        for d in chain:
            d.close()
    for member in ("entry.json", *pickles):
        check_private(root, [*names, member])


def _walk(parent: StorageDir, name: str, path: str, members: ty.Mapping) -> None:
    """Every object below a handed-out tree is a listed regular file or a directory."""
    try:
        mode = parent.lstat(name).st_mode
        if stat.S_ISREG(mode):
            if path not in members:
                raise CorruptEntry(f"{path} is in the entry but not in entry.json")
        elif stat.S_ISDIR(mode):
            with parent.child(name) as d:
                for e in list(d.scandir()):
                    _walk(d, e.name, f"{path}/{e.name}", members)
        else:
            raise CorruptEntry(
                f"{path} is a symlink or another object that is not a file"
            )
    except WrongKindError as err:
        raise CorruptEntry(f"{path} is not what it should be: {err}") from err


def _check_trees(
    entry: StorageDir, trees: list[tuple[str, list[str]]], members: ty.Mapping
) -> None:
    """Walk each file set handed out by descriptor and refuse anything that is not a listed
    regular file or a directory, and anything under ``files/`` no output names."""
    try:
        files = entry.child("files")
    except FileNotFoundError:
        if trees:
            raise CorruptEntry("the entry has no files/ directory") from None
        return
    except WrongKindError as err:
        raise CorruptEntry(f"files is not a directory: {err}") from err
    with files:
        tops: dict[str, set[str]] = {}
        for set_hex, names in trees:
            tops.setdefault(set_hex, set()).update(names)
        for e in files.scandir():
            if e.name not in tops or not e.is_dir():
                raise CorruptEntry(f"files/{e.name} is not a file set of the entry")
        for set_hex, names in tops.items():
            try:
                set_dir = files.child(set_hex)
            except (FileNotFoundError, WrongKindError) as err:
                raise CorruptEntry(f"files/{set_hex} cannot be read: {err}") from err
            with set_dir:
                for e in set_dir.scandir():
                    if e.name not in names:
                        raise CorruptEntry(
                            f"files/{set_hex}/{e.name} is not named by an output"
                        )
                for name in sorted(names):
                    _walk(set_dir, name, f"files/{set_hex}/{name}", members)


# -- declared types ----------------------------------------------------------------------


def _fileset_leaves(value: ty.Any) -> ty.Iterator[ty.Any]:
    from fileformats.core.fileset import FileSet

    if isinstance(value, FileSet):
        yield value
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            yield from _fileset_leaves(v)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _fileset_leaves(k)
            yield from _fileset_leaves(v)


def _has_fileset(tp: ty.Any) -> bool:
    from fileformats.core.fileset import FileSet

    if isinstance(tp, type) and issubclass(tp, FileSet):
        return True
    return any(_has_fileset(a) for a in ty.get_args(tp))


def _declared(
    task: "Task | None", outputs: dict[str, ty.Any], entry_path: str
) -> dict[str, ty.Any]:
    """Decode each output to the type its field declares (a tuple, a set), where the task is
    known; a value that does not take the type is left as decoded, except for a field that
    declares a file set. A file set handed out is always a path inside the entry: one that
    coercion made from anything else is `CorruptEntry`."""
    if task is None:
        return outputs
    from pydra.utils.general import get_fields
    from pydra.utils.typing import TypeParser

    base = os.path.join(os.path.abspath(entry_path), "files") + os.sep
    fields = {f.name: f for f in get_fields(task.Outputs)}
    out = {}
    for name, value in outputs.items():
        field = fields.get(name)
        if field is None or field.type in (ty.Any, object):
            out[name] = value
            continue
        try:
            out[name] = TypeParser(field.type)(value)
        except Exception as err:
            if _has_fileset(field.type):
                raise CorruptEntry(
                    f"output {name!r} is declared {field.type} and its value is not one: {err}"
                ) from err
            logger.warning(
                "output %r does not take its declared type %s: %s",
                name,
                field.type,
                err,
            )
            out[name] = value
        for leaf in _fileset_leaves(out[name]):
            for p in leaf.fspaths:
                if not os.path.abspath(os.fspath(p)).startswith(base):
                    raise CorruptEntry(
                        f"output {name!r} would be handed the path {p}, which is outside the entry"
                    )
    return out


def check_encodable(task: "type[Task] | Task", allow_pickle: bool) -> None:
    """Check each declared output field of ``task`` against the codecs, before any
    computation.

    A field whose declared type has no codec raises `UnencodableOutput` naming the field and the
    type, unless the store may pickle. A field typed ``Any``, or a value whose runtime type
    differs from its declaration, can only be found at publish (RFC 0003 section 6.4).
    """
    if allow_pickle:
        return
    from pydra.utils.general import get_fields

    for field in get_fields(task.Outputs):
        if not type_is_encodable(field.type):
            raise UnencodableOutput(field.name, field.type)
