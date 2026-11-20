"""Errors of the portable store's entry format (RFC 0003 section 6).

`CorruptEntry` and `MisfiledEntry` are the failures a store answers by quarantining the entry;
`EntryFormatUnknown` and `EntryNotTrusted` leave it in place, not served.
"""

from __future__ import annotations

import typing as ty

from pydra.utils.storagedir import StoreUnavailable

__all__ = [
    "CorruptEntry",
    "MisfiledEntry",
    "EntryFormatUnknown",
    "EntryNotTrusted",
    "UnencodableOutput",
]


class CorruptEntry(StoreUnavailable):
    """An entry failed one of the checks that bind it to its address and its members
    (RFC 0003 section 6.2). It is quarantined."""


class MisfiledEntry(CorruptEntry):
    """An entry's recorded checksum or fingerprint is not the address it is found at."""


class EntryFormatUnknown(StoreUnavailable):
    """An entry holds a member of another format version, or a codec tag that this process has
    not registered. It is left in place and not served."""


class EntryNotTrusted(StoreUnavailable):
    """An entry holds a pickle member that the root's trust does not admit (RFC 0003 section
    6.4). It is left in place and not served."""


class UnencodableOutput(TypeError):
    """An output cannot be stored: it has no codec and the store cannot pickle it, or the
    reason given says why.

    Attributes
    ----------
    field : str
        The output field.
    type : type
        The declared type (at hand-out) or the runtime type (at publish) with no codec.
    """

    def __init__(self, field: str, type: ty.Any, reason: str = "") -> None:
        name = getattr(type, "__qualname__", None) or repr(type)
        module = getattr(type, "__module__", "")
        shown = name if module in ("builtins", "") else f"{module}.{name}"
        if reason:
            message = f"output {field!r} of type {shown} cannot be stored: {reason}"
        else:
            message = f"output {field!r} of type {shown} has no codec and this store cannot pickle it"
        super().__init__(message)
        self.field = field
        self.type = type
        self.reason = reason

    def __reduce__(self) -> tuple:
        return (type(self), (self.field, self.type, self.reason))
