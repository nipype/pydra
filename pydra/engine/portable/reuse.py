"""The reuse hook of a `PortableStore`: which stored entry a job may use (RFC 0003 section 7)."""

from __future__ import annotations

import enum
import typing as ty

import attrs

from pydra.engine.portable.entry import EntryRef, StoredEntry
from pydra.engine.portable.identity import Context
from pydra.engine.result import Result
from pydra.engine.store import (
    AnyReuse,
    ConsistentReuse,
    ExactReuse,
    ReusePolicy,
    reuse_policy,
)

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.portable.store import PortableStore

__all__ = [
    "Verdict",
    "Candidate",
    "Reuse",
    "Hit",
    "Verify",
    "Miss",
    "Lookup",
    "PolicyReuse",
    "as_reuse",
]


class Verdict(enum.Enum):
    """What a `Reuse` hook says of one candidate."""

    SERVE = "serve"
    """return the stored result"""
    VERIFY = "verify"
    """run the job, compare its outputs with the stored ones, then publish"""
    REFUSE = "refuse"
    """do not serve this entry; run the job (ends the lookup)"""
    SKIP = "skip"
    """do not serve this entry; consider the next candidate"""


@attrs.frozen
class Candidate:
    """A stored entry that may be reused.

    Attributes
    ----------
    entry : EntryRef
    context : Mapping
        The stored context.
    context_digest : str
        The sha256 of the ``context.json`` bytes that were ranked.
    rank : int
        0 is the narrowest admission (an equal fingerprint).
    admitted_by : str
        Why it was admitted, e.g. ``"exact"``, ``"consistent"``.
    """

    entry: EntryRef
    context: Context
    context_digest: str
    rank: int
    admitted_by: str


@ty.runtime_checkable
class Reuse(ty.Protocol):
    """The hook that decides which stored entries a job may reuse (RFC 0003 section 7.1). It is
    picklable and runs in whichever process looks up."""

    def bind(self, store: "PortableStore") -> None:
        """Give the hook its store, once at construction and again after unpickling in a
        worker, so it can reach ``store.sidecar(name)``."""

    def rank(
        self,
        current: Context,
        stored: ty.Sequence[tuple[EntryRef, Context, str]],
        task: "Task | None",
    ) -> list[Candidate]:
        """The admitted candidates, best first, from every stored ``(entry, context, digest of
        its context.json)`` of the job's checksum in every root (own root first)."""

    def verdict(
        self, candidate: Candidate, current: Context, task: "Task | None"
    ) -> Verdict:
        """What to do with ``candidate``."""

    def served(
        self,
        candidate: Candidate,
        current: Context,
        task: "Task | None",
        entry: StoredEntry,
    ) -> None:
        """The reuse record, called before a result is returned. If it raises, the result is
        not served and the job runs."""

    def compare(
        self,
        candidate: Candidate,
        stored: StoredEntry,
        fresh: ty.Mapping[str, ty.Any],
        fresh_digest: str,
        task: "Task | None",
    ) -> None:
        """Compare a job marked to verify with the entry it verifies, after staging and before
        the commit hook and the rename. If it raises, the job errors and nothing is committed.
        """


@attrs.frozen
class Hit:
    """A stored result to serve."""

    result: Result
    candidate: ty.ClassVar[None] = None


@attrs.frozen
class Verify:
    """Run the job, then compare its outputs with ``candidate``'s (``Job.verify``)."""

    candidate: Candidate
    result: ty.ClassVar[None] = None


@attrs.frozen
class Miss:
    """Nothing to reuse; run the job."""

    result: ty.ClassVar[None] = None
    candidate: ty.ClassVar[None] = None


Lookup = ty.Union[Hit, Verify, Miss]

_NAMES = {ConsistentReuse: "consistent", AnyReuse: "any"}


class PolicyReuse:
    """A `ReusePolicy` as a `Reuse` hook: the order of `PickleStore.lookup` (the job's own
    fingerprint first, then every stored context the policy accepts, in path order), always
    ``SERVE``, no record and no comparison. The store marks a result from another fingerprint
    with ``Result.cross_context`` and ``Result.reused_in``."""

    def __init__(self, policy: ReusePolicy) -> None:
        self._policy = policy
        self._store: "PortableStore | None" = None

    def __getstate__(self) -> dict[str, ty.Any]:
        state = dict(self.__dict__)
        state["_store"] = None  # bound again where it is unpickled
        return state

    def __setstate__(self, state: dict[str, ty.Any]) -> None:
        self.__dict__.update(state)

    def bind(self, store: "PortableStore") -> None:
        self._store = store

    def rank(
        self,
        current: Context,
        stored: ty.Sequence[tuple[EntryRef, Context, str]],
        task: "Task | None",
    ) -> list[Candidate]:
        if self._store is None:
            raise RuntimeError("the hook is not bound to a store")
        mine = self._store.identity.fingerprint(current)
        own = [s for s in stored if s[0].fingerprint == mine]
        out = [Candidate(e, c, d, 0, "exact") for e, c, d in own]
        if type(self._policy) is ExactReuse:
            return out
        name = _NAMES.get(type(self._policy), "policy")
        for e, c, d in stored:
            if e.fingerprint != mine and self._policy.accept(c, current, task):
                out.append(Candidate(e, c, d, 1, name))
        return out

    def verdict(
        self, candidate: Candidate, current: Context, task: "Task | None"
    ) -> Verdict:
        return Verdict.SERVE

    def served(
        self,
        candidate: Candidate,
        current: Context,
        task: "Task | None",
        entry: StoredEntry,
    ) -> None:
        return None

    def compare(
        self,
        candidate: Candidate,
        stored: StoredEntry,
        fresh: ty.Mapping[str, ty.Any],
        fresh_digest: str,
        task: "Task | None",
    ) -> None:
        return None


def as_reuse(reuse: ty.Any) -> Reuse:
    """The hook ``PortableStore(reuse=...)`` stands for: a `Reuse` as it is, a policy name or a
    `ReusePolicy` wrapped in `PolicyReuse`."""
    if isinstance(reuse, Reuse):
        return reuse
    return PolicyReuse(reuse_policy(reuse))
