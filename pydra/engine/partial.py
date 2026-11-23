"""Partial ``State`` (proposal 0001 §3.1 step 2, RFC 0002 E2): the bookkeeping of a node
that splits over an upstream output, which is expanded one *block* per upstream element
as each succeeds, rather than once its upstream nodes are done.

A :class:`~pydra.engine.runner.NodeExecution` classifies itself once its upstream nodes
have started (:func:`classify`) and, when it is partial, delegates its blocks, element
keys, groups and seal to a :class:`PartialExpansion`. The index algebra itself (paths,
entries, the RPN evaluation) is in :mod:`pydra.engine.state`.

Terms
-----
path
    A tuple naming a splitter field through the prev-state terms that lead to it:
    ``("_A", "A.x")`` is field ``A.x`` reached through the prev-state term ``_A``.
element key (``state_key``)
    The coordinates of an element, one per path of ``elem_paths``. It never changes;
    its flat ``state_index`` is assigned when the node is sealed.
block
    One combination of the final keys of the node's sources (its inner sources, and the
    partial nodes it reads through prev-state terms).
settled / sealed
    Settled: every element the node will have is known. Sealed: settled, with every
    block expanded and every source sealed.
"""

import itertools
import typing as ty
from collections import defaultdict
from traceback import format_exc
import attrs
from pydra.engine.absent import Absent
from pydra.engine.lazy import LazyField, LazyOutField
from pydra.engine.progress import UNSET, changed
from pydra.engine.state import (
    Entry,
    PydraStateError,
    compatible,
    expand_rpn,
    flatten,
    is_sequence,
    remove_inp_from_splitter_rpn,
    rpn_terms,
    scalar_operands,
)
from pydra.utils.general import attrs_values, ensure_list
from pydra.utils.typing import StateArray

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.runner import NodeExecution


class _AllElements:
    """Reads every element of a node (a whole read)"""

    def __repr__(self):
        return "ALL"


ALL = _AllElements()


@attrs.frozen
class GroupRead:
    """Reads the elements of one combined group of a partial node"""

    key: tuple


@attrs.frozen
class Block:
    """A block of a partial node that failed before it was expanded, so has no element.
    It is a key of ``NodeExecution.unrunnable``

    Attributes
    ----------
    source : str
        The node the failure came from: the node itself for its own block, else the
        upstream node whose failed block it inherits
    key : tuple
        The key of the block in ``source``
    fields : tuple
        The coordinates fixed by the block, as ``(path, coordinate)`` pairs in the path
        space of the node that holds it
    """

    source: str
    key: tuple
    fields: tuple

    def __str__(self):
        return f"{self.source}[block {key_text(self.key)}]"


def key_text(key: tuple) -> str:
    return ",".join(str(k) for k in key)


def sort_key(key: tuple) -> tuple:
    """Orders block keys, whose positions may be None (a stateless source)"""
    return tuple((k is not None, k) for k in key)


def _dedupe(paths: ty.Iterable[tuple]) -> list[tuple]:
    return list(dict.fromkeys(paths))


class _BlockState:
    __slots__ = ("fields", "status")

    def __init__(self, fields: dict):
        self.fields = fields
        self.status = "pending"  # "expanded", "failed" or "errored"


class _Spec:
    """The structure of a partial node, worked out from its splitter and its upstream
    nodes once they have started

    Attributes
    ----------
    inner : dict
        Each of the node's split inputs that is an upstream output, ``"<node>.<input>"``
        mapped to its lazy field
    inner_sources : frozenset
        The names of the nodes the inner inputs read
    rpn : list
        The splitter in RPN without the prev-state term of each inner source
        (``state.py`` ``prepare_states_ind``), which fixes the element order
    sources : list
        The nodes whose final keys make up a block key: the inner sources, then the
        partial upstream nodes of prev-state terms
    linked : set
        The sources linked through the node's ``State`` (an inner source that is
        neither stateless nor combined to nothing)
    elem_paths, group_paths, final_paths : list
        The paths of an element key, of a group key (None without a combiner), and of
        the keys the node's readers see
    units : list
        ``(inner path, source paths)`` of each inner input still split in the final
        keys: an inner position means something only beside its source's coordinates
    prev_fields : dict
        The node's non-split inputs read through each prev-state term
    group_by_block : bool
        Whether the groups of a block are known from the block alone, so that a block
        with no element still has its groups
    """

    def __init__(self):
        self.inner_sources: frozenset = frozenset()
        self.inner: dict[str, LazyOutField] = {}
        self.rpn: list[str] = []
        self.sources: list[str] = []
        self.linked: set[str] = set()
        self.elem_paths: list[tuple] = []
        self.group_paths: list[tuple] | None = None
        self.final_paths: list[tuple] = []
        self.units: list[tuple[tuple, list[tuple]]] = []
        self.prev_fields: dict[str, list[str]] = {}
        self.group_by_block = False


# ----------------------------------------------------------------------------------
# Classification


def source_paths(ex: "NodeExecution", name: str) -> list[tuple]:
    """The paths of the final keys of an upstream node of ``ex``"""
    node = ex.graph.node(name)
    if node._classify() == "partial":
        return node._partial.spec.final_paths
    if not node.state:
        return []
    return [(n,) for n in node.state.keys_final]


def classify(ex: "NodeExecution") -> tuple[str, ty.Any]:
    """Classify a node whose upstream nodes have started.

    Returns
    -------
    mode : str
        ``"static"``, ``"partial"``, ``"barrier"`` or ``"error"``
    detail : PartialExpansion | Exception | None
        The expansion of a partial node, or the error of an ``"error"`` node
    """
    state = ex.state
    if not state:
        return "static", None
    graph = ex.graph
    rpn = list(state.splitter_rpn_compact)
    inner = {
        name: value
        for name, value in ex.node.state_values.items()
        if name in state.current_splitter_rpn and isinstance(value, LazyOutField)
    }
    inner_sources = list(dict.fromkeys(v._node.name for v in inner.values()))
    prev_terms = [
        t for t in rpn_terms(rpn) if t.startswith("_") and t[1:] in state.other_states
    ]
    partial_prevs = [
        t[1:]
        for t in prev_terms
        if t[1:] not in inner_sources and graph.node(t[1:])._classify() == "partial"
    ]
    if not inner and not partial_prevs:
        return "static", None

    spec = _Spec()
    spec.inner = inner
    spec.inner_sources = frozenset(inner_sources)
    spec.linked = {u for u in inner_sources if u in state.other_states}
    spec.rpn = remove_inp_from_splitter_rpn(rpn, [f"_{u}" for u in spec.linked])
    spec.sources = inner_sources + partial_prevs
    # (3) a container dimension set by the user on an inner input
    user_ndim = ex.node._task._container_ndim or {}
    if any(user_ndim.get(name.split(".", 1)[1], 1) > 1 for name in inner):
        return "barrier", None
    # (5) a non-split input read from an inner source, or from a stateful upstream
    # node that the splitter does not reach through a prev-state term
    reduced_terms = set(rpn_terms(spec.rpn))
    for name, value in attrs_values(ex.node._task).items():
        if not isinstance(value, LazyOutField) or f"{ex.name}.{name}" in inner:
            continue
        source = graph.node(value._node.name)
        if value._node.name in inner_sources or (
            source.state
            and source.state.splitter_rpn_final
            and f"_{source.name}" not in reduced_terms
        ):
            return "barrier", None
    names = set()
    for u in inner_sources:
        names |= {p[-1] for p in source_paths(ex, u)}
    for t in prev_terms:
        if t[1:] not in inner_sources and names & {
            p[-1] for p in source_paths(ex, t[1:])
        }:
            return "barrier", None

    # (1) and (2): ``.`` pairs only the node's own static terms, or the inner inputs of
    # one source
    def tag(term):
        if term in inner:
            return "inner:" + inner[term]._node.name
        if term.startswith("_") and term[1:] in partial_prevs:
            return "partial:" + term[1:]
        return "static"

    for left, right in scalar_operands(spec.rpn, tag):
        if not (
            left == right
            and len(left) == 1
            and not next(iter(left)).startswith("partial:")
        ):
            return "barrier", None

    def term_paths(term):
        if term in inner:
            u = inner[term]._node.name
            base = (
                [("_" + u,) + p for p in source_paths(ex, u)]
                if u in spec.linked
                else []
            )
            return base + [(term,)]
        if term.startswith("_"):
            return [(term,) + p for p in source_paths(ex, term[1:])]
        return [(term,)]

    spec.elem_paths = _dedupe(p for t in rpn_terms(spec.rpn) for p in term_paths(t))
    units = []
    for term, value in inner.items():
        u = value._node.name
        base = [("_" + u,) + p for p in source_paths(ex, u)] if u in spec.linked else []
        units.append(((term,), base))
    for t in rpn_terms(spec.rpn):
        if t.startswith("_") and t[1:] in partial_prevs:
            prefix = (t,)
            for j, srcs in graph.node(t[1:])._partial.spec.units:
                units.append((prefix + j, [prefix + s for s in srcs]))
    for u in spec.linked:
        if graph.node(u)._classify() == "partial":
            prefix = ("_" + u,)
            for j, srcs in graph.node(u)._partial.spec.units:
                units.append((prefix + j, [prefix + s for s in srcs]))
    combiner = list(state.combiner or [])
    if combiner:
        keep = {
            u
            for u in spec.linked
            if all(n in combiner for n, v in inner.items() if v._node.name == u)
        }
        group_rpn = remove_inp_from_splitter_rpn(
            rpn, [f"_{u}" for u in spec.linked if u not in keep]
        )
        candidates = _dedupe(
            p
            for t in rpn_terms(group_rpn)
            if not (t in inner and t in combiner)
            for p in term_paths(t)
        )
        every = set(spec.elem_paths) | set(candidates)
        for name in combiner:
            if len({p for p in every if p[-1] == name}) != 1:
                return "barrier", None
        spec.group_paths = [p for p in candidates if p[-1] not in combiner]
        removed = {p for p in every if p[-1] in combiner}
        final_units = []
        for j, srcs in units:
            if j in removed:
                continue
            if any(s in removed for s in srcs):
                if len(j) == 1:
                    return "error", PydraStateError(
                        f"Node {ex.name!r} combines {combiner} but keeps its split "
                        f"input {j[-1]!r}: combining the block axis without its "
                        "inner input has no element order"
                    )
                return "barrier", None
            final_units.append((j, srcs))
        spec.units = final_units
        spec.final_paths = spec.group_paths
        spec.group_by_block = not ({j for j, _ in units} & set(spec.group_paths))
    else:
        spec.units = units
        spec.final_paths = spec.elem_paths
    for t in rpn_terms(spec.rpn):
        if t.startswith("_"):
            spec.prev_fields[t[1:]] = [
                f for f in state.other_states[t[1:]][1] if f"{ex.name}.{f}" not in inner
            ]
    return "partial", PartialExpansion(ex, spec)


# ----------------------------------------------------------------------------------
# Expansion


class _Index:
    """Counts of items by their coordinates on a set of paths, built on first use for
    each set of paths and kept up to date as items are added and removed, so that "is
    there an item compatible with this assignment" costs one lookup"""

    def __init__(self):
        self._by_paths: dict[tuple, dict[tuple, set]] = {}

    def add(self, item, fields: dict) -> None:
        for paths, index in self._by_paths.items():
            index[tuple(fields.get(p) for p in paths)].add(item)

    def remove(self, item, fields: dict) -> None:
        for paths, index in self._by_paths.items():
            bucket = index.get(tuple(fields.get(p) for p in paths))
            if bucket is not None:
                bucket.discard(item)

    def matching(self, paths: tuple, assignment: dict, items: dict) -> set:
        """The items whose fields agree with ``assignment`` on ``paths``; ``items``
        maps every current item to its fields (used to build an index the first time
        a set of paths is asked for)"""
        if paths not in self._by_paths:
            index = defaultdict(set)
            for item, fields in items.items():
                index[tuple(fields.get(p) for p in paths)].add(item)
            self._by_paths[paths] = index
        return self._by_paths[paths].get(tuple(assignment[p] for p in paths), set())


class PartialExpansion:
    """The blocks, element keys, groups and seal of one partial node

    Parameters
    ----------
    ex : NodeExecution
        The node, whose ``_tasks``, ``blocked``, ``unrunnable``, ``errored`` and
        ``successful`` it fills in
    spec : _Spec
        The node's structure, from :func:`classify`
    """

    def __setattr__(self, name, value):
        if name in ("version", "is_sealed") and self.__dict__.get(name, UNSET) != value:
            changed()  # progress for `GraphRunner.ready`
        super().__setattr__(name, value)

    def __init__(self, ex: "NodeExecution", spec: _Spec):
        self.ex = ex
        self.spec = spec
        self.blocks: dict[tuple, _BlockState] | None = None
        self.pending: dict[tuple, dict] = {}  # pending block key -> its fields
        self._pending_index = _Index()
        self.failed: list[tuple[ty.Any, dict]] = []  # own and inherited failed blocks
        self.inherited: set = set()
        self.block_errors: list[str] = []
        self.vals: dict[tuple, dict] = {}
        self.inputs_ind: dict[tuple, dict] = {}
        self.block_of: dict[tuple, tuple] = {}  # element key -> the key of its block
        self.group_of: dict[tuple, tuple] = {}  # element key -> its group
        # pending blocks by (inner source, its final key), so that ``expand_ready``
        # visits only the blocks whose inner sources' keys changed (see ``_changed``)
        self._waiting: dict[tuple, set] = defaultdict(set)
        self._fresh: set = set()  # blocks registered since the last ``expand_ready``
        self._cursors: dict[str, tuple] = {}
        self._reverse: dict[str, dict] = {}
        self.groups: dict[tuple, list[tuple]] = {}
        self.closed: set[tuple] = set()
        self.open_groups: dict[tuple, dict] = {}  # open group key -> its fields
        self._open_index = _Index()
        self.final_log: list = []  # the node's final keys, in the order they appeared
        self._seen: list[int] = [0] * len(spec.sources)  # offsets in sources' logs
        self._seen_keys: list[list] = [[] for _ in spec.sources]
        self.static_entries: dict[str, list] = {}
        self.version = 0
        self.is_sealed = False
        self.rank: dict[tuple, int] = {}
        self.keys_by_rank: list[tuple] = []
        self.legacy: "bool | Exception | None" = None
        self.legacy_groups: list[tuple] = []
        self._reg_token: tuple | None = None

    @property
    def graph(self):
        return self.ex.graph

    def sources(self) -> list["NodeExecution"]:
        return [self.graph.node(n) for n in self.spec.sources]

    # ------------------------------------------------------------------------------
    # Starting

    def start(self) -> None:
        """No element is known yet; blocks are registered as their source keys
        appear"""
        ex, spec = self.ex, self.spec
        ex._tasks = {}
        ex.blocked = {}
        self.blocks = {}
        for term in rpn_terms(spec.rpn):
            if term in spec.inner:
                continue
            if term.startswith("_"):
                name = term[1:]
                if name in spec.sources:
                    continue
                source = self.graph.node(name)
                self.static_entries[term] = [
                    Entry({(term, n): c for n, c in ind.items()}, {}, {name: g})
                    for g, ind in enumerate(source.state.states_ind_final)
                ]
            else:
                value = ex.node.state_values[term]
                if isinstance(value, LazyField):
                    value = value._get_value(workflow=ex.workflow, graph=self.graph)
                ndim = (ex.state.container_ndim or {}).get(term, 1)
                items = list(flatten(ensure_list(value), max_depth=ndim))
                self.static_entries[term] = [
                    Entry({(term,): i}, {term: v}, {}) for i, v in enumerate(items)
                ]
        self.register()

    # ------------------------------------------------------------------------------
    # The final keys the node shows its readers

    def final_keys(self) -> list:
        """The final keys of the node that exist now, in the order they appeared
        (append-only, so a reader can take only the new ones)"""
        if self.spec.group_paths is not None:
            for group in list(self.open_groups):
                self.group_closed(group)
        return self.final_log

    def final_fields(self, key) -> dict:
        return dict(zip(self.spec.final_paths, key))

    # ------------------------------------------------------------------------------
    # Blocks

    def _inner_positions(self, bkey: tuple) -> list[tuple[str, ty.Any]]:
        """The inner sources of a block, each with the final key the block reads"""
        names = self.spec.inner_sources
        return [(n, k) for n, k in zip(self.spec.sources, bkey) if n in names]

    def _add_pending(self, bkey: tuple, block: _BlockState) -> None:
        self.blocks[bkey] = block
        self.pending[bkey] = block.fields
        self._pending_index.add(bkey, block.fields)
        self._fresh.add(bkey)
        for item in self._inner_positions(bkey):
            self._waiting[item].add(bkey)

    def _set_status(self, bkey: tuple, status: str) -> None:
        block = self.blocks[bkey]
        block.status = status
        self.pending.pop(bkey, None)
        self._pending_index.remove(bkey, block.fields)
        self._fresh.discard(bkey)
        for item in self._inner_positions(bkey):
            waiting = self._waiting.get(item)
            if waiting is not None:
                waiting.discard(bkey)
                if not waiting:
                    del self._waiting[item]
        if status in ("failed", "errored"):
            self.failed.append((("block", bkey), block.fields))
        self.version += 1

    def register(self) -> None:
        """Register a block for each new combination of source keys, and a failed
        block for each new failed block of a source"""
        ex, spec = self.ex, self.spec
        sources = self.sources()
        token = tuple(s._token() for s in sources)
        if token == self._reg_token:
            return
        self._reg_token = token
        new_keys = []
        for i, (name, source) in enumerate(zip(spec.sources, sources)):
            if name in spec.linked or source.is_partial:
                log = source._final_keys()
            else:
                log = [None]
            new_keys.append(list(log[self._seen[i] :]))
            self._seen[i] = len(log)
        # every combination with at least one new key: new keys at position i, keys
        # seen before at positions before i, all keys at positions after i
        for i in range(len(sources)):
            if not new_keys[i]:
                continue
            lists = (
                self._seen_keys[:i]
                + [new_keys[i]]
                + [self._seen_keys[k] + new_keys[k] for k in range(i + 1, len(sources))]
            )
            for combo in itertools.product(*lists):
                if combo in self.blocks:
                    continue
                fields = {}
                for name, source, key in zip(spec.sources, sources, combo):
                    for p, c in source._final_fields(key).items():
                        fields[("_" + name,) + p] = c
                self._add_pending(combo, _BlockState(fields))
                self.version += 1
        for i in range(len(sources)):
            self._seen_keys[i].extend(new_keys[i])
        for name, source in zip(spec.sources, sources):
            for ident, fields in source._failed_blocks():
                if (name, ident) in self.inherited:
                    continue
                self.inherited.add((name, ident))
                prefixed = {("_" + name,) + p: c for p, c in fields.items()}
                # a block whose fields all lie on paths the source combines away fixes
                # no final key: the readers of the groups it fails report them, so it
                # is not an element of its own (it still keeps groups from sealing)
                if self._fixes_final_key(source, fields):
                    block = Block(name, ident[1:], tuple(prefixed.items()))
                    ex.unrunnable[block] = [source]
                self.failed.append((("from", name, ident), prefixed))
                self.version += 1

    @staticmethod
    def _fixes_final_key(source: "NodeExecution", fields: dict) -> bool:
        """Whether a failed block of ``source`` fixes a coordinate of its final keys, or
        the source has a single final key (combined whole), which then never appears"""
        final = source._partial.spec.final_paths if source.is_partial else None
        if not final:
            return True
        return any(p in fields for p in final)

    def _changed(self, name: str) -> "set | None":
        """The final keys of inner source ``name`` whose status may have changed since
        the last call, from the append-only logs of its tables (and, for a partial
        source, its closed groups); None if any of them may have (a whole-node or block
        failure, or logs that were replaced)"""
        source = self.graph.node(name)
        logs = (source.successful.log, source.errored.log, source.unrunnable.log)
        partial = source._partial if source.is_partial else None
        extra = (
            (partial.final_log, len(partial.failed), len(partial.block_errors))
            if partial is not None
            else (None, 0, 0)
        )
        before = self._cursors.get(name)
        now = tuple((id(log), len(log)) for log in logs) + (
            len(extra[0]) if extra[0] is not None else 0,
            extra[1],
            extra[2],
        )
        self._cursors[name] = now
        if before is None or any(b[0] != n[0] for b, n in zip(before[:3], now[:3])):
            return None
        if before[4:] != now[4:]:  # a failed block, or a block error, of the source
            return None
        entries = [
            key for log, (_, start) in zip(logs, before[:3]) for key in log[start:]
        ]
        if any(e is None or isinstance(e, Block) for e in entries):
            return None if source.state else {None}
        keyed = name in self.spec.linked or partial is not None
        if not keyed:  # its whole output is read: one key, None
            return {None} if entries else set()
        changed = set()
        if partial is not None:
            if partial.spec.group_paths is None:
                changed.update(entries)
            else:
                changed.update(
                    partial.group_of[e] for e in entries if e in partial.group_of
                )
                changed.update(partial.final_log[before[3] :])  # closed groups
            return changed
        reverse = self._reverse.get(name)
        if reverse is None:  # a static source: its elements -> its final keys
            reverse = self._reverse[name] = defaultdict(list)
            for k in source._final_keys():
                for e in source._read_elements(source._read_indices(k)):
                    reverse[e].append(k)
        for e in entries:
            changed.update(reverse.get(e, ()))
        return changed

    def _candidates(self) -> list:
        """The pending blocks whose status may have changed: those registered since the
        last call, and those that read a changed key of an inner source"""
        found = set(self._fresh)
        self._fresh = set()
        for name in self.spec.inner_sources:
            changed = self._changed(name)
            if changed is None:
                return list(self.pending)
            for key in changed:
                found.update(self._waiting.get((name, key), ()))
        return [b for b in found if b in self.pending]

    def expand_ready(self) -> None:
        """Expand each pending block whose inner sources have succeeded, and fail each
        whose inner source has failed. Only the blocks that may have changed since the
        last call are visited (``_candidates``)"""
        ex, spec = self.ex, self.spec
        inner_sources = spec.inner_sources
        for bkey in sorted(self._candidates(), key=sort_key):
            statuses = []
            for name, key in zip(spec.sources, bkey):
                if name not in inner_sources:
                    continue
                source = self.graph.node(name)
                if name in spec.linked or source.is_partial:
                    statuses.append((source, source._final_status(key)))
                else:
                    statuses.append((source, source._final_status(None)))
            if failed := [s for s, status in statuses if status == "failed"]:
                fields = self.blocks[bkey].fields
                ex.unrunnable[Block(ex.name, bkey, tuple(fields.items()))] = failed
                self._set_status(bkey, "failed")
            elif all(status == "ok" for _, status in statuses):
                self._expand(bkey, self.blocks[bkey])
                if ex.absent is not None:  # the node became absent as a whole
                    return

    def _expand(self, bkey: tuple, block: _BlockState) -> None:
        from pydra.engine.runner import _check_splittable, error_entry

        ex, spec = self.ex, self.spec
        entries = dict(self.static_entries)
        skeleton = dict(self.static_entries)
        positions = dict(zip(spec.sources, bkey))
        for term, lf in spec.inner.items():
            if positions[lf._node.name] is None:  # the whole output of a source
                value = lf._get_value(workflow=ex.workflow, graph=self.graph)
                if isinstance(value, Absent) and not ex._tasks:
                    # every block reads this one value: the node is absent as a whole
                    ex._become_absent(value)
                    return
                _check_splittable(ex.name, term, value)
        try:
            for term in rpn_terms(spec.rpn):
                if term in spec.inner:
                    lf = spec.inner[term]
                    u = lf._node.name
                    key = positions[u]
                    value = lf._get_value(
                        workflow=ex.workflow, graph=self.graph, state_index=key
                    )
                    items = list(value) if is_sequence(value) else [value]
                    base = {p: c for p, c in block.fields.items() if p[0] == "_" + u}
                    terms = {u: key} if u in spec.linked else {}
                    entries[term] = [
                        Entry({**base, (term,): j}, {term: v}, terms)
                        for j, v in enumerate(items)
                    ]
                    skeleton[term] = [Entry(dict(base), {}, terms)]
                elif term.startswith("_") and term[1:] in spec.sources:
                    name = term[1:]
                    base = {p: c for p, c in block.fields.items() if p[0] == term}
                    entries[term] = skeleton[term] = [
                        Entry(base, {}, {name: positions[name]})
                    ]
            elements = expand_rpn(spec.rpn, entries)
        except Exception as exc:
            label = f"{ex.name}[block {key_text(bkey)}]"
            try:
                raise exc
            except Exception:
                trace = format_exc()
            self.block_errors.append(
                f"Job {label!r}, could not be expanded, errored:\n{trace}"
            )
            ex._resolution_errors.append(self.block_errors[-1])
            ex._resolution_entries.append(
                error_entry(ex.name, None, exc, phase="expansion", block=bkey)
            )
            self._set_status(bkey, "errored")
            return
        if spec.group_paths is not None and spec.group_by_block:
            for entry in expand_rpn(spec.rpn, skeleton):
                self._add_group(tuple(entry.fields[p] for p in spec.group_paths))
        for entry in elements:
            key = tuple(entry.fields[p] for p in spec.elem_paths)
            ex._tasks[key] = None
            self.block_of[key] = bkey
            self.vals[key] = entry.values
            self.inputs_ind[key] = {
                f"{ex.name}.{field}": entry.terms[name]
                for name, fields in spec.prev_fields.items()
                if name in entry.terms
                for field in fields
            }
            if spec.group_paths is not None:
                group = tuple(entry.fields[p] for p in spec.group_paths)
                self._add_group(group)
                self.groups[group].append(key)
                self.group_of[key] = group
            else:
                self.final_log.append(key)
            ex.blocked[key] = ex._upstream(key)
        self._set_status(bkey, "expanded")

    def inner_reads(self, key: tuple) -> dict[str, ty.Any]:
        """For each inner input of an element, the final key of the upstream element
        whose output its block split (None for a whole read of a source without a
        state)"""
        positions = dict(zip(self.spec.sources, self.block_of[key]))
        return {term: positions[lf._node.name] for term, lf in self.spec.inner.items()}

    def _add_group(self, group: tuple) -> None:
        if group in self.groups:
            return
        self.groups[group] = []
        fields = dict(zip(self.spec.group_paths, group))
        self.open_groups[group] = fields
        self._open_index.add(group, fields)

    # ------------------------------------------------------------------------------
    # Settled, sealed, groups

    def settled(self) -> bool:
        if self.blocks is None:
            return False
        self.register()
        if self.pending:
            return False
        return all(s.settled for s in self.sources())

    def sealed(self) -> bool:
        if self.is_sealed:
            return True
        if not self.settled() or self.failed:
            return False
        if not all(s.sealed for s in self.sources()):
            return False
        self.is_sealed = True
        self.keys_by_rank = sorted(self.ex._tasks)
        self.rank = {k: i for i, k in enumerate(self.keys_by_rank)}
        for key, job in self.ex._tasks.items():
            if job is not None:
                job.state_index = self.rank[key]
        return True

    def may_produce(self, assignment: dict) -> bool:
        """Whether an element whose final coordinates agree with ``assignment`` may still
        appear (conservative: True when it cannot be ruled out)"""
        if self.blocks is None:
            return True
        self.register()
        if self.pending:
            paths = tuple(
                p for p in next(iter(self.pending.values())) if p in assignment
            )
            if self._pending_index.matching(paths, assignment, self.pending):
                return True
        for source in self.sources():
            prefix = "_" + source.name
            sub = {p[1:]: c for p, c in assignment.items() if p[0] == prefix}
            if source._may_produce(sub) or source._has_open_group(sub):
                return True
        return False

    def has_open_group(self, assignment: dict) -> bool:
        if self.spec.group_paths is None or not self.open_groups:
            return False
        paths = tuple(p for p in self.spec.group_paths if p in assignment)
        candidates = self._open_index.matching(paths, assignment, self.open_groups)
        return any(not self.group_closed(g) for g in list(candidates))

    def group_closed(self, group: tuple) -> bool:
        if group in self.closed:
            return True
        fields = self.open_groups.get(group)
        if fields is None:  # a group with no element yet
            return not self.may_produce(dict(zip(self.spec.group_paths, group)))
        if self.may_produce(fields):
            return False
        self.closed.add(group)
        self._open_index.remove(group, self.open_groups.pop(group))
        self.final_log.append(group)
        self.version += 1
        return True

    def group_failed(self, group: tuple) -> bool:
        ex = self.ex
        if any(
            m in ex.errored or m in ex.unrunnable for m in self.groups.get(group, ())
        ):
            return True
        assignment = dict(zip(self.spec.group_paths, group))
        return any(compatible(fields, assignment) for _, fields in self.failed)

    def group_succeeded(self, group: tuple) -> bool:
        return self.group_closed(group) and all(
            m in self.ex.successful for m in self.groups.get(group, ())
        )

    def token(self) -> tuple:
        return (self.version, tuple(s._token() for s in self.sources()))

    def summary(self) -> str:
        """The blocks that are not expanded, for ``stuck()``"""
        if self.blocks is None:
            return ""
        by_status = defaultdict(list)
        for key, block in self.blocks.items():
            if block.status != "expanded":
                by_status[block.status].append(key)
        unsettled = [s.name for s in self.sources() if not s.settled]
        return "".join(
            f", {status} blocks={keys}" for status, keys in sorted(by_status.items())
        ) + (f", unsettled sources={unsettled}" if unsettled else "")

    # ------------------------------------------------------------------------------
    # Reading

    def read_spec(self, state_index: ty.Any) -> ty.Any:
        """What of the node a reader reads at ``state_index`` (see
        ``NodeExecution._read_indices``)"""
        spec = self.spec
        if state_index is None:
            return ALL
        if isinstance(state_index, int):  # a barrier reader, after the seal
            if spec.group_paths is None:
                return [self.keys_by_rank[state_index]]
            state_index = self.legacy_groups[state_index]
        if spec.group_paths is None:
            return [state_index]
        if not spec.group_paths:
            return ALL
        return GroupRead(state_index)

    def read_value(self, retrieve: ty.Callable[["Job"], ty.Any], state_index: ty.Any):
        """The value of the node's output for a reader: one element, one group, or all
        of them once sealed"""
        ex, spec = self.ex, self.spec
        if isinstance(state_index, int):
            if spec.group_paths is None:
                state_index = self.keys_by_rank[state_index]
            else:
                state_index = self.legacy_groups[state_index]
        if state_index is None:
            if not self.sealed():
                raise RuntimeError(
                    f"Cannot read every element of node {ex.name!r}: it is not sealed "
                    "(an upstream element has not completed, or has failed)"
                )
            if spec.group_paths is None:
                return StateArray(retrieve(ex.job(k)) for k in self.keys_by_rank)
            if not spec.group_paths:
                return [retrieve(ex.job(k)) for k in self.keys_by_rank]
            return StateArray(
                [retrieve(ex.job(m)) for m in sorted(self.groups[g])]
                for g in sorted(self.groups)
            )
        if spec.group_paths is None:
            return retrieve(ex.job(state_index))
        return [retrieve(ex.job(m)) for m in sorted(self.groups[state_index])]

    # ------------------------------------------------------------------------------
    # Today's ``State`` for a barrier reader

    def legacy_prepare(self) -> "bool | Exception":
        """Prepare the node's ``State`` as today's barrier path would, for a barrier
        reader, once the node is sealed. Returns True, or the reason it cannot be (a
        ragged or combined source, or an order today's ``State`` gets wrong)"""
        ex = self.ex
        if self.legacy is not None:
            return self.legacy
        if not self.sealed():
            self.legacy = PydraStateError(f"node {ex.name!r} is not sealed")
            return self.legacy
        try:
            for source in self.sources():
                if (
                    source.is_partial
                    and (error := source._legacy_prepare()) is not True
                ):
                    raise PydraStateError(str(error))
            values = {}
            for name, value in ex.node.state_values.items():
                if name in ex.state.current_splitter_rpn and name not in ("*", "."):
                    if isinstance(value, LazyField):
                        values[name] = value._get_value(
                            workflow=ex.workflow, graph=self.graph
                        )
                    else:
                        values[name] = value
            ex._check_barrier(values)
            ex.state.prepare_states(values)
            ex.state.prepare_inputs()
            own = list(self.vals[self.keys_by_rank[0]]) if ex._tasks else []
            legacy = [{k: v[k] for k in own if k in v} for v in ex.state.states_val]
            if legacy != [self.vals[k] for k in self.keys_by_rank]:
                raise PydraStateError(
                    f"today's State orders the elements of {ex.name!r} differently"
                )
            if self.spec.group_paths is not None:
                groups = sorted(self.groups)
                mine = [sorted(self.rank[m] for m in self.groups[g]) for g in groups]
                mapping = ex.state.final_combined_ind_mapping
                theirs = [sorted(mapping[i]) for i in range(len(mapping))]
                if ex.state.ind_l_final and mine != theirs:
                    raise PydraStateError(
                        f"today's State groups the elements of {ex.name!r} differently"
                    )
                self.legacy_groups = groups
        except Exception as exc:
            self.legacy = exc
        else:
            self.legacy = True
        return self.legacy
