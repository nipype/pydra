"""`Reuse` verdicts, `find(job)`, the verify mark, the lost-race table, the commit hook, sidecars
and read-only roots of a `PortableStore` (RFC 0003 section 7, P5)."""

import os
import pickle
import stat
import sys
import typing as ty
import uuid
from pathlib import Path

import cloudpickle
import pytest

from pydra.compose import python
from pydra.engine.job import Job
from pydra.engine.portable.identity import PydraIdentityV1
from pydra.engine.portable.layout import StoreLayoutMismatch
from pydra.engine.portable.reuse import (
    Candidate,
    Hit,
    Miss,
    PolicyReuse,
    Verdict,
    Verify,
)
from pydra.engine.portable.store import CommitHookError, PortableStore
from pydra.engine.portable.trust import Root
from pydra.engine.result import Result
from pydra.engine.store import AnyReuse, ExactReuse
from pydra.engine.submitter import Submitter
from pydra.utils.storagedir import StorageDir

from .test_portable_store import IDENT, TRUST, _Lost, make, posix_only, staging_names
from .test_portable_workdirs import _Count, job_in

# Worker processes cannot import the test modules, so they receive their classes by value
cloudpickle.register_pickle_by_value(sys.modules[__name__])


@python.define
def _Inc(a: int) -> int:
    return a + 1


class Scripted(PolicyReuse):
    """A hook that answers by the ``licence`` field of the stored context and records what it
    was asked."""

    def __init__(
        self,
        verdicts: ty.Mapping[ty.Any, Verdict] | None = None,
        default: Verdict = Verdict.SERVE,
        policy: ty.Any = None,
        fail_served: bool = False,
    ) -> None:
        super().__init__(policy if policy is not None else AnyReuse())
        self.verdicts = dict(verdicts or {})
        self.default = default
        self.fail_served = fail_served
        self.log: list[tuple] = []
        self.on_rank: ty.Callable | None = None

    def rank(self, current, stored, task):
        ranked = super().rank(current, stored, task)
        self.log.append(("rank", [c.context.get("licence") for c in ranked]))
        if self.on_rank is not None:
            hook, self.on_rank = self.on_rank, None
            hook(ranked)
        return ranked

    def verdict(self, candidate, current, task):
        key = candidate.context.get("licence")
        self.log.append(("verdict", key))
        return self.verdicts.get(key, self.default)

    def served(self, candidate, current, task, entry):
        self.log.append(("served", candidate.context.get("licence")))
        if self.fail_served:
            raise RuntimeError("the reuse record could not be written")

    def compare(self, candidate, stored, fresh, fresh_digest, task):
        self.log.append(("compare", candidate.rank, dict(stored.outputs), dict(fresh)))

    def calls(self, name: str) -> list:
        return [c for c in self.log if c[0] == name]


def put(store, job, licence=None, out=5, **fields):
    """Publish a result of ``job``'s checksum under the job's context with ``licence`` (and
    other fields) changed; no field is the job's own address."""
    context = dict(job.context, **fields)
    if licence is not None:
        context["licence"] = licence
    return store.publish(
        job.checksum, None, context, {"out": out}, job.task, None, replace=False
    )


def jobfor(tmp_path, store, a=1):
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = Job(task=_Inc(a=a), name="j", submitter=sub)
    job.attempt = uuid.uuid4().hex
    return job


def licences(store, job):
    return [
        c.context.get("licence")
        for c in store.reuse.rank(job.context, store.stored(job.checksum), job.task)
    ]


# -- ranking and verdicts ------------------------------------------------------------------------------


def test_policy_wrapper_matches_picklestore_order(tmp_path):
    store = make(tmp_path, reuse=AnyReuse())
    job = jobfor(tmp_path, store)
    for licence in ("c", "a", "b"):
        put(store, job, licence)
    own = put(store, job, None)
    ranked = store.reuse.rank(job.context, store.stored(job.checksum), job.task)
    mine = IDENT.fingerprint(job.context)
    others = sorted(IDENT.fingerprint(c.context) for c in ranked[1:])
    assert [c.entry.fingerprint for c in ranked] == [mine, *others]
    assert [c.rank for c in ranked] == [0, 1, 1, 1]
    assert ranked[0].admitted_by == "exact" and ranked[1].admitted_by == "any"
    assert ranked[0].entry == own.entry
    assert store.reuse.verdict(ranked[1], job.context, job.task) is Verdict.SERVE
    hit = store.find(job)
    assert isinstance(hit, Hit) and hit.result.cross_context is False
    # without the job's own slot, the first of the rest in path order is reused
    store.evict(own.entry, own.own_digest)
    hit = store.find(job)
    assert isinstance(hit, Hit)
    assert hit.result.cross_context is True
    assert hit.result.reused_in == dict(job.context)
    assert hit.result.context["licence"] == ranked[1].context["licence"]
    assert hit.result.outputs.out == 5


def test_exact_policy_serves_only_the_own_slot(tmp_path):
    store = make(tmp_path, reuse="exact")
    assert isinstance(store.reuse, PolicyReuse)
    assert type(store.reuse._policy) is ExactReuse
    job = jobfor(tmp_path, store)
    put(store, job, "x")
    assert isinstance(store.find(job), Miss)
    put(store, job, None, out=9)
    hit = store.find(job)
    assert isinstance(hit, Hit) and hit.result.outputs.out == 9


def test_lookup_is_the_adapter(tmp_path):
    store = make(tmp_path, reuse=AnyReuse())
    job = jobfor(tmp_path, store)
    assert store.lookup(job.checksum, job.context, job.task) is None
    put(store, job, "x")
    got = store.lookup(job.checksum, job.context, job.task)
    assert got is not None and got.outputs.out == 5 and got.cross_context
    verify = make(tmp_path, "v", reuse=Scripted({"x": Verdict.VERIFY}))
    vjob = jobfor(tmp_path, verify)
    put(verify, vjob, "x")
    assert isinstance(verify.find(vjob), Verify)
    assert verify.lookup(vjob.checksum, vjob.context, vjob.task) is None


def test_refuse_ends_lookup(tmp_path):
    hook = Scripted({"a": Verdict.REFUSE})
    store = make(tmp_path, reuse=hook)
    job = jobfor(tmp_path, store)
    for licence in ("a", "b", "c"):
        put(store, job, licence)
    first = licences(store, job)[0]
    hook.verdicts = {first: Verdict.REFUSE}
    hook.log.clear()
    assert isinstance(store.find(job), Miss)
    assert hook.calls("verdict") == [("verdict", first)]
    assert hook.calls("served") == []


def test_refuse_is_miss_and_ends_lookup(tmp_path, monkeypatch):
    """A verdict keyed on a field of the stored context (a licence) is a `Miss`, and no
    further candidate is read."""
    hook = Scripted({"closed": Verdict.REFUSE})
    store = make(tmp_path, reuse=hook)
    job = jobfor(tmp_path, store)
    put(store, job, "closed")
    put(store, job, "open")
    order = licences(store, job)
    hook.verdicts = {order[0]: Verdict.REFUSE}
    reads = []
    real = store._read_entry
    monkeypatch.setattr(
        store, "_read_entry", lambda *a, **k: reads.append(a) or real(*a, **k)
    )
    assert isinstance(store.find(job), Miss)
    assert reads == []


def test_skip_next_candidate(tmp_path):
    hook = Scripted()
    store = make(tmp_path, reuse=hook)
    job = jobfor(tmp_path, store)
    put(store, job, "a", out=1)
    put(store, job, "b", out=2)
    first, second = licences(store, job)
    hook.verdicts = {first: Verdict.SKIP}
    hook.log.clear()
    got = store.find(job)
    assert isinstance(got, Hit)
    assert got.result.context["licence"] == second
    assert hook.calls("verdict") == [("verdict", first), ("verdict", second)]
    assert hook.calls("served") == [("served", second)]


def test_served_raising_blocks_serve(tmp_path):
    hook = Scripted(fail_served=True)
    store = make(tmp_path, reuse=hook)
    job = jobfor(tmp_path, store)
    put(store, job, "a")
    put(store, job, "b")
    assert isinstance(store.find(job), Miss)
    assert len(hook.calls("served")) == 1  # and the next candidate is not tried
    assert store.lookup(job.checksum, job.context, job.task) is None


def test_context_changed_between_rank_and_read_reranks(tmp_path):
    ident = PydraIdentityV1(recorded_only={"host"})
    hook = Scripted()
    store = make(tmp_path, identity=ident, reuse=hook)
    job = jobfor(tmp_path, store)
    first = put(store, job, "a", host="one")

    def swap(_ranked):
        store.evict(first.entry, first.own_digest)
        put(store, job, "a", host="two")

    hook.on_rank = swap
    got = store.find(job)
    assert isinstance(got, Hit)
    assert len(hook.calls("rank")) == 2
    assert got.result.context["host"] == "two"  # the verdict applied to what was read
    assert hook.calls("served") == [("served", "a")]


def test_corrupt_candidate_is_quarantined_and_the_next_taken(tmp_path):
    store = make(tmp_path, reuse=AnyReuse())
    job = jobfor(tmp_path, store)
    put(store, job, "a")
    put(store, job, "b")
    ranked = store.reuse.rank(job.context, store.stored(job.checksum), job.task)
    victim = store.entry_path(job.checksum, ranked[0].context)
    os.chmod(victim, 0o755)
    os.chmod(victim / "entry.json", 0o644)
    (victim / "entry.json").write_text("{")
    got = store.find(job)
    assert isinstance(got, Hit)
    assert got.result.context["licence"] != ranked[0].context["licence"]
    assert list((Path(store.path) / "quarantine").iterdir())
    assert not victim.exists()


# -- the verify mark ----------------------------------------------------------------------------------


def test_verify_marks_job_and_compare_runs_before_rename(tmp_path):
    log = tmp_path / "log"
    hook = Scripted({"old": Verdict.VERIFY})
    store = make(tmp_path, reuse=hook)
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = job_in(sub, _Count(a=1, log=str(log), pause=0.0))
    put(store, job, "old", out=2)
    seen = {}
    real = hook.compare

    def compare(candidate, stored, fresh, fresh_digest, task):
        seen["published"] = store.entry_path(job.checksum, job.context).exists()
        seen["staged"] = len(staging_names(store.path))
        seen["candidate"] = candidate
        seen["digest"] = fresh_digest
        real(candidate, stored, fresh, fresh_digest, task)

    hook.compare = compare
    assert job.verify is None
    result = job.run()
    assert log.read_text() == "ran\n"  # verify runs the job
    assert isinstance(job.verify, Candidate) and job.verify is seen["candidate"]
    assert job.verify.context["licence"] == "old" and job.verify.rank == 1
    assert seen["published"] is False and seen["staged"] == 1
    assert hook.calls("compare")[0][2:] == ({"out": 2}, {"out": 2})
    assert result.committed.won and result.committed.own_digest == seen["digest"]
    assert store.entry_path(job.checksum, job.context).is_dir()


def test_a_served_hit_clears_the_verify_mark(tmp_path):
    hook = Scripted()
    store = make(tmp_path, reuse=hook)
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = job_in(sub, _Inc(a=1))
    put(store, job, "x", out=7)
    job.verify = object()
    got = job.run()
    assert got.outputs.out == 7 and job.verify is None


def test_verify_at_rank0_compare_then_lost_race(tmp_path):
    log = tmp_path / "log"
    hook = Scripted({None: Verdict.VERIFY})
    store = make(tmp_path, reuse=hook)
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = job_in(sub, _Count(a=1, log=str(log), pause=0.0))
    occupant = put(store, job, None, out=99)  # the job's own address, other outputs
    result = job.run()
    assert log.read_text() == "ran\n"
    assert job.verify is not None and job.verify.rank == 0
    compares = hook.calls("compare")
    assert len(compares) == 1 and compares[0][1] == 0
    assert compares[0][2:] == ({"out": 99}, {"out": 2})
    # the rename failed (the address is occupied), the hook did not SERVE the occupant
    assert result.committed.won is False and result.stored is False
    assert result.outputs.out == 2
    assert hook.calls("served") == []
    assert store.read(occupant.entry, job.task).outputs.out == 99


def test_lost_race_table(tmp_path):
    hook = Scripted()
    store = make(tmp_path, reuse=hook)
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        job.attempt = uuid.uuid4().hex

        def commit(out):
            r = Result(
                outputs=_Inc.Outputs(out=out),
                cache_dir=job.cache_dir,
                task=job.task,
                context=job.context,
            )
            store.commit(_Lost(), r, job)
            return r

        first = commit(2)
        assert first.committed.won
        # row 1: equal digests: an execution, its own outputs, served is not called
        equal = commit(2)
        assert (equal.committed.won, equal.stored, equal.committed.served) == (
            False, True, False,
        )  # fmt: skip
        assert equal.outputs.out == 2 and hook.calls("served") == []
        # row 2: unequal digests and the hook serves the occupant as a rank-0 candidate
        served = commit(3)
        assert served.stored is True and served.committed.served is True
        assert served.outputs.out == 2
        assert hook.calls("served") == [("served", None)]
        assert store.read_result(job).outputs.out == 2  # a later reader agrees
        # row 3: REFUSE, and SKIP, give the job its own outputs, uncommitted
        for verdict in (Verdict.REFUSE, Verdict.SKIP):
            hook.verdicts = {None: verdict}
            hook.log.clear()
            own = commit(4)
            assert own.stored is False and own.outputs.out == 4
            assert own.committed.kept is not None and own.committed.served is False
            assert hook.calls("served") == []
        assert store.read(first.committed.entry, job.task).outputs.out == 2


def test_lost_race_with_a_failing_record_keeps_its_own_outputs(tmp_path):
    hook = Scripted()
    store = make(tmp_path, reuse=hook)
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        job.attempt = uuid.uuid4().hex
        store.publish(job.checksum, None, job.context, {"out": 2}, job.task, None)
        hook.fail_served = True
        r = Result(
            outputs=_Inc.Outputs(out=3),
            cache_dir=job.cache_dir,
            task=job.task,
            context=job.context,
        )
        store.commit(_Lost(), r, job)
    assert r.stored is False and r.outputs.out == 3 and r.committed.kept is not None


# -- the commit hook -----------------------------------------------------------------------------------


def test_on_commit_raising_commits_nothing(tmp_path):
    def refuse(*args):
        raise RuntimeError("no provenance, no result")

    store = make(tmp_path, on_commit=refuse)
    with pytest.raises(RuntimeError, match="no provenance"):
        store.publish("python-abc", None, {"n": 1}, {"a": 1}, None, None)
    assert staging_names(store.path) == []
    assert not any((Path(store.path) / "results@1").rglob("entry.json"))


def test_on_commit_sees_staging_before_rename(tmp_path):
    """The hook keeps its record in a sidecar, written before the entry is visible; the entry
    reads back and a job runs through it."""
    calls = []

    def hook(staged, checksum, key, context, digest):
        staged = Path(staged)
        calls.append(
            (
                checksum, key, dict(context), digest,
                (Path(store.path) / "staging") == staged.parent,
                any((Path(store.path) / "results@1").rglob("entry.json")),
                sorted(p.name for p in staged.iterdir()),
            )
        )  # fmt: skip
        store.write_sidecar("prov", f"{digest}.json", b"{}")

    store = make(tmp_path, on_commit=hook)
    c = store.publish("python-abc", None, {"n": 1}, {"a": 1}, None, None)
    (call,) = calls
    assert call[:6] == ("python-abc", None, {"n": 1}, c.own_digest, True, False)
    assert "entry.json" in call[6] and "context.json" in call[6]
    assert store.read(c.entry, None).outputs == {"a": 1}
    side = Path(store.path) / "sidecars" / "prov@1" / f"{c.own_digest}.json"
    assert side.read_bytes() == b"{}"
    assert c.won and staging_names(store.path) == []
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = job_in(sub, _Inc(a=1))
    assert job.run().outputs.out == 2
    assert isinstance(store.find(jobfor(tmp_path, store)), Hit)


def test_on_commit_adding_a_member_is_refused(tmp_path):
    def provenance(staged, *args):
        (Path(staged) / "note.txt").write_text("not a member")

    store = make(tmp_path, on_commit=provenance)
    with pytest.raises(CommitHookError, match=r"provenance.*note\.txt"):
        store.publish("python-abc", None, {"n": 1}, {"a": 1}, None, None)
    assert staging_names(store.path) == []
    assert not any((Path(store.path) / "results@1").rglob("entry.json"))


def test_on_commit_runs_after_compare(tmp_path):
    order = []
    hook = Scripted({"old": Verdict.VERIFY})
    real = hook.compare
    hook.compare = lambda *a: (order.append("compare"), real(*a))[1]
    store = make(tmp_path, reuse=hook, on_commit=lambda *a: order.append("on_commit"))
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = job_in(sub, _Inc(a=1))
    put(store, job, "old", out=2)
    order.clear()
    job.run()
    assert order == ["compare", "on_commit"]


# -- binding, sidecars, read-only roots ---------------------------------------------------------------


def test_reuse_bound_after_unpickle(tmp_path):
    hook = Scripted()
    store = make(tmp_path, reuse=hook)
    assert hook._store is store
    again = pickle.loads(pickle.dumps(store))
    assert again.reuse is not hook
    assert again.reuse._store is again
    assert again.reuse.default is Verdict.SERVE
    wrapped = pickle.loads(pickle.dumps(make(tmp_path, "w", reuse="exact")))
    assert wrapped.reuse._store is wrapped
    assert type(wrapped.reuse._policy) is ExactReuse


def test_sidecar_is_a_storage_dir_collect_leaves_alone(tmp_path):
    store = make(tmp_path)
    with store.sidecar("verify") as side:
        assert isinstance(side, StorageDir)
        assert side.path == os.path.join(store.path, "sidecars", "verify@1")
        side.child("kept", create=True).close()
    store.collect()
    assert (Path(store.path) / "sidecars" / "verify@1" / "kept").is_dir()


def test_hook_can_write_its_record_to_a_sidecar_through_bind(tmp_path):
    class Recording(Scripted):
        def served(self, candidate, current, task, entry):
            self._store.write_sidecar(
                "reuse", "r-" + candidate.entry.fingerprint[:8] + ".json", b"{}"
            )

    store = make(tmp_path, reuse=Recording())
    job = jobfor(tmp_path, store)
    put(store, job, "x")
    assert isinstance(store.find(job), Hit)
    assert len(list((Path(store.path) / "sidecars" / "reuse@1").iterdir())) == 1


def test_write_sidecar_is_atomic_and_fsynced(tmp_path, monkeypatch):
    store = make(tmp_path)
    fsyncs, syncs = [], []
    real_fsync, real_sync = os.fsync, StorageDir.sync
    monkeypatch.setattr(os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd))[1])
    monkeypatch.setattr(
        StorageDir, "sync", lambda self: (syncs.append(self.path), real_sync(self))[1]
    )
    store.write_sidecar("verify", "a.json", b"one")
    side = Path(store.path) / "sidecars" / "verify@1"
    assert (side / "a.json").read_bytes() == b"one"
    assert not [p for p in side.iterdir() if p.name.endswith(".tmp")]
    assert len(fsyncs) >= 1  # the file
    # the directories made by the call are flushed through their parents, and the sidecar
    # directory after the rename
    assert str(side) in syncs
    assert os.path.join(store.path, "sidecars") in syncs or store.path in syncs
    store.write_sidecar("verify", "a.json", b"two")  # replaces whole
    assert (side / "a.json").read_bytes() == b"two"
    with pytest.raises(ValueError):
        store.write_sidecar("verify", "../x", b"")


def _snapshot(root):
    return sorted(
        (str(p), p.lstat().st_mtime_ns, stat.S_IMODE(p.lstat().st_mode))
        for p in [root, *root.rglob("*")]
    )


def test_readonly_roots_ranked_together_corrupt_skipped(tmp_path, caplog):
    donor = make(tmp_path, "donor")
    donor_job = jobfor(tmp_path, donor)
    for licence in ("a", "b", "c"):
        put(donor, donor_job, licence, out=ord(licence))
    hook = Scripted()
    store = make(
        tmp_path,
        "main",
        reuse=hook,
        readonly=[Root(tmp_path / "donor", trust=TRUST)],
    )
    job = jobfor(tmp_path, store)
    mine = put(store, job, "own", out=1)
    ranked = store.reuse.rank(job.context, store.stored(job.checksum), job.task)
    roots = [c.entry.root for c in ranked]
    assert roots[0] == store.path  # the main root first, both ranked in one list
    assert sorted(set(roots)) == sorted({store.path, str(tmp_path / "donor")})
    assert len(ranked) == 4 and ranked[0].entry == mine.entry
    # the main root's entry is withdrawn, and the first read-only candidate is corrupt
    store.evict(mine.entry, mine.own_digest)
    candidates = store.reuse.rank(job.context, store.stored(job.checksum), job.task)
    assert {c.entry.root for c in candidates} == {str(tmp_path / "donor")}
    victim = donor.entry_path(job.checksum, candidates[0].context)
    os.chmod(victim, 0o755)
    os.chmod(victim / "entry.json", 0o644)
    (victim / "entry.json").write_text("{")
    before = _snapshot(tmp_path / "donor")
    with caplog.at_level("WARNING", logger="pydra"):
        got = store.find(job)
    assert isinstance(got, Hit)
    assert got.result.context["licence"] == candidates[1].context["licence"]
    assert Path(got.result.cache_dir).is_relative_to(tmp_path / "donor")
    assert _snapshot(tmp_path / "donor") == before  # never written, quarantined into
    assert not list((tmp_path / "donor" / "quarantine").iterdir())
    assert any("skipped" in r.message for r in caplog.records)


def test_readonly_root_header_must_match(tmp_path):
    make(tmp_path, "donor", identity=PydraIdentityV1(recorded_only={"host"}))
    with pytest.raises(StoreLayoutMismatch):
        make(tmp_path, "main", readonly=[Root(tmp_path / "donor", trust=TRUST)])


def test_find_through_a_job_run_serves_in_workers(any_worker, tmp_path):
    """The reuse record is written by whichever process looks up."""
    store = make(tmp_path, reuse=Scripted())
    log = tmp_path / "log"
    with Submitter(worker=any_worker, store=store, cache_root=tmp_path / "pc") as sub:
        first = job_in(sub, _Count(a=1, log=str(log), pause=0.0))
        sub.submit(first)
    with Submitter(worker=any_worker, store=store, cache_root=tmp_path / "pc2") as sub:
        second = job_in(sub, _Count(a=1, log=str(log), pause=0.0))
        sub.submit(second)
        got = second.result(settled=True)
    assert log.read_text() == "ran\n" and got.outputs.out == 2


# -- a served hit is recorded so that another process reads it ---------------------------------------


def _fresh(job):
    """The job as another process holds it: nothing cached, nothing committed."""
    other = cloudpickle.loads(cloudpickle.dumps(job))
    other._cached_result = None
    other.committed = None
    return other


def _read_back_worker(job):
    import asyncio

    from pydra.engine.executor import WorkerExecutor

    from pydra.workers.debug import Worker

    executor = WorkerExecutor(Worker(), visibility_timeout=0.5)
    return asyncio.run(executor._read_back(job))


def test_cross_fingerprint_hit_reads_back_from_another_process(tmp_path):
    store = make(tmp_path, reuse="any")
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = job_in(sub, _Inc(a=1))
    put(store, job, "other", out=7)
    ran = job.run()
    assert ran.cross_context and ran.outputs.out == 7
    assert job.committed is not None and job.committed.served and job.committed.stored
    for settled in (True, False):
        got = _fresh(job).result(settled=settled)
        assert got is not None and not got.errored and got.outputs.out == 7
        assert got.committed.served
    got = _read_back_worker(_fresh(job))  # the slurm and sge adapters' read
    assert isinstance(got, Result) and got.outputs.out == 7


def test_readonly_root_hit_reads_back_from_another_process(tmp_path):
    donor = make(tmp_path, "donor")
    store = make(tmp_path, "main", readonly=[Root(tmp_path / "donor", trust=TRUST)])
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = job_in(sub, _Inc(a=1))
    put(donor, job, None, out=9)  # the job's own fingerprint, in the read-only root
    before = _snapshot(tmp_path / "donor")
    ran = job.run()
    assert ran.outputs.out == 9 and Path(ran.cache_dir).is_relative_to(
        tmp_path / "donor"
    )
    assert job.committed.entry.root == str(tmp_path / "donor")
    got = _fresh(job).result(settled=True)
    assert not got.errored and got.outputs.out == 9
    assert _read_back_worker(_fresh(job)).outputs.out == 9
    assert _snapshot(tmp_path / "donor") == before


def test_an_own_address_hit_is_recorded_too(tmp_path):
    store = make(tmp_path)
    sub = Submitter(worker="debug", store=store, cache_root=tmp_path / "pc")
    job = job_in(sub, _Inc(a=1))
    put(store, job, None, out=3)
    assert job.run().outputs.out == 3
    assert (store.work_dir(job) / "_committed.json").is_file()
    assert _fresh(job).result(settled=True).outputs.out == 3


# -- read-only roots that cannot be reached -----------------------------------------------------------


def test_unreachable_readonly_root_is_skipped_and_retried(tmp_path, caplog):
    gone = tmp_path / "donor"
    hook = Scripted()
    store = make(tmp_path, "main", reuse=hook, readonly=[Root(gone, trust=TRUST)])
    job = jobfor(tmp_path, store)
    with caplog.at_level("WARNING", logger="pydra"):
        assert isinstance(store.find(job), Miss)
        again = pickle.loads(
            pickle.dumps(store)
        )  # a worker's unpickle does not fail either
    assert any("skipped" in r.message for r in caplog.records)
    donor = make(tmp_path, "donor")
    put(donor, job, "x", out=4)
    for s in (store, again):
        got = s.find(job)
        assert isinstance(got, Hit) and got.result.outputs.out == 4


def test_readonly_root_header_mismatch_found_late_still_raises(tmp_path):
    gone = tmp_path / "donor"
    store = make(tmp_path, "main", readonly=[Root(gone, trust=TRUST)])
    job = jobfor(tmp_path, store)
    make(tmp_path, "donor", identity=PydraIdentityV1(recorded_only={"host"}))
    with pytest.raises(StoreLayoutMismatch):
        store.find(job)
