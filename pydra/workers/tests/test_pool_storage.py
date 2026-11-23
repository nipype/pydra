"""The worker-pool protocol over shared storage (RFC 0005 sections 5.1-5.3, W4).

The queue's contract (oldest-first claims, idempotent put, one outcome, an expired claim reclaimed,
a double claim committing once, a lost reply on the outcome write, claim cost flat in queue size,
nothing pickled) and the W4 rows of RFC 0005 section 11.
"""

import ast
import errno
import hashlib
import json
import os
import pickle
import stat
import subprocess
import sys
import threading
from pathlib import Path

import attrs
import cloudpickle
import pytest

from pydra.engine.lease import LeaseSettings
from pydra.utils.digest import jcs
from pydra.utils.storagedir import StorageDir, StorePolicy
from pydra.workers.pool import storage as pool_storage
from pydra.workers.pool.storage import (
    Claim,
    JobHeld,
    PoolRoot,
    PoolSettingsMismatch,
    PoolStorageUnsupported,
    Ticket,
    TicketJob,
    queue_label,
)

FAST = LeaseSettings(
    stale_after=2.0, renew_fraction=0.1, attribute_delay=0.0, timeout=5.0
)
OLD = 10**9
POSIX = sys.platform != "win32"


@pytest.fixture(autouse=True)
def release_claims(monkeypatch):
    """Release every claim lease a test leaves held, so no heartbeat thread outlives the test."""
    made = []
    init = Claim.__init__

    def recording(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(Claim, "__init__", recording)
    yield
    for claim in made:
        claim.lease.__exit__(None, None, None)


def _pool(tmp_path: Path, segment_items: int = 3, **kwargs) -> PoolRoot:
    return PoolRoot.open(
        tmp_path / "pool", FAST, probe=False, segment_items=segment_items, **kwargs
    )


def _job(name: str) -> TicketJob:
    digest = hashlib.sha256(name.encode()).hexdigest()
    return TicketJob(
        checksum=f"python-{digest}",
        fingerprint=hashlib.sha256(f"fp-{name}".encode()).hexdigest(),
        payload={
            "codec": "private-pickle@1",
            "path": f"/work/{name}/_job.pklz",
            "sha256": digest,
        },
        runner_attempt=0,
        units=1.0,
    )


def _ticket(name: str, jobs: int = 1, queue: str = "cpu", **kwargs) -> Ticket:
    return Ticket(
        queue=queue,
        node="wf/node",
        jobs=tuple(_job(f"{name}-{i}") for i in range(jobs)),
        **kwargs,
    )


def _age(path: Path) -> None:
    """Stamp a lease directory and everything in it `OLD`, as a holder that stopped long ago leaves it."""
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in dirnames + filenames:
            os.utime(Path(dirpath) / name, (OLD, OLD))
    os.utime(path, (OLD, OLD))


def _claim_dir(pool: PoolRoot, queue: str, digest: str) -> Path:
    return pool.path / "queues" / queue_label(queue) / "claims" / f"{digest}.lease"


def _finish(claim: Claim, kind: str = "done") -> None:
    for job in claim.ticket.jobs:
        attempt = claim.start(job, "w")
        if attempt is not None:
            assert claim.outcome(job, attempt, kind)
    claim.release()


def _drain(pool: PoolRoot, queue: str = "cpu") -> list[str]:
    order = []
    while (claim := pool.claim(queue)) is not None:
        order.append(claim.digest)
        _finish(claim)
    return order


def _files(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir() if not p.name.startswith("."))


# -- the queue's contract ----------------------------------------------------------------------


def test_claims_oldest_first_across_segments(tmp_path):
    """Seven puts take slots in three segments of three; claims come back in put order, a released
    claim is next, and a segment whose slots are all resolved is retired."""
    pool = _pool(tmp_path)
    digests = [pool.put("cpu", _ticket(f"t{i}")) for i in range(7)]
    segments = pool.path / "queues" / queue_label("cpu") / "segments"
    assert _files(segments) == [f"{s:012d}" for s in range(3)]
    assert (segments / f"{1:012d}" / "slot-0").read_bytes() == digests[3].encode()
    assert pool.depth("cpu") == 7 and pool.work("cpu") == 7

    first = pool.claim("cpu")
    second = pool.claim("cpu")
    assert (first.digest, second.digest) == (digests[0], digests[1])
    _finish(first)
    second.release()  # unfinished: ready again, and next
    again = pool.claim("cpu")
    assert again.digest == digests[1]
    _finish(again)
    assert _drain(pool) == digests[2:]
    assert pool.depth("cpu") == 0
    retired = [(segments / f"{s:012d}" / "retired").exists() for s in range(3)]
    assert retired == [True, True, False]


def test_put_is_idempotent_and_names_the_queue_by_label(tmp_path):
    """Two puts of one ticket write one ticket file and one slot; a queue lives under the sha256 of
    its name, which is written once beside it."""
    pool = _pool(tmp_path)
    queue = "gpu-h100#" + "3f" * 32
    label = hashlib.sha256(queue.encode()).hexdigest()
    ticket = _ticket("a", jobs=2, queue=queue)
    first = pool.put(queue, ticket)
    assert pool.put(queue, ticket) == first
    qdir = pool.path / "queues" / label
    assert (qdir / "name").read_text() == queue
    assert _files(qdir / "tickets") == [f"{first}.json"]
    assert pool.depth(queue) == 1 and pool.work(queue) == 2
    with pytest.raises(ValueError, match="queue name"):
        pool.queue("gpu h100")
    with pytest.raises(ValueError, match="queue"):
        pool.put("cpu", ticket)


def test_outcome_first_writer_wins(tmp_path):
    """`expired` then `done` for one attempt: `expired` stands, the `done` writes nothing, and the
    worker skips the job afterwards."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a"))
    claim = pool.claim("cpu")
    (job,) = claim.ticket.jobs
    attempt = claim.start(job, "w")
    assert pool.outcome("cpu", digest, job, attempt, {"outcome": "expired"}) is True
    assert claim.outcome(job, attempt, "done") is False
    assert pool.read_outcome("cpu", digest, job, attempt)["outcome"] == "expired"
    assert claim.start(job, "w") is None
    assert not claim.resolved()  # expired waits for the taker


def test_expired_claim_reclaimed(tmp_path):
    """A dead claimer's stale lease is taken over and its loss recorded; a live claim is never taken."""
    pool = _pool(tmp_path)
    first, second = (pool.put("cpu", _ticket(f"t{i}")) for i in range(2))
    dead = pool.claim("cpu", beat="external")
    assert dead.digest == first and not dead.took_over
    _age(_claim_dir(pool, "cpu", first))

    taker = _pool(tmp_path).claim("cpu")
    assert taker.digest == first and taker.took_over and taker.loss == 0
    assert _files(pool.path / "queues" / queue_label("cpu") / "losses" / first) == ["0"]
    assert not dead.lease.prove()
    other = _pool(tmp_path).claim("cpu")
    assert other.digest == second
    other.release()
    taker.release()


def test_double_claim_commits_once(tmp_path):
    """A claimer stalled past the stale bound is taken over; the taker runs the job under a fresh
    attempt and the stalled claimer's late outcome is refused by its failed proof."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a"))
    slow = pool.claim("cpu", beat="external")
    (job,) = slow.ticket.jobs
    lost_attempt = slow.start(job, "slow")
    _age(_claim_dir(pool, "cpu", digest))

    fast = _pool(tmp_path).claim("cpu")
    assert fast.took_over
    attempt = fast.takeover(job, charged=True)
    assert attempt != lost_attempt and fast.start(job, "fast") == attempt
    assert fast.outcome(job, attempt, "done") is True
    assert slow.outcome(job, lost_attempt, "done") is None
    outcomes = pool.path / "queues" / queue_label("cpu") / "outcome" / digest
    assert len(_files(outcomes)) == 1
    assert pool.read_outcome("cpu", digest, job, attempt)["outcome"] == "done"
    newest = pool.newest(job.checksum, job.fingerprint)[1]
    assert (newest["cause"], newest["prior"], newest["charged"]) == (
        "takeover",
        lost_attempt,
        True,
    )
    fast.release()
    assert pool.follow(job.checksum, job.fingerprint) is None


def test_lost_link_reply_read_back_as_own(tmp_path, monkeypatch):
    """A create the storage made but answered `EEXIST` (a retransmitted request) is read back by its
    uuid as this write's; another writer's record is not."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a", jobs=2))
    claim = pool.claim("cpu")
    one, two = claim.ticket.jobs
    attempts = [claim.start(job, "w") for job in (one, two)]
    name = "rename" if not POSIX else "link"
    real = getattr(os, name)

    def made_then_lost(*args, **kwargs):
        real(*args, **kwargs)
        raise FileExistsError(errno.EEXIST, "lost reply")

    with monkeypatch.context() as patch:
        patch.setattr(os, name, made_then_lost)
        assert claim.outcome(one, attempts[0], "done") is True
    assert pool.outcome("cpu", digest, two, attempts[1], {"outcome": "failed"})
    assert claim.outcome(two, attempts[1], "done") is False


SEGMENT = 10


def test_claim_cost_flat_in_queue_size(tmp_path, monkeypatch):
    """Claiming and settling 100 tickets, through segment retirements, costs the same calls with
    10 and 100 segments in the queue."""

    def cost(root: Path, tickets: int) -> int:
        pool = PoolRoot.open(root, FAST, probe=False, segment_items=SEGMENT)
        with (
            monkeypatch.context() as patch
        ):  # puts are not under test: skip their flushes
            patch.setattr(os, "fsync", lambda fd: None)
            for i in range(tickets):
                pool.put("cpu", _ticket(f"{root.name}-{i}"))
        counted, me = 0, threading.get_ident()
        names = ("open", "stat", "lstat", "listdir", "scandir", "mkdir", "rmdir")
        names += ("rename", "link", "unlink", "fsync", "utime", "replace")
        with monkeypatch.context() as patch:
            for fn in names:
                real = getattr(os, fn)

                def counting(*args, _real=real, **kwargs):
                    nonlocal counted
                    counted += (
                        threading.get_ident() == me
                    )  # not another test's heartbeat
                    return _real(*args, **kwargs)

                patch.setattr(os, fn, counting)
            for _ in range(100):
                claim = pool.claim("cpu", beat="external")
                assert claim is not None
                _finish(claim)
        return counted

    # the scale is in segments, not tickets: small segments keep the puts, which are not
    # under test, from taking most of a loaded runner's test timeout
    small = cost(tmp_path / "small", 10 * SEGMENT)
    large = cost(tmp_path / "large", 100 * SEGMENT)
    assert small == large, (small, large)


def test_ticket_never_contains_pickle(tmp_path, monkeypatch):
    """A ticket is its JSON document as JCS; putting and claiming never call pickle, and a payload
    that would carry anything but a reference or JSON is refused."""

    def refuse(*args, **kwargs):
        raise AssertionError("a ticket was pickled")

    for module in (pickle, cloudpickle):
        for name in ("dump", "dumps", "load", "loads"):
            monkeypatch.setattr(module, name, refuse)
    pool = _pool(tmp_path)
    ticket = _ticket("a")
    digest = pool.put("cpu", ticket)
    data = (
        pool.path / "queues" / queue_label("cpu") / "tickets" / f"{digest}.json"
    ).read_bytes()
    stored = Ticket.from_json(json.loads(data))
    assert data == jcs(stored.to_json()) and hashlib.sha256(data).hexdigest() == digest
    assert pool.claim("cpu").ticket == stored
    job = _job("b")
    for payload in (
        {"codec": "private-pickle@1", "data": "gASV"},
        {
            "codec": "private-pickle@1",
            "path": "/w/_job.pklz",
            "sha256": "0" * 64,
            "json": 1,
        },
        {"codec": "other@1", "sha256": "0" * 64},
        {"codec": "private-pickle@1", "path": "/w/_job.pklz", "sha256": b"\x80\x05"},
    ):
        with pytest.raises((ValueError, TypeError)):
            attrs.evolve(job, payload=payload)


# -- the W4 rows of RFC 0005 section 11 ----------------------------------------------------------


def test_every_index_outcome_settle_write_syncs(tmp_path, monkeypatch):
    """A counting fake `StorageDir`: every exclusive create of the protocol (ticket, generation,
    started mark, slot, job count, resolved mark, outcome, settled mark, loss, hold) is a staged file
    flushed before it is linked, and its directory is flushed after; so is every directory made.
    """
    events: list[tuple] = []
    fds: dict[int, tuple[str, str]] = {}
    real_open_new, real_link, real_rename = (
        StorageDir.open_new,
        StorageDir.link,
        StorageDir.rename,
    )
    real_sync, real_child, real_fsync = StorageDir.sync, StorageDir.child, os.fsync

    def open_new(self, name, mode):
        f = real_open_new(self, name, mode)
        fds[f.fileno()] = (self.path, name)
        return f

    def fsync(fd):
        if fd in fds:
            events.append(("fsync",) + fds[fd])
        return real_fsync(fd)

    def link(self, name, new_name):
        real_link(self, name, new_name)
        events.append(("link", self.path, name, new_name))

    def rename(self, name, target, new_name):
        real_rename(self, name, target, new_name)
        events.append(("link", self.path, name, new_name))

    def sync(self):
        events.append(("sync", self.path))
        return real_sync(self)

    def child(self, name, *, create=False, mode=None):
        existed = os.path.lexists(os.path.join(self.path, name))
        handle = real_child(self, name, create=create, mode=mode)
        if create and not existed:
            events.append(("mkdir", self.path, name))
        return handle

    for name, fn in (
        ("open_new", open_new),
        ("link", link),
        ("rename", rename),
        ("sync", sync),
        ("child", child),
    ):
        monkeypatch.setattr(StorageDir, name, fn)
    monkeypatch.setattr(os, "fsync", fsync)

    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a", jobs=2))
    pool.hold("cpu", 1, "f" * 64)
    claim = pool.claim("cpu", beat="external")
    one, two = claim.ticket.jobs
    attempt = claim.start(one, "w")
    _age(_claim_dir(pool, "cpu", digest))  # the claimer died
    taker = _pool(tmp_path).claim("cpu", epoch=2)
    assert taker.took_over
    attempt = taker.start(one, "w")  # restart: a new generation and its mark
    assert taker.outcome(one, attempt, "done")
    (half,) = taker.move([[two]], "cpu", cause="replaced")
    taker.release()

    links = [e for e in events if e[0] == "link"]
    finals = [e[3] for e in links]
    made = {
        "header": "pool.json" in finals,
        "queue name": "name" in finals,
        "tickets": {f"{digest}.json", f"{half}.json"} <= set(finals),
        "slot": any(f.startswith("slot-") for f in finals),
        "job count": any(f.startswith("jobs-") for f in finals),
        "slotted": digest in finals,
        "resolved": any(f.startswith("res-") for f in finals),
        "hold": any(f.startswith("held-") for f in finals),
        "generations": {"0", "1"} <= set(finals),
        "started": any(f.endswith(".started") for f in finals),
        "settled": any(f.startswith(digest + "-") for f in finals),
        "outcome": any(f.startswith(one.checksum) for f in finals),
        "loss": any(
            Path(e[1]).name == digest and Path(e[1]).parent.name == "losses"
            for e in links
        ),
    }
    assert all(made.values()), made
    for _, where, staged, final in links:
        at = events.index(("link", where, staged, final))
        assert ("fsync", where, staged) in events[:at], final
        assert ("sync", where) in events[at + 1 :], final
    for _, where, name in (e for e in events if e[0] == "mkdir"):
        at = events.index(("mkdir", where, name))
        assert ("sync", where) in events[at + 1 :], name


def test_estale_reads_absent(tmp_path, monkeypatch):
    """`ESTALE` (a directory another NFS client removed) reads as absent, never raised."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a"))
    claim = pool.claim("cpu")
    (job,) = claim.ticket.jobs
    attempt = claim.start(job, "w")
    claim.outcome(job, attempt, "done")

    def stale(*args, **kwargs):
        raise OSError(errno.ESTALE, "stale file handle")

    with monkeypatch.context() as patch:
        patch.setattr(StorageDir, "open_read", stale)
        assert pool.read_outcome("cpu", digest, job, attempt) is None
        assert pool.newest(job.checksum, job.fingerprint) is None
    with monkeypatch.context() as patch:
        patch.setattr(StorageDir, "names", stale)
        assert pool.settled("cpu") == {}
        assert pool.generations(job.checksum, job.fingerprint) == []
    with monkeypatch.context() as patch:
        real_child = StorageDir.child

        def child(self, name, **kwargs):
            if name == digest or name.startswith("python-"):
                stale()
            return real_child(self, name, **kwargs)

        patch.setattr(StorageDir, "child", child)
        assert pool.read_outcome("cpu", digest, job, attempt) is None
        assert pool.follow(job.checksum, job.fingerprint) is None
    assert pool.read_outcome("cpu", digest, job, attempt)["outcome"] == "done"


def test_no_symlink_made(tmp_path, monkeypatch):
    """No step of the protocol makes a symlink: slots, follow generations and every mark are regular
    files."""

    def refuse(*args, **kwargs):
        raise AssertionError("a symlink was made")

    monkeypatch.setattr(os, "symlink", refuse)
    pool = _pool(tmp_path)
    pool.put("cpu", _ticket("a", jobs=3))
    pool.hold("cpu", 1, "e" * 64)
    claim = pool.claim("cpu", beat="external")
    for job in claim.ticket.jobs[:1]:
        claim.outcome(job, claim.start(job, "w"), "failed")
    _age(_claim_dir(pool, "cpu", claim.digest))
    taker = _pool(tmp_path).claim("cpu")
    taker.split()
    taker.release()
    pool.take_epoch()
    for dirpath, dirnames, filenames in os.walk(pool.path):
        for name in dirnames + filenames:
            assert not os.path.islink(os.path.join(dirpath, name)), name


def test_settle_count_race(tmp_path, monkeypatch):
    """Two writers computing one count: the second's create meets the first's mark, recounts and
    writes the mark of the new count, so both outcomes are covered by a written mark."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a", jobs=3))
    claim = pool.claim("cpu")
    one, two, three = claim.ticket.jobs
    attempts = [claim.start(job, "w") for job in (one, two, three)]
    assert claim.outcome(one, attempts[0], "done")
    assert pool.settled("cpu") == {digest: 1}

    real_count = pool_storage._count_outcomes
    stale = {"left": 1}

    def count(d):
        if stale["left"]:  # a listing that has not yet seen the newest outcome
            stale["left"] -= 1
            return real_count(d) - 1
        return real_count(d)

    monkeypatch.setattr(pool_storage, "_count_outcomes", count)
    assert claim.outcome(two, attempts[1], "done")
    marks = _files(pool.path / "queues" / queue_label("cpu") / "settled")
    assert marks == [
        f"{digest}-2"
    ]  # the writer of a mark removes the ticket's previous one
    assert pool.settled("cpu") == {digest: 2}
    # both compute 3: the second finds the first's mark, which covers its own outcome
    pool.outcome("cpu", digest, three, attempts[2], {"outcome": "done"}, settle=False)
    assert pool.settle("cpu", digest) == 3 and pool.settle("cpu", digest) == 3
    assert pool.settled("cpu") == {digest: 3}


ALLOCATE = """
import json, sys
from pydra.engine.lease import LeaseSettings
from pydra.workers.pool.storage import PoolRoot, queue_label
settings = LeaseSettings.from_json(json.loads(sys.argv[2]))
pool = PoolRoot.open(sys.argv[1], settings, probe=False, segment_items=3)
key = (sys.argv[3], sys.argv[4])
found = []
for _ in range(int(sys.argv[5])):
    found.append(pool._allocate(key, ticket="0" * 64, queue=queue_label("cpu"), cause="put")[1]["attempt"])
print(json.dumps(found))
"""


def test_generation_allocation_race(tmp_path, monkeypatch):
    """Two writers of one job's next generation get distinct numbers and distinct attempts, whether
    one's listing is stale or four processes allocate at once."""
    one, two = _pool(tmp_path), _pool(tmp_path)
    job = _job("a")
    key = (job.checksum, job.fingerprint)
    one._allocate(key, ticket="0" * 64, queue=queue_label("cpu"), cause="put")
    real = pool_storage._generation_numbers
    monkeypatch.setattr(pool_storage, "_generation_numbers", lambda d: [])
    g1, r1 = one._allocate(key, ticket="1" * 64, queue=queue_label("cpu"), cause="put")
    g2, r2 = two._allocate(key, ticket="2" * 64, queue=queue_label("cpu"), cause="put")
    monkeypatch.setattr(pool_storage, "_generation_numbers", real)
    assert (g1, g2) == (1, 2) and r1["attempt"] != r2["attempt"]
    assert r1["attempt"] == f"{one.pool_id}.1"

    root = tmp_path / "race"
    PoolRoot.open(root, FAST, probe=False, segment_items=3).close()
    job = _job("shared")
    argv = [sys.executable, "-c", ALLOCATE, str(root), json.dumps(FAST.to_json())]
    argv += [job.checksum, job.fingerprint, "10"]
    procs = [subprocess.Popen(argv, stdout=subprocess.PIPE) for _ in range(4)]
    attempts = []
    for proc in procs:
        out, _ = proc.communicate(timeout=120)
        assert proc.returncode == 0
        attempts += json.loads(out)
    assert len(attempts) == 40 == len(set(attempts))


def test_move_never_unheld(tmp_path, monkeypatch):
    """A move killed after each of its five steps: a coordinator reading the index always finds a
    ticket holding each job, and puts nothing twice. A move killed between its generations and its
    slot is repaired by the next process that takes the old ticket."""
    for step in range(1, 6):
        root = tmp_path / f"step{step}"
        pool = _pool(root)
        ticket = _ticket("a", jobs=2)
        old = pool.put("cpu", ticket)
        claim = pool.claim("cpu", beat="external")

        def die(self, at, _step=step):
            if at == _step:
                raise KeyboardInterrupt(f"killed after step {at}")

        with monkeypatch.context() as patch:
            patch.setattr(Claim, "_checkpoint", die)
            with pytest.raises(KeyboardInterrupt):
                claim.move([list(claim.ticket.jobs)], "gpu", cause="replaced")
        coordinator = _pool(root)
        held = {coordinator.follow(j.checksum, j.fingerprint) for j in ticket.jobs}
        assert None not in held and len(held) == 1, (step, held)
        (holder,) = held
        assert (holder == old) == (step == 1), step
        if step == 2:  # no slot yet: the taker of the old ticket writes it
            assert coordinator.claim("gpu") is None
            _age(_claim_dir(pool, "cpu", old))
            assert coordinator.claim("cpu") is None  # nothing left to run there
            moved = coordinator.claim("gpu")
            assert moved is not None and moved.digest == holder
            moved.release()
        tickets = sorted(p.name for p in root.rglob("tickets/*.json"))
        for queue in ("cpu", "gpu"):
            try:
                again = coordinator.put(queue, attrs.evolve(ticket, queue=queue))
            except JobHeld as err:
                assert err.ticket == holder
            else:
                assert again == holder
        assert sorted(p.name for p in root.rglob("tickets/*.json")) == tickets
        assert {coordinator.follow(j.checksum, j.fingerprint) for j in ticket.jobs} == {
            holder
        }


def test_split_of_split_attempts_distinct(tmp_path):
    """A taker splits a lost four-job ticket into halves and a half into quarters: every placement
    has a fresh attempt, distinct from the loser's, the other half's and each other's.
    """
    pool = _pool(tmp_path)
    old = pool.put("cpu", _ticket("a", jobs=4))
    loser = pool.claim("cpu", beat="external")
    lost = {j.checksum: loser.start(j, "loser") for j in loser.ticket.jobs}
    _age(_claim_dir(pool, "cpu", old))
    taker = _pool(tmp_path).claim("cpu", beat="external")
    assert taker.took_over
    halves = taker.split()
    taker.release()
    assert len(halves) == 2
    half = _pool(tmp_path).claim("cpu", beat="external")
    assert half.digest == halves[0] and len(half.ticket.jobs) == 2
    _age(_claim_dir(pool, "cpu", half.digest))
    retaker = _pool(tmp_path).claim("cpu")
    assert retaker.digest == halves[0] and retaker.took_over
    quarters = retaker.split()
    retaker.release()
    assert len(quarters) == 2
    attempts = []
    for job in _ticket("a", jobs=4).jobs:
        gens = pool.generations(job.checksum, job.fingerprint)
        attempts += [(job.checksum, record["attempt"]) for _, record in gens]
        assert gens[-1][1]["prior"] is not None
        assert lost[job.checksum] == gens[0][1]["attempt"]
    assert len(attempts) == len(set(attempts)) == 4 + 4 + 2
    old_outcomes = pool.outcomes("cpu", old)
    assert {r["outcome"] for r in old_outcomes.values()} == {"withdrawn"}
    assert {r["reason"] for r in old_outcomes.values()} == {"split"}
    assert {r["by"] for r in old_outcomes.values()} == set(halves)


def test_withdraw_needs_claim(tmp_path):
    """`try_take` on a ticket with a live claim fails and nothing is written; once the claim is
    released the coordinator takes it and withdraws."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a"))
    holder = pool.claim("cpu")
    coordinator = _pool(tmp_path)
    assert coordinator.take("cpu", digest) is None
    outcome = pool.path / "queues" / queue_label("cpu") / "outcome"
    assert not (outcome / digest).exists() and coordinator.settled("cpu") == {}
    holder.release()
    claim = coordinator.take("cpu", digest)
    (job,) = claim.ticket.jobs
    with pytest.raises(ValueError, match="reason"):
        claim.withdraw(job, "because")
    claim.withdraw(job, "cancel")
    claim.release()
    record = next(iter(coordinator.outcomes("cpu", digest).values()))
    assert (record["outcome"], record["reason"]) == ("withdrawn", "cancel")
    assert coordinator.follow(job.checksum, job.fingerprint) is None
    assert coordinator.claim("cpu") is None
    assert coordinator.take("cpu", digest) is None  # resolved: nothing to take


def test_started_attempt_restarts_under_next_generation(tmp_path):
    """A start of an attempt that already started writes a `restart` generation first."""
    pool = _pool(tmp_path)
    pool.put("cpu", _ticket("a"))
    claim = pool.claim("cpu")
    (job,) = claim.ticket.jobs
    first = claim.start(job, "w1")
    claim.release()
    again = pool.claim("cpu")
    second = again.start(job, "w2")
    assert first == f"{pool.pool_id}.0" and second == f"{pool.pool_id}.1"
    g, record = pool.newest(job.checksum, job.fingerprint)
    assert (g, record["cause"], record["prior"]) == (1, "restart", first)
    assert pool.started(job.checksum, job.fingerprint, 0)["worker"] == "w1"
    assert pool.started(job.checksum, job.fingerprint, 1)["worker"] == "w2"
    again.release()


def test_collect_keeps_highest_generation(tmp_path):
    """`collect` removes a settled job's generations except the highest, so a rerun continues from
    it under a fresh attempt; a held job's generations are kept."""
    pool = _pool(tmp_path)
    pool.put("cpu", _ticket("a"))
    claim = pool.claim("cpu")
    (job,) = claim.ticket.jobs
    claim.start(job, "w")
    claim.release()
    claim = pool.claim("cpu")
    attempt = claim.start(job, "w")  # restart: generation 1
    pending = pool.put("cpu", _ticket("b"))
    assert pool.collect() == 0  # job a is still held
    claim.outcome(job, attempt, "done")
    claim.release()
    assert pool.collect() == 2  # generation 0 and its started mark
    follow = pool.path / "follow" / f"{job.checksum}-{job.fingerprint}"
    assert _files(follow) == ["1", "1.started"]
    again = pool.put("cpu", _ticket("a"))
    first = pool.claim("cpu")
    assert first.digest == pending
    _finish(first)
    rerun = pool.claim("cpu")
    assert rerun.digest == again
    assert rerun.start(job, "w") == f"{pool.pool_id}.2"


def test_ticket_carries_ladder_and_moves_up_it(tmp_path):
    """A ticket carries its ladder through JSON; a ladder move goes to the next class under a fresh
    generation and `resource_moves` counts the call-phase ladder generations."""
    ticket = _ticket("a", ladder=("gpu-h100-80", "gpu-b200"), batch=4, roots=("/data",))
    assert Ticket.from_json(ticket.to_json()) == ticket
    assert ticket.to_json()["ladder"] == ["gpu-h100-80", "gpu-b200"]
    pool = _pool(tmp_path)
    pool.put("cpu", ticket)
    claim = pool.claim("cpu")
    (job,) = claim.ticket.jobs
    (moved,) = claim.move(
        [[job]], "gpu-h100-80", cause="ladder", phase="call", ladder=("gpu-b200",)
    )
    claim.release()
    new = pool.ticket("gpu-h100-80", moved)
    assert new.ladder == ("gpu-b200",) and new.queue == "gpu-h100-80"
    assert pool.charges(job.checksum, job.fingerprint) == {
        "crashes": 0,
        "infrastructure": 0,
        "resource_moves": 1,
    }
    assert pool.follow(job.checksum, job.fingerprint) == moved


def test_pool_json_records_settings_and_refuses_mismatch(tmp_path):
    """`pool.json` records the pool id, the four lease settings, the codec and the owner; an open with
    any of the four different is refused, one differing only in `max_wait` is not."""
    pool = _pool(tmp_path)
    header = json.loads((pool.path / "pool.json").read_bytes())
    assert header["format"] == "pool@1" and header["pool_id"] == pool.pool_id
    assert header["lease"] == FAST.to_json() and header["codec"] == "private-pickle@1"
    if POSIX:
        assert header["owner_uid"] == os.geteuid()
        assert stat.S_IMODE(os.stat(pool.path).st_mode) == 0o700
    for field, value in (
        ("stale_after", 3.0),
        ("renew_fraction", 0.05),
        ("attribute_delay", 0.1),
        ("timeout", 6.0),
    ):
        with pytest.raises(PoolSettingsMismatch, match=field):
            PoolRoot.open(pool.path, attrs.evolve(FAST, **{field: value}), probe=False)
    PoolRoot.open(pool.path, attrs.evolve(FAST, max_wait=1.0), probe=False).close()


@pytest.mark.skipif(not POSIX, reason="owners and modes are POSIX")
def test_pool_root_writable_by_others_refused(tmp_path):
    """A pool is single-user: a root that group or others can write is refused."""
    pool = _pool(tmp_path)
    pool.close()
    os.chmod(pool.path, 0o770)
    with pytest.raises(OSError, match="private"):
        PoolRoot.open(pool.path, FAST, probe=False)


def test_epoch_fence(tmp_path):
    """Coordinator epochs are taken by `mkdir`, never twice; an older epoch is fenced by a newer one."""
    one, two = _pool(tmp_path), _pool(tmp_path)
    assert one.take_epoch() == 1 and two.take_epoch() == 2
    assert one.fenced(1) and not two.fenced(2) and one.epochs() == [1, 2]
    one.end_epoch(1)
    assert (one.path / "coordinator" / "1" / "ended").exists()
    lease = two.coordinator_lease(2)
    assert lease.lockfile == two.path / "coordinator" / "2" / "lease"


def test_hold_counts_for_its_epoch_or_newer(tmp_path):
    """A hold of a residency on a queue keeps its tickets unclaimed for its epoch, not a newer one."""
    pool = _pool(tmp_path)
    residency = ({"kind": "provider", "name": "hf-model", "spec": "json@1:x"},)
    ticket = _ticket("a", residency=residency)
    pool.put("cpu", ticket)
    pool.hold("cpu", 2, ticket.residency_label)
    assert pool.holds("cpu", 2) == {ticket.residency_label: 2}
    assert pool.claim("cpu", epoch=2) is None
    assert pool.holds("cpu", 3) == {}
    claim = pool.claim("cpu", epoch=3)
    assert claim is not None
    claim.release()


def test_probe_refuses_root_without_oexcl(tmp_path, monkeypatch):
    """`probe` checks the primitives the protocol uses; a root whose `O_EXCL` create does not refuse
    an existing name is refused by `PoolRoot.open` (no `pool.json` is written) and by
    `pydracli pool probe`; a pool created on a good root records the probe in `pool.json`.
    """
    from click.testing import CliRunner

    from pydra.scripts.cli import cli
    from pydra.workers.pool import probe as probing

    good = probing.probe(tmp_path, limit=5.0)
    assert good.missing() == [] and probing.refusal(good, 0.0) is None
    assert good.mtime_delay is not None
    late = attrs.evolve(good, mtime_delay=5.0)
    assert "attribute_delay" in probing.refusal(late, 1.0)
    real_open = os.open

    def ignores_oexcl(path, flags, *args, **kwargs):
        if os.fspath(path).endswith("oexcl"):
            flags &= ~os.O_EXCL
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", ignores_oexcl)
        assert probing.probe(tmp_path, measure_delay=False).missing() == [
            "oexcl_create"
        ]
        with pytest.raises(PoolStorageUnsupported, match="oexcl_create"):
            PoolRoot.open(tmp_path / "pool", FAST, probe=True)
        assert not (tmp_path / "pool" / "pool.json").exists()
        refused = CliRunner().invoke(cli, ["pool", "probe", str(tmp_path)])
        assert refused.exit_code == 1 and "oexcl_create" in refused.output
    found = CliRunner().invoke(
        cli, ["pool", "probe", str(tmp_path), "--attribute-delay", "0"]
    )
    assert found.exit_code == 0, found.output
    assert json.loads(found.output)["refused"] is None
    pool = PoolRoot.open(tmp_path / "pool", FAST, probe=True)
    assert pool.header["probe"]["oexcl_create"] is True
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".pydra")] == []


def test_no_pool_module_takes_host_bound_lock():
    """No pool module takes a host-bound lock (`flock`, `lockf`, `msvcrt.locking`, `filelock`):
    every exclusion in a pool is a `DirectoryLease` judged on the storage clock."""
    banned_modules = {"fcntl", "filelock", "msvcrt", "portalocker", "lockfile"}
    banned_names = {"flock", "lockf", "locking", "SoftFileLock", "FileLock"}
    directory = Path(pool_storage.__file__).parent
    modules = sorted(directory.glob("*.py"))
    assert {"storage.py", "codec.py", "probe.py"} <= {m.name for m in modules}
    for module in modules:
        tree = ast.parse(module.read_text(), str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found = {a.name.split(".")[0] for a in node.names} & banned_modules
            elif isinstance(node, ast.ImportFrom):
                found = {(node.module or "").split(".")[0]} & banned_modules
                found |= {a.name for a in node.names} & banned_names
            elif isinstance(node, ast.Attribute):
                found = {node.attr} & banned_names
            elif isinstance(node, ast.Name):
                found = {node.id} & banned_names
            else:
                continue
            assert (
                not found
            ), f"{module.name}:{node.lineno} takes a host-bound lock: {found}"


# -- follow, settle and outcome listings-------------------------------------------------------------------------


def _segment(pool: PoolRoot, queue: str, seq: int = 0) -> Path:
    return pool.path / "queues" / queue_label(queue) / "segments" / f"{seq:012d}"


def test_stale_follow_listing_never_resolves_a_live_ticket(tmp_path, monkeypatch):
    """A claimant whose listing of a job's follow directory is stale (it shows only generation 0,
    which names the ticket the job moved out of) still runs the job in the ticket that now holds it:
    a slot is resolved only on a final outcome written in its own ticket."""
    a, b = _pool(tmp_path), _pool(tmp_path)
    old = a.put("cpu", _ticket("a"))
    moving = b.take("cpu", old)
    (job,) = moving.ticket.jobs
    (new,) = moving.move([[job]], "gpu", cause="replaced")
    moving.release()
    with monkeypatch.context() as patch:
        patch.setattr(pool_storage, "_generation_numbers", lambda d: [0])
        claim = a.claim("gpu")
        assert claim is not None and claim.digest == new
        attempt = claim.start(job, "w")
        assert attempt == f"{a.pool_id}.1"
        claim.release()
        assert not (_segment(a, "gpu") / "res-0").exists()
    assert a.follow(job.checksum, job.fingerprint) == new
    again = a.claim("gpu")
    assert again is not None and again.digest == new


def test_estale_follow_listing_never_resolves_a_ticket(tmp_path, monkeypatch):
    """An `ESTALE` listing of a job's follow directory reads as unknown, never as moved: the ticket
    is claimed, the job runs, and nothing is resolved until its outcome is written."""
    pool = _pool(tmp_path)
    pool.put("cpu", _ticket("a"))
    real_names, real_scandir = StorageDir.names, StorageDir.scandir

    def stale(real):
        def call(self):
            if Path(self.path).parent.name == "follow":  # a job's generations
                raise OSError(errno.ESTALE, "stale file handle")
            return real(self)

        return call

    monkeypatch.setattr(StorageDir, "names", stale(real_names))
    monkeypatch.setattr(StorageDir, "scandir", stale(real_scandir))
    claim = pool.claim("cpu")
    assert claim is not None
    (job,) = claim.ticket.jobs
    attempt = claim.start(job, "w")
    assert attempt == f"{pool.pool_id}.0"
    claim.release()
    assert not (_segment(pool, "cpu") / "res-0").exists()
    claim = pool.claim("cpu")
    attempt = claim.start(job, "w")  # restart: generation 1, found by name
    assert claim.outcome(job, attempt, "done")
    claim.release()
    assert (_segment(pool, "cpu") / "res-0").exists()


def test_claimant_completes_a_half_done_move(tmp_path, monkeypatch):
    """A move that died after its slot (step 3) and before its withdrawals (step 4): the taker of the
    old ticket writes the `withdrawn` itself, naming the new ticket, and only then resolves it.
    """
    pool = _pool(tmp_path)
    old = pool.put("cpu", _ticket("a"))
    claim = pool.claim("cpu", beat="external")
    (job,) = claim.ticket.jobs

    def die(self, at):
        if at == 3:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(Claim, "_checkpoint", die)
        with pytest.raises(KeyboardInterrupt):
            claim.move([[job]], "gpu", cause="replaced")
    new = pool.follow(job.checksum, job.fingerprint)
    assert pool.outcomes("cpu", old) == {}
    _age(_claim_dir(pool, "cpu", old))
    assert _pool(tmp_path).claim("cpu") is None
    (record,) = pool.outcomes("cpu", old).values()
    assert (record["outcome"], record["reason"], record["by"]) == (
        "withdrawn",
        "replaced",
        new,
    )
    assert (_segment(pool, "cpu") / "res-0").exists()


def test_settle_repaired_after_crash(tmp_path):
    """A writer killed between its outcome and its settled mark: the next claimant settles the
    ticket, so the follower's count covers the outcome."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a", jobs=2))
    claim = pool.claim("cpu", beat="external")
    one, two = claim.ticket.jobs
    attempt = claim.start(one, "w")
    pool.outcome("cpu", digest, one, attempt, {"outcome": "done"}, settle=False)
    assert pool.settled("cpu") == {}
    _age(_claim_dir(pool, "cpu", digest))
    taker = _pool(tmp_path).claim("cpu")
    assert taker is not None and pool.settled("cpu") == {digest: 1}
    taker.release()
    # and a release settles too
    claim = pool.claim("cpu")
    attempt = claim.start(two, "w")
    pool.outcome("cpu", digest, two, attempt, {"outcome": "done"}, settle=False)
    claim.release()
    assert pool.settled("cpu") == {digest: 2}


def test_withdrawn_needs_the_claim_lease(tmp_path):
    """`PoolRoot.outcome` refuses a `withdrawn` written without the ticket's claim lease."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a"))
    (job,) = pool.ticket("cpu", digest).jobs
    attempt = f"{pool.pool_id}.0"
    with pytest.raises(ValueError, match="claim"):
        pool.outcome(
            "cpu", digest, job, attempt, {"outcome": "withdrawn", "reason": "cancel"}
        )
    assert pool.outcomes("cpu", digest) == {}
    claim = pool.take("cpu", digest)
    record = {"outcome": "withdrawn", "reason": "cancel"}
    assert pool.outcome("cpu", digest, job, attempt, record, lease=claim.lease)


def test_outcomes_wait_for_the_expected_count(tmp_path, monkeypatch):
    """`outcomes(expect=n)` lists again until n records show (a listing that lags a settled mark),
    and reports a listing still behind at its bound instead of returning fewer."""
    pool = _pool(tmp_path)
    digest = pool.put("cpu", _ticket("a", jobs=2))
    claim = pool.claim("cpu")
    for job in claim.ticket.jobs:
        claim.outcome(job, claim.start(job, "w"), "done")
    assert pool.settled("cpu") == {digest: 2}
    real = StorageDir.names
    lag = {"left": 2}

    def lagging(self):
        found = real(self)
        if Path(self.path).name == digest and lag["left"]:
            lag["left"] -= 1
            return sorted(found)[:-1]
        return found

    monkeypatch.setattr(StorageDir, "names", lagging)
    assert len(pool.outcomes("cpu", digest, expect=2)) == 2 and lag["left"] == 0
    lag["left"] = 10**6
    with pytest.raises(pool_storage.ListingBehind) as behind:
        pool.outcomes("cpu", digest, expect=2, within=0.2)
    assert len(behind.value.found) == 1 and behind.value.expected == 2


def test_claim_stats_flat_on_a_large_live_segment(tmp_path, monkeypatch):
    """A claim on a segment of many taken, resolved slots stats no listed name: a cold claim costs
    the same stats at 30 and 300 slots; and a queue whose residency is held is not walked.
    """

    def stats(root: Path, tickets: int) -> int:
        pool = PoolRoot.open(root, FAST, probe=False, segment_items=1000)
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", lambda fd: None)
            for i in range(tickets):
                pool.put("cpu", _ticket(f"{root.name}-{i}"))
            for _ in range(tickets - 1):
                _finish(pool.claim("cpu", beat="external"))
        cold = PoolRoot.open(root, FAST, probe=False)
        counted, me = 0, threading.get_ident()
        with monkeypatch.context() as patch:
            for fn in ("stat", "lstat"):
                real = getattr(os, fn)

                def counting(*args, _real=real, **kwargs):
                    nonlocal counted
                    counted += threading.get_ident() == me
                    return _real(*args, **kwargs)

                patch.setattr(os, fn, counting)
            claim = cold.claim("cpu", beat="external")
        assert claim is not None
        return counted

    assert stats(tmp_path / "small", 30) == stats(tmp_path / "large", 300)

    pool = _pool(tmp_path / "held", segment_items=1000)
    ticket = _ticket("h", residency=({"kind": "provider", "name": "m"},))
    for i in range(5):
        pool.put("cpu", _ticket(f"h{i}", residency=ticket.residency))
    assert pool.claim("cpu") is not None  # learns the queue's residency
    pool.hold("cpu", 1, ticket.residency_label)
    walked = []
    real_target = PoolRoot._target
    monkeypatch.setattr(
        PoolRoot, "_target", lambda self, *a: walked.append(a) or real_target(self, *a)
    )
    assert pool.claim("cpu", epoch=1) is None and walked == []


def test_half_done_move_repaired_after_collect(tmp_path, monkeypatch):
    """A move dies after step 3, the job finishes in its new ticket, and ``collect`` removes the
    generation that named the old ticket; the next claimant of the old ticket still withdraws the job
    there under the ticket's own attempt, so the old ticket resolves and is never claimable
    again."""
    pool = _pool(tmp_path)
    old = pool.put("cpu", _ticket("a"))
    claim = pool.claim("cpu", beat="external")
    (job,) = claim.ticket.jobs

    def die(self, at):
        if at == 3:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(Claim, "_checkpoint", die)
        with pytest.raises(KeyboardInterrupt):
            claim.move([[job]], "gpu", cause="replaced")
    _finish(_pool(tmp_path).claim("gpu"))
    assert _pool(tmp_path).collect(payload_age=0) >= 1
    _age(_claim_dir(pool, "cpu", old))
    assert _pool(tmp_path).claim("cpu") is None  # repaired and resolved, not handed out
    (withdrawn,) = pool.outcomes("cpu", old).values()
    assert (withdrawn["outcome"], withdrawn["reason"]) == ("withdrawn", "replaced")
    assert withdrawn["attempt"] == job.attempt
    assert _pool(tmp_path).claim("cpu") is None


def test_settled_listing_bounded_and_collected(tmp_path):
    """A queue's ``settled/`` holds one mark per ticket, the highest, however many outcomes it had, so
    the listing the follower reads each poll is bounded by the tickets within a run;
    ``collect_settled`` removes resolved tickets with their marks, outcomes and slotted entries, so
    across runs it holds only live tickets."""
    pool = _pool(tmp_path)
    digests = [pool.put("cpu", _ticket(f"t{i}", jobs=3)) for i in range(4)]
    while (claim := pool.claim("cpu")) is not None:
        _finish(claim)
    q = pool.path / "queues" / queue_label("cpu")
    # one mark per ticket, the highest, however many outcomes it had
    assert sorted(p.name for p in (q / "settled").iterdir()) == sorted(
        f"{d}-3" for d in digests
    )
    assert pool.settled("cpu") == {d: 3 for d in digests}
    live = pool.put("cpu", _ticket("live", jobs=2))
    assert pool.collect_settled(age=0) == 4
    assert list((q / "settled").iterdir()) == []
    assert [p.name for p in (q / "tickets").iterdir()] == [f"{live}.json"]
    assert list((q / "outcome").iterdir()) == []
    assert [p.name for p in (q / "slotted").iterdir()] == [live]
    claim = _pool(tmp_path).claim("cpu")
    assert (
        claim is not None and claim.digest == live
    )  # the index still serves live tickets
    claim.release()
    assert pool.collect_settled(age=3600) == 0


def test_a_claim_lease_refused_for_good_on_windows_raises(tmp_path, monkeypatch):
    """Each claim makes a new claim lease. A refusal of its mkdir that outlasts the release
    patience ends in LeaseRefused, rather than passing the ticket over on every claim.
    """
    from pydra.engine import lease as lease_module
    from pydra.engine.lease import LeaseRefused, release_patience

    pool = _pool(tmp_path)
    pool.put("cpu", _ticket("a"))
    now = [0.0]
    pool.monotonic = lambda: now[0]
    real_mkdir = os.mkdir

    def mkdir(path, *args, **kwargs):
        path_ = Path(path)
        if path_.suffix == ".lease" and path_.parent.name == "claims":
            raise PermissionError(errno.EACCES, "Access is denied", str(path), 5)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(lease_module, "IS_WINDOWS", True)
    assert pool.claim("cpu") is None
    now[0] = max(release_patience(FAST), FAST.renew_interval)  # judged again
    with pytest.raises(LeaseRefused):
        pool.claim("cpu")


def test_visibility_delay_counts_only_looks_that_missed():
    from pydra.workers.pool.probe import visibility_delay

    # the watcher was descheduled: its first look after the change already saw it
    assert visibility_delay(made=100.0, reply="None", limit=61.0) == 0.0
    assert visibility_delay(made=100.0, reply="99.5", limit=61.0) == 0.0
    # it looked 2 s after the change and still missed it: the storage was late
    assert visibility_delay(made=100.0, reply="102.0", limit=61.0) == 2.0
    assert visibility_delay(made=100.0, reply="none", limit=61.0) == 61.0


@pytest.mark.parametrize("refusals, read", [(2, True), (10, False)])
def test_a_record_being_replaced_is_read_again_on_windows(
    tmp_path, monkeypatch, refusals, read
):
    """A record another process is linking into place refuses an open on Windows for a moment
    (a sharing violation): the read is tried again, within `SHARING_RETRY`."""
    import attrs
    from pydra.engine.portable import store as portable_store

    monkeypatch.setattr(pool_storage, "IS_WINDOWS", True)
    monkeypatch.setattr(
        portable_store,
        "SHARING_RETRY",
        attrs.evolve(portable_store.SHARING_RETRY, first=0, cap=0),
    )
    d = StorageDir.open_root(tmp_path, StorePolicy(), trust="trusted")
    with d.open_new("rec", 0o600) as f:
        f.write(b"data")
    real, calls = StorageDir.open_read, []

    def refusing(self, name):
        calls.append(name)
        if len(calls) <= refusals:
            raise PermissionError(13, "in use", name)
        return real(self, name)

    monkeypatch.setattr(StorageDir, "open_read", refusing)
    if read:
        assert pool_storage._read(d, "rec") == b"data"
    else:
        with pytest.raises(PermissionError):
            pool_storage._read(d, "rec")
