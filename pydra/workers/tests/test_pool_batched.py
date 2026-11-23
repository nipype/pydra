"""Batched calls (RFC 0005 sections 4.4, 6, 8; W10).

A Python task declared ``batched=`` runs several jobs of one ticket in one call, at most the
ticket's ``batch`` (`Placement.batch`), ordered by ``units``. Each job keeps its own checksum, stored
result, outcome and audit; an exception element fails only its job; a crashing item is isolated by
bisection, in-process (a raised `WorkerLost`) or by takeovers (an item that kills its worker); a half
that completed keeps its results.

Nothing is timed: workers in this process are driven to idle, and waits on other processes poll for a
state with a bound that only fails a hung test.
"""

import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import attrs
import time
from types import SimpleNamespace

import pytest

from pydra.engine.job import Job
from pydra.engine.lease import LeaseSettings
from pydra.engine.submitter import Submitter

POSIX = sys.platform != "win32"
LEASE = LeaseSettings(
    stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
)
BOUND = 120.0  # seconds any wait on another process may take before the test fails

TASKS = """
import json, os, signal, time
from pathlib import Path
from uuid import uuid4
from pydra.compose import python, workflow
from pydra.engine.resident import Provided
from pydra.workers.pool.faults import WorkerLost

CALLS = []  # each batched call's x values, in this process
LOADS = []


def _mark(where, xs):
    # one file per call, so a call a killed worker made is still counted
    if where:
        Path(where, uuid4().hex + ".json").write_text(json.dumps(xs))


def _one(x, mode):
    if mode == "error":
        raise ValueError(f"item {x} is malformed")
    return {"x": x, "square": x * x, "root": x ** 0.5}


def square_many(x, where, mode):
    xs = list(x)
    CALLS.append(xs)
    _mark(where[0] if where else "", xs)
    if "kill" in mode:
        os.kill(os.getpid(), signal.SIGKILL)
    if "hang" in mode:
        time.sleep(600)
    if "lost" in mode:
        raise WorkerLost("the provider's process ended mid-request")
    out = []
    for v, m in zip(xs, mode):
        try:
            if m == "batch-error":
                raise ValueError(f"item {v} failed in its batch")
            out.append(_one(v, m))
        except Exception as err:
            out.append(err)
    return out


def size(x):
    return float(x)


@python.define(batched=square_many, units=size, batch=4)
def Square(x: int, where: str = "", mode: str = "") -> dict:
    _mark(where, [x])
    if mode == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "hang":
        time.sleep(600)
    if mode == "batch-error":
        mode = ""
    if mode == "lost":
        raise WorkerLost("the provider's process ended mid-request")
    return _one(x, mode)


@python.define
def PlainSquare(x: int, where: str = "", mode: str = "") -> dict:
    return _one(x, mode)


def short_many(x, where, mode):
    return [{"x": v} for v in list(x)[:-1]]


@python.define(batched=short_many)
def Short(x: int, where: str = "", mode: str = "") -> dict:
    return {"x": x}


def whole_many(x, where, mode):
    raise RuntimeError("the batch could not be assembled")


@python.define(batched=whole_many)
def Whole(x: int, where: str = "", mode: str = "") -> dict:
    return {"x": x}


class Model:
    # an in-process provider whose state scales its input
    name = "model"

    def load(self, spec, device):
        LOADS.append(spec)
        return {"scale": int(spec)}

    def close(self, obj):
        pass

    def footprint(self, spec):
        return None

    def measure(self, obj):
        return None

    def version(self):
        return "1"


def scale_many(model, x):
    CALLS.append((model["scale"], list(x)))
    return [model["scale"] * v for v in x]


@python.define(batched=scale_many)
def Scale(model: Provided[str, "model"], x: int) -> int:
    return model["scale"] * x


@workflow.define
def Squares(xs: list) -> list:
    return workflow.add(Square().split(x=xs), name="S").out
"""


# -- fixtures and helpers ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def tasks(tmp_path_factory):
    """The task module, importable here and in workers started from here."""
    where = tmp_path_factory.mktemp("pool_batched_tasks")
    (where / "_pydra_pool_batched_tasks.py").write_text(TASKS)
    sys.path.insert(0, str(where))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(where), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    import _pydra_pool_batched_tasks

    return _pydra_pool_batched_tasks


@pytest.fixture(autouse=True)
def clean(monkeypatch, tasks):
    """Release every claim a test leaves held and clear the call records, pass or fail."""
    from pydra.workers.pool.storage import Claim

    made = []
    init = Claim.__init__

    def recording(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(Claim, "__init__", recording)
    tasks.CALLS.clear()
    tasks.LOADS.clear()
    yield
    tasks.CALLS.clear()
    tasks.LOADS.clear()
    for claim in made:
        claim.lease.__exit__(None, None, None)


@pytest.fixture
def processes():
    """Processes a test starts, each in its own session; killed with their group at the end."""
    started: list[subprocess.Popen] = []
    yield started
    for proc in started:
        if proc.poll() is None:
            try:
                if POSIX:
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(BOUND)
        except subprocess.TimeoutExpired:
            pass


def _settings(**kwargs):
    from pydra.workers.pool.settings import PoolSettings

    fields = dict(lease=LEASE, poll_s=0.05, idle_polls=2, timeout_grace=60.0)
    fields.update(kwargs)
    return PoolSettings(**fields)


def _class(name="cpu", **kwargs):
    from pydra.workers.pool.classes import WorkerClass

    fields = dict(name=name, device="cpu", quantum_jobs=16, quantum_s=600.0)
    fields.update(kwargs)
    return WorkerClass(**fields)


class Pool:
    """A pool, a store and the jobs put on the pool, for one test."""

    def __init__(self, root: Path):
        from pydra.workers.pool.storage import PoolRoot

        self.pool = PoolRoot.open(root / "pool", LEASE, probe=False)
        self.submitter = Submitter(cache_root=root / "store", worker="debug")
        self.store = self.submitter.store
        self.tickets: list[tuple[str, str]] = []

    @property
    def path(self) -> Path:
        return self.pool.path

    def job(self, task, name: str) -> Job:
        return Job(task=task, submitter=self.submitter, name=name)

    def entry(self, job: Job):
        from pydra.workers.pool.codec import PrivatePickle, write_payload
        from pydra.workers.pool.placement import job_units
        from pydra.workers.pool.storage import TicketJob

        fingerprint = self.store.identity.fingerprint(job.context)
        g = self.pool._next_generation((job.checksum, fingerprint))
        job.attempt = f"{self.pool.pool_id}.{g}"
        return TicketJob(
            checksum=job.checksum,
            fingerprint=fingerprint,
            payload=write_payload(PrivatePickle(), job, self.path),
            attempt=job.attempt,
            runner_attempt=0,
            units=job_units(job),
        )

    def put(self, jobs, queue="cpu", **kwargs) -> str:
        import attrs

        from pydra.engine import resident
        from pydra.workers.pool.storage import Ticket

        residency = tuple(
            attrs.asdict(resident.spec_key(jobs[0].task, f, p))
            for f, p in resident.provided_fields(jobs[0].task)
        )
        ticket = Ticket(
            queue=queue,
            node="wf/node",
            jobs=[self.entry(j) for j in jobs],
            residency=residency,
            **kwargs,
        )
        digest = self.pool.put(queue, ticket)
        self.tickets.append((queue, digest))
        return digest

    def key(self, job: Job) -> tuple[str, str]:
        return job.checksum, self.store.identity.fingerprint(job.context)

    def newest(self, job: Job) -> dict | None:
        """The job's outcome under its newest generation."""
        _, record = self.pool.newest(*self.key(job))
        queue = self.pool._by_label(record["queue"]).name
        return self.pool.read_outcome(
            queue, record["ticket"], self.key(job), record["attempt"]
        )

    def generations(self, job: Job) -> list[dict]:
        return [r for _, r in self.pool.generations(*self.key(job))]

    def result(self, job: Job):
        return getattr(self.store.find(job), "result", None)

    def outputs_bytes(self, job: Job) -> tuple[str, bytes]:
        """The job's one stored entry: its slot (the fingerprint it was published under) and the
        canonical bytes of its outputs; the writer's token and the entry digest over it differ
        between any two writers."""
        from pydra.engine.portable.layout import cdir_for
        from pydra.utils.digest import jcs

        found = sorted(
            Path(self.store.path).rglob(f"{cdir_for(job.checksum)}/*/entry.json")
        )
        assert len(found) == 1, found
        entry = json.loads(found[0].read_text())
        return entry["fingerprint"], jcs(entry["outputs"])


@pytest.fixture
def pool(tmp_path):
    p = Pool(tmp_path)
    yield p
    p.pool.close()


def _run(pool: Pool, cls=None, **kwargs):
    from pydra.workers.pool.worker import Worker

    kwargs.setdefault("settings", _settings())
    kwargs.setdefault("device_memory", lambda: None)
    worker = Worker(pool.path, cls or _class(), **kwargs)
    return worker, worker.run()


def _samples(pool: Pool) -> list[dict]:
    return [
        json.loads(line)
        for path in (pool.path / "workers").glob("*/samples.jsonl")
        for line in path.read_text().splitlines()
    ]


def _age(path: Path) -> None:
    """Stamp a lease directory and everything in it as a holder that stopped long ago leaves it."""
    old = 10**9
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in dirnames + filenames:
            os.utime(Path(dirpath) / name, (old, old))
    os.utime(path, (old, old))


def _hand_worker(pool: Pool, processes, tmp_path, cls=None, settings=None):
    from pydra.workers.pool.worker import class_record

    cls = cls or _class()
    pool.pool.put_class(cls.name, class_record(cls))
    path = tmp_path / f"settings-{len(processes)}.json"
    path.write_text(json.dumps((settings or _settings()).to_json()))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pydra.workers.pool.worker",
            "--root",
            str(pool.path),
            "--class",
            cls.name,
            "--settings",
            str(path),
        ],
        start_new_session=POSIX,
    )
    processes.append(proc)
    return proc


# -- identity ----------------------------------------------------------------------------------


def test_units_do_not_change_the_checksum_and_batched_code_does(tasks):
    """``units=`` is an execution fact (RFC 0005 section 8): a job's units never enter the checksum.
    The ``batched=`` callable's code joins the task's key (RFC 0005 W10)."""
    from pydra.compose import python

    def Square(x: int, where: str = "", mode: str = "") -> dict:  # noqa: N802
        return {"x": x}

    plain = python.define(Square)
    batched = python.define(Square, batched=tasks.square_many)
    sized = python.define(Square, batched=tasks.square_many, units=tasks.size)
    assert batched._batched is tasks.square_many and plain._batched is None
    assert sized(x=3)._checksum == batched(x=3)._checksum != plain(x=3)._checksum


def test_batched_must_be_callable():
    from pydra.compose import python

    def f(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="batched must be callable"):
        python.define(f, batched=3)
    with pytest.raises(TypeError, match="units must be callable"):
        python.define(f, units="seconds")


def test_units_read_the_inputs_the_function_names(tasks, pool):
    from pydra.workers.pool.placement import fit_request, job_units

    job = pool.job(tasks.Square(x=7), "s")
    assert job_units(job) == 7.0 and fit_request(job).units == 7.0
    assert job_units(pool.job(tasks.PlainSquare(x=7), "p")) is None


# -- results -----------------------------------------------------------------------------------


def test_batched_results_equal_unbatched(tasks, tmp_path):
    """The same jobs run one per call (``batch`` 1: the function) and four per call (the batched
    function) give equal checksums, equal results and byte-equal stored outputs, each job under its
    own checksum."""
    xs = [3, 1, 4, 10, 9, 2, 6, 5]
    stored = {}
    for batch in (1, 4):
        p = Pool(tmp_path / f"b{batch}")
        try:
            jobs = [p.job(tasks.Square(x=x), f"s{i}") for i, x in enumerate(xs)]
            p.put(jobs, batch=batch)
            tasks.CALLS.clear()
            _, code = _run(p)
            assert code == 0
            assert all(p.newest(j)["outcome"] == "done" for j in jobs)
            if batch == 1:
                assert tasks.CALLS == []  # the function, never the batched one
            else:
                assert sorted(len(c) for c in tasks.CALLS) == [4, 4]
                # ordered by units: the four smallest share a call
                assert sorted(tasks.CALLS)[0] == [1, 2, 3, 4]
            stored[batch] = {
                j.checksum: (p.result(j).outputs.out, p.outputs_bytes(j)) for j in jobs
            }
        finally:
            p.pool.close()
    assert stored[1] == stored[4]
    assert len(stored[4]) == len(xs)


def test_batch_is_recorded_never_keyed(tasks, pool):
    """Each job's pool facts and sample record the call's size and the job's place in it; its slot
    is the one its lookup finds."""
    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (5, 2, 8)]
    pool.put(jobs, batch=3)
    _, code = _run(pool)
    assert code == 0
    facts = {}
    for job in jobs:
        result = pool.result(job)
        assert result is not None and result.outputs.out["square"] == job.task.x**2
        outcome = pool.newest(job)
        assert outcome["sample"]["batch_size"] == 3
        assert (
            outcome["sample"]["units"] == 8.0 * 3
        )  # the longest job padded over the call
        facts[job.task.x] = outcome
    samples = _samples(pool)
    assert len(samples) == 3 and {s["batch_size"] for s in samples} == {3}
    records = pool.path.glob("workers/*/calls/*.json")
    [call] = [json.loads(r.read_text()) for r in records]
    assert len(call["jobs"]) == 3


def test_pool_facts_carry_the_batch(tasks, pool, monkeypatch):
    from pydra.workers.pool.worker import Worker

    seen = {}
    real = Worker.pool_facts

    def spy(self, claim, entry, job, attempt, group, batch=(1, 0), batched=None):
        facts = real(self, claim, entry, job, attempt, group, batch, batched)
        seen[job.task.x] = facts["batch"]
        return facts

    monkeypatch.setattr(Worker, "pool_facts", spy)
    pool.put([pool.job(tasks.Square(x=x), f"s{x}") for x in (5, 2, 8)], batch=4)
    pool.put([pool.job(tasks.Square(x=9), "alone")])  # one job per call: the function
    _, code = _run(pool)
    assert code == 0
    implementation = {
        "batched": True,
        "function": "_pydra_pool_batched_tasks:square_many",
    }
    assert seen == {
        2: {"size": 3, "index": 0, **implementation},
        5: {"size": 3, "index": 1, **implementation},
        8: {"size": 3, "index": 2, **implementation},
        9: {"size": 1, "index": 0},
    }


def test_a_call_takes_jobs_of_one_residency(tasks, pool):
    """Jobs of two provided specs in one ticket never share a call: each call binds one state."""
    pool.submitter.providers = (tasks.Model(),)
    jobs = [
        pool.job(tasks.Scale(model=str(m), x=x), f"m{m}x{x}")
        for m in (2, 3)
        for x in (1, 2, 3)
    ]
    pool.put(jobs, batch=8)
    _, code = _run(pool)
    assert code == 0
    assert sorted(tasks.CALLS) == [(2, [1, 2, 3]), (3, [1, 2, 3])]
    for job in jobs:
        assert pool.result(job).outputs.out == int(job.task.model) * job.task.x


def test_a_ticket_of_batch_one_runs_the_function(tasks, pool):
    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2)]
    pool.put(jobs)  # no batch
    _, code = _run(pool)
    assert code == 0 and tasks.CALLS == []
    assert {s["batch_size"] for s in _samples(pool)} == {1}


# -- faults ------------------------------------------------------------------------------------


def test_content_fault_in_batched_call_fails_one_job(tasks, pool):
    """An exception element is its job's content fault: that job fails, with an errored result, and
    the others of the call are done; nothing is bisected."""
    jobs = [
        pool.job(tasks.Square(x=x, mode="error" if x == 2 else ""), f"s{x}")
        for x in (1, 2, 3, 4)
    ]
    pool.put(jobs, batch=4)
    _, code = _run(pool)
    assert code == 0
    assert [len(c) for c in tasks.CALLS] == [4]
    for job in jobs:
        outcome = pool.newest(job)
        if job.task.x == 2:
            assert outcome["outcome"] == "failed"
            assert (outcome["fault"]["kind"], outcome["fault"]["type"]) == (
                "content",
                "ValueError",
            )
            assert pool.result(job) is None
        else:
            assert outcome["outcome"] == "done"
            assert pool.result(job).outputs.out["square"] == job.task.x**2


def test_a_whole_call_content_raise_fails_each_job(tasks, pool):
    """A batched call that raises as a whole with a content fault fails each of its jobs, unsplit."""
    jobs = [pool.job(tasks.Whole(x=x), f"w{x}") for x in (1, 2, 3)]
    pool.put(jobs, batch=4)
    _, code = _run(pool)
    assert code == 0
    for job in jobs:
        outcome = pool.newest(job)
        assert outcome["outcome"] == "failed"
        assert outcome["fault"]["type"] == "RuntimeError"


def test_a_return_of_the_wrong_length_fails_each_job(tasks, pool):
    jobs = [pool.job(tasks.Short(x=x), f"w{x}") for x in (1, 2, 3)]
    pool.put(jobs, batch=4)
    _, code = _run(pool)
    assert code == 0
    for job in jobs:
        outcome = pool.newest(job)
        assert (outcome["outcome"], outcome["fault"]["type"]) == (
            "failed",
            "ValueError",
        )


def test_completed_half_keeps_results(tasks, pool):
    """A call that crashes as a whole (`WorkerLost`) is bisected on a reloaded residency: each half
    that completes keeps its results, committed to the store and settled ``done``, and the crashing
    job alone is charged ``JobCrashedWorker``."""
    xs = list(range(1, 9))
    jobs = [
        pool.job(tasks.Square(x=x, mode="lost" if x == 6 else ""), f"s{x}") for x in xs
    ]
    pool.put(jobs, batch=8)
    _, code = _run(pool)
    assert code == 0
    for job in jobs:
        outcome = pool.newest(job)
        if job.task.x == 6:
            assert (outcome["outcome"], outcome["fault"]["type"]) == (
                "failed",
                "JobCrashedWorker",
            )
        else:
            assert outcome["outcome"] == "done"
            assert pool.result(job).outputs.out["square"] == job.task.x**2
    # calls holding the crashing job halve until it is alone: [1..8], [5..8], [5, 6], [6], [6]
    holding = [c for c in tasks.CALLS if 6 in c]
    assert [len(c) for c in holding] == [8, 4, 2, 1, 1]
    assert [1, 2, 3, 4] in tasks.CALLS and [7, 8] in tasks.CALLS and [5] in tasks.CALLS


@pytest.mark.skipif(not POSIX, reason="the task kills its worker by signal")
def test_crashing_item_isolated_in_log2_calls(tasks, pool, processes, tmp_path):
    """A batched item that kills its worker process: each taker splits the lost call's jobs in
    halves (fresh generations), so the crashing job is alone after at most ``ceil(log2 n) + 1``
    calls that hold it; the other jobs complete, keep their results and are never charged; the
    crashing job, lost alone ``crashes_charged`` times, fails ``JobCrashedWorker``."""
    marks = tmp_path / "marks"
    marks.mkdir()
    n = 8
    jobs = [
        pool.job(
            tasks.Square(x=x, where=str(marks), mode="kill" if x == 3 else ""), f"s{x}"
        )
        for x in range(1, n + 1)
    ]
    crashing = jobs[2]
    digest_dirs = pool.path / "queues"
    pool.put(jobs, batch=n)
    for _ in range(4 * n):  # a bound on worker starts that only fails a runaway test
        proc = _hand_worker(pool, processes, tmp_path)
        code = proc.wait(BOUND)
        if code == 0 and pool.newest(crashing) is not None:
            break
        assert code == -signal.SIGKILL, code
        for d in digest_dirs.glob("*/claims/*.lease"):
            _age(d)
    else:
        raise AssertionError("the crashing job was never settled")
    calls = [json.loads(p.read_text()) for p in marks.iterdir()]
    holding = [c for c in calls if 3 in c]
    sizes = sorted((len(c) for c in holding), reverse=True)
    first_alone = sizes.index(1) + 1
    assert first_alone <= math.ceil(math.log2(n)) + 1, sizes
    assert sizes == [8, 4, 2, 1, 1]  # alone, then lost alone again: charged twice
    for job in jobs:
        outcome = pool.newest(job)
        if job is crashing:
            assert (outcome["outcome"], outcome["fault"]["type"]) == (
                "failed",
                "JobCrashedWorker",
            )
            continue
        assert outcome["outcome"] in ("done",), outcome
        assert pool.result(job).outputs.out["square"] == job.task.x**2
        assert not any(g["charged"] for g in pool.generations(job))
    charged = [g["charged"] for g in pool.generations(crashing)]
    assert charged.count(True) == 2


def test_a_lost_batched_call_is_split_apart_from_the_rest(tasks, pool):
    """A taker splits the jobs of the lost holder's in-flight call apart from the ticket's other
    open jobs, and halves them, so a batch smaller than its ticket isolates its crashing job within
    its own call's halvings; no job of a call of several is charged."""
    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in range(1, 7)]
    digest = pool.put(jobs, batch=4)
    dead = "d" * 32
    claim = pool.pool.take("cpu", digest, beat="external")
    entries = {e.key: e for e in claim.ticket.jobs}
    listed = []
    for job in jobs[:4]:  # the dead worker's call held the first four
        entry = entries[pool.key(job)]
        attempt = claim.start(entry, dead)
        listed.append(
            {
                "checksum": entry.checksum,
                "fingerprint": entry.fingerprint,
                "attempt": attempt,
            }
        )
    pool.pool.write_call(dead, 1, {"jobs": listed})
    _age(claim.lease.lockfile)
    _, code = _run(pool)
    assert code == 0
    assert sorted(sorted(c) for c in tasks.CALLS) == [[1, 2], [3, 4], [5, 6]]
    for job in jobs:
        assert pool.newest(job)["outcome"] == "done"
        assert not any(g["charged"] for g in pool.generations(job))


def test_a_lost_claim_settles_nothing_of_the_call(tasks, pool, monkeypatch):
    """A claim lost while a batched call runs: no job of it publishes or settles."""
    from pydra.workers.pool import worker as module

    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2, 3)]
    pool.put(jobs, batch=4)
    real = module.Worker._batched_call

    def losing(self, call_jobs):
        for d in pool.path.glob("queues/*/claims/*.lease"):
            for dirpath, dirnames, filenames in os.walk(d, topdown=False):
                for name in filenames:
                    os.unlink(Path(dirpath) / name)
                for name in dirnames:
                    os.rmdir(Path(dirpath) / name)
        return real(self, call_jobs)

    monkeypatch.setattr(module.Worker, "_batched_call", losing)
    _, code = _run(pool)
    assert code == 0
    for job in jobs:
        assert pool.newest(job) is None and pool.result(job) is None


def _lost_call(pool, digest, jobs, *, expired=False, published=()):
    """Leave ticket ``digest`` as a worker killed mid-call leaves it: every job of ``jobs`` started
    under the dead worker's token and listed in its in-flight call record, each given an
    ``expired`` outcome when ``expired``, those of ``published`` committed to the store; the claim
    aged."""
    dead = "d" * 32
    claim = pool.pool.take("cpu", digest, beat="external")
    entries = {e.key: e for e in claim.ticket.jobs}
    listed = []
    for job in jobs:
        entry = entries[pool.key(job)]
        attempt = claim.start(entry, dead)
        listed.append(
            {
                "checksum": entry.checksum,
                "fingerprint": entry.fingerprint,
                "attempt": attempt,
            }
        )
        if expired:
            claim.outcome(entry, attempt, "expired")
        if job in published:
            job.attempt = attempt
            job.run()  # what the call published before its worker was killed
    pool.pool.write_call(dead, 1, {"jobs": listed})
    _age(claim.lease.lockfile)


def test_an_expired_batched_call_is_bisected_not_laddered(tasks, pool):
    """A batched call that passed its deadline is bisected by its taker (RFC 0005 sections 5.3, 6): a
    job its call published settles ``done``; the others are split in halves, uncharged, as
    ``split``, and none of them is moved up the ladder or spends ``resource_moves``."""
    from pydra.workers.pool.worker import class_record

    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2, 3, 4)]
    digest = pool.put(jobs, batch=4, ladder=("big",))
    pool.pool.put_class("big", class_record(_class("big", time=7200)))
    _lost_call(pool, digest, jobs, expired=True, published=jobs[:1])
    _, code = _run(pool, _class(time=60))
    assert code == 0
    assert sorted(tasks.CALLS) == [[2, 3], [4]]
    first, *rest = jobs
    assert pool.newest(first)["outcome"] == "done"
    for job in rest:
        gens = pool.generations(job)
        assert [g["cause"] for g in gens] == ["put", "takeover", "split"]
        assert not any(g["charged"] for g in gens)
        assert pool.pool.charges(*pool.key(job))["resource_moves"] == 0
        assert pool.newest(job)["outcome"] == "done"


def test_a_lone_expired_job_still_moves_up_the_ladder(tasks, pool):
    from pydra.workers.pool.worker import class_record

    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2)]
    digest = pool.put(jobs, batch=4, ladder=("big",))
    pool.pool.put_class("big", class_record(_class("big", time=7200)))
    _lost_call(pool, digest, jobs, expired=True, published=jobs[:1])
    _, code = _run(pool, _class(time=60))
    assert code == 0 and tasks.CALLS == []
    published, slow = jobs
    assert pool.newest(published)["outcome"] == "done"
    assert [g["cause"] for g in pool.generations(slow)] == [
        "put",
        "takeover",
        "ladder",
    ]


@pytest.mark.skipif(not POSIX, reason="the beater kills its hung worker by signal")
def test_one_hanging_item_fails_alone_on_a_one_class_pool(
    tasks, pool, processes, tmp_path
):
    """A batch of four with one item that hangs, on a pool whose only class has no larger ``time``:
    each expiry bisects the call, so the hanging job is alone after three calls and only it fails
    ``ResourceExhausted`` (a timeout); the other three are done."""
    marks = tmp_path / "marks"
    marks.mkdir()
    jobs = [
        pool.job(
            tasks.Square(x=x, where=str(marks), mode="hang" if x == 2 else ""),
            f"s{x}",
        )
        for x in (1, 2, 3, 4)
    ]
    hanging = jobs[1]
    pool.put(jobs, batch=4)
    settings = _settings(call_timeout=1.0, timeout_grace=1.0)
    for _ in range(12):  # a bound on worker starts that only fails a runaway test
        proc = _hand_worker(pool, processes, tmp_path, settings=settings)
        code = proc.wait(BOUND)
        if code == 0 and all(pool.newest(j) is not None for j in jobs):
            break
        for d in (pool.path / "queues").glob("*/claims/*.lease"):
            _age(d)
    else:
        raise AssertionError("the hanging job was never settled")
    calls = [json.loads(p.read_text()) for p in marks.iterdir()]
    assert sorted(len(c) for c in calls if 2 in c) == [1, 2, 4]
    for job in jobs:
        outcome = pool.newest(job)
        if job is hanging:
            assert (outcome["outcome"], outcome["fault"]["type"]) == (
                "failed",
                "ResourceExhausted",
            )
            assert outcome["fault"]["reason"] == "timeout"
        else:
            assert outcome["outcome"] == "done"
            assert pool.result(job).outputs.out["square"] == job.task.x**2


def test_a_failed_batched_jobs_pickle_reruns_the_function(tasks, pool):
    """A failed batched job's ``_job.pklz`` carries no element of its call: a rerun from it runs the
    task's function."""
    from pydra.engine.job import load_job

    jobs = [
        pool.job(tasks.Square(x=x, mode="batch-error" if x == 2 else ""), f"s{x}")
        for x in (1, 2)
    ]
    pool.put(jobs, batch=4)
    _, code = _run(pool)
    assert code == 0
    failed = jobs[1]
    assert pool.newest(failed)["outcome"] == "failed"
    pickles = [load_job(p) for p in Path(pool.store.path).rglob("_job.pklz")]
    [loaded] = [j for j in pickles if j.checksum == failed.checksum]
    assert loaded.batched_element is None
    result = loaded.run(rerun=True)
    assert not result.errored and result.outputs.out["square"] == 4


def test_a_jobs_own_run_error_goes_through_the_classifier(tasks, pool, monkeypatch):
    """An exception a job's own ``Job.run`` raises (not its element) is typed by the worker's
    classifier, as a lone call's is: an infrastructure fault requeues the job rather than failing
    it as content, and the job runs again."""
    from pydra.engine.job import Job as JobClass
    from pydra.workers.pool import faults

    def classify(exc, phase):
        if isinstance(exc, OSError):
            return faults.FaultRecord("infrastructure", "OSError", str(exc), phase)
        return faults.classify(exc, phase)

    raised = []
    real = JobClass._record_pool_facts

    def flaky(self, result):
        if self.pool_facts is not None and self.task.x == 2 and not raised:
            raised.append(self.task.x)
            raise OSError(5, "input/output error")
        return real(self, result)

    monkeypatch.setattr(JobClass, "_record_pool_facts", flaky)
    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2, 3)]
    pool.put(jobs, batch=4)
    _, code = _run(pool, classify=classify)
    assert code == 0 and raised == [2]
    for job in jobs:
        assert pool.newest(job)["outcome"] == "done"
    causes = [(g["cause"], g.get("phase")) for g in pool.generations(jobs[1])]
    assert ("restart", "call") in causes


def test_one_calls_infrastructure_faults_count_once(tasks, pool, monkeypatch):
    """Infrastructure faults of several jobs of one batched call are one fault of the residency:
    the three jobs are requeued together, the residency is not held, and all three end done.
    """
    from pydra.engine.job import Job as JobClass
    from pydra.workers.pool import faults

    def classify(exc, phase):
        if isinstance(exc, OSError):
            return faults.FaultRecord("infrastructure", "OSError", str(exc), phase)
        return faults.classify(exc, phase)

    raised = []
    real = JobClass._record_pool_facts

    def flaky(self, result):
        if self.pool_facts is not None and self.task.x not in raised:
            raised.append(self.task.x)
            raise OSError(5, "input/output error")
        return real(self, result)

    monkeypatch.setattr(JobClass, "_record_pool_facts", flaky)
    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2, 3)]
    pool.put(jobs, batch=4)
    _, code = _run(pool, classify=classify)
    assert code == 0 and sorted(raised) == [1, 2, 3]
    assert not pool.pool.holds("cpu")
    for job in jobs:
        assert pool.newest(job)["outcome"] == "done"
        causes = [(g["cause"], g.get("phase")) for g in pool.generations(job)]
        assert causes.count(("restart", "call")) == 1


def test_committed_jobs_are_served_not_recomputed(tasks, pool):
    """A job the store already holds settles ``done`` (served) before the batched call and is left
    out of it."""
    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2, 3, 4)]
    pool.put(jobs, batch=4)
    for job in jobs[:2]:
        job.run()
    _, code = _run(pool)
    assert code == 0
    assert tasks.CALLS == [[3, 4]]
    for job in jobs:
        outcome = pool.newest(job)
        assert outcome["outcome"] == "done"
        assert outcome.get("served", False) is (job in jobs[:2])


def test_a_taker_serves_what_the_lost_call_published(tasks, pool):
    """Jobs a lost batched call published are served by the taker before it splits the rest."""
    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2, 3, 4)]
    digest = pool.put(jobs, batch=4)
    _lost_call(pool, digest, jobs, published=jobs[:2])
    _, code = _run(pool)
    assert code == 0
    assert sorted(tasks.CALLS) == [[3], [4]]
    for job in jobs:
        assert pool.newest(job)["outcome"] == "done"


def test_a_lost_lease_mid_call_stops_the_later_jobs(tasks, pool, monkeypatch):
    """A job whose proof before publish fails (`LeaseLost`) ends the batched call: no later job of it
    runs or publishes, and nothing of it is settled; the ticket, claimed again, runs every job.
    """
    from pydra.engine import job as job_module

    runs = []
    real = job_module.Job.run

    def counting(self, *args, **kwargs):
        runs.append(self.task.x)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(job_module.Job, "run", counting)
    proofs = []
    real_proof = job_module.lease_still_held

    def lost_once(lease):
        proofs.append(lease)
        return len(proofs) > 1 and real_proof(lease)

    monkeypatch.setattr(job_module, "lease_still_held", lost_once)
    jobs = [pool.job(tasks.Square(x=x), f"s{x}") for x in (1, 2, 3)]
    pool.put(jobs, batch=4)
    _, code = _run(pool)
    assert code == 0
    assert runs == [1, 1, 2, 3]  # the lost call ran one job; the next call ran them all
    for job in jobs:
        outcome = pool.newest(job)
        assert outcome["outcome"] == "done" and not outcome.get("served")


# -- the declared batch ------------------------------------------------------------------------


def test_a_declared_batch_is_the_fits_batch(tasks, pool):
    """``python.define(batch=N)`` is unhashed and is the batch `DeclaredFit` gives; `MeasuredFit`
    gives it before a per-unit memory is measured and the smaller of it and the memory bound after;
    no declared batch is 1."""
    from pydra.compose import python
    from pydra.workers.pool.calibration import MemoryFit
    from pydra.workers.pool.fit import MeasuredFit
    from pydra.workers.pool.placement import DeclaredFit, FitRequest, fit_request
    from pydra.utils.resources import ComputeResources

    def Twice(x: int) -> int:  # noqa: N802
        return 2 * x

    assert (
        python.define(Twice, batched=tasks.square_many, batch=8)(x=1)._checksum
        == python.define(Twice, batched=tasks.square_many)(x=1)._checksum
    )
    with pytest.raises(ValueError, match="batch must be a positive int"):
        python.define(Twice, batch=0)
    cpu = _class()
    declared = fit_request(pool.job(tasks.Square(x=3), "s"))
    assert declared.batch == 4
    assert DeclaredFit().batch(declared, cpu, [3.0]) == 4
    plain = fit_request(pool.job(tasks.PlainSquare(x=3), "p"))
    assert plain.batch is None and DeclaredFit().batch(plain, cpu, [None]) == 1

    G = 2**30
    figures = {"fit": None}

    class Fixed:
        def memory(self, key, hardware, task=None):
            return figures["fit"]

        def unfit(self, key):
            return ()

        def times(self, key, hardware, task=None):
            return ()

    gpu = _class("gpu", device="cuda", gpus=1, gpu_mem=80 * G, quantum_jobs=64)
    request = FitRequest(
        residency=(), resources=ComputeResources(gpus=1), key="k", batch=4
    )
    fit = MeasuredFit(Fixed(), headroom=0.9)
    assert fit.batch(request, gpu, [4.0]) == 4  # nothing measured: the declared batch
    figures["fit"] = MemoryFit(resident=10 * G, m0=2 * G, m1=None, n=6)
    assert fit.batch(request, gpu, [4.0]) == 4  # per-unit memory unknown
    figures["fit"] = MemoryFit(resident=10 * G, m0=2 * G, m1=0.5 * G, n=6)
    assert fit.batch(request, gpu, [4.0]) == 4  # a memory bound of 30: the declared 4
    figures["fit"] = MemoryFit(resident=64 * G, m0=2 * G, m1=0.5 * G, n=6)
    assert fit.batch(request, gpu, [4.0]) == 3  # a memory bound of 3 below it


def test_no_declared_batch_runs_one_job_per_call(tasks, tmp_path):
    """A task is placed, for ``plan()`` as for its ticket, with its fit's batch (the probe's or the
    declared one); the fit's memory bound is kept as ``memory_bound``."""
    from pydra.workers.pool.pool import _ticket_batch
    from pydra.workers.pool.placement import DeclaredFit, place

    sub = Submitter(cache_root=tmp_path / "store", worker="debug")

    class EightFit(DeclaredFit):
        def judge(self, request, cls):
            from pydra.workers.pool.placement import ClassTried

            return ClassTried(cls.name, "fits", "")

        def batch(self, request, cls, units):
            return 8 if request.batch else 1  # the probe starts at one job per call

        def memory_bound(self, request, cls, units):
            return 8

    def placed(task):
        job = Job(task=task, submitter=sub, name="j")
        return job, place(job, [_class()], EightFit(), lambda cls: True)

    class J:
        def __init__(self, job, placement):
            self.job, self.batch = job, placement.batch

    job, undeclared = placed(tasks.Short(x=1))
    assert (undeclared.batch, undeclared.memory_bound) == (1, 8)
    assert _ticket_batch([J(job, undeclared)]) == 1
    job, declared = placed(tasks.Square(x=1))
    assert (declared.batch, declared.memory_bound) == (8, 8)
    assert _ticket_batch([J(job, declared)]) == 8


# -- through a WarmPool ------------------------------------------------------------------------


def test_a_pool_writes_its_placements_batch_and_units(tasks, tmp_path):
    """A `WarmPool` writes each ticket's ``batch`` from its jobs' placements (the task's declared
    batch, through `DeclaredFit`) and each job's units from the task's ``units``, so its workers
    batch the node's jobs; ``max_concurrent=1`` does not cap a pool's batched calls."""
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.storage import Ticket
    from pydra.workers.pool.substrate import LocalSubstrate

    store = Submitter(cache_root=tmp_path / "store", worker="debug").store
    substrate = LocalSubstrate()
    executor = WarmPool(
        tmp_path / "pool",
        [_class(quantum_jobs=8)],
        substrate,
        store=store,
        settings=_settings(idle_polls=40),
    )
    try:
        with Submitter(store=store, worker=executor, max_concurrent=1) as sub:
            result = sub(tasks.Squares(xs=[4, 3, 2, 1, 8, 7, 6, 5]))
    finally:
        executor.close()
        for handle in substrate.handles:
            if handle.process is not None and handle.process.poll() is None:
                handle.process.kill()
                handle.process.wait(BOUND)
    tickets = [
        Ticket.from_json(json.loads(p.read_text()))
        for p in (tmp_path / "pool" / "queues").glob("*/tickets/*.json")
    ]
    assert tickets and all(t.batch == 4 for t in tickets)
    assert sorted(j.units for t in tickets for j in t.jobs) == [
        float(x) for x in range(1, 9)
    ]
    sizes = [
        json.loads(line)["batch_size"]
        for p in (tmp_path / "pool" / "workers").glob("*/samples.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert sorted(sizes) == [4] * 8
    assert [o["square"] for o in result.outputs.out] == [
        x * x for x in [4, 3, 2, 1, 8, 7, 6, 5]
    ]


def _double(x: int) -> int:
    return 2 * x


def _double_all(x):
    return [2 * v for v in x]


def _double_all_edited(x):
    return [v + v for v in x]


def test_the_batched_callable_joins_the_task_key():
    from pydra.compose import python

    plain = python.define(_double)
    batched = python.define(_double, batched=_double_all)
    edited = python.define(_double, batched=_double_all_edited)
    assert batched(x=1)._checksum != plain(x=1)._checksum
    assert edited(x=1)._checksum != batched(x=1)._checksum
    again = python.define(_double, batched=_double_all)
    assert again(x=1)._checksum == batched(x=1)._checksum
    assert python.define(_double)(x=1)._checksum == plain(x=1)._checksum


def test_a_probing_worker_moves_through_sizes_within_one_run(tasks, tmp_path):
    """The batch of each call is probed from the calibration as it stands when the call is
    made, so one run of many jobs moves through 1, 2, 4 as its own calls are measured"""
    from types import SimpleNamespace

    from pydra.workers.pool.calibration import TimeRecord
    from pydra.workers.pool.settings import PoolSettings
    from pydra.workers.pool.storage import Ticket
    from pydra.workers.pool.worker import Worker

    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    jobs = [
        (None, Job(task=tasks.Short(x=i), submitter=sub, name="j")) for i in range(2)
    ]
    measured: list = []

    class Growing:
        def times(self, key, hardware, task=None):
            return measured

    worker = SimpleNamespace(
        calibration=Growing(), cls=_class(), settings=PoolSettings(probe_samples=1)
    )
    claim = SimpleNamespace(ticket=SimpleNamespace(residency=(), batch=16, probe=True))
    sizes = []
    for record in [None, TimeRecord(1, 1.0, 1), TimeRecord(2, 0.5, 1), None]:
        if record is not None:
            measured.append(record)
        sizes.append(Worker._batch_size(worker, claim, jobs))
    assert sizes == [1, 2, 4, 4]
    declared = SimpleNamespace(
        ticket=SimpleNamespace(residency=(), batch=16, probe=False)
    )
    assert Worker._batch_size(worker, declared, jobs) == 16  # a declared size is kept


def test_a_probing_ticket_sizes_each_call_from_the_calibration_of_its_earlier_calls(
    tasks, tmp_path, monkeypatch
):
    """One ticket's calls move through the probe's sizes: each call is sized when it is made"""
    from pydra.workers.pool.calibration import TimeRecord
    from pydra.workers.pool.settings import PoolSettings
    from pydra.workers.pool.worker import Worker

    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    jobs = [
        (
            SimpleNamespace(units=None),
            Job(task=tasks.Short(x=i), submitter=sub, name="j"),
        )
        for i in range(16)
    ]
    measured: dict = {}

    class Growing:
        def times(self, key, hardware, task=None):
            return [TimeRecord(b, 1.0 / b, n) for b, n in sorted(measured.items())]

    worker = Worker(tmp_path / "pool", _class(), settings=PoolSettings(probe_samples=1))
    worker.calibration = Growing()
    worker.cls = _class()
    sizes = []

    def run_batch(claim, members, group):
        sizes.append(len(members))
        measured[len(members)] = measured.get(len(members), 0) + 1

    monkeypatch.setattr(worker, "_run_batch", run_batch)
    for name in ("_lost_claim", "_held_here"):
        monkeypatch.setattr(worker, name, lambda claim: False)
    monkeypatch.setattr(worker, "_within_crash_budget", lambda claim, entry: True)
    claim = SimpleNamespace(
        ticket=SimpleNamespace(residency=(), batch=16, probe=True),
        lease=SimpleNamespace(prove=lambda: True),
    )
    worker._run_batches(claim, jobs, {})
    assert sizes[:4] == [1, 2, 4, 8] and sum(sizes) == 16


def test_a_batched_call_the_time_left_does_not_hold_recycles_the_worker(
    tasks, tmp_path, monkeypatch
):
    """Before each call the worker compares the time left with its jobs' declared time"""
    from pydra.utils.resources import ComputeResources
    from pydra.workers.pool.settings import PoolSettings
    from pydra.workers.pool.worker import Worker

    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    jobs = []
    for i in range(4):
        job = Job(task=tasks.Short(x=i), submitter=sub, name="j")
        job.resources = ComputeResources(time=100)
        jobs.append((SimpleNamespace(units=None), job))
    worker = Worker(tmp_path / "pool", _class(), settings=PoolSettings())
    worker.cls = _class()
    worker.ends_at = time.time() + 150.0
    ran = []
    monkeypatch.setattr(
        worker, "_run_batch", lambda c, members, g: ran.append(len(members))
    )
    for name in ("_lost_claim", "_held_here"):
        monkeypatch.setattr(worker, name, lambda claim: False)
    monkeypatch.setattr(worker, "_within_crash_budget", lambda claim, entry: True)
    claim = SimpleNamespace(
        ticket=SimpleNamespace(residency=(), batch=2, probe=False),
        lease=SimpleNamespace(prove=lambda: True),
    )
    worker._run_batches(claim, jobs, {})
    assert ran == [2, 2]
    worker.ends_at = time.time() + 50.0
    ran.clear()
    worker._run_batches(claim, jobs, {})
    assert ran == [] and worker.recycle is not None


def test_a_batched_call_is_calibrated_once(tasks, tmp_path):
    """A call of several jobs gives the calibration one sample, not one per job"""
    from pydra.workers.pool.worker import Worker

    recorded = []
    sample = SimpleNamespace(key="k", to_json=lambda: {})

    class Cal:
        def record(self, s):
            recorded.append(s)

    worker = SimpleNamespace(calibration=Cal(), _calibrated=None)
    for _ in range(3):  # the call's three jobs settle with the call's one sample
        Worker._calibrate(worker, sample)
    assert recorded == [sample]
