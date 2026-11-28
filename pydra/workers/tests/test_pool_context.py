"""The ``pool`` section of the execute context, and a pool that outlives a run (RFC 0005 section 8, W11a).

A worker computes ``pool-facts@1`` for each call. Under a store whose identity lists ``pool`` in
``recorded_only`` the section enters the execute context, where it is recorded and never splits a slot;
under `PickleStore` it reaches only the audit record and ``Result.runtime``. A store whose identity
would key it is refused (`PoolFactsWouldSplitSlots`).

`Submitter(worker=<WarmPool>)` leaves the pool to its caller, so its workers stay resident from one run to
the next; a pool the `Submitter` makes by name is closed with the `Submitter`. Loads are counted by a
provider that writes one file per load, never timed.
"""

import json
import os
import signal
import subprocess
import sys
import typing as ty
from pathlib import Path

import attrs
import pytest

from pydra.engine.job import Job
from pydra.engine.lease import LeaseSettings
from pydra.engine.portable.identity import PydraIdentityV1
from pydra.engine.portable.store import PortableStore
from pydra.engine.submitter import Submitter

POSIX = sys.platform != "win32"
LEASE = LeaseSettings(
    stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
)
BOUND = 120.0  # seconds any wait on another process may take before the test fails

TASKS = """
import json, os
from pathlib import Path
from uuid import uuid4
from pydra.compose import python, workflow
from pydra.engine.resident import Provided

LOADS = []
COMMITS = []


class Counting:
    # writes one file per load into the directory its spec names, so loads are counted across processes
    name = "counting"

    def load(self, spec, device):
        LOADS.append(spec)
        Path(spec, uuid4().hex).touch()
        return spec

    def close(self, obj):
        pass

    def footprint(self, spec):
        return None

    def measure(self, obj):
        return None

    def version(self):
        return "7"


class RefusesPoolFacts:
    # a messenger that fails on the pool-facts message, and keeps the rest
    def send(self, message, **kwargs):
        if message.get("@type") == "pool-facts":
            raise OSError("the audit is unwritable")


def record_commit(staged, checksum, key, context, digest):
    COMMITS.append({"checksum": checksum, "context": json.loads(json.dumps(context))})


@python.define
def UseModel(model: Provided[str, "counting"], x: int) -> int:
    return x


@python.define
def Echo(x: int) -> int:
    return x


@workflow.define
def UseModels(model: str, xs: list) -> list:
    return workflow.add(UseModel(model=model).split(x=xs), name="U").out


@python.define
def MaybeOom(model: Provided[str, "counting"], x: int) -> int:
    # the second job runs out of memory: a resource fault, after which the residency reloads
    if x == 2:
        raise MemoryError("out of memory in the call")
    return x


@python.define
def Gate(where: str) -> str:
    # the first execution waits for "go"; any later one returns at once
    import time

    d = Path(where)
    if not (d / "first").exists():
        (d / "first").touch()
        while not (d / "go").exists():
            time.sleep(0.05)
        return "first"
    return "later"


@workflow.define
def Echoes(xs: list) -> list:
    return workflow.add(Echo().split(x=xs), name="E").out


@python.define
def Hold(x: int, where: str) -> int:
    # marks that it started, then waits for "go"
    import time

    d = Path(where)
    (d / "held").touch()
    while not (d / "go").exists():
        time.sleep(0.05)
    return x + 1


@workflow.define(outputs=["a1", "a2"])
def Interleaved(model: str, where: str):
    # A1 and A2 run on the pool, M between them elsewhere
    a1 = workflow.add(UseModel(model=model, x=1), name="A1")
    m = workflow.add(Hold(x=a1.out, where=where), name="M")
    a2 = workflow.add(UseModel(model=model, x=m.out), name="A2")
    return a1.out, a2.out


@python.define
def Inc(x: int) -> int:
    return x + 1


@workflow.define
def ThreeRounds(model: str, x: int) -> int:
    # three rounds on the pool, each reading the one before
    a = workflow.add(UseModel(model=model, x=x), name="A")
    b = workflow.add(Inc(x=a.out), name="B")
    return workflow.add(UseModel(model=model, x=b.out), name="C").out


@workflow.define(outputs=["g", "e"])
def GateAndEcho(where: str):
    g = workflow.add(Gate(where=where), name="G")
    e = workflow.add(Echo(x=7), name="E")
    return g.out, e.out
"""


@pytest.fixture(scope="session")
def tasks(tmp_path_factory):
    """The task module, importable here and in workers started from here."""
    where = tmp_path_factory.mktemp("pool_context_tasks")
    (where / "_pydra_pool_context_tasks.py").write_text(TASKS)
    sys.path.insert(0, str(where))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(where), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    import _pydra_pool_context_tasks

    return _pydra_pool_context_tasks


@pytest.fixture(autouse=True)
def clean(monkeypatch, tasks):
    """Release every claim a test leaves held, empty the process's resident set and the task module's
    records, pass or fail."""
    from pydra.engine import resident
    from pydra.workers.pool.storage import Claim

    made = []
    init = Claim.__init__

    def recording(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(Claim, "__init__", recording)
    device = resident.process_device()
    tasks.LOADS.clear()
    tasks.COMMITS.clear()
    yield
    tasks.LOADS.clear()
    tasks.COMMITS.clear()
    resident.set_process_device(device)
    resident.close_process_set()
    for claim in made:
        claim.lease.__exit__(None, None, None)


@pytest.fixture
def groups():
    """Process groups of workers a test started: killed at the end, and none may be left."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        if POSIX:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    if POSIX:
        for pid in pids:
            _wait(lambda: not _group_alive(pid), f"process group {pid} to exit")


def _group_alive(pid: int) -> bool:
    found = subprocess.run(["pgrep", "-g", str(pid)], capture_output=True, text=True)
    return found.stdout.strip() != ""


def _wait(predicate, what: str, bound: float = BOUND) -> None:
    import time

    deadline = time.monotonic() + bound
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _settings(**kwargs):
    from pydra.workers.pool.settings import PoolSettings

    fields = dict(lease=LEASE, poll_s=0.05, idle_polls=2, timeout_grace=60.0)
    fields.update(kwargs)
    return PoolSettings(**fields)


def _class(name="cpu", **kwargs):
    from pydra.workers.pool.classes import WorkerClass

    fields = dict(name=name, device="cpu", quantum_jobs=8, quantum_s=600.0)
    fields.update(kwargs)
    return WorkerClass(**fields)


@attrs.frozen
class ClientIdentity:
    """A client's identity (RFC 0003 P9): pydra's key, context and fingerprint under another name."""

    recorded_only: frozenset = attrs.field(factory=frozenset, converter=frozenset)
    name: ty.ClassVar[str] = "client-identity@1"
    fingerprint_name: ty.ClassVar[str] = "pydra-context@1"
    scheme: ty.ClassVar[None] = None

    def supplied_key(self, checksum):
        return None

    @property
    def _v1(self) -> PydraIdentityV1:
        return PydraIdentityV1(recorded_only=self.recorded_only)

    def key(self, job):
        return self._v1.key(job)

    def check_key(self, checksum, key):
        return self._v1.check_key(checksum, key)

    def context(self, job, phase):
        return self._v1.context(job, phase)

    def fingerprint(self, context):
        return self._v1.fingerprint(context)


def _store(tmp_path: Path, identity=None, **kwargs) -> PortableStore:
    root = tmp_path / "store"
    root.mkdir(mode=0o700, exist_ok=True)
    kwargs.setdefault("pickle", True)
    if identity is not None:
        kwargs["identity"] = identity
    return PortableStore(root, trust="private", **kwargs)


class Queue:
    """A pool root, a submitter and the tickets put on the pool, for an in-process worker."""

    def __init__(self, tmp_path: Path, tasks, store):
        from pydra.workers.pool.storage import PoolRoot

        self.pool = PoolRoot.open(tmp_path / "pool", LEASE, probe=False)
        self.submitter = Submitter(
            store=store,
            worker="debug",
            providers=[tasks.Counting()],
            audit_flags=_prov(),
            messengers=_file_messenger(),
        )
        self.store = store

    def job(self, task, name: str, rerun: bool = False) -> Job:
        job = Job(task=task, submitter=self.submitter, name=name)
        job.rerun = rerun
        return job

    def put(self, jobs, queue="cpu") -> str:
        from pydra.engine import resident
        from pydra.workers.pool.codec import PrivatePickle, write_payload
        from pydra.workers.pool.storage import Ticket, TicketJob

        entries = []
        for job in jobs:
            fingerprint = _fingerprint(self.store, job)
            g = self.pool._next_generation((job.checksum, fingerprint))
            job.attempt = f"{self.pool.pool_id}.{g}"
            entries.append(
                TicketJob(
                    checksum=job.checksum,
                    fingerprint=fingerprint,
                    payload=write_payload(PrivatePickle(), job, self.pool.path),
                    attempt=job.attempt,
                    runner_attempt=0,
                    units=1.0,
                )
            )
        residency = tuple(
            attrs.asdict(resident.spec_key(jobs[0].task, f, p))
            for f, p in resident.provided_fields(jobs[0].task)
        )
        return self.pool.put(
            queue,
            Ticket(queue=queue, node="wf/node", jobs=entries, residency=residency),
        )

    def classes(self, *classes) -> None:
        from pydra.workers.pool.worker import class_record

        for cls in classes:
            self.pool.put_class(cls.name, class_record(cls))

    def work(self, cls, **kwargs):
        """Run a worker of ``cls`` in this process until it is idle."""
        from pydra.workers.pool.worker import Worker

        worker = Worker(
            self.pool.path,
            cls,
            settings=_settings(),
            device_memory=lambda: None,
            **kwargs,
        )
        assert worker.run() == 0
        return worker


def _prov():
    from pydra.utils.messenger import AuditFlag

    return AuditFlag.PROV


def _file_messenger():
    from pydra.utils.messenger import FileMessenger

    return FileMessenger()


def _fingerprint(store, job) -> str:
    import hashlib

    from pydra.utils.digest import jcs

    identity = getattr(store, "identity", None)
    if identity is not None:
        return identity.fingerprint(job.context)
    return hashlib.sha256(jcs(job.context)).hexdigest()


@pytest.fixture
def queue_of(tmp_path, tasks):
    made = []

    def make(store):
        q = Queue(tmp_path, tasks, store)
        made.append(q)
        return q

    yield make
    for q in made:
        q.pool.close()
        q.submitter.close()


def _messages(root: Path, kind: str) -> list[dict]:
    found = []
    for path in Path(root).rglob("*.jsonld"):
        message = json.loads(path.read_text())
        if message.get("@type") == kind:
            found.append(message)
    return found


# -- the pool section --------------------------------------------------------------------------


def test_cold_and_warm_share_one_slot(tmp_path, tasks, queue_of, monkeypatch):
    """Under an identity listing ``pool``, the cold and the warm call of one worker publish ``pool-facts@1``
    in their execute contexts, into the slot the lookup finds: the fingerprint is the one of the context
    without the section, so both results are served afterwards with no load."""
    identity = PydraIdentityV1(recorded_only={"pool"})
    store = _store(tmp_path, identity, on_commit=tasks.record_commit)
    q = queue_of(store)
    q.classes(_class())
    models = tmp_path / "loads"
    models.mkdir()
    jobs = [q.job(tasks.UseModel(model=str(models), x=x), f"j{x}") for x in (1, 2)]
    q.put(jobs)
    from pydra.workers.pool import worker as worker_module
    from pydra.workers.pool.samples import HardwareClass

    monkeypatch.setattr(worker_module, "_host_token", lambda: "0123456789abcdef")
    worker = q.work(_class(), hardware=lambda: HardwareClass(cpus=3))

    assert len(tasks.LOADS) == 1
    assert [c["checksum"] for c in tasks.COMMITS] == [j.checksum for j in jobs]
    facts = [c["context"]["pool"] for c in tasks.COMMITS]
    assert [f["residency"][0]["state"] for f in facts] == ["cold", "warm"]
    assert [f["residency"][0]["since_load"] for f in facts] == [0, 1]
    for f, job in zip(facts, jobs):
        assert f["scheme"] == "pool-facts@1"
        assert f["worker"] == worker.token and f["class"] == "cpu"
        assert f["attempt"] == job.attempt
        assert f["batch"] == {"size": 1, "index": 0}
        assert f["losses_survived"] == 0
        assert f["hardware"] == {
            "device": "cpu",
            "name": None,
            "memory_mib": None,
            "compute_capability": None,
            "cpus": 3,
        }
        assert f["host"] == "0123456789abcdef"
        (entry,) = f["residency"]
        assert entry["kind"] == "provider" and entry["name"] == "counting"
        assert entry["version"] == "7"
        assert entry["key"] == _resident_key(job)
    assert isinstance(facts[0]["residency"][0]["load_s"], float)
    slots = {identity.fingerprint(c["context"]) for c in tasks.COMMITS}
    assert slots == {identity.fingerprint(jobs[0].context)}
    assert all("pool" not in j.context for j in jobs)
    # the lookup finds both entries in that slot: served, never loaded again
    with Submitter(store=store, worker="debug", providers=[tasks.Counting()]) as sub:
        assert [
            sub(tasks.UseModel(model=str(models), x=x)).outputs.out for x in (1, 2)
        ] == [1, 2]
    assert len(list(models.iterdir())) == 1


def _resident_key(job) -> str:
    from pydra.engine import resident

    [(field, name)] = resident.provided_fields(job.task)
    return resident.spec_key(job.task, field, name).digest()


def test_two_workers_share_one_slot(tmp_path, tasks, queue_of):
    """One job run again on a second worker publishes into the slot of the first: the facts name two
    workers, and the slot is one."""
    identity = PydraIdentityV1(recorded_only={"pool"})
    store = _store(tmp_path, identity, on_commit=tasks.record_commit)
    q = queue_of(store)
    q.classes(_class())
    models = tmp_path / "loads"
    models.mkdir()
    task = tasks.UseModel(model=str(models), x=1)
    q.put([q.job(task, "first")])
    first = q.work(_class())
    q.put([q.job(task, "again", rerun=True)])
    second = q.work(_class())

    assert first.token != second.token
    workers = [c["context"]["pool"]["worker"] for c in tasks.COMMITS]
    assert workers == [first.token, second.token]
    slots = {identity.fingerprint(c["context"]) for c in tasks.COMMITS}
    assert len(slots) == 1
    stored = store.stored(task._checksum)
    assert {ref.fingerprint for ref, _, _ in stored} == slots
    [(_, context, _)] = stored
    assert context["pool"]["worker"] in (first.token, second.token)


def test_store_keying_pool_refused(tmp_path, tasks):
    """A store whose identity would key ``pool`` is refused before the pool is touched; the same client
    identity listing ``pool`` in ``recorded_only`` is accepted."""
    from pydra.workers.pool.pool import PoolFactsWouldSplitSlots, WarmPool
    from pydra.workers.pool.substrate import ExternalSubstrate

    keyed = _store(tmp_path, ClientIdentity())
    with pytest.raises(PoolFactsWouldSplitSlots, match="client-identity@1"):
        WarmPool(
            tmp_path / "pool",
            [_class()],
            ExternalSubstrate(),
            store=keyed,
            settings=_settings(),
        )
    assert not (tmp_path / "pool").exists()
    with pytest.raises(PoolFactsWouldSplitSlots):
        Submitter(
            store=keyed,
            worker="pool",
            root=tmp_path / "pool",
            classes=[_class()],
            substrate=ExternalSubstrate(),
            settings=_settings(),
        )
    assert not (tmp_path / "pool").exists()

    (tmp_path / "other").mkdir()
    listed = _store(tmp_path / "other", ClientIdentity(recorded_only={"pool"}))
    pool = WarmPool(
        tmp_path / "pool",
        [_class()],
        ExternalSubstrate(),
        store=listed,
        settings=_settings(),
    )
    pool.close()


@pytest.mark.parametrize("kind", ["v1", "pickle"])
def test_v1_and_picklestore_facts_audit_only(tmp_path, tasks, queue_of, kind):
    """Under ``pydra-identity@1`` without ``pool`` in ``recorded_only``, and under `PickleStore`, the
    facts go to the audit record and ``Result.runtime``, never into the execute context.
    """
    from pydra.engine.store import PickleStore

    if kind == "v1":
        store = _store(tmp_path, PydraIdentityV1(), on_commit=tasks.record_commit)
    else:
        store = PickleStore(tmp_path / "store")
    q = queue_of(store)
    q.classes(_class())
    models = tmp_path / "loads"
    models.mkdir()
    job = q.job(tasks.UseModel(model=str(models), x=1), "j")
    q.put([job])
    worker = q.work(_class())

    if kind == "v1":
        [commit] = tasks.COMMITS
        assert "pool" not in commit["context"]
    [message] = _messages(tmp_path / "store", "pool-facts")
    facts = message["pool"]
    assert facts["scheme"] == "pool-facts@1" and facts["worker"] == worker.token
    assert message["checksum"] == job.checksum and message["attempt"] == job.attempt
    assert facts["residency"][0]["state"] == "cold"
    if kind == "pickle":
        result = store.lookup(job.checksum, job.context, task=job.task)
        assert result is not None and "pool" not in result.context
        assert result.runtime is not None and result.runtime.pool == facts


# -- a pool that outlives a run ------------------------------------------------------------------


def _local_pool(tmp_path, tasks, store, groups, **kwargs):
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.substrate import LocalSubstrate

    substrate = LocalSubstrate()
    kwargs.setdefault("settings", _settings(idle_polls=100_000))
    pool = WarmPool(
        tmp_path / "pool",
        [_class()],
        substrate,
        store=store,
        providers=["_pydra_pool_context_tasks:Counting"],
        **kwargs,
    )
    return pool, substrate


def _track(substrate, groups) -> None:
    for handle in substrate.handles:
        if handle.process is not None and handle.process.pid not in groups:
            groups.append(handle.process.pid)


def test_two_runs_on_a_caller_owned_pool_load_once(tmp_path, tasks, groups):
    """Two successive ``sub(task)`` runs on one caller-owned `WarmPool` use one resident load: neither
    the run nor the `Submitter`'s close closes the pool, which its caller closes."""
    store = _store(tmp_path)
    pool, substrate = _local_pool(tmp_path, tasks, store, groups)
    models = tmp_path / "loads"
    models.mkdir()
    try:
        with Submitter(store=store, worker=pool, providers=[tasks.Counting()]) as sub:
            first = sub(tasks.UseModel(model=str(models), x=1))
            _track(substrate, groups)
            assert first.outputs.out == 1
            assert not pool._closed
            second = sub(tasks.UseModels(model=str(models), xs=[2, 3]))
            _track(substrate, groups)
            assert second.outputs.out == [2, 3]
        assert not pool._closed
        assert len(list(models.iterdir())) == 1
        assert len(substrate.handles) == 1
        assert substrate.handles[0].process.poll() is None  # still resident
    finally:
        pool.close()
    _track(substrate, groups)
    _wait(
        lambda: all(h.process.poll() is not None for h in substrate.handles),
        "the pool's workers to stop when its caller closes it",
    )


def test_max_concurrent_does_not_cap_a_pool(tmp_path, tasks, groups, monkeypatch):
    """``Submitter(max_concurrent=1)`` hands a `WarmPool` every ready job of a round in one
    ``submit``: the cap bounds a scheduler worker's submissions, not a pool's jobs"""
    store = _store(tmp_path)
    pool, substrate = _local_pool(tmp_path, tasks, store, groups)
    models = tmp_path / "loads"
    models.mkdir()
    submitted = []
    real = type(pool).submit

    def submit(self, jobs):
        submitted.append(len(jobs))
        return real(self, jobs)

    monkeypatch.setattr(type(pool), "submit", submit)
    try:
        with Submitter(
            store=store, worker=pool, providers=[tasks.Counting()], max_concurrent=1
        ) as sub:
            result = sub(tasks.UseModels(model=str(models), xs=[1, 2, 3]))
            _track(substrate, groups)
        assert result.outputs.out == [1, 2, 3]
        assert submitted == [3]
    finally:
        pool.close()
    _track(substrate, groups)
    _wait(
        lambda: all(h.process.poll() is not None for h in substrate.handles),
        "the pool's workers to stop when its caller closes it",
    )


def test_pool_by_name_closed_on_submitter_exit(tmp_path, tasks, groups):
    """A pool the `Submitter` makes by name stays open between runs and is closed with the `Submitter`,
    which stops its workers."""
    from pydra.workers.pool.substrate import LocalSubstrate

    store = _store(tmp_path)
    substrate = LocalSubstrate()
    models = tmp_path / "loads"
    models.mkdir()
    with Submitter(
        store=store,
        worker="pool",
        root=tmp_path / "pool",
        classes=[_class()],
        substrate=substrate,
        settings=_settings(idle_polls=100_000),
        providers=[tasks.Counting()],
    ) as sub:
        pool = sub.worker.executor
        assert sub(tasks.UseModel(model=str(models), x=1)).outputs.out == 1
        _track(substrate, groups)
        assert not pool._closed
        assert sub(tasks.UseModel(model=str(models), x=2)).outputs.out == 2
        _track(substrate, groups)
        assert not pool._closed
        assert len(list(models.iterdir())) == 1
    assert pool._closed
    assert (tmp_path / "pool" / "coordinator" / str(pool.epoch) / "ended").exists()
    _wait(
        lambda: all(h.process.poll() is not None for h in substrate.handles),
        "the pool's workers to stop when the Submitter closes",
    )


def test_drive_leaves_an_executor_instance_open(tmp_path, tasks, groups):
    """``drive`` never closes an executor it was handed: its owner does."""
    import asyncio

    from pydra.engine.drivers import drive
    from pydra.engine.runner import GraphRunner

    store = _store(tmp_path)
    pool, substrate = _local_pool(tmp_path, tasks, store, groups)
    try:
        sub = Submitter(store=store, worker="debug")
        runner = GraphRunner(
            tasks.UseModels(model=str(tmp_path), xs=[1]),
            store,
            on_error="raise_at_end",
            submitter=sub,
        )
        sub.providers = (tasks.Counting(),)
        asyncio.run(
            drive(
                runner,
                {"pool": pool},
                lambda job: "pool",
                on_error="raise_at_end",
            )
        )
        _track(substrate, groups)
        assert not runner.errors and not runner.not_run
        assert not pool._closed
        sub.close()
    finally:
        pool.close()
    assert pool._closed


def test_reload_after_a_resource_fault_is_cold(tmp_path, tasks, queue_of):
    """A call whose residency was reloaded after a resource fault is recorded cold, with no call since
    the load: the facts follow the load that really happened."""
    identity = PydraIdentityV1(recorded_only={"pool"})
    store = _store(tmp_path, identity, on_commit=tasks.record_commit)
    q = queue_of(store)
    q.classes(_class(), _class("big", mem="64G"))
    models = tmp_path / "loads"
    models.mkdir()
    jobs = [q.job(tasks.MaybeOom(model=str(models), x=x), f"j{x}") for x in (1, 2, 3)]
    q.put(jobs)
    q.work(_class())

    assert (
        len(list(models.iterdir())) == 2
    )  # the first load, and the reload after the fault
    facts = {c["checksum"]: c["context"]["pool"]["residency"][0] for c in tasks.COMMITS}
    first, third = facts[jobs[0].checksum], facts[jobs[2].checksum]
    assert (first["state"], first["since_load"]) == ("cold", 0)
    assert (third["state"], third["since_load"]) == ("cold", 0)


def test_losses_survived_counts_takeovers():
    """``losses_survived`` counts the job's takeover generations; a job still on its first put lists
    nothing."""
    from pydra.workers.pool.worker import Worker

    listed = []

    class Pool:
        def generations(self, checksum, fingerprint):
            listed.append((checksum, fingerprint))
            return [
                (0, {"cause": "put"}),
                (1, {"cause": "takeover"}),
                (2, {"cause": "ladder"}),
                (3, {"cause": "takeover"}),
            ]

    class Entry:
        key = ("ck", "fp")

    class Claim:
        def __init__(self, cause):
            self.cause = cause

        def current(self, entry):
            return 3, {"cause": self.cause}

    worker = Worker.__new__(Worker)
    worker.pool = Pool()
    assert worker._losses(Claim("takeover"), Entry()) == 2
    assert listed == [("ck", "fp")]
    assert worker._losses(Claim("put"), Entry()) == 0
    assert listed == [("ck", "fp")]


def test_submitter_takes_the_executors_store(tmp_path, tasks):
    """A `Submitter` given an executor uses its store; another store object, or another named store,
    is refused."""
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.substrate import ExternalSubstrate

    store = _store(tmp_path)
    pool = WarmPool(
        tmp_path / "pool",
        [_class()],
        ExternalSubstrate(),
        store=store,
        settings=_settings(),
    )
    try:
        with Submitter(worker=pool) as sub:
            assert sub.store is store
        (tmp_path / "other").mkdir()
        with pytest.raises(ValueError, match="executor"):
            Submitter(worker=pool, store=_store(tmp_path / "other"))
        with pytest.raises(ValueError, match="executor"):
            Submitter(worker=pool, store="pickle")
        assert not pool._closed
    finally:
        pool.close()


def test_failed_construction_closes_a_pool_made_by_name(tmp_path, tasks):
    """A `Submitter` whose construction fails after it made a pool by name closes that pool, so its
    coordinator lease is not left held; a bad ``on_error`` is refused before any pool is made.
    """
    from pydra.workers.pool.substrate import ExternalSubstrate

    store = _store(tmp_path)
    args = dict(
        store=store,
        worker="pool",
        root=tmp_path / "pool",
        classes=[_class()],
        substrate=ExternalSubstrate(),
        settings=_settings(),
    )
    with pytest.raises(ValueError, match="on_error"):
        Submitter(on_error="bogus", **args)
    assert not (tmp_path / "pool").exists()
    with pytest.raises(ValueError, match="No worker matches"):
        Submitter(workers={"other": "no-such-worker"}, **args)
    assert (tmp_path / "pool" / "coordinator" / "1" / "ended").exists()


def _in_thread(outcome: dict, name: str, call) -> "threading.Thread":
    import threading

    def run() -> None:
        try:
            outcome[name] = call()
        except BaseException as exc:  # reported to the test
            outcome[name] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _sub(store, pool, tasks, task, **kwargs):
    kwargs.setdefault("worker", pool)
    with Submitter(store=store, providers=[tasks.Counting()], **kwargs) as sub:
        return sub(task).outputs


def test_another_run_between_a_runs_pool_rounds(tmp_path, tasks, groups):
    """Run A puts A1 on the pool, then M elsewhere, then A2 on the pool. While A is in M, with nothing
    open on the pool, run B binds the pool and completes; A2's outcome still goes to A, which
    completes with its own results."""
    store = _store(tmp_path)
    pool, substrate = _local_pool(tmp_path, tasks, store, groups)
    models, where = tmp_path / "loads", tmp_path / "where"
    models.mkdir()
    where.mkdir()
    outcome: dict = {}
    threads = []
    try:
        threads.append(
            _in_thread(
                outcome,
                "A",
                lambda: _sub(
                    store,
                    pool,
                    tasks,
                    tasks.Interleaved(model=str(models), where=str(where)),
                    workers={"pool": pool},
                    router=lambda job: "pool" if job.name in ("A1", "A2") else None,
                    worker="debug",
                ),
            )
        )
        _wait(lambda: (where / "held").exists(), "run A to reach M")
        _track(substrate, groups)
        b = _sub(store, pool, tasks, tasks.UseModels(model=str(models), xs=[5, 6]))
        assert b.out == [5, 6]
        (where / "go").touch()
        threads[0].join(BOUND)
        assert not threads[0].is_alive(), "run A never received A2's outcome"
        a = outcome["A"]
        assert not isinstance(a, BaseException), a
        assert (a.a1, a.a2) == (1, 2)
        assert len(list(models.iterdir())) == 1
    finally:
        (where / "go").touch()
        pool.close()
        for thread in threads:
            thread.join(BOUND)


def test_two_runs_bind_one_pool_at_once(tmp_path, tasks, groups):
    """Two threads each run a workflow on one pool at the same moment (a barrier): both complete
    with their own results, a job they share included."""
    import threading

    store = _store(tmp_path)
    pool, substrate = _local_pool(tmp_path, tasks, store, groups)
    models = tmp_path / "loads"
    models.mkdir()
    barrier = threading.Barrier(2)
    outcome: dict = {}

    def run(xs):
        barrier.wait(BOUND)
        return _sub(store, pool, tasks, tasks.UseModels(model=str(models), xs=xs)).out

    threads = [
        _in_thread(outcome, "A", lambda: run([1, 2, 100])),
        _in_thread(outcome, "B", lambda: run([3, 4, 100])),
    ]
    try:
        for thread in threads:
            thread.join(BOUND)
            _track(substrate, groups)
        assert not any(t.is_alive() for t in threads)
        assert outcome == {"A": [1, 2, 100], "B": [3, 4, 100]}
    finally:
        pool.close()
        for thread in threads:
            thread.join(BOUND)


def test_two_drives_on_one_event_loop(tmp_path, tasks, groups):
    """Two runs driven on one event loop (``asyncio.gather``), each over three pool rounds: each
    gets its own outcomes and completes; afterwards both are served from the store with no load.
    """
    import asyncio

    from pydra.engine.drivers import drive
    from pydra.engine.runner import GraphRunner

    store = _store(tmp_path)
    pool, substrate = _local_pool(tmp_path, tasks, store, groups)
    models = tmp_path / "loads"
    models.mkdir()
    sub = Submitter(store=store, worker="debug", providers=[tasks.Counting()])
    try:
        runners = [
            GraphRunner(
                tasks.ThreeRounds(model=str(models), x=x),
                store,
                on_error="raise_at_end",
                submitter=sub,
            )
            for x in (10, 20)
        ]

        async def both():
            await asyncio.wait_for(
                asyncio.gather(
                    *(
                        drive(
                            runner,
                            {"pool": pool},
                            lambda job: "pool",
                            on_error="raise_at_end",
                        )
                        for runner in runners
                    )
                ),
                BOUND,
            )

        # a binding made in the calling thread first (as an earlier run, or a test, leaves one):
        # the tasks inherit its token, and must still be runs of their own
        placeholder = lambda job, outcome: None  # noqa: E731
        pool.bind(placeholder)
        asyncio.run(both())
        pool.unbind(placeholder)
        _track(substrate, groups)
        for runner in runners:
            assert not runner.errors and not runner.not_run
        loads = len(list(models.iterdir()))
        assert loads == 1
        with Submitter(store=store, worker="debug", providers=[tasks.Counting()]) as s:
            assert s(tasks.ThreeRounds(model=str(models), x=10)).outputs.out == 11
            assert s(tasks.ThreeRounds(model=str(models), x=20)).outputs.out == 21
        assert len(list(models.iterdir())) == loads
        assert not pool._reports and not pool._open
    finally:
        pool.close()
        sub.close()


def test_a_twin_in_another_run_gets_the_result(tmp_path, tasks, groups):
    """Run B submits a job run A has open on the pool (the same checksum and slot): it waits on A's
    ticket, and both runs get the one result."""
    store = _store(tmp_path)
    pool, substrate = _local_pool(tmp_path, tasks, store, groups)
    gate = tmp_path / "gate"
    gate.mkdir()
    outcome: dict = {}
    threads = []
    try:
        threads.append(
            _in_thread(
                outcome,
                "A",
                lambda: _sub(store, pool, tasks, tasks.Gate(where=str(gate))),
            )
        )
        _wait(lambda: (gate / "first").exists(), "run A's job to start")
        _track(substrate, groups)
        threads.append(
            _in_thread(
                outcome,
                "B",
                lambda: _sub(store, pool, tasks, tasks.Gate(where=str(gate))),
            )
        )
        _wait(
            lambda: any(j.twin_of is not None for j in list(pool._open.values())),
            "run B's twin to be open on the pool",
        )
        (gate / "go").touch()
        for thread in threads:
            thread.join(BOUND)
        assert not any(t.is_alive() for t in threads)
        assert outcome["A"].out == "first" and outcome["B"].out == "first"
    finally:
        (gate / "go").touch()
        pool.close()
        for thread in threads:
            thread.join(BOUND)


def test_one_runs_cancel_leaves_the_other_run_alone(tmp_path, tasks, groups):
    """A run that cancels its jobs, one of them the twin another run waits on, is reported
    ``NotRun`` for its own jobs only; the other run completes with the result."""
    from pydra.engine.runner import GraphRunner, NotRun

    store = _store(tmp_path)
    pool, substrate = _local_pool(tmp_path, tasks, store, groups)
    gate = tmp_path / "gate"
    gate.mkdir()
    outcome: dict = {}
    threads = []
    sub = Submitter(store=store, worker="debug")
    try:
        runner = GraphRunner(
            tasks.GateAndEcho(where=str(gate)),
            store,
            on_error="raise_at_end",
            submitter=sub,
        )
        reported = []
        pool.bind(lambda job, o: reported.append((job.name, o)))
        handle = pool.submit(runner.ready().dispatch)
        _wait(
            lambda: (gate / "first").exists(), "the cancelled run's gate job to start"
        )
        _track(substrate, groups)
        threads.append(
            _in_thread(
                outcome,
                "A",
                lambda: _sub(store, pool, tasks, tasks.Gate(where=str(gate))),
            )
        )
        _wait(
            lambda: any(j.twin_of is not None for j in list(pool._open.values())),
            "run A's twin to be open on the pool",
        )
        pool.cancel(handle)
        # E may have settled already; G, open under a live claim, is cancelled for this run only
        outcomes = dict(reported)
        assert set(outcomes) == {"E", "G"}
        assert outcomes["G"] == NotRun("cancelled", held=False)
        (gate / "go").touch()
        threads[0].join(BOUND)
        assert not threads[0].is_alive()
        assert outcome["A"].out == "first"
        assert len(reported) == 2
        runner.close()
    finally:
        (gate / "go").touch()
        pool.close()
        sub.close()
        for thread in threads:
            thread.join(BOUND)


def test_a_failing_audit_never_skips_the_commit(tmp_path, tasks, queue_of):
    """A messenger that fails on the pool-facts message leaves the facts on the result and the commit
    in place."""
    identity = PydraIdentityV1(recorded_only={"pool"})
    store = _store(tmp_path, identity, on_commit=tasks.record_commit)
    q = queue_of(store)
    q.submitter.audit.messengers = [tasks.RefusesPoolFacts()]
    q.classes(_class())
    models = tmp_path / "loads"
    models.mkdir()
    job = q.job(tasks.UseModel(model=str(models), x=1), "j")
    q.put([job])
    q.work(_class())
    [commit] = tasks.COMMITS
    assert commit["checksum"] == job.checksum and "pool" in commit["context"]


def test_an_unbound_submit_needs_exactly_one_binding(tmp_path, tasks):
    """A submit from a thread that never bound goes to the one bound run; with two runs bound it
    is refused, never sent to either."""
    import threading

    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.substrate import ExternalSubstrate

    pool = WarmPool(
        tmp_path / "pool",
        [_class()],
        ExternalSubstrate(),
        store=_store(tmp_path),
        settings=_settings(),
    )
    outcome: dict = {}

    def elsewhere(name, call):
        thread = threading.Thread(
            target=lambda: outcome.__setitem__(name, _try(call)), daemon=True
        )
        thread.start()
        thread.join(BOUND)

    try:
        pool.bind(lambda job, o: None)
        elsewhere("sole", lambda: pool.submit([]))
        assert not isinstance(outcome["sole"], BaseException)
        elsewhere("other", lambda: pool.bind(lambda job, o: None))
        elsewhere("two", lambda: pool.submit([]))
        assert isinstance(outcome["two"], RuntimeError)
        assert "2 runs are bound" in str(outcome["two"])
    finally:
        pool.close()


def _try(call):
    try:
        return call()
    except BaseException as exc:  # returned to the test
        return exc


def test_pool_facts_follow_recorded_only_never_an_identity_name():
    from pydra.workers.pool.pool import PoolFactsWouldSplitSlots, pool_facts_recorded

    class Store:
        identity = PydraIdentityV1()

    assert "pool" not in Store.identity.recorded_only
    with pytest.raises(PoolFactsWouldSplitSlots):
        pool_facts_recorded(Store())
