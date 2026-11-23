"""The warm worker of a pool (RFC 0005 sections 4.3, 5.3-5.5, W5).

The worker's contract (idle exit, quantum yield, gathering once warm, resident memory out of the per-job samples, device memory once per claim, settling before
the next call, a reclaimed claim logged, a claim kept through a long job, SIGTERM releasing claims, a
lost claim abandoning its call). The rest are the W5 rows of RFC 0005 section 11.

Tasks live in a module written to a temporary directory and put on ``sys.path`` and ``PYTHONPATH``, so
a worker in another process imports them as this one does (a wheel install carries no test modules).
In-process tests reach a running task through that module's ``HOOK`` table.
"""

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import attrs
import pytest

from pydra.engine.job import Job
from pydra.engine.lease import DirectoryLease, LeaseLost, LeaseSettings
from pydra.engine.submitter import Submitter
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.codec import PrivatePickle, write_payload
from pydra.workers.pool.storage import Claim, PoolRoot, Ticket, TicketJob, queue_label

POSIX = sys.platform != "win32"
# Acknowledgements are due within one renewal interval (1.6 s), which loaded CI runners meet.
LEASE = LeaseSettings(
    stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
)
BOUND = 60.0  # seconds any wait on another process may take before the test fails

TASKS = """
import os, sys, threading, time, subprocess
from pathlib import Path
from pydra.compose import python
from pydra.engine.procgroup import ProcessProvider
from pydra.engine.resident import Provided

HOOK = {}


def _hook(name, *args):
    fn = HOOK.get(name)
    return None if fn is None else fn(*args)


@python.define
def Echo(x: int) -> int:
    _hook("echo", x)
    return x


@python.define
def Sleep(seconds: float, flag: str = "") -> float:
    if flag:
        Path(flag).touch()
    _hook("sleep", seconds)
    time.sleep(seconds)
    return seconds


@python.define
def HoldGil(seconds: int) -> int:
    import ctypes
    if sys.platform == "win32":
        ctypes.PyDLL("kernel32").Sleep(seconds * 1000)
    else:
        ctypes.PyDLL(None).sleep(seconds)
    return seconds


@python.define
def Wedge(flag: str) -> int:
    Path(flag).touch()
    threading.Event().wait()
    return 0


@python.define
def Child(seconds: int, marker: str) -> int:
    if os.path.exists(marker):
        Path(marker + ".started").touch()
        subprocess.run(["sleep", str(seconds)], check=True)
    return seconds


@python.define
def Fail(x: int) -> int:
    raise RuntimeError("this task always fails")


LOADS = []


class SleepServer(ProcessProvider):
    # a provider whose state is a child process that sleeps
    name = "sleepserver"

    def argv(self, spec, device):
        return [sys.executable, "-c", "import time; time.sleep(600)"]

    def load(self, spec, device):
        LOADS.append(("sleepserver", device))
        return super().load(spec, device)


class Recorder:
    # an in-process provider that records the device it loads on
    name = "recorder"

    def load(self, spec, device):
        LOADS.append(("recorder", device))
        return spec

    def close(self, obj):
        pass

    def footprint(self, spec):
        return None

    def measure(self, obj):
        return None

    def version(self):
        return "1"


class Broken:
    # a provider whose load always fails, as a missing model file does
    name = "broken"

    def load(self, spec, device):
        LOADS.append(("broken", device))
        raise FileNotFoundError(2, "no such model", spec)

    def close(self, obj):
        pass

    def footprint(self, spec):
        return None

    def measure(self, obj):
        return None

    def version(self):
        return "1"


@python.define
def UseBroken(model: Provided[object, "broken"], x: int) -> int:
    return x


@python.define
def UseServer(model: Provided[object, "sleepserver"], x: int) -> int:
    _hook("server", model, x)
    if model.group.proc.poll() is not None:
        raise RuntimeError("the provider's process is dead")
    return x


@python.define
def UseRecorder(model: Provided[str, "recorder"], x: int) -> int:
    return x


@python.define
def FailOnce(marker: str) -> str:
    if not os.path.exists(marker):
        Path(marker).touch()
        raise ValueError("the first attempt fails")
    return marker
"""


@pytest.fixture(scope="session")
def tasks(tmp_path_factory):
    """The task module, importable here and in workers started from here."""
    where = tmp_path_factory.mktemp("pool_tasks")
    (where / "_pydra_pool_tasks.py").write_text(TASKS)
    sys.path.insert(0, str(where))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(where), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    import _pydra_pool_tasks

    return _pydra_pool_tasks


@pytest.fixture(autouse=True)
def release_claims(monkeypatch, tasks):
    """Release every claim a test leaves held and clear the task hooks, pass or fail."""
    made = []
    init = Claim.__init__

    def recording(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(Claim, "__init__", recording)
    yield
    tasks.HOOK.clear()
    for claim in made:
        claim.lease.__exit__(None, None, None)


@pytest.fixture
def processes():
    """Worker processes a test starts, each in its own session; killed with their group at the end."""
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


def _class(name="cpu", **kwargs) -> WorkerClass:
    fields = dict(name=name, device="cpu", quantum_jobs=2, quantum_s=600.0)
    fields.update(kwargs)
    return WorkerClass(**fields)


class Pool:
    """A pool, a store and the jobs put on the pool, for one test."""

    def __init__(self, tmp_path: Path):
        self.pool = PoolRoot.open(tmp_path / "pool", LEASE, probe=False)
        self.submitter = Submitter(cache_root=tmp_path / "store", worker="debug")
        self.store = self.submitter.store
        self.jobs: dict[str, Job] = {}
        self.tickets: list[tuple[str, str]] = []

    @property
    def path(self) -> Path:
        return self.pool.path

    def job(self, task, name: str, *, resources=None) -> Job:
        job = Job(task=task, submitter=self.submitter, name=name, resources=resources)
        self.jobs[name] = job
        return job

    def entry(self, job: Job) -> TicketJob:
        fingerprint = self.store.identity.fingerprint(job.context)
        g = self.pool._next_generation((job.checksum, fingerprint))
        job.attempt = f"{self.pool.pool_id}.{g}"
        payload = write_payload(PrivatePickle(), job, self.path)
        return TicketJob(
            checksum=job.checksum,
            fingerprint=fingerprint,
            payload=payload,
            attempt=job.attempt,
            runner_attempt=0,
            units=1.0,
        )

    def put(self, jobs, queue="cpu", **kwargs) -> str:
        ticket = Ticket(
            queue=queue, node="wf/node", jobs=[self.entry(j) for j in jobs], **kwargs
        )
        digest = self.pool.put(queue, ticket)
        self.tickets.append((queue, digest))
        return digest

    def key(self, job: Job) -> tuple[str, str]:
        return job.checksum, self.store.identity.fingerprint(job.context)

    def outcomes(self, queue="cpu", digest=None) -> list[dict]:
        found = []
        for q, d in self.tickets:
            if q == queue and (digest is None or d == digest):
                found.extend(self.pool.outcomes(q, d).values())
        return found

    def kinds(self, queue="cpu") -> list[str]:
        return sorted(o["outcome"] for o in self.outcomes(queue))

    def generations(self, job: Job) -> list[dict]:
        return [r for _, r in self.pool.generations(*self.key(job))]

    def found(self, job: Job) -> bool:
        """Whether the store holds a result for the job."""
        return getattr(self.store.find(job), "result", None) is not None

    def claim_dirs(self, queue="cpu") -> list[Path]:
        claims = self.path / "queues" / queue_label(queue) / "claims"
        return (
            [d for d in claims.glob("*.lease/*") if d.is_dir()]
            if claims.exists()
            else []
        )


@pytest.fixture
def pool(tmp_path):
    p = Pool(tmp_path)
    yield p
    p.pool.close()


def _run(pool: Pool, cls=None, **kwargs):
    from pydra.workers.pool.worker import Worker

    settings = kwargs.pop("settings", None) or _settings()
    kwargs.setdefault("device_memory", lambda: None)
    worker = Worker(pool.path, cls or _class(), settings=settings, **kwargs)
    return worker, worker.run()


def _wait(predicate, what: str, bound: float = BOUND) -> None:
    deadline = time.monotonic() + bound
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _start(
    pool: Pool, processes, tmp_path, cls=None, settings=None
) -> subprocess.Popen:
    """Start ``pydracli pool worker`` on the pool, in its own session."""
    cls = cls or _class()
    pool.pool.put_class(cls.name, _record(cls))
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


def _record(cls: WorkerClass) -> dict:
    from pydra.workers.pool.worker import class_record

    return class_record(cls)


# -- the worker's contract ---------------------------------------------------------------------


@pytest.mark.parametrize("polls", [2, 5])
def test_idle_exit_after_idle_polls(pool, polls):
    """With nothing to claim, a worker polls ``idle_polls`` times and exits 0."""
    worker, code = _run(pool, settings=_settings(idle_polls=polls))
    assert code == 0 and worker.polls == polls


def test_quantum_yields_to_older_queue(pool, tasks):
    """After a quantum of 2 jobs on ``cpu``, the worker claims the older ticket waiting on ``cpu#b``,
    then goes back."""
    pool.put([pool.job(tasks.Echo(x=0), "b")], queue="cpu#b")
    for i in range(1, 5):
        pool.put([pool.job(tasks.Echo(x=i), f"a{i}")])
    other = pool.pool.claim("cpu#b")  # held, so cpu is served first
    assert other is not None

    def release_once(x):
        if other.lease.held:
            other.release()
            # a ticket found under a live claim is not judged again for one renewal interval
            time.sleep(LEASE.renew_interval)

    tasks.HOOK["echo"] = release_once
    worker, code = _run(pool)
    assert code == 0
    assert [q for q, _ in worker.served] == ["cpu", "cpu", "cpu#b", "cpu", "cpu"]
    assert pool.kinds() == ["done"] * 4 and pool.kinds("cpu#b") == ["done"]


def test_gathers_one_job_tickets_once_warm(pool, tasks):
    """A cold queue's first ticket is claimed alone; once warm one-job tickets are gathered up to the
    quantum, and a 4-job ticket is never gathered."""
    for i in range(3):
        pool.put([pool.job(tasks.Echo(x=i), f"s{i}")])
    for t in range(2):
        pool.put([pool.job(tasks.Echo(x=10 + 4 * t + i), f"m{t}{i}") for i in range(4)])
    worker, code = _run(pool, _class(quantum_jobs=4))
    assert code == 0
    assert worker.claims == [1, 2, 4, 4]
    samples = [o["sample"] for o in pool.outcomes()]
    assert sorted(s["cold"] for s in samples) == [False] * 10 + [True]


def test_samples_leave_resident_memory_out_of_the_per_job_figure(pool, tasks):
    """A stand-in model of 1000 bytes loaded on the first job, and 10 bytes per job while it runs: the
    warm samples hold the model as resident and their peaks add only the job's bytes."""
    for i in range(3):
        pool.put([pool.job(tasks.Echo(x=i), f"t{i}")])
    memory = {"current": 100, "peak": 100}

    def standing_in(x):
        if memory["current"] == 100:
            memory["current"] += 1000
        memory["peak"] = max(memory["peak"], memory["current"] + 10)

    tasks.HOOK["echo"] = standing_in
    _, code = _run(pool, memory=lambda: (memory["current"], memory["peak"]))
    assert code == 0
    samples = [o["sample"] for o in pool.outcomes()]
    warm = [s for s in samples if not s["cold"]]
    assert min(s["resident_bytes"] for s in warm) == 1100
    assert max(s["peak_bytes"] for s in samples) == 1110
    assert {s["baseline_bytes"] for s in samples} == {100}


def test_device_memory_read_once_per_claim_on_a_cuda_class(pool, tasks):
    """On a CUDA class the device memory is read once before the first claim (the baseline) and once
    per claim, never per job."""
    cls = _class("gpu", device="cuda", gpus=1, gpu_mem="16G", quantum_jobs=4)
    for i in range(4):
        pool.put([pool.job(tasks.Echo(x=i), f"t{i}")], queue="gpu")
    reads = iter([0, 1000, 1010])
    _, code = _run(
        pool,
        cls,
        hardware=_gpu_hardware(16 << 30),
        device_memory=lambda: next(reads),
    )
    assert code == 0 and next(reads, None) is None
    samples = [o["sample"] for o in pool.outcomes("gpu")]
    assert len(samples) == 4
    assert {s["device_baseline_bytes"] for s in samples} == {0}
    assert sorted(s["device_bytes"] for s in samples) == [1000, 1010, 1010, 1010]


def test_a_jobs_outcome_is_settled_before_the_next_call(pool, tasks):
    """Four gathered one-job tickets: each job's outcome and settled mark exist when the next job
    starts."""
    jobs = [pool.job(tasks.Echo(x=i), f"t{i}") for i in range(4)]
    for job in jobs:
        pool.put([job])
    seen = []
    tasks.HOOK["echo"] = lambda x: seen.append(
        (x, len(pool.outcomes()), sum(pool.pool.settled("cpu").values()))
    )
    worker, code = _run(pool, _class(quantum_jobs=4))
    assert code == 0 and worker.claims == [1, 3]
    assert seen == [(i, i, i) for i in range(4)]


def test_reclaimed_logged_before_claimed(pool, tasks, caplog):
    """A dead holder's ticket logs ``reclaimed`` (the lost token and its age) before ``claimed``; the
    other ticket logs only ``claimed``."""
    a, b = pool.job(tasks.Echo(x=0), "a"), pool.job(tasks.Echo(x=1), "b")
    dead = pool.put([a])
    pool.put([b])
    held = pool.pool.claim("cpu", beat="external")
    assert held is not None and held.digest == dead
    token = held.lease.token_dir.name
    _age(held.lease.lockfile)
    caplog.set_level(logging.INFO, logger="pydra.pool")
    _, code = _run(pool)
    assert code == 0
    events = [
        (r.event, r.ticket)
        for r in caplog.records
        if getattr(r, "event", None) in ("claimed", "reclaimed")
    ]
    assert [e for e, _ in events] == ["reclaimed", "claimed", "claimed"]
    assert events[0][1] == dead
    reclaimed = next(
        r for r in caplog.records if getattr(r, "event", "") == "reclaimed"
    )
    assert reclaimed.tokens == [token] and reclaimed.age > LEASE.stale_after
    assert pool.kinds() == ["done", "done"]
    assert pool.generations(a)[-1]["cause"] == "takeover"


def test_claim_kept_through_a_long_job(pool, tasks):
    """A job running past twice the stale bound keeps its claim: a thief's takes all fail and the
    worker settles it."""
    job = pool.job(tasks.Sleep(seconds=2.5 * LEASE.stale_after), "slow")
    digest = pool.put([job])
    stolen = []

    def thief(seconds):
        def steal():
            other = PoolRoot.open(pool.path, LEASE, create=False, probe=False)
            try:
                end = time.monotonic() + seconds - 0.5
                while time.monotonic() < end:
                    claim = other.take("cpu", digest, beat="external")
                    stolen.append(claim is not None)
                    if claim is not None:
                        claim.release()
                    time.sleep(0.25)
            finally:
                other.close()

        threading.Thread(target=steal, daemon=True).start()

    tasks.HOOK["sleep"] = thief
    _, code = _run(pool)
    assert code == 0
    assert len(stolen) > 10 and not any(stolen)
    [outcome] = pool.outcomes()
    assert outcome["outcome"] == "done" and pool.pool.losses("cpu", digest) == []


@pytest.mark.skipif(not POSIX, reason="SIGTERM to a process group")
def test_sigterm_settles_finished_and_releases_the_rest(
    pool, tasks, processes, tmp_path
):
    """On SIGTERM a worker settles the job that finished, releases the ticket whose call is running
    and exits 0; the released ticket is claimable at once."""
    flag = tmp_path / "slow-started"
    pool.put([pool.job(tasks.Echo(x=1), "fast")])
    slow = pool.put([pool.job(tasks.Sleep(seconds=600.0, flag=str(flag)), "slow")])
    proc = _start(pool, processes, tmp_path, settings=_settings(drain_s=0.5))
    _wait(flag.exists, "the slow job to start")
    os.killpg(proc.pid, signal.SIGTERM)
    assert proc.wait(BOUND) == 0
    assert pool.kinds() == ["done"]
    again = pool.pool.take("cpu", slow, beat="external")
    assert again is not None and not again.took_over
    again.release()


def test_lost_claim_abandons_its_call(pool, tasks):
    """A claim broken in its first job's call: nothing is published or settled, and the ticket's other
    job is never run; the ticket is claimable by another worker."""
    first, second = pool.job(tasks.Echo(x=1), "one"), pool.job(tasks.Echo(x=2), "two")
    digest = pool.put([first, second])
    ran = []

    def broken(x):
        ran.append(x)
        for d in pool.claim_dirs():
            _remove_tree(d)

    tasks.HOOK["echo"] = broken
    _, code = _run(pool)
    assert code == 0 and ran == [1]
    assert pool.outcomes() == [] and not pool.found(first)
    assert [r.get("worker") for r in _started(pool, second)] == []
    taken = []  # the emptied lease is taken once it is judged stale
    _wait(
        lambda: (
            taken.append(pool.pool.take("cpu", digest, beat="external")) or taken[-1]
        ),
        "the ticket to be claimable",
    )
    taken[-1].release()


# -- RFC 0005 section 11, W5 ------------------------------------------------------------------


def test_gil_bound_call_keeps_claim(pool, tasks, tmp_path):
    """A task holding the GIL past the stale bound (well over three renewal intervals) keeps its claim:
    the beater beats it from another process, a rival process cannot take it over, and the worker
    settles the job with no loss recorded."""
    seconds = int(1.5 * LEASE.stale_after)
    digest = pool.put([pool.job(tasks.HoldGil(seconds=seconds), "gil")])
    claims = pool.path / "queues" / queue_label("cpu") / "claims"
    rival = subprocess.Popen(
        [
            sys.executable,
            "-c",
            RIVAL,
            str(pool.path),
            digest,
            str(claims),
            str(seconds),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        _, code = _run(pool)
        out, _ = rival.communicate(timeout=BOUND)
    finally:
        rival.kill()
        rival.wait()
    assert code == 0
    report = json.loads(out)
    assert report["tries"] >= 1 and report["stolen"] == 0
    assert pool.kinds() == ["done"] and pool.pool.losses("cpu", digest) == []


RIVAL = """
import json, sys, time
from pathlib import Path
from pydra.engine.lease import LeaseSettings
from pydra.workers.pool.storage import PoolRoot
root, digest, claims, seconds = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
pool = PoolRoot.open(root, LeaseSettings(stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0,
                     timeout=10.0), create=False, probe=False)
end = time.monotonic() + 60
while not list(claims.glob(digest + ".lease/*")) and time.monotonic() < end:
    time.sleep(0.05)
tries = stolen = 0
end = time.monotonic() + 60  # a bound only: it tries until the worker releases the claim
while list(claims.glob(digest + ".lease/*")) and time.monotonic() < end:
    claim = pool.take("cpu", digest, beat="external")
    tries += 1
    if claim is not None:
        stolen += not claim.resolved()  # a settled ticket taken after release is no steal
        claim.release()
    time.sleep(0.5)
print(json.dumps({"tries": tries, "stolen": stolen}))
"""


@pytest.mark.parametrize(
    "field", ["stale_after", "renew_fraction", "attribute_delay", "timeout"]
)
def test_settings_mismatch_any_field_refuses(pool, field):
    """A worker whose lease settings differ from the pool's in any of the four fields exits 3 before
    starting anything."""
    from pydra.workers.pool.worker import EXIT_SETTINGS

    changed = {
        "stale_after": 5.0,
        "renew_fraction": 0.3,
        "attribute_delay": 0.5,
        "timeout": 11.0,
    }[field]
    lease = attrs.evolve(LEASE, **{field: changed})
    worker, code = _run(pool, settings=_settings(lease=lease))
    assert code == EXIT_SETTINGS == 3
    assert worker.token is None and not (pool.path / "workers").exists()


def test_settings_differing_in_max_wait_accepted(pool):
    """``max_wait`` caps a waiter, not staleness: a worker differing only there starts."""
    _, code = _run(pool, settings=_settings(lease=attrs.evolve(LEASE, max_wait=3.0)))
    assert code == 0


def _gpu_hardware(memory: int, product: str = "NVIDIA H100 80GB HBM3", count: int = 1):
    from pydra.workers.pool.samples import Gpu, HardwareClass

    return lambda: HardwareClass(
        device="cuda",
        gpus=tuple(Gpu(product, memory, "9.0") for _ in range(count)),
        cpus=4,
    )


@pytest.mark.parametrize(
    "cls,hardware",
    [
        (dict(gpus=1, gpu_type="h100"), dict(product="NVIDIA A100-SXM4-40GB")),
        (dict(gpus=2, gpu_type="h100"), dict(count=1)),
        (dict(gpus=1), dict(count=0)),
    ],
    ids=["gpu-type", "gpu-count", "no-gpu"],
)
def test_mistyped_gpu_grant_fails_start(pool, cls, hardware):
    """A grant that does not match the class's ``gpu_type`` or ``gpus`` refuses to start (exit 4)
    with no ready mark and no claim."""
    from pydra.workers.pool.worker import EXIT_HARDWARE

    pool.put([pool.job(_echo(), "e")], queue="gpu")
    worker, code = _run(
        pool,
        _class("gpu", device="cuda", **cls),
        hardware=_gpu_hardware(80 << 30, **hardware),
    )
    assert code == EXIT_HARDWARE == 4
    assert pool.pool.worker_record(worker.token, "ready.json") is None
    assert pool.outcomes("gpu") == [] and worker.served == []
    assert not worker.beater_alive()


def test_ready_mark_fills_gpu_mem(pool):
    """A class with ``gpu_mem=None`` takes the device memory the worker reads; the ready mark records
    the hardware."""
    worker, code = _run(
        pool, _class("gpu", device="cuda", gpus=1), hardware=_gpu_hardware(16 << 30)
    )
    assert code == 0
    ready = pool.pool.worker_record(worker.token, "ready.json")
    assert ready["class"] == "gpu" and ready["gpu_mem"] == 16 << 30
    assert ready["hardware"]["gpus"][0]["product"] == "NVIDIA H100 80GB HBM3"


def test_wrong_hardware_rejudgement_withdraws_refused(pool, tasks):
    """On a ``gpu_mem=None`` class, a ticket declaring more device memory than the worker reads is
    withdrawn (reason ``refused``, the memory read) and nothing runs."""
    from pydra.utils.resources import ComputeResources

    job = pool.job(
        tasks.Echo(x=1),
        "big",
        resources=ComputeResources(gpus=1, gpu_mem="40G"),
    )
    digest = pool.put([job], queue="gpu")
    ran = []
    tasks.HOOK["echo"] = ran.append
    _, code = _run(
        pool, _class("gpu", device="cuda", gpus=1), hardware=_gpu_hardware(16 << 30)
    )
    assert code == 0 and ran == []
    [outcome] = pool.outcomes("gpu")
    assert (outcome["outcome"], outcome["reason"]) == ("withdrawn", "refused")
    assert outcome["memory"] == 16 << 30
    assert _started(pool, job) == [] and pool.pool.claim("gpu") is None
    assert pool.pool.losses("gpu", digest) == []


@pytest.mark.skipif(not POSIX, reason="the beater kills the worker with SIGKILL")
def test_wedged_call_times_out(pool, tasks, processes, tmp_path):
    """An in-process call that never returns past its deadline: the beater writes ``expired``, the
    worker (which cannot stop the call) is killed after ``timeout_grace``, and a taker places the job
    under a fresh attempt and types the timeout a resource fault, without running it."""
    from pydra.utils.resources import ComputeResources

    flag = tmp_path / "wedged"
    job = pool.job(
        tasks.Wedge(flag=str(flag)), "wedge", resources=ComputeResources(time=1)
    )
    digest = pool.put([job])
    proc = _start(pool, processes, tmp_path, settings=_settings(timeout_grace=1.0))
    assert proc.wait(BOUND) == -signal.SIGKILL
    assert flag.exists()
    [expired] = pool.outcomes()
    assert expired["outcome"] == "expired" and expired["attempt"].endswith(".0")
    flag.unlink()
    _wait(lambda: _stale(pool.claim_dirs()), "the unbeaten claim to go stale")
    _, code = _run(pool)  # takes the stale claim over
    assert code == 0
    gens = pool.generations(job)
    assert [g["cause"] for g in gens] == ["put", "takeover"]
    assert gens[1]["charged"] is False and gens[1]["prior"] == expired["attempt"]
    failed = pool.pool.read_outcome("cpu", digest, pool.key(job), gens[1]["attempt"])
    assert failed["outcome"] == "failed"
    assert failed["fault"]["kind"] == "resource"
    assert failed["fault"]["type"] == "ResourceExhausted"
    assert not flag.exists()  # never run again


@pytest.mark.skipif(not POSIX, reason="kills or stops the beater by signal")
@pytest.mark.parametrize("signum", ["SIGKILL", "SIGSTOP"])
def test_dead_beater_under_a_hung_call_expires_and_recycles(
    pool, tasks, processes, tmp_path, signum
):
    """A worker whose beater dies (killed) or hangs (stopped) while a call that never returns runs
    stops proving its claim, writes the call's ``expired`` outcomes itself once the deadline passes,
    and exits to be recycled; the claim goes stale and a taker types the timeout. It does not keep
    the claim alive forever."""
    flag = tmp_path / "wedged"
    job = pool.job(tasks.Wedge(flag=str(flag)), "wedge")
    digest = pool.put([job])
    proc = _start(
        pool,
        processes,
        tmp_path,
        settings=_settings(call_timeout=10.0, timeout_grace=1.0),
    )
    _wait(flag.exists, "the call to start")
    out = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True)
    beaters = [int(p) for p in out.stdout.split()]
    assert beaters, "the worker has a beater"
    for pid in beaters:
        os.kill(pid, getattr(signal, signum))
    try:
        assert proc.wait(BOUND) == 5  # EXIT_RECYCLE: neither hung nor killed
    finally:
        for pid in beaters:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    [expired] = pool.outcomes()
    assert expired["outcome"] == "expired" and expired["attempt"].endswith(".0")
    flag.unlink()
    _wait(lambda: _stale(pool.claim_dirs()), "the unproved claim to go stale")
    _, code = _run(pool)  # takes the stale claim over
    assert code == 0
    gens = pool.generations(job)
    assert [g["cause"] for g in gens] == ["put", "takeover"]
    failed = pool.pool.read_outcome("cpu", digest, pool.key(job), gens[1]["attempt"])
    assert (failed["outcome"], failed["fault"]["type"]) == (
        "failed",
        "ResourceExhausted",
    )
    assert not flag.exists()


def test_worker_skips_withdrawn_and_superseded(pool, tasks):
    """While the worker holds a ticket, a job whose newest generation names another ticket and a job
    with a ``withdrawn`` outcome are skipped: nothing started, nothing written."""
    a, b, c = (pool.job(tasks.Echo(x=i), n) for i, n in enumerate("abc"))
    digest = pool.put([a, b, c])
    label = queue_label("cpu")

    def interfere(x):
        if x != 0:
            return
        pool.pool._allocate(
            pool.key(b), ticket="f" * 64, queue=label, cause="replaced", prior=None
        )
        attempt = pool.generations(c)[-1]["attempt"]
        pool.pool.outcome(
            "cpu",
            digest,
            pool.key(c),
            attempt,
            {"outcome": "withdrawn", "reason": "cancel"},
            lease=_Proving(),
        )

    tasks.HOOK["echo"] = interfere
    _, code = _run(pool)
    assert code == 0
    assert (
        len(_started(pool, a)) == 1
        and _started(pool, b) == []
        and _started(pool, c) == []
    )
    found = {(o["checksum"], o["outcome"]) for o in pool.outcomes()}
    assert found == {(a.checksum, "done"), (c.checksum, "withdrawn")}


@pytest.mark.skipif(not POSIX, reason="SIGTERM to a process group")
def test_released_started_job_gets_fresh_attempt(pool, tasks, processes, tmp_path):
    """SIGTERM during a subprocess call stops the call and releases the ticket unsettled; the next
    claim finds the attempt started, writes a ``restart`` generation and runs in a new work
    directory."""
    marker = tmp_path / "marker"
    marker.touch()
    job = pool.job(tasks.Child(seconds=600, marker=str(marker)), "child")
    pool.put([job])
    proc = _start(pool, processes, tmp_path)
    _wait(Path(str(marker) + ".started").exists, "the subprocess call to start")
    os.killpg(proc.pid, signal.SIGTERM)
    assert proc.wait(BOUND) == 0
    assert pool.outcomes() == []
    marker.unlink()
    _, code = _run(pool)
    assert code == 0
    gens = pool.generations(job)
    assert [g["cause"] for g in gens] == ["put", "restart"]
    assert gens[1]["prior"] == gens[0]["attempt"] and gens[1]["charged"] is False
    [done] = pool.outcomes()
    assert done["outcome"] == "done" and done["attempt"] == gens[1]["attempt"]
    first, second = (_work_dir(pool, job, g["attempt"]) for g in gens)
    assert first != second


def test_content_retry_runs_under_a_new_attempt(pool, tasks, tmp_path):
    """With ``content_attempts=2`` a task that raises once is rerun under a ``restart`` generation; the
    failure is recorded in the first attempt's work directory, the result under the second.
    """
    from pydra.workers.pool.faults import FaultBudgets

    job = pool.job(tasks.FailOnce(marker=str(tmp_path / "failed")), "flaky")
    pool.put([job])
    _, code = _run(pool, faults=FaultBudgets(content_attempts=2))
    assert code == 0
    gens = pool.generations(job)
    assert [g["cause"] for g in gens] == ["put", "restart"]
    [done] = pool.outcomes()
    assert done["outcome"] == "done" and done["attempt"] == gens[1]["attempt"]
    assert (_work_dir(pool, job, gens[0]["attempt"]) / "_error.json").exists()
    assert pool.found(job)


def test_content_fault_fails_without_retry_by_default(pool, tasks, tmp_path):
    """The default budget (``content_attempts=1``) writes ``failed`` with a content fault record."""
    job = pool.job(tasks.FailOnce(marker=str(tmp_path / "failed")), "flaky")
    pool.put([job])
    _, code = _run(pool)
    assert code == 0
    [failed] = pool.outcomes()
    assert failed["outcome"] == "failed" and failed["fault"]["kind"] == "content"
    assert failed["fault"]["type"] == "ValueError"
    assert [g["cause"] for g in pool.generations(job)] == ["put"]


def test_call_record_written_before_and_done_after(pool, tasks):
    """The call record naming the job and its attempt exists while the call runs, without ``done``;
    ``done`` is written when it returns."""
    job = pool.job(tasks.Echo(x=1), "e")
    pool.put([job])
    seen = []
    holder = {}

    def look(x):
        seen.extend(holder["worker"].pool.calls(holder["worker"].token))

    tasks.HOOK["echo"] = look
    from pydra.workers.pool.worker import Worker

    worker = Worker(
        pool.path, _class(), settings=_settings(), device_memory=lambda: None
    )
    holder["worker"] = worker
    assert worker.run() == 0
    [(seq, record, done)] = seen
    attempt = pool.generations(job)[0]["attempt"]
    assert not done and record["jobs"] == [
        {"checksum": job.checksum, "fingerprint": pool.key(job)[1], "attempt": attempt}
    ]
    assert [d for _, _, d in pool.pool.calls(worker.token)] == [True]


def test_no_heartbeat_thread_in_worker(pool, tasks, tmp_path):
    """Under the worker every lease is beaten by the beater process: no ``_Heartbeat`` thread runs
    while a job runs, the store's own lease included."""
    from pydra.engine.lease import _Heartbeat

    pool.put([pool.job(tasks.Echo(x=1), "e")])
    seen = []
    mine = str(
        tmp_path
    )  # threads other tests of this process left are not this worker's
    tasks.HOOK["echo"] = lambda x: seen.append(
        [
            t
            for t in threading.enumerate()
            if isinstance(t, _Heartbeat) and str(t.token_dir).startswith(mine)
        ]
    )
    _, code = _run(pool)
    assert code == 0 and seen == [[]]


def test_lost_claim_publish_refused(pool, tasks):
    """A job whose claim is lost while it runs is not published: the proof before ``publish`` fails,
    nothing is stored and no outcome is written."""
    job = pool.job(tasks.Echo(x=1), "e")
    pool.put([job])

    def lose(x):
        for d in pool.claim_dirs():
            _remove_tree(d)

    tasks.HOOK["echo"] = lose
    _, code = _run(pool)
    assert code == 0
    assert pool.outcomes() == [] and not pool.found(job)


def test_job_run_proves_the_claim_before_publish(pool, tasks, tmp_path):
    """``Job.run`` under a lost claim raises `LeaseLost` before the store publishes."""
    job = pool.job(tasks.Echo(x=1), "e")
    job.attempt = "abc.0"
    lease = DirectoryLease.at(tmp_path / "claim.lease", LEASE, beat="external")
    assert lease.try_take() is None
    job.claim = lease
    _remove_tree(lease.token_dir)
    with pytest.raises(LeaseLost):
        job.run()
    assert not pool.found(job)
    lease.__exit__(None, None, None)


def test_cli_runs_a_worker(pool, tasks, processes, tmp_path):
    """``python -m pydra.workers.pool.worker`` runs the class named in the pool until idle."""
    job = pool.job(tasks.Echo(x=3), "e")
    pool.put([job])
    proc = _start(pool, processes, tmp_path)
    assert proc.wait(BOUND) == 0
    assert pool.kinds() == ["done"] and pool.found(job)


def test_crash_charged_only_to_the_lone_job_of_the_call(pool, tasks):
    """A taker charges a crash to a lost attempt that was alone in its worker's in-flight call, and
    not to one the dead worker never called."""
    from pydra.workers.pool.worker import Worker

    a, b = pool.job(tasks.Echo(x=1), "a"), pool.job(tasks.Echo(x=2), "b")
    da, db = pool.put([a]), pool.put([b])
    dead = "d" * 32
    for digest, job, called in ((da, a, True), (db, b, False)):
        claim = pool.pool.take("cpu", digest, beat="external")
        attempt = claim.start(job_entry(claim), dead)
        if called:
            pool.pool.write_call(
                dead,
                1,
                {
                    "jobs": [
                        {
                            "checksum": a.checksum,
                            "fingerprint": pool.key(a)[1],
                            "attempt": attempt,
                        }
                    ]
                },
            )
        _age(claim.lease.lockfile)
    worker = Worker(
        pool.path, _class(), settings=_settings(), device_memory=lambda: None
    )
    assert worker.run() == 0
    assert [g["charged"] for g in pool.generations(a)] == [False, True]
    assert [g["charged"] for g in pool.generations(b)] == [False, False]
    assert pool.kinds() == ["done", "done"]


# -- helpers -----------------------------------------------------------------------------------


class _Proving:
    def prove(self) -> bool:
        return True


def job_entry(claim: Claim) -> TicketJob:
    return claim.ticket.jobs[0]


def _echo():
    import _pydra_pool_tasks

    return _pydra_pool_tasks.Echo(x=0)


def _started(pool: Pool, job: Job) -> list[dict]:
    d = pool.path / "follow" / f"{pool.key(job)[0]}-{pool.key(job)[1]}"
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.started"))]


def _work_dir(pool: Pool, job: Job, attempt: str) -> Path:
    job.attempt = attempt
    job._work_dir_memo = None
    return pool.store.work_dir(job)


def _remove_tree(path: Path) -> None:
    """Remove a lease's token directory as a taker's rename does: gone from its name at once."""
    import shutil

    gone = path.with_name(f"gone-{path.name}")
    try:
        os.rename(path, gone)
    except FileNotFoundError:
        return
    shutil.rmtree(gone, ignore_errors=True)


def _stale(dirs: list[Path]) -> bool:
    """Whether nothing in these lease directories changed within the stale bound (one host's clock:
    the test's storage is local)."""
    newest = 0.0
    for d in dirs:
        for path in [d, *d.rglob("*")]:
            try:
                newest = max(newest, path.stat().st_mtime)
            except FileNotFoundError:
                pass
    return time.time() - newest > LEASE.stale_after + 0.5


def _age(path: Path) -> None:
    """Stamp a lease directory and everything in it as a holder that stopped long ago leaves it."""
    old = 10**9
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in dirnames + filenames:
            os.utime(Path(dirpath) / name, (old, old))
    os.utime(path, (old, old))


def test_pydracli_pool_worker(pool, tasks, tmp_path):
    """``pydracli pool worker --root --class --settings`` runs the class until idle, exit 0."""
    from click.testing import CliRunner

    from pydra.scripts.cli import cli

    job = pool.job(tasks.Echo(x=4), "e")
    pool.put([job])
    pool.pool.put_class("cpu", _record(_class()))
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(_settings().to_json()))
    result = CliRunner().invoke(
        cli,
        [
            "pool",
            "worker",
            "--root",
            str(pool.path),
            "--class",
            "cpu",
            "--settings",
            str(settings),
        ],
    )
    assert result.exit_code == 0, result.output
    assert pool.kinds() == ["done"]


def test_unknown_class_refuses_to_start(pool):
    """A class the pool has no record of: exit 2, nothing started."""
    from pydra.workers.pool.worker import EXIT_USAGE, Worker

    worker = Worker(pool.path, "nosuch", settings=_settings())
    assert worker.run() == EXIT_USAGE and worker.client is None


def test_substrate_worker_exits_when_orphaned(pool):
    """A substrate-started worker with no live coordinator exits on ``orphan_after_s`` before its
    idle polls run out; with a live coordinator lease it exits on idle polls only."""
    from pydra.workers.pool.worker import Worker

    settings = _settings(idle_polls=10_000, orphan_after_s=0.2)
    orphan = Worker(pool.path, _class(), settings=settings, substrate=True)
    assert orphan.run() == 0 and orphan.polls < 10_000
    epoch = pool.pool.take_epoch()
    with pool.pool.coordinator_lease(epoch):
        attended = Worker(
            pool.path,
            _class(),
            settings=_settings(idle_polls=40, orphan_after_s=0.2),
            substrate=True,
        )
        assert attended.run() == 0 and attended.polls == 40


def test_beater_death_recycles_after_settling(pool, tasks):
    """The beater dies during a call: the call finishes under the proof, its outcome is written, no
    new claim is taken, and the worker exits `EXIT_RECYCLE`."""
    from pydra.workers.pool.worker import EXIT_RECYCLE, Worker

    first, second = pool.job(tasks.Echo(x=1), "a"), pool.job(tasks.Echo(x=2), "b")
    pool.put([first])
    pool.put([second])
    worker = Worker(
        pool.path, _class(), settings=_settings(), device_memory=lambda: None
    )

    def kill_beater(x):
        if x == 1:
            worker.client.process.kill()
            assert worker.client.wait_dead(BOUND)

    tasks.HOOK["echo"] = kill_beater
    assert worker.run() == EXIT_RECYCLE
    assert pool.kinds() == ["done"] and worker.served == [("cpu", pool.tickets[0][1])]


# -- lost claims, drains and fault budgets ---------------------------------------------------------


def _providers(pool: Pool, tasks) -> None:
    """Jobs made from here on name the test providers."""
    pool.submitter.providers = (tasks.SleepServer(), tasks.Recorder(), tasks.Broken())
    tasks.LOADS.clear()


@pytest.mark.skipif(not POSIX, reason="process-owning providers are stopped by group")
def test_lost_claim_evicts_the_stopped_provider(pool, tasks):
    """A claim lost while a provider's process runs the call: the worker stops that process and
    evicts it, so the next job loads the provider afresh and succeeds."""
    _providers(pool, tasks)
    pool.put([pool.job(tasks.UseServer(model="m", x=1), "lost")])
    second = pool.put([pool.job(tasks.UseServer(model="m", x=2), "next")])
    stopped = []

    def lose(model, x):
        if x != 1:
            return
        for d in pool.claim_dirs():
            _remove_tree(d)
        deadline = time.monotonic() + BOUND
        while model.group.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        stopped.append(model.group.proc.poll() is not None)

    tasks.HOOK["server"] = lose
    _, code = _run(pool)
    assert code == 0 and stopped == [True]
    assert [o["outcome"] for o in pool.outcomes(digest=second)] == ["done"]
    assert pool.kinds() == ["done"]
    assert [kind for kind, _ in tasks.LOADS] == ["sleepserver", "sleepserver"]


@pytest.mark.skipif(not POSIX, reason="SIGTERM to this process")
def test_sigterm_drains_a_call_that_ends_in_time(pool, tasks):
    """SIGTERM while a provider's call runs: the call is not stopped while ``drain_s`` lasts, so it
    settles ``done``; the next ticket is not run."""
    from pydra.workers.pool.worker import Worker

    _providers(pool, tasks)
    pool.put([pool.job(tasks.UseServer(model="m", x=1), "drained")])
    rest = pool.put([pool.job(tasks.UseServer(model="m", x=2), "rest")])
    worker = Worker(
        pool.path, _class(), settings=_settings(), device_memory=lambda: None
    )

    def terminate(model, x):
        os.kill(os.getpid(), signal.SIGTERM)
        _wait(worker.stop.is_set, "the worker to see SIGTERM")
        # give an early stop the time to happen: the provider must outlive two renewal intervals
        end = time.monotonic() + 2 * LEASE.renew_interval
        while model.group.proc.poll() is None and time.monotonic() < end:
            time.sleep(0.05)

    tasks.HOOK["server"] = terminate
    assert worker.run() == 0
    assert pool.kinds() == ["done"] and pool.outcomes(digest=rest) == []


def _infrastructure(exc, phase):
    from pydra.workers.pool.faults import FaultRecord

    return FaultRecord("infrastructure", type(exc).__name__, str(exc), phase, gap=True)


def test_infrastructure_faults_hold_the_residency(pool, tasks):
    """Under a classifier that types every exception infrastructure, a job that always fails is
    requeued under ``restart`` generations at the ``call`` phase and, at
    ``infrastructure_replacements`` faults of its residency, the residency is held and the job is
    settled ``held``.
    """
    job = pool.job(tasks.Fail(x=1), "infra")
    pool.put([job])
    _, code = _run(pool, classify=_infrastructure)
    assert code == 0
    gens = pool.generations(job)
    assert [(g["cause"], g["phase"]) for g in gens] == [
        ("put", None),
        ("restart", "call"),
        ("restart", "call"),
    ]
    [held] = pool.outcomes()
    assert held["outcome"] == "held" and held["attempt"] == gens[-1]["attempt"]
    assert held["fault"]["type"] == "RuntimeError"
    assert "3 infrastructure faults" in held["reason"]
    assert len(pool.pool.holds("cpu")) == 1


def test_requeued_ticket_reclaimed_while_gathering(pool, tasks):
    """A warm worker that gathers one-job tickets passes over the ticket it still holds; once that
    ticket is released unsettled (an infrastructure requeue) the same worker claims it again at
    once, rather than skipping it for a renewal interval and exiting idle with the job open.
    """
    job = pool.job(tasks.Fail(x=1), "requeued")
    pool.put([job])
    _, code = _run(pool, _class(quantum_jobs=4), classify=_infrastructure)
    assert code == 0
    assert [g["cause"] for g in pool.generations(job)] == ["put", "restart", "restart"]
    assert [o["outcome"] for o in pool.outcomes()] == ["held"]


def test_crash_budget_fails_a_job_crashed_worker(pool, tasks):
    """A job with ``crashes_charged`` charged generations is ``failed`` (``JobCrashedWorker``)
    without running."""
    job = pool.job(tasks.Echo(x=1), "crashed")
    digest = pool.put([job])
    ran = []
    tasks.HOOK["echo"] = ran.append
    label = queue_label("cpu")
    for _ in range(2):
        prior = pool.generations(job)[-1]["attempt"]
        pool.pool._allocate(
            pool.key(job),
            ticket=digest,
            queue=label,
            cause="takeover",
            prior=prior,
            charged=True,
        )
    _, code = _run(pool)
    assert code == 0 and ran == []
    [failed] = pool.outcomes()
    assert failed["outcome"] == "failed"
    assert failed["fault"]["type"] == "JobCrashedWorker"


def test_takeover_settles_a_committed_result_uncharged(pool, tasks):
    """A worker that died after publishing, before settling, alone in its call: the taker finds the
    result in the store first, settles ``done`` and charges nothing."""
    from pydra.workers.pool.worker import Worker

    job = pool.job(tasks.Echo(x=7), "published")
    digest = pool.put([job])
    claim = pool.pool.take("cpu", digest, beat="external")
    attempt = claim.start(claim.ticket.jobs[0], "d" * 32)
    pool.pool.write_call(
        "d" * 32,
        1,
        {
            "jobs": [
                {
                    "checksum": job.checksum,
                    "fingerprint": pool.key(job)[1],
                    "attempt": attempt,
                }
            ]
        },
    )
    job.attempt = attempt
    job.run()  # what the dead worker published
    _age(claim.lease.lockfile)
    ran = []
    tasks.HOOK["echo"] = ran.append
    worker = Worker(
        pool.path, _class(), settings=_settings(), device_memory=lambda: None
    )
    assert worker.run() == 0 and ran == []
    gens = pool.generations(job)
    assert [g["cause"] for g in gens] == ["put", "takeover"]
    assert [g["charged"] for g in gens] == [False, False]
    [done] = pool.outcomes()
    assert done["outcome"] == "done" and done["served"] is True


def test_providers_load_on_the_granted_device(pool, tasks):
    """On a CUDA class a provider loads on the worker's granted device, the same device the job's
    own binding asks for; the process default is the CPU again after the worker exits.
    """
    from pydra.engine.resident import Device, process_device

    _providers(pool, tasks)
    pool.put([pool.job(tasks.UseRecorder(model="m", x=1), "on-gpu")], queue="gpu")
    _, code = _run(
        pool,
        _class("gpu", device="cuda", gpus=1, gpu_mem="16G"),
        hardware=_gpu_hardware(16 << 30),
    )
    assert code == 0 and pool.kinds("gpu") == ["done"]
    assert tasks.LOADS == [("recorder", Device("cuda", 0))]
    assert process_device() == Device()


@pytest.mark.parametrize(
    "gpu_type,product,starts",
    [
        ("h100", "NVIDIA H100 80GB HBM3", True),
        ("a100", "NVIDIA A100-SXM4-80GB", True),
        ("a10", "NVIDIA A100-SXM4-80GB", False),
        ("l4", "NVIDIA L40S", False),
        ("l40s", "NVIDIA L40S", True),
        ("a40", "NVIDIA A400", False),
        ("a100-80gb", "NVIDIA A100-SXM4-80GB", True),
        ("80gb-a100", "NVIDIA A100-SXM4-80GB", False),
    ],
)
def test_gpu_type_matches_whole_tokens(pool, gpu_type, product, starts):
    """A class's ``gpu_type`` matches the product's name as whole tokens: ``a10`` is not an A100."""
    from pydra.workers.pool.worker import EXIT_HARDWARE

    _, code = _run(
        pool,
        _class("gpu", device="cuda", gpus=1, gpu_type=gpu_type),
        hardware=_gpu_hardware(16 << 30, product=product),
    )
    assert code == (0 if starts else EXIT_HARDWARE)


def test_long_queued_ticket_survives_store_collect(pool, tasks):
    """A ticket queued past every age the store's collect judges by keeps its payload: payloads live
    under the pool root, which the store's collect never touches."""
    from pydra.utils.storagedir import StoreAges, StorePolicy

    job = pool.job(tasks.Echo(x=5), "queued")
    pool.put([job])
    for root in (pool.store.path, pool.path):
        _age_tree(Path(root))
    pool.store.policy = StorePolicy(ages=StoreAges(work=1, staging=1, quarantine=1))
    pool.store.collect()
    _, code = _run(pool)
    assert code == 0 and pool.kinds() == ["done"] and pool.found(job)


def test_pool_collect_removes_only_payloads_nothing_unsettled_names(pool, tasks):
    """The pool's collect removes the payload of a settled ticket, keeps that of an unsettled one,
    and keeps a young payload no ticket names yet."""
    settled = pool.job(tasks.Echo(x=1), "settled")
    pool.put([settled])
    assert _run(pool)[1] == 0
    waiting = pool.job(tasks.Echo(x=2), "waiting")
    pool.put([waiting])
    young = write_payload(
        PrivatePickle(), pool.job(tasks.Echo(x=3), "young"), pool.path
    )
    payloads = {Path(p) for p in (pool.path / "payloads").iterdir()}
    first = next(p for p in payloads if p.name != Path(young["path"]).name)
    for p in payloads - {Path(young["path"])}:
        _age_tree(p)
    assert pool.pool.collect(payload_age=3600.0) >= 1
    left = {p.name for p in (pool.path / "payloads").iterdir()}
    names = {o["checksum"]: o for o in pool.outcomes()}
    assert Path(young["path"]).name in left and len(left) == 2
    assert names[settled.checksum]["outcome"] == "done"
    _ = first
    assert _run(pool)[1] == 0 and pool.kinds() == ["done", "done"]


def test_orphan_exit_after_a_live_coordinator_goes_stale(pool):
    """A coordinator lease beaten by nobody is live until ``stale_after``, then the substrate worker
    waits ``orphan_after_s`` more and exits, before its idle polls run out."""
    from pydra.workers.pool.worker import Worker

    epoch = pool.pool.take_epoch()
    lease = pool.pool.coordinator_lease(epoch, beat="external")
    assert lease.try_take() is None
    try:
        worker = Worker(
            pool.path,
            _class(),
            settings=_settings(idle_polls=400, orphan_after_s=1.0),
            substrate=True,
        )
        assert worker.run() == 0
        assert worker.live_polls > 0 and worker.polls < 400
        # it waited orphan_after_s from the last live judgement, however slow each poll was
        assert time.monotonic() - worker._last_live > 1.0
    finally:
        lease.__exit__(None, None, None)


def test_orphan_exit_when_the_coordinator_ended(pool):
    """A live coordinator lease whose epoch has its ``ended`` mark is no coordinator."""
    from pydra.workers.pool.worker import Worker

    epoch = pool.pool.take_epoch()
    with pool.pool.coordinator_lease(epoch):
        pool.pool.end_epoch(epoch)
        worker = Worker(
            pool.path,
            _class(),
            settings=_settings(idle_polls=400, orphan_after_s=0.2),
            substrate=True,
        )
        assert worker.run() == 0
        assert worker.live_polls == 0 and worker.polls < 400


def test_substrate_flag_through_python_m(pool, processes, tmp_path):
    """``--substrate`` reaches the worker of ``python -m``: with no coordinator it exits orphaned."""
    pool.pool.put_class("cpu", _record(_class()))
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(_settings(idle_polls=100, orphan_after_s=0.2).to_json()))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pydra.workers.pool.worker",
            "--root",
            str(pool.path),
            "--class",
            "cpu",
            "--settings",
            str(path),
            "--substrate",
        ],
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=POSIX,
    )
    processes.append(proc)
    _, err = proc.communicate(timeout=BOUND)
    assert proc.returncode == 0 and "no live coordinator" in err


def test_substrate_flag_through_pydracli(pool, tmp_path, caplog):
    """``pydracli pool worker --substrate`` passes the flag: with no coordinator it exits orphaned."""
    from click.testing import CliRunner

    from pydra.scripts.cli import cli

    pool.pool.put_class("cpu", _record(_class()))
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(_settings(idle_polls=100, orphan_after_s=0.2).to_json()))
    caplog.set_level(logging.INFO, logger="pydra.pool")
    result = CliRunner().invoke(
        cli,
        [
            "pool",
            "worker",
            "--root",
            str(pool.path),
            "--class",
            "cpu",
            "--settings",
            str(path),
            "--substrate",
        ],
    )
    assert result.exit_code == 0, result.output
    assert any("no live coordinator" in r.getMessage() for r in caplog.records)


@pytest.mark.skipif(not POSIX, reason="process-owning providers are stopped by group")
def test_expired_provider_call_is_stopped_and_its_claim_kept(pool, tasks):
    """A provider call past its deadline: the worker stops the provider's process, acknowledges,
    recycles, and leaves the expired claim in place to go stale (never released)."""
    from pydra.utils.resources import ComputeResources
    from pydra.workers.pool.worker import EXIT_RECYCLE

    _providers(pool, tasks)
    pool.put(
        [
            pool.job(
                tasks.UseServer(model="m", x=3),
                "wedged",
                resources=ComputeResources(time=1),
            )
        ]
    )
    stopped = []

    def wait_for_stop(model, x):
        try:
            model.group.proc.wait(BOUND)
        except subprocess.TimeoutExpired:
            pass
        stopped.append(model.group.proc.poll() is not None)

    tasks.HOOK["server"] = wait_for_stop
    _, code = _run(pool, settings=_settings(timeout_grace=5.0))
    assert code == EXIT_RECYCLE and stopped == [True]
    assert pool.kinds() == ["expired"]
    assert pool.claim_dirs() != []  # the expired claim was not released


def test_loser_stops_before_the_next_job_of_its_ticket(pool, tasks, monkeypatch):
    """A claim lost between two jobs of a ticket: the proof before the next job fails, so it is
    never started."""
    a, b = pool.job(tasks.Echo(x=1), "a"), pool.job(tasks.Echo(x=2), "b")
    pool.put([a, b])
    write = Claim.outcome

    def then_lose(self, job, attempt, kind, **fields):
        written = write(self, job, attempt, kind, **fields)
        if job.checksum == a.checksum:
            for d in pool.claim_dirs():
                _remove_tree(d)
        return written

    monkeypatch.setattr(Claim, "outcome", then_lose)
    _, code = _run(pool)
    assert code == 0
    assert len(_started(pool, a)) == 1 and _started(pool, b) == []
    assert pool.kinds() == ["done"]


def _age_tree(path: Path) -> None:
    old = 10**9
    paths = [path] if path.is_file() else [*path.rglob("*"), path]
    for p in paths:
        try:
            os.utime(p, (old, old), follow_symlinks=False)
        except (FileNotFoundError, NotImplementedError):
            os.utime(p, (old, old))


def test_bound_loads_on_the_process_device(tmp_path, tasks):
    """``resident.bound`` (a job's own binding) loads on the process device the worker set, the
    device `Worker._load` uses, so either may load the state."""
    from pydra.engine import resident
    from pydra.engine.resident import Device

    tasks.LOADS.clear()
    resident.set_process_device(Device("cuda", 0))
    try:
        with Submitter(
            cache_root=tmp_path / "store", worker="debug", providers=[tasks.Recorder()]
        ) as sub:
            result = sub(tasks.UseRecorder(model="bound-here", x=2))
        assert not result.errored and result.outputs.out == 2
    finally:
        resident.set_process_device(None)
        resident.close_process_set()
    assert tasks.LOADS == [("recorder", Device("cuda", 0))]


def test_load_infrastructure_faults_hold_the_residency(pool, tasks):
    """A provider whose load always fails (an infrastructure fault under the default classifier):
    each fault writes a ``restart`` generation at the ``load`` phase, and at
    ``infrastructure_replacements`` the residency is held and every job of the ticket is ``held``, so the worker stops reclaiming it and exits on idle.
    """
    _providers(pool, tasks)
    jobs = [pool.job(tasks.UseBroken(model="m", x=i), f"b{i}") for i in range(2)]
    pool.put(jobs)
    worker, code = _run(pool)
    assert code == 0
    assert worker.claims == [2, 2, 2]
    for job in jobs:
        gens = pool.generations(job)
        assert [(g["cause"], g["phase"]) for g in gens] == [
            ("put", None),
            ("restart", "load"),
            ("restart", "load"),
        ]
    held = pool.outcomes()
    assert sorted(o["outcome"] for o in held) == ["held", "held"]
    for o in held:
        assert o["fault"]["type"] == "FileNotFoundError"
        assert "3 infrastructure faults" in o["reason"]
        assert "at load" in o["reason"]
    assert [kind for kind, _ in tasks.LOADS] == ["broken"] * 3


def _fake_smi(tmp_path, monkeypatch, indices, visible, mig=None):
    """A ``nvidia-smi`` on PATH listing GPUs ``indices`` (uuid ``GPU-<index>-aaaa``); ``mig`` maps
    a parent index to its MIG UUIDs, which only ``-L`` lists, as the real tool does."""
    rows = "\n".join(f"{i}, GPU-{i}-aaaa, H100-{i}, 81559, 9.0" for i in indices)
    listing = []
    for i in indices:
        listing.append(f"GPU {i}: NVIDIA H100 (UUID: GPU-{i}-aaaa)")
        for n, u in enumerate((mig or {}).get(i, [])):
            listing.append(f"  MIG 1g.10gb Device {n}: (UUID: {u})")
    # a Python script, run through a shebang on POSIX and a .cmd shim on Windows
    script = tmp_path / "nvidia_smi.py"
    script.write_text(
        "import sys\n"
        f"print({chr(10).join(listing)!r} if sys.argv[1:2] == ['-L'] else {rows!r})\n"
    )
    if sys.platform == "win32":
        (tmp_path / "nvidia-smi.cmd").write_text(
            f'@"{sys.executable}" "{script}" %*\r\n'
        )
    else:
        exe = tmp_path / "nvidia-smi"
        exe.write_text(f"#!{sys.executable}\n" + script.read_text())
        exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)


def _hw(tmp_path, monkeypatch, indices, visible, mig=None):
    from pydra.workers.pool.samples import read_hardware

    _fake_smi(tmp_path, monkeypatch, indices, visible, mig)
    return read_hardware().gpus


def _read(tmp_path, monkeypatch, indices, visible):
    gpus = _hw(tmp_path, monkeypatch, indices, visible)
    assert all(g.uuid == f"GPU-{g.index}-aaaa" for g in gpus)
    return [g.index for g in gpus]


def test_visible_indices_on_an_unconstrained_host(tmp_path, monkeypatch):
    """(a) ``2,3`` on a four-GPU host: logical 0, 1 are physical 2, 3."""
    assert _read(tmp_path, monkeypatch, [0, 1, 2, 3], "2,3") == [2, 3]


def test_unset_visible_devices_grants_all(tmp_path, monkeypatch):
    assert _read(tmp_path, monkeypatch, [0, 1], None) == [0, 1]


def test_constrained_devices_map_local_indices_exactly(tmp_path, monkeypatch):
    """(b) Under ConstrainDevices Slurm sets local ``0,1`` and ``nvidia-smi`` lists 0, 1."""
    assert _read(tmp_path, monkeypatch, [0, 1], "0,1") == [0, 1]


def test_visible_order_is_logical_order(tmp_path, monkeypatch):
    """``1,0``: logical 0 is physical 1, logical 1 is physical 0."""
    assert _read(tmp_path, monkeypatch, [0, 1], "1,0") == [1, 0]


def test_uuids_are_matched_by_uuid(tmp_path, monkeypatch):
    """(c) UUID entries select by UUID, in ``CUDA_VISIBLE_DEVICES`` order, not by position."""
    assert _read(tmp_path, monkeypatch, [0, 1, 2, 3], "GPU-3-aaaa,GPU-1-aaaa") == [3, 1]


def test_uuid_prefix_matches_when_unambiguous(tmp_path, monkeypatch):
    assert _read(tmp_path, monkeypatch, [0, 1, 2, 3], "GPU-2,gpu-0-a") == [2, 0]


@pytest.mark.parametrize("visible", ["-1", "", "NoDevFiles"])
def test_no_device_values_grant_nothing_quietly(tmp_path, monkeypatch, caplog, visible):
    with caplog.at_level(logging.WARNING):
        assert _read(tmp_path, monkeypatch, [0, 1], visible) == []
    assert caplog.text == ""


def test_mig_uuid_maps_to_its_parent_with_unknown_memory(tmp_path, monkeypatch):
    """A MIG slice takes the parent's product and index, and no memory (the parent's total overstates it)."""
    mig = {1: ["MIG-aaaa1111", "MIG-bbbb2222"]}
    gpus = _hw(tmp_path, monkeypatch, [0, 1], "MIG-bbbb2222,GPU-0-aaaa", mig)
    assert [(g.uuid, g.index, g.product) for g in gpus] == [
        ("MIG-bbbb2222", 1, "H100-1"),
        ("GPU-0-aaaa", 0, "H100-0"),
    ]
    assert gpus[0].memory is None and gpus[1].memory == 81559 << 20


def test_unknown_mig_uuid_grants_nothing(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        assert _hw(tmp_path, monkeypatch, [0, 1], "MIG-zzzz", {1: ["MIG-aaaa"]}) == ()
    assert "CUDA_VISIBLE_DEVICES" in caplog.text


@pytest.mark.parametrize(
    "indices,visible",
    [
        ([0, 1, 2, 3], "GPU-9-aaaa"),  # unknown uuid
        ([0, 1, 2, 3], "2,9"),  # index not listed
        ([0, 1], "2,3"),  # host indices under a cgroup: CUDA cannot see them either
        ([0, 1], "GPU-0-aaaa,5"),  # mixed with a missing index
        ([0, 1], "1,1"),  # duplicate
        ([0, 1, 2], "GPU-"),  # ambiguous prefix
    ],
)
def test_unmappable_visible_devices_grant_nothing(
    tmp_path, monkeypatch, caplog, indices, visible
):
    """(d) An entry that cannot be mapped warns and grants no device; never a guessed one."""
    with caplog.at_level(logging.WARNING):
        assert _read(tmp_path, monkeypatch, indices, visible) == []
    assert "CUDA_VISIBLE_DEVICES" in caplog.text


def test_gpu_identity_does_not_enter_equality():
    from pydra.workers.pool.samples import Gpu

    assert Gpu("H100", 1, "9.0", "GPU-a", 0) == Gpu("H100", 1, "9.0", "GPU-b", 3)


def test_ready_mark_records_uuid_and_index(pool, tmp_path, monkeypatch):
    from pydra.workers.pool.samples import read_hardware

    _fake_smi(tmp_path, monkeypatch, [0, 1], "1")
    worker, code = _run(
        pool, _class("gpu", device="cuda", gpus=1), hardware=read_hardware
    )
    assert code == 0
    gpu = pool.pool.worker_record(worker.token, "ready.json")["hardware"]["gpus"][0]
    assert (gpu["uuid"], gpu["index"]) == ("GPU-1-aaaa", 1)


def test_the_stop_mark_watch_outlives_a_storage_error(tmp_path):
    """A storage error reading the stop mark (ESTALE on network storage) does not end the watch:
    the next read finds the mark and the worker drains."""
    import errno
    from types import SimpleNamespace

    from pydra.workers.pool.worker import Worker

    reads = []

    def worker_record(token, name):
        reads.append(name)
        if len(reads) == 1:
            raise OSError(getattr(errno, "ESTALE", errno.EIO), "Stale file handle")
        return {"at": 0}

    stub = SimpleNamespace(
        pool=SimpleNamespace(worker_record=worker_record, path=tmp_path),
        token="d" * 32,
        stop=threading.Event(),
        settings=SimpleNamespace(poll_s=0.0),
    )
    drained, done = threading.Event(), threading.Event()
    watch = threading.Thread(
        target=Worker._watch_stop_mark, args=(stub, done, drained.set), daemon=True
    )
    watch.start()
    try:
        assert drained.wait(BOUND)
    finally:
        done.set()
        watch.join(BOUND)
    assert len(reads) == 2


def test_a_drain_past_drain_s_marks_the_call_drained(pool, tasks):
    """A drain (what SIGTERM starts) while a provider's call outlives ``drain_s``: the call is
    stopped and its attempt marked drained, so its taker places it again uncharged (a scheduler
    preemption). The drain is started as the SIGTERM handler starts it, without signalling the test
    process."""
    from pydra.workers.pool.worker import Worker

    _providers(pool, tasks)
    job = pool.job(tasks.UseServer(model="m", x=1), "cut")
    pool.put([job])
    worker = Worker(
        pool.path,
        _class(),
        settings=_settings(drain_s=0.2),
        device_memory=lambda: None,
    )

    def terminate(model, x):
        worker.stop.set()
        worker._wake.set()
        _wait(lambda: model.group.proc.poll() is not None, "the drain to stop the call")

    tasks.HOOK["server"] = terminate
    worker.run()
    g, _ = pool.pool.newest(*pool.key(job))
    assert pool.pool.drained(*pool.key(job), g)
