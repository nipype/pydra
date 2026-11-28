"""The warm pool executor and the local substrate (RFC 0005 sections 4.1, 4.2, 4.5, 5.5, 7, W6).

The tests drive a `GraphRunner` with the asynchronous driver and a `WarmPool`, as an external driver
does (RFC 0005 section 7.2). Workers are real processes: started by a `LocalSubstrate`, or by hand.
Tasks live in a module written to a temporary directory and put on ``sys.path`` and ``PYTHONPATH``, so
a worker in another process imports them as this one does.
"""

import asyncio
import json
import logging
import os
import signal
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pydra.engine.drivers import drive
from pydra.engine.lease import LeaseSettings
from pydra.engine.runner import GraphRunner, NotRun
from pydra.engine.submitter import RunIncomplete, Submitter, run_incomplete
from pydra.workers.pool.faults import FaultBudgets

POSIX = sys.platform != "win32"
LEASE = LeaseSettings(
    stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
)
BOUND = 120.0  # seconds any wait on another process may take before the test fails

TASKS = """
import os
import time
from pathlib import Path
from uuid import uuid4
from pydra.compose import python, workflow
from pydra.utils.resources import ComputeResources


@python.define
def Echo(x: int) -> int:
    return x


@python.define
def Mark(x: int, where: str) -> int:
    # one file per execution, so a test counts how often each job ran
    Path(where, f"{x}-{uuid4().hex}").touch()
    return x


@python.define
def Gate(where: str) -> str:
    # the first execution waits for "go"; any later one returns at once
    d = Path(where)
    if not (d / "first").exists():
        (d / "first").touch()
        while not (d / "go").exists():
            time.sleep(0.05)
        return "first"
    return "later"


@python.define
def Fail(x: int) -> int:
    raise RuntimeError("this task always fails")


@workflow.define
def Fails(xs: list) -> list:
    return workflow.add(Fail().split(x=xs), name="F").out


@workflow.define(outputs=["a", "b"])
def TwoNodes(xs: list):
    a = workflow.add(Echo().split(x=xs), name="A")
    b = workflow.add(Mark(where="unused").split(x=xs), name="B")
    return a.out, b.out


@workflow.define
def Many(xs: list) -> list:
    return workflow.add(Echo().split(x=xs), name="E").out


@python.define
def Forget(where: str) -> str:
    # removes a directory from inside a pool job, once the outer job has started
    import shutil

    shutil.rmtree(where, ignore_errors=True)
    return where


@python.define
def EchoAfter(x: int, after: str) -> int:
    return x


@workflow.define
def ForgetThenMany(where: str, xs: list) -> list:
    gone = workflow.add(Forget(where=where), name="F")
    return workflow.add(EchoAfter(after=gone.out).split(x=xs), name="E").out


@workflow.define
def Marks(xs: list, where: str) -> list:
    return workflow.add(Mark(where=where).split(x=xs), name="M").out


@workflow.define
def OneGate(where: str) -> str:
    return workflow.add(Gate(where=where), name="G").out


@python.define
def IsPos(x: int) -> bool:
    return x > 0


@python.define
def Twice(x: int) -> int:
    return 2 * x


@workflow.define
def GatedTwins(xs: list) -> list:
    a = workflow.add(Echo().split(x=xs), name="A")
    c = workflow.add(IsPos(x=a.out), name="C")
    return workflow.add(Twice(x=a.out), name="D", when=c.out).out


@workflow.define
def GpuMany(xs: list) -> list:
    gpu = ComputeResources(gpus=1, gpu_type="h100")
    return workflow.add(Echo().split(x=xs), name="E", resources=gpu).out


@workflow.define
def BigGpuMark(x: int, where: str) -> int:
    need = ComputeResources(gpus=1, gpu_mem="100G")
    return workflow.add(Mark(x=x, where=where), name="M", resources=need).out


@python.define
def DeviceHold(x: int, where: str, devices: int) -> str:
    # announce this worker's device, then hold until ``devices`` workers have announced theirs, so
    # no worker can drain the queue before the others start; the bound only stops a hang
    d = Path(where)
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "none")
    (d / f"device-{device}").touch()
    deadline = time.monotonic() + 120
    while len(list(d.glob("device-*"))) < devices:
        if time.monotonic() > deadline:
            raise RuntimeError("the other workers never ran a job")
        time.sleep(0.05)
    return device


@workflow.define
def GpuHolds(xs: list, where: str, devices: int) -> list:
    gpu = ComputeResources(gpus=1)
    node = DeviceHold(where=where, devices=devices).split(x=xs)
    return workflow.add(node, name="H", resources=gpu).out
"""

FAKE_NVIDIA_SMI = """#!{python}
import os, sys
args = " ".join(sys.argv[1:])
gpus = {gpus!r}
if "--query-gpu=index,uuid,name" in args:
    for i, gpu in enumerate(gpus):
        name, mib = gpu if isinstance(gpu, (list, tuple)) else (gpu, 81559)
        print(f"{{i}}, GPU-fake-{{i}}, {{name}}, {{mib}}, 9.0")
elif "--query-gpu=index,name" in args:
    for i, gpu in enumerate(gpus):
        name, mib = gpu if isinstance(gpu, (list, tuple)) else (gpu, 81559)
        print(f"{{i}}, {{name}}, {{mib}}, 9.0")
elif "--query-gpu=index" in args:
    for i, _ in enumerate(gpus):
        print(i)
"""


@pytest.fixture(scope="session")
def tasks(tmp_path_factory):
    """The task module, importable here and in workers started from here."""
    where = tmp_path_factory.mktemp("warm_pool_tasks")
    (where / "_pydra_warm_pool_tasks.py").write_text(TASKS)
    sys.path.insert(0, str(where))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(where), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    import _pydra_warm_pool_tasks

    return _pydra_warm_pool_tasks


@pytest.fixture
def processes():
    """Processes a test starts by hand, each in its own session; killed with their group at the end."""
    started: list[subprocess.Popen] = []
    yield started
    for proc in started:
        _kill_group(proc)


@pytest.fixture
def pools():
    """Every `WarmPool` a test makes is closed, and every worker its substrate started is killed."""
    made = []
    yield made
    for pool in made:
        try:
            pool.close()
        finally:
            for handle in getattr(pool.substrate, "handles", []):
                if handle.process is not None:
                    _kill_group(handle.process)


def _kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            if POSIX:
                os.killpg(proc.pid, signal.SIGCONT)
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

    fields = dict(lease=LEASE, poll_s=0.05, idle_polls=40, timeout_grace=60.0)
    fields.update(kwargs)
    return PoolSettings(**fields)


def _class(name="cpu", **kwargs):
    from pydra.workers.pool.classes import WorkerClass

    fields = dict(name=name, device="cpu", quantum_jobs=8, quantum_s=600.0)
    fields.update(kwargs)
    return WorkerClass(**fields)


def _warm(tmp_path, pools, sub, classes=None, substrate=None, **kwargs):
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.substrate import LocalSubstrate

    kwargs.setdefault("settings", _settings())
    pool = WarmPool(
        tmp_path / "pool",
        classes or [_class()],
        substrate if substrate is not None else LocalSubstrate(),
        store=sub.store,
        **kwargs,
    )
    pools.append(pool)
    return pool


def _submitter(tmp_path):
    return Submitter(cache_root=tmp_path / "store", worker="debug")


def _drive(sub, task, executor, check=True, rerun=False, **kwargs):
    """Run ``task``'s nodes on ``executor`` through the asynchronous driver, as an external driver
    that owns the executor does: it closes the executor after the run (``drive`` never does). Raise as
    `Submitter` does when jobs did not run."""
    runner = GraphRunner(
        task, sub.store, on_error="raise_at_end", submitter=sub, rerun=rerun
    )
    limits = []
    ready = runner.ready

    def recording(limit=None):
        limits.append(limit)
        return ready(limit)

    runner.ready = recording
    reported = []
    bind = executor.bind

    def recording_bind(report):
        def recorded(job, outcome):
            reported.append((job, outcome))
            report(job, outcome)

        bind(recorded)

    executor.bind = recording_bind
    try:
        asyncio.run(
            drive(
                runner,
                {"pool": executor},
                lambda job: "pool",
                on_error="raise_at_end",
                **kwargs,
            )
        )
    finally:
        executor.close()
    runner.limits, runner.reported = limits, reported
    if not check:
        return runner
    if runner.errors:
        raise AssertionError("\n".join(runner.errors))
    if runner.not_run:
        raise run_incomplete(runner.job or task, runner.not_run)
    return runner


def _wait(predicate, what: str, bound: float = BOUND) -> None:
    import time

    deadline = time.monotonic() + bound
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _tickets(root: Path, queue: str = "cpu") -> list[dict]:
    from pydra.workers.pool.storage import queue_label

    d = root / "queues" / queue_label(queue) / "tickets"
    return (
        [json.loads(p.read_bytes()) for p in sorted(d.glob("*.json"))]
        if d.exists()
        else []
    )


def _all_tickets(root: Path) -> list[dict]:
    d = root / "queues"
    return [json.loads(p.read_bytes()) for p in sorted(d.glob("*/tickets/*.json"))]


def _key(sub, job):
    return job.checksum, sub.store.identity.fingerprint(job.context)


def _raw(root: Path):
    """The pool at ``root``, opened afresh: `_drive` closes its executor, and with it the pool's."""
    from pydra.workers.pool.storage import PoolRoot

    return PoolRoot.open(root, LEASE, create=False, probe=False)


def _generations(root: Path, sub, job) -> list[dict]:
    with _raw(root) as raw:
        return [r for _, r in raw.generations(*_key(sub, job))]


def _hand_worker(root: Path, cls, processes, tmp_path, settings=None, substrate=False):
    """``python -m pydra.workers.pool.worker`` on the pool, started by hand, in its own session."""
    from pydra.workers.pool.storage import PoolRoot
    from pydra.workers.pool.worker import class_record

    with PoolRoot.open(root, LEASE, probe=False) as pool:
        pool.put_class(cls.name, class_record(cls))
    path = tmp_path / f"hand-settings-{len(processes)}.json"
    path.write_text(json.dumps((settings or _settings()).to_json()))
    argv = [
        sys.executable,
        "-m",
        "pydra.workers.pool.worker",
        "--root",
        str(root),
        "--class",
        cls.name,
        "--settings",
        str(path),
    ]
    if substrate:
        argv.append("--substrate")
    proc = subprocess.Popen(argv, start_new_session=POSIX)
    processes.append(proc)
    return proc


def _ready_marks(root: Path) -> list[dict]:
    return [
        json.loads(p.read_bytes()) for p in sorted(root.glob("workers/*/ready.json"))
    ]


@pytest.fixture
def fake_nvidia_smi(tmp_path, monkeypatch):
    """Put an ``nvidia-smi`` on PATH that lists the GPUs a test names."""

    def install(*names):
        where = tmp_path / "fakebin"
        where.mkdir(exist_ok=True)
        exe = where / "nvidia-smi"
        exe.write_text(FAKE_NVIDIA_SMI.format(python=sys.executable, gpus=list(names)))
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("PATH", f"{where}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        return exe

    return install


class Recording:
    """A substrate that starts nothing and records what it is asked to stop."""

    def __init__(self, live=()):
        self.handles = []
        self._live = list(live)
        self.stopped = []

    def attach(self, attachment):
        pass

    def check_root(self, root):
        pass

    def start(self, cls, n):
        return []

    def live(self):
        return list(self._live)

    def stop(self, workers):
        self.stopped.extend(workers)

    def can_start(self, cls):
        return True


# -- tickets and placement -----------------------------------------------------------------------


def test_round_of_100_becomes_tickets_of_quantum(tmp_path, tasks, pools):
    """A 100-element round of one node becomes two tickets of 50 jobs on one queue, cut in node order,
    each a JSON document naming every job's payload (never a pickle)."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path, pools, sub, [_class(quantum_jobs=50)], substrate=ExternalSubstrate()
    )
    runner = GraphRunner(
        tasks.Many(xs=list(range(100))),
        sub.store,
        on_error="raise_at_end",
        submitter=sub,
    )
    jobs = runner.ready().dispatch
    assert len(jobs) == 100
    pool.bind(lambda job, outcome: None)
    pool.submit(jobs)
    _wait(
        lambda: sum(len(t["jobs"]) for t in _tickets(pool.root)) == 100,
        "the round's tickets",
    )
    tickets = _tickets(pool.root)
    assert len(tickets) == 2 and [len(t["jobs"]) for t in tickets] == [50, 50]
    assert {t["queue"] for t in tickets} == {"cpu"} and {
        t["node"] for t in tickets
    } == {"E"}
    order = {j.checksum: j.state_index for j in jobs}
    for t in tickets:
        indices = [order[j["checksum"]] for j in t["jobs"]]
        assert indices == sorted(indices)
        assert all(set(j["payload"]) == {"codec", "path", "sha256"} for j in t["jobs"])
    assert _all_tickets(pool.root) == tickets  # nothing on any other queue


def test_resubmission_follows_live_tickets(tmp_path, tasks, pools, processes):
    """A resubmitted run finds its predecessor's live tickets and follows them: it puts no ticket
    again, and every job runs once."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    first = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    runner = GraphRunner(
        tasks.Marks(xs=list(range(6)), where=str(marks)),
        sub.store,
        on_error="raise_at_end",
        submitter=sub,
    )
    first.bind(lambda job, outcome: None)
    first.submit(runner.ready().dispatch)
    _wait(lambda: sum(len(t["jobs"]) for t in _tickets(first.root)) == 6, "tickets")
    put = _tickets(first.root)
    first.close()  # the first coordinator is gone; no worker has run anything

    _hand_worker(first.root, _class(), processes, tmp_path)
    second = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    resumed = _drive(sub, tasks.Marks(xs=list(range(6)), where=str(marks)), second)
    assert resumed.done and not resumed.failed
    assert _tickets(second.root) == put  # followed: no ticket was put again
    ran = sorted(p.name.split("-")[0] for p in marks.iterdir())
    assert ran == [str(i) for i in range(6)]  # each job ran once
    for ticket in put:
        for entry in ticket["jobs"]:
            with _raw(second.root) as raw:
                gens = raw.generations(entry["checksum"], entry["fingerprint"])
            assert [g["cause"] for _, g in gens] == ["put"]


def test_live_ticket_of_unstarted_class_withdrawn_and_replaced(tmp_path, tasks, pools):
    """An unclaimed ticket of a class the resumed coordinator does not start is taken by `try_take`,
    its jobs moved to a ticket of a class it starts under fresh attempts (``withdrawn``, reason
    ``replaced``), and run there once."""
    from pydra.workers.pool.substrate import ExternalSubstrate, LocalSubstrate

    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    first = _warm(
        tmp_path, pools, sub, [_class("cpu-a")], substrate=ExternalSubstrate()
    )
    runner = GraphRunner(
        tasks.Marks(xs=[1, 2, 3], where=str(marks)),
        sub.store,
        on_error="raise_at_end",
        submitter=sub,
    )
    jobs = runner.ready().dispatch
    first.bind(lambda job, outcome: None)
    first.submit(jobs)
    _wait(lambda: len(_tickets(first.root, "cpu-a")) == 1, "the cpu-a ticket")
    first.close()

    second = _warm(tmp_path, pools, sub, [_class("cpu-b")], substrate=LocalSubstrate())
    resumed = _drive(sub, tasks.Marks(xs=[1, 2, 3], where=str(marks)), second)
    assert resumed.done and not resumed.failed
    assert sorted(p.name.split("-")[0] for p in marks.iterdir()) == ["1", "2", "3"]
    (old,) = _tickets(second.root, "cpu-a")
    (new,) = _tickets(second.root, "cpu-b")
    for job in jobs:
        gens = _generations(second.root, sub, job)
        assert [g["cause"] for g in gens] == ["put", "replaced"]
        assert gens[1]["prior"] == gens[0]["attempt"] != gens[1]["attempt"]
    with _raw(second.root) as raw:
        outcomes = raw.outcomes("cpu-a", _digest(old))
    assert {o["outcome"] for o in outcomes.values()} == {"withdrawn"}
    assert {o["reason"] for o in outcomes.values()} == {"replaced"}
    assert {o["by"] for o in outcomes.values()} == {_digest(new)}


def _digest(ticket: dict) -> str:
    import hashlib

    from pydra.utils.digest import jcs

    return hashlib.sha256(jcs(ticket)).hexdigest()


# -- recovery ------------------------------------------------------------------------------------


def _children(pid: int) -> list[int]:
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
    return [int(p) for p in out.stdout.split()]


@pytest.mark.skipif(not POSIX, reason="stops the beater with SIGSTOP")
def test_job_outlives_stale_after(tmp_path, tasks, pools, processes):
    """A job runs past ``stale_after`` while its worker's beater is stopped: the worker finds its
    beater dead and stops proving, so the claim goes stale; a second worker takes it over under a new
    attempt, in its own work directory, and publishes; the loser's ``publish`` is refused by its
    failed proof, so the store holds one entry and the run reports the taker's result.
    """
    import threading

    from pydra.workers.pool.storage import queue_label
    from pydra.workers.pool.substrate import LocalSubstrate

    gate = tmp_path / "gate"
    gate.mkdir()
    sub = _submitter(tmp_path)
    settings = _settings(idle_polls=400)
    pool = _warm(tmp_path, pools, sub, substrate=LocalSubstrate(), settings=settings)
    box = {}

    def run():
        try:
            box["runner"] = _drive(sub, tasks.OneGate(where=str(gate)), pool)
        except BaseException as exc:  # noqa: B036 - re-raised by the test
            box["error"] = exc

    driver = threading.Thread(target=run)
    driver.start()
    losses = pool.root / "queues" / queue_label("cpu") / "losses"
    try:
        _wait(lambda: (gate / "first").exists(), "the first execution to start")
        (loser,) = pool.substrate.live()
        beaters = _children(loser.process.pid)
        assert beaters, "the worker has a beater"
        for pid in beaters:
            os.kill(pid, signal.SIGSTOP)
        _hand_worker(pool.root, _class(), processes, tmp_path, settings=settings)
        _wait(lambda: any(losses.glob("*/*")), "the takeover of the stale claim")
    finally:
        (gate / "go").touch()
        driver.join(BOUND)
    assert not driver.is_alive(), "the run did not end"
    if "error" in box:
        raise box["error"]
    runner = box["runner"]
    assert runner.done and not runner.failed
    assert (
        loser.process.poll() is not None
    )  # recycled, or stopped by the coordinator's close

    (ticket,) = _tickets(pool.root)
    entry = ticket["jobs"][0]
    with _raw(pool.root) as raw:
        gens = [r for _, r in raw.generations(entry["checksum"], entry["fingerprint"])]
        outcomes = raw.outcomes("cpu", _digest(ticket))
    assert [g["cause"] for g in gens] == ["put", "takeover"]
    assert gens[1]["prior"] == gens[0]["attempt"]
    loser_dir = sub.store._work_path(
        entry["checksum"], entry["fingerprint"], gens[0]["attempt"]
    )
    taker_dir = sub.store._work_path(
        entry["checksum"], entry["fingerprint"], gens[1]["attempt"]
    )
    assert loser_dir != taker_dir
    # the loser ran in its own directory and its publish was refused
    assert loser_dir.exists() and not (loser_dir / "_committed.json").exists()
    assert "was lost" in loser.log.read_text()
    assert [(o["outcome"], o["attempt"]) for o in outcomes.values()] == [
        ("done", gens[1]["attempt"])
    ]
    ((_, result),) = runner.reported
    assert result.outputs.out == "later" and not result.errored  # the taker's
    slots = list(Path(sub.store.path).glob(f"results@1/*/{entry['checksum']}/*"))
    assert len(slots) == 1 and (slots[0] / "entry.json").exists()  # one entry
    assert json.loads((slots[0] / "entry.json").read_bytes())["outputs"] == {
        "out": "later"
    }


# -- the follower --------------------------------------------------------------------------------


def test_follower_cost_10k_jobs(tmp_path, tasks, pools, monkeypatch):
    """Following 10 000 jobs in 200 tickets on two queues lists one directory per queue per poll (the
    settled marks), plus the epoch fence; a ticket's outcomes (and its jobs' newest generations) are
    listed only when its settled count moved."""
    import hashlib

    from pydra.workers.pool import storage
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        [_class("cpu"), _class("cpu2")],
        substrate=ExternalSubstrate(),
        settings=_settings(max_queued_jobs=20000),
    )
    for q in ("cpu", "cpu2"):
        for t in range(100):
            digest = hashlib.sha256(f"{q}-{t}".encode()).hexdigest()
            for j in range(50):
                key = (f"{q}{t:03d}{j:02d}".ljust(64, "0"), "f" * 64)
                pool._follow_key(q, digest, key, None)
    # one real ticket of three jobs on cpu, whose settled count moves
    runner = GraphRunner(
        tasks.Many(xs=[7, 8, 9]), sub.store, on_error="raise_at_end", submitter=sub
    )
    jobs = runner.ready().dispatch
    reported = []
    pool.bind(lambda j, outcome: reported.append((j, outcome)))
    pool._queue_jobs(jobs, handle=None)
    pool._write()
    (ticket,) = _tickets(pool.root)
    digest = _digest(ticket)

    listed, entries = [], []
    names = storage._names

    def counting(d):
        listed.append(d)
        found = names(d)
        entries.append(len(found))
        return found

    monkeypatch.setattr(storage, "_names", counting)
    for _ in range(3):
        listed.clear()
        entries.clear()
        pool._poll()
        # settled/ of each queue, and coordinator/ for the fence; nothing is settled yet
        assert len(listed) == 2 + 1 and sum(entries) == 1
    claim = pool.pool.take("cpu", digest)
    entry = [e for e in claim.ticket.jobs if e.checksum == jobs[0].checksum][0]
    claim.outcome(
        entry, claim.current(entry)[1]["attempt"], "failed", fault={"kind": "x"}
    )
    claim.release()
    listed.clear()
    pool._poll()
    # and, once, that ticket's outcome directory and the settled job's follow directory (not
    # its two open neighbours')
    assert len(listed) == 2 + 1 + 2
    assert [j for j, _ in reported] == [jobs[0]]
    listed.clear()
    pool._poll()
    assert len(listed) == 2 + 1


# -- substrates, scaling and failed starts -------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="the fake nvidia-smi is a shebang script")
def test_pool_never_starting_holds_jobs(tmp_path, tasks, pools, fake_nvidia_smi):
    """A class whose workers never start (an A100 granted to an H100 class) backs off and, after
    ``start_failures`` consecutive failed starts, is broken: its jobs are ``NotRun(held=True)``
    naming the class, the count and the latest cause, and the run ends `RunIncomplete`.
    """
    from pydra.workers.pool.substrate import LocalSubstrate

    fake_nvidia_smi("NVIDIA A100-SXM4-40GB")
    sub = _submitter(tmp_path)
    gpu = _class("gpu", device="cuda", gpus=1, gpu_type="h100", gpu_mem="80G")
    substrate = LocalSubstrate()
    pool = _warm(
        tmp_path,
        pools,
        sub,
        [gpu],
        substrate=substrate,
        faults=FaultBudgets(start_failures=2),
    )
    with pytest.raises(RunIncomplete) as info:
        _drive(sub, tasks.GpuMany(xs=[1, 2]), pool)
    not_run = info.value.not_run
    assert len(not_run) == 2
    for _, outcome in not_run:
        assert isinstance(outcome, NotRun) and outcome.held
        assert "'gpu'" in outcome.reason and "2 consecutive" in outcome.reason
        assert "exit code 4" in outcome.reason
    assert len(substrate.handles) == 2
    assert _ready_marks(pool.root) == []


@pytest.mark.skipif(not POSIX, reason="the fake nvidia-smi is a shebang script")
def test_one_worker_per_gpu_local(tmp_path, tasks, pools, fake_nvidia_smi):
    """On a host whose ``nvidia-smi`` lists two GPUs, a one-GPU class runs one worker per device,
    each given its device through ``CUDA_VISIBLE_DEVICES``; a third cannot start. Every job holds
    until both workers have run one, so the placement is asserted, not a race between the first
    worker draining the queue and the second marking ready."""
    from pydra.workers.pool.substrate import LocalSubstrate

    fake_nvidia_smi("NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3")
    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    gpu = _class("gpu", device="cuda", gpus=1, max_workers=4, quantum_jobs=1)
    substrate = LocalSubstrate()
    pool = _warm(
        tmp_path,
        pools,
        sub,
        [gpu],
        substrate=substrate,
        settings=_settings(unmeasured_workers=4),
    )
    holds = tasks.GpuHolds(xs=list(range(4)), where=str(marks), devices=2)
    runner = _drive(sub, holds, pool)
    assert runner.done and not runner.failed
    assert sorted(p.name for p in marks.iterdir()) == ["device-0", "device-1"]
    assert len(substrate.handles) == 2
    assert sorted(h.devices for h in substrate.handles) == [("0",), ("1",)]
    assert substrate.can_start(gpu) is not True or not substrate.live()
    ready = _ready_marks(pool.root)
    assert len(ready) == 2
    assert all(len(r["hardware"]["gpus"]) == 1 for r in ready)


def test_drive_hands_the_pool_every_ready_job(tmp_path, tasks, pools):
    """``drive`` asks the runner for every ready job of a round routed to the pool: one round,
    one ticket"""
    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub)
    assert pool.batches is True and pool.checks_store is True and pool.inline is False
    runner = _drive(sub, tasks.Many(xs=list(range(4))), pool)
    assert runner.done and not runner.failed
    assert runner.limits and set(runner.limits) == {None}
    assert len(_tickets(pool.root)) == 1


def test_gated_and_shared_elements_never_reach_the_pool(tmp_path, tasks, pools):
    """Through ``drive`` and a ``WarmPool``: a gated-off element and a shared twin are settled by the
    runner and never handed to the pool, and the run is done only after both are reported (RFC 0002
    R7, R10b)"""
    from pydra.engine.absent import Absent

    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub)
    runner = _drive(sub, tasks.GatedTwins(xs=[5, 5, -1]), pool)
    assert runner.done and not runner.failed
    ran = sorted((j.name, j.state_index) for j, _ in runner.reported)
    assert ran == [("A", 0), ("A", 2), ("C", 0), ("C", 2), ("D", 0)]
    assert not any(j._propagated_absent is not None for j, _ in runner.reported)
    result = sub(tasks.GatedTwins(xs=[5, 5, -1]))
    assert result.outputs.out[:2] == [10, 10]
    assert isinstance(result.outputs.out[2], Absent)


# -- coordinators and lifecycle ------------------------------------------------------------------


def test_close_with_second_live_coordinator_stops_nothing(tmp_path, tasks, pools):
    """A coordinator fenced by a newer epoch writes its ``ended`` mark on ``close`` and stops no
    worker, since they serve the newer one; one that holds the newest epoch stops the workers its
    substrate lists."""
    from pydra.workers.pool.substrate import WorkerHandle

    sub = _submitter(tmp_path)
    live = [WorkerHandle(token="a" * 32, worker_class="cpu")]
    older = _warm(tmp_path, pools, sub, substrate=Recording(live))
    epoch = older.pool.take_epoch()  # a second coordinator raced for the free pool
    newer = older.pool.coordinator_lease(epoch)
    newer.__enter__()
    try:
        older.close()
        assert older.substrate.stopped == []
        assert (older.root / "coordinator" / str(older.epoch) / "ended").exists()
    finally:
        newer.__exit__(None, None, None)
    sole = _warm(tmp_path / "other", pools, sub, substrate=Recording(live))
    sole.close()
    assert sole.substrate.stopped == live


def test_hand_started_worker_survives_no_coordinator(tmp_path, tasks, pools, processes):
    """A worker started by hand serves the pool before any coordinator exists, through a run, and after
    that coordinator ends; it is not stopped by ``close`` and exits only on idle."""
    from pydra.workers.pool.storage import PoolRoot, Ticket, TicketJob
    from pydra.workers.pool.substrate import ExternalSubstrate

    marks = tmp_path / "marks"
    marks.mkdir()
    root = tmp_path / "pool"
    PoolRoot.open(root, LEASE, probe=False).close()
    settings = _settings(idle_polls=100000, orphan_after_s=0.1)
    worker = _hand_worker(root, _class(), processes, tmp_path, settings=settings)
    _wait(lambda: len(_ready_marks(root)) == 1, "the hand-started worker to be ready")

    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate(), settings=settings)
    runner = _drive(sub, tasks.Marks(xs=[1, 2], where=str(marks)), pool)
    assert runner.done and not runner.failed
    pool.close()
    assert worker.poll() is None

    # after the coordinator ended, the worker still serves the pool
    job_runner = GraphRunner(
        tasks.Marks(xs=[3], where=str(marks)),
        sub.store,
        on_error="raise_at_end",
        submitter=sub,
    )
    (job,) = job_runner.ready().dispatch
    from pydra.workers.pool.codec import PrivatePickle, write_payload

    with PoolRoot.open(root, LEASE, probe=False) as raw:
        fingerprint = sub.store.identity.fingerprint(job.context)
        entry = TicketJob(
            checksum=job.checksum,
            fingerprint=fingerprint,
            payload=write_payload(PrivatePickle(), job, root),
        )
        raw.put("cpu", Ticket(queue="cpu", node="M", jobs=[entry]))
    _wait(
        lambda: len(list(marks.glob("3-*"))) == 1,
        "the job put after the coordinator ended",
    )
    assert worker.poll() is None
    assert sorted(p.name.split("-")[0] for p in marks.iterdir()) == ["1", "2", "3"]


def test_plain_workflow_untouched(tmp_path):
    """``import pydra`` and a ``cf`` run import nothing of the pool and write nothing outside the cache
    root."""
    script = textwrap.dedent("""
        import json, os, sys, threading
        import pydra
        from pydra.compose import python, workflow
        from pydra.engine.submitter import Submitter

        @python.define
        def Add(x: int) -> int:
            return x + 1

        @workflow.define
        def Wf(xs: list) -> list:
            return workflow.add(Add().split(x=xs), name="A").out

        def _orphaned(parent=os.getppid()):
            # exit with the test, however it ends (a timeout kills pytest, not this script)
            while os.getppid() == parent:
                threading.Event().wait(1.0)
            os._exit(3)

        if __name__ == "__main__":
            threading.Thread(target=_orphaned, daemon=True).start()
            with Submitter(worker="cf", cache_root=sys.argv[1]) as sub:
                out = sub(Wf(xs=[1, 2, 3])).outputs.out
            pool = [m for m in sys.modules if m.startswith("pydra.workers.pool")]
            print(json.dumps({"out": out, "pool": pool}))
        """)
    work = tmp_path / "cwd"
    work.mkdir()
    (tmp_path / "run.py").write_text(script)
    before = sorted(p.name for p in work.iterdir())
    proc = subprocess.Popen(
        [sys.executable, str(tmp_path / "run.py"), str(tmp_path / "cache")],
        cwd=work,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=POSIX,
    )
    try:
        stdout, stderr = proc.communicate(timeout=BOUND)
    finally:
        _kill_group(proc)  # the cf worker's processes go with the script's group
    assert proc.returncode == 0, stderr
    if POSIX:
        left = subprocess.run(
            ["pgrep", "-g", str(proc.pid)], capture_output=True, text=True
        )
        assert left.stdout.strip() == "", "the cf run left processes behind"
    out = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    record = json.loads(out.stdout.strip().splitlines()[-1])
    assert record == {"out": [2, 3, 4], "pool": []}
    assert sorted(p.name for p in work.iterdir()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache", "cwd", "run.py"]
    cache = tmp_path / "cache"
    assert not list(cache.rglob("pool.json")) and not list(cache.rglob("payloads"))


def test_second_coordinator_waits_then_pool_busy(tmp_path, tasks, pools):
    """A second coordinator on a pool whose coordinator is live waits; with ``coordinator_wait`` it
    raises `PoolBusy` naming the live one; once that one closes, a waiting one takes the next epoch.
    """
    import threading

    from pydra.workers.pool.pool import PoolBusy
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    first = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    with pytest.raises(PoolBusy, match="epoch 1"):
        _warm(
            tmp_path,
            pools,
            sub,
            substrate=ExternalSubstrate(),
            settings=_settings(coordinator_wait=0.2),
        )
    made = {}
    waiting = threading.Thread(
        target=lambda: made.update(
            pool=_warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
        )
    )
    waiting.start()
    first.close()
    waiting.join(BOUND)
    assert not waiting.is_alive()
    assert made["pool"].epoch == 2


def test_fenced_coordinator_ends_run_incomplete(tmp_path, tasks, pools):
    """A coordinator that sees a newer epoch writes nothing more and reports its open jobs
    ``NotRun(held=True)`` naming `PoolFenced`: the run ends `RunIncomplete`, and nothing ran.
    """
    from pydra.workers.pool.substrate import ExternalSubstrate

    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    pool.pool.take_epoch()  # a coordinator that started on the free pool at the same time
    with pytest.raises(RunIncomplete) as info:
        _drive(sub, tasks.Marks(xs=[1, 2], where=str(marks)), pool)
    reasons = [o.reason for _, o in info.value.not_run]
    assert len(reasons) == 2 and all(r.startswith("PoolFenced") for r in reasons)
    assert all(o.held for _, o in info.value.not_run)
    assert list(marks.iterdir()) == [] and _all_tickets(pool.root) == []


def test_cancel_withdraws_only_unclaimed_tickets(tmp_path, tasks, pools):
    """``cancel`` withdraws (reason ``cancel``) the jobs of a ticket whose claim it can take and leaves
    a ticket under a live claim to its holder; every cancelled job is reported ``NotRun``.
    """
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path, pools, sub, [_class(quantum_jobs=1)], substrate=ExternalSubstrate()
    )
    runner = GraphRunner(
        tasks.Many(xs=[1, 2]), sub.store, on_error="raise_at_end", submitter=sub
    )
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    handle = pool.submit(runner.ready().dispatch)
    _wait(lambda: len(_tickets(pool.root)) == 2, "two tickets")
    held, free = sorted(_tickets(pool.root), key=lambda t: t["jobs"][0]["checksum"])
    claim = pool.pool.take("cpu", _digest(held))
    try:
        pool.cancel(handle)
        assert handle.open == set()
        assert len(reported) == 2
        assert all(o == NotRun("cancelled", held=False) for _, o in reported)
        with _raw(pool.root) as raw:
            assert raw.outcomes("cpu", _digest(held)) == {}
            (withdrawn,) = raw.outcomes("cpu", _digest(free)).values()
        assert (withdrawn["outcome"], withdrawn["reason"]) == ("withdrawn", "cancel")
    finally:
        claim.release()


@pytest.mark.skipif(not POSIX, reason="the fake nvidia-smi is a shebang script")
def test_refused_ticket_replaced_one_class_up(tmp_path, tasks, pools, fake_nvidia_smi):
    """A worker of a class whose device memory was unknown at placement reads less than the job
    declares and refuses the ticket; the follower re-places the job on the next class of the ladder,
    under a fresh attempt, where it runs."""
    from pydra.workers.pool.substrate import LocalSubstrate

    fake_nvidia_smi(("NVIDIA H100", 40960), ("NVIDIA H100", 163840))
    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    small = _class("small", device="cuda", gpus=1, mem="1G")
    large = _class("large", device="cuda", gpus=1, mem="2G")
    pool = _warm(
        tmp_path,
        pools,
        sub,
        [small, large],
        substrate=LocalSubstrate(),
        settings=_settings(idle_polls=400),
    )
    runner = _drive(sub, tasks.BigGpuMark(x=1, where=str(marks)), pool)
    assert runner.done and not runner.failed
    assert len(list(marks.iterdir())) == 1
    (refused,) = _tickets(pool.root, "small")
    (ran,) = _tickets(pool.root, "large")
    assert refused["ladder"] == ["large"]
    entry = refused["jobs"][0]
    with _raw(pool.root) as raw:
        gens = [g for _, g in raw.generations(entry["checksum"], entry["fingerprint"])]
        (withdrawn,) = raw.outcomes("small", _digest(refused)).values()
    assert (withdrawn["outcome"], withdrawn["reason"]) == ("withdrawn", "refused")
    assert [g["ticket"] for g in gens] == [_digest(refused), _digest(ran)]
    assert gens[0]["attempt"] != gens[1]["attempt"]


def test_pydracli_pool_status_release_hold_collect(tmp_path, tasks, pools):
    """``pydracli pool status`` reports epochs, queues and holds; ``release-hold`` lifts a queue's
    holds; ``collect`` removes ended coordinators below the newest epoch and keeps a payload an
    unsettled ticket names."""
    from click.testing import CliRunner

    from pydra.scripts.cli import cli
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    first = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    runner = GraphRunner(
        tasks.Many(xs=[5]), sub.store, on_error="raise_at_end", submitter=sub
    )
    first.bind(lambda job, outcome: None)
    first.submit(runner.ready().dispatch)
    _wait(lambda: len(_tickets(first.root)) == 1, "the ticket")
    label = "ab" * 32
    first.pool.hold("cpu", 1, label)
    first.close()
    _warm(tmp_path, pools, sub, substrate=ExternalSubstrate()).close()
    root = str(first.root)
    cli_runner = CliRunner()

    status = json.loads(cli_runner.invoke(cli, ["pool", "status", root]).output)
    assert status["epochs"] == [
        {"epoch": 1, "ended": True},
        {"epoch": 2, "ended": True},
    ]
    assert status["queues"]["cpu"]["waiting_tickets"] == 1
    assert status["queues"]["cpu"]["holds"] == {label: 1}

    out = cli_runner.invoke(cli, ["pool", "release-hold", root, "--queue", "cpu"])
    assert out.exit_code == 0 and "1 hold(s) lifted" in out.output
    status = json.loads(cli_runner.invoke(cli, ["pool", "status", root]).output)
    assert status["queues"]["cpu"]["holds"] == {}

    out = cli_runner.invoke(cli, ["pool", "collect", root, "--payload-age", "0"])
    assert out.exit_code == 0 and "1 ended coordinator(s)" in out.output
    assert sorted(p.name for p in (first.root / "coordinator").iterdir()) == ["2"]
    assert (
        len(list((first.root / "payloads").iterdir())) == 1
    )  # its ticket is unsettled


# -- broken classes, blocking reads, queue caps and lagging listings --------------------


class NeverStarts:
    """A substrate whose workers never become live: every start is a failed start."""

    def __init__(self):
        self.handles = []
        self.starts = 0

    def attach(self, attachment):
        pass

    def check_root(self, root):
        pass

    def start(self, cls, n):
        from pydra.workers.pool.substrate import WorkerHandle

        self.starts += 1
        return [WorkerHandle(token=f"{self.starts:032x}", worker_class=cls.name)]

    def live(self):
        return []

    def stop(self, workers):
        pass

    def can_start(self, cls):
        return True


def _round(sub, tasks, xs, task=None, **kwargs):
    task = task or tasks.Many(xs=xs)
    runner = GraphRunner(
        task, sub.store, on_error="raise_at_end", submitter=sub, **kwargs
    )
    return runner.ready().dispatch


def _settle_ticket(root: Path, queue: str, digest: str, kind="failed", **fields):
    """Write ``kind`` for every job of a ticket under its current attempt, as a worker would."""
    with _raw(root) as raw:
        claim = raw.take(queue, digest)
        assert claim is not None
        for entry in claim.ticket.jobs:
            current = claim.current(entry)
            if current is not None:
                claim.outcome(entry, current[1]["attempt"], kind, **fields)
        claim.release()


NOT_CONTENT = {"kind": "infrastructure", "type": "Probe", "message": "a probe"}


def test_job_placed_after_its_class_broke_is_held(tmp_path, tasks, pools):
    """Jobs submitted to a class after it broke are reported ``NotRun(held=True)`` at once, naming
    the broken class; none is written or left unreported."""
    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=NeverStarts(),
        faults=FaultBudgets(start_failures=2),
    )
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    pool.submit(_round(sub, tasks, [1]))
    _wait(lambda: len(reported) == 1, "the first job held when the class broke")
    second = _round(sub, tasks, [2])
    pool.submit(second)
    assert len(reported) == 2  # held by submit itself, before it returns
    for _, outcome in reported:
        assert outcome.held and "worker class 'cpu' is broken" in outcome.reason
    assert reported[1][0] is second[0]
    assert len(_tickets(pool.root)) == 1  # only the first job's ticket was written


def test_submit_and_cancel_stay_responsive_while_a_result_read_blocks(
    tmp_path, tasks, pools, monkeypatch
):
    """While the follower is blocked reading a result, ``submit`` and ``cancel`` return: the pool's
    lock is not held across a result read."""
    import threading

    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    entered, release = threading.Event(), threading.Event()

    def blocking(job, attempt, settled):
        entered.set()
        release.wait()  # set by the test's finally, pass or fail
        return NotRun("read", held=False)

    monkeypatch.setattr(pool, "_result", blocking)
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    (first,) = _round(sub, tasks, [1])
    pool.submit([first])
    # followed by the pool: its put is whole (the ticket file is written before the job's
    # generation, and a ticket settled in between would get no outcome for the job)
    _wait(lambda: len(pool._tickets) == 1, "the first ticket followed")
    _settle_ticket(pool.root, "cpu", _digest(_tickets(pool.root)[0]), "done")
    try:
        assert entered.wait(BOUND), "the follower did not read the result"
        returned = threading.Event()

        def submit_and_cancel():
            handle = pool.submit(_round(sub, tasks, [2]))
            pool.cancel(handle)
            returned.set()

        threading.Thread(target=submit_and_cancel, daemon=True).start()
        assert returned.wait(
            BOUND / 4
        ), "submit or cancel blocked behind the result read"
        assert not release.is_set()
        assert [o.reason for _, o in reported] == ["cancelled"]
    finally:
        release.set()
    _wait(lambda: len(reported) == 2, "the blocked read's report")
    assert reported[1][0] is first


def test_max_queued_jobs_caps_the_tickets_written(tmp_path, tasks, pools):
    """With ``max_queued_jobs=2`` five one-job tickets are written two at a time, as tickets
    settle."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        [_class(quantum_jobs=1)],
        substrate=ExternalSubstrate(),
        settings=_settings(max_queued_jobs=2),
    )
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    pool._queue_jobs(_round(sub, tasks, list(range(5))), handle=None)
    pool._poll()
    assert len(_tickets(pool.root)) == 2
    pool._poll()
    assert len(_tickets(pool.root)) == 2  # nothing settled: no room
    for ticket in _tickets(pool.root):
        _settle_ticket(pool.root, "cpu", _digest(ticket), fault=NOT_CONTENT)
    pool._poll()  # reads the two outcomes
    pool._poll()  # writes two more
    assert len(reported) == 2 and len(_tickets(pool.root)) == 4


def test_listing_behind_is_read_at_the_next_poll(tmp_path, tasks, pools, monkeypatch):
    """A settled mark ahead of the outcomes listed (an attribute cache) does not stall the poll: the
    follower does not wait for the listing, and reads the ticket again at a later poll.
    """
    from pydra.workers.pool import storage
    from pydra.workers.pool.storage import queue_label
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    pool._queue_jobs(_round(sub, tasks, [1]), handle=None)
    pool._poll()
    digest = _digest(_tickets(pool.root)[0])
    settled = pool.root / "queues" / queue_label("cpu") / "settled"
    (settled / f"{digest}-1").touch()  # a mark whose outcome is not listed yet

    def no_waiting(seconds):
        raise AssertionError("the follower waited for a listing")

    monkeypatch.setattr(storage.time, "sleep", no_waiting)
    pool._poll()
    assert reported == []
    monkeypatch.undo()
    (settled / f"{digest}-1").unlink()
    _settle_ticket(pool.root, "cpu", digest, fault=NOT_CONTENT)
    pool._poll()
    assert len(reported) == 1


def test_failed_starts_back_off(tmp_path, tasks, pools):
    """After a failed start the class waits ``poll_s`` before the next, doubling, on the storage-free
    clock the follower passes."""
    sub = _submitter(tmp_path)
    substrate = NeverStarts()
    pool = _warm(tmp_path, pools, sub, substrate=substrate)
    pool.bind(lambda job, outcome: None)
    pool._queue_jobs(_round(sub, tasks, [1]), handle=None)
    pool._write()
    poll = pool.settings.poll_s
    starts = []
    for now in (100.0, 100.0, 1.5, 1.5, 3.0, 4.0):
        pool._scale(now if now == 100.0 else 100.0 + now * poll)
        starts.append(substrate.starts)
    # a start, its failure; after poll_s the next, its failure; then 2 x poll_s
    assert starts == [1, 1, 2, 2, 2, 3]


class RefusesStarts(NeverStarts):
    """A substrate whose every start raises, as an ``sbatch`` the scheduler refuses does."""

    def start(self, cls, n):
        from pydra.workers.pool.substrate import SubstrateError

        self.starts += 1
        raise SubstrateError("sbatch: error: invalid partition specified: gpuq")


class Pending(NeverStarts):
    """A substrate whose workers stay live, never ready, until `leave` ends them."""

    def start(self, cls, n):
        handles = super().start(cls, n)
        self.handles.extend(handles)
        return handles

    def live(self):
        return list(self.handles)

    def leave(self):
        self.handles.clear()


class _Events(logging.Handler):
    """The ``pydra.pool`` records emitted while it is installed."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        logging.getLogger("pydra.pool").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("pydra.pool").removeHandler(self)

    def named(self, event: str) -> list[logging.LogRecord]:
        return [r for r in self.records if getattr(r, "event", None) == event]


def test_a_failed_ticket_write_puts_the_jobs_back(tmp_path, tasks, pools, monkeypatch):
    """A ticket write that raises leaves its jobs waiting, never dropped: they are still wanted, the
    next write puts them, and the failure is a ``write_failed`` pool event."""
    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub, substrate=Recording())
    pool.bind(lambda job, outcome: None)
    pool._queue_jobs(_round(sub, tasks, [1, 2]), handle=None)
    put = pool.pool.put
    calls = []

    def failing_once(queue, ticket):
        calls.append(queue)
        if len(calls) == 1:
            raise OSError("No space left on device")
        return put(queue, ticket)

    monkeypatch.setattr(pool.pool, "put", failing_once)
    with _Events() as events:
        pool._write()
    assert _tickets(pool.root) == []
    assert sorted(j.job.state_index for j in pool._waiting["cpu"]) == [0, 1]
    assert pool._demand() == {"cpu": 1}  # still wanted: a worker is started for them
    [event] = events.named("write_failed")
    assert "No space left on device" in event.cause and event.queue == "cpu"
    pool._write()
    assert sum(len(t["jobs"]) for t in _tickets(pool.root)) == 2
    assert "cpu" not in pool._waiting


def test_ticket_writes_that_keep_failing_hold_the_jobs(
    tmp_path, tasks, pools, monkeypatch
):
    """Ticket writes that keep failing (``start_deadline_failures`` times over ``start_deadline_s``)
    break the class: the submitter gets ``NotRun(held=True)`` naming the write error, and the run
    ends rather than hanging.
    """

    def failing(queue, ticket):
        raise OSError("Read-only file system")

    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=Recording(),
        settings=_settings(start_deadline_s=0.5),
    )
    monkeypatch.setattr(pool.pool, "put", failing)
    with pytest.raises(RunIncomplete) as info:
        _drive(sub, tasks.Many(xs=[1, 2]), pool)
    not_run = info.value.not_run
    assert len(not_run) == 2
    for _, outcome in not_run:
        assert isinstance(outcome, NotRun) and outcome.held
        assert "Read-only file system" in outcome.reason


def test_refused_starts_hold_the_jobs_before_the_backoff_ladder_ends(
    tmp_path, tasks, pools
):
    """A class whose every start is refused breaks once ``start_deadline_failures`` starts have failed
    over ``start_deadline_s``: its jobs are ``NotRun(held=True)`` naming the latest start error,
    well before ``start_failures`` backed-off starts would have broken it."""
    budget = FaultBudgets().start_failures
    settings = _settings(poll_s=0.5, start_deadline_s=1.0)
    # the backoff after failed starts 1 and 2 (0.5 s, then 1 s) puts start 3 past the
    # deadline, so the deadline breaks the class at start 3, whatever the machine's speed
    assert settings.start_deadline_failures == 3 < budget
    sub = _submitter(tmp_path)
    substrate = RefusesStarts()
    pool = _warm(tmp_path, pools, sub, substrate=substrate, settings=settings)
    with pytest.raises(RunIncomplete) as info:
        _drive(sub, tasks.Many(xs=[1, 2]), pool)
    assert substrate.starts == settings.start_deadline_failures
    assert len(info.value.not_run) == 2
    for _, outcome in info.value.not_run:
        assert isinstance(outcome, NotRun) and outcome.held
        assert "worker class 'cpu' is broken" in outcome.reason
        assert "invalid partition specified: gpuq" in outcome.reason


def test_a_pending_start_is_not_cut_short(tmp_path, tasks, pools):
    """A start still pending when ``start_deadline_s`` has passed (a slow model load, a queued Slurm
    job) is waited for; once it too leaves without a ready mark, the class breaks at once.
    """
    sub = _submitter(tmp_path)
    substrate = Pending()
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=substrate,
        settings=_settings(
            poll_s=1.0, start_deadline_s=10.0, start_deadline_failures=2
        ),
    )
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    pool._queue_jobs(_round(sub, tasks, [1]), handle=None)
    pool._write()
    pool._scale(100.0)  # starts one
    substrate.leave()
    pool._scale(100.5)  # it left without a ready mark: failed start 1
    pool._scale(102.0)  # after the backoff, a second start, which stays pending
    assert substrate.starts == 2 and not pool._broken
    pool._scale(150.0)  # long past start_deadline_s, but a start is pending
    assert not pool._broken and reported == []
    substrate.leave()
    pool._scale(151.0)  # the pending start failed too
    assert "cpu" in pool._broken
    assert len(reported) == 1 and reported[0][1].held


def test_start_and_listing_failures_are_pool_events(tmp_path, tasks, pools):
    """A failed listing, a refused start and a broken class are each a pool event (a ``pydra.pool``
    log record carrying ``event``), not only a warning's text."""

    class ListingFails(RefusesStarts):
        listing = True

        def live(self):
            if self.listing:
                raise OSError("squeue: error: slurm_load_jobs error: timed out")
            return []

    sub = _submitter(tmp_path)
    substrate = ListingFails()
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=substrate,
        settings=_settings(poll_s=1.0, start_deadline_s=5.0, start_deadline_failures=2),
    )
    pool.bind(lambda job, outcome: None)
    pool._queue_jobs(_round(sub, tasks, [1]), handle=None)
    pool._write()
    with _Events() as events:
        pool._scale(100.0)
        substrate.listing = False
        pool._scale(101.0)
        pool._scale(110.0)
    [listing] = events.named("listing_failed")
    assert "timed out" in listing.cause
    started = events.named("start_failed")
    assert "invalid partition" in started[0].cause
    assert started[0].worker_class == "cpu" and started[0].failures == 1
    [broken] = events.named("class_broken")
    assert broken.worker_class == "cpu" and "invalid partition" in broken.cause


def test_a_stale_streak_never_breaks_a_class_without_a_start(tmp_path, tasks, pools):
    """A failure streak ends when its class has no waiting jobs: after a failed start, a cancelled
    run and an hour idle, a new run's first poll starts a worker rather than breaking the class.
    """
    from pydra.engine.executor import Handle

    sub = _submitter(tmp_path)
    substrate = Pending()
    substrate.stop = lambda workers: [
        substrate.handles.remove(w) for w in workers if w in substrate.handles
    ]
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=substrate,
        settings=_settings(
            poll_s=1.0, start_deadline_s=120.0, start_deadline_failures=1
        ),
    )
    pool.bind(lambda job, outcome: None)
    jobs = _round(sub, tasks, [1])
    handle = Handle(pool, tuple(jobs), set(jobs))
    pool._queue_jobs(jobs, handle=handle)
    pool._write()
    pool._scale(100.0)  # start 1
    substrate.leave()
    pool._scale(100.5)  # failed start 1
    pool._scale(102.0)  # start 2, pending
    pool.cancel(handle)  # the run goes away
    pool._scale(
        103.0
    )  # no waiting work: the pending surplus is stopped, the streak ends
    starts = substrate.starts
    pool._queue_jobs(_round(sub, tasks, [2]), handle=None)
    pool._write()
    pool._scale(4000.0)  # an hour later, a new run
    assert "cpu" not in pool._broken
    assert substrate.starts == starts + 1


def test_ticket_write_streak_ends_with_its_waiting_jobs(
    tmp_path, tasks, pools, monkeypatch
):
    """A queue's write streak ends when nothing waits to be written: one failed write, its jobs
    cancelled, and a failed write an hour later does not break the class."""
    from pydra.engine.executor import Handle

    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=Recording(),
        settings=_settings(start_deadline_s=0.05, start_deadline_failures=2),
    )
    pool.bind(lambda job, outcome: None)

    def failing(queue, ticket):
        raise OSError("Stale file handle")

    monkeypatch.setattr(pool.pool, "put", failing)
    import time
    import threading

    jobs = _round(sub, tasks, [1])
    handle = Handle(pool, tuple(jobs), set(jobs))
    pool._queue_jobs(jobs, handle=handle)
    pool._write()  # failed write 1
    pool.cancel(handle)
    pool._write()  # nothing waits: the streak ends
    # an hour on, on the monotonic clock as this thread (the code under test) reads it
    real_monotonic, me = time.monotonic, threading.current_thread()
    with monkeypatch.context() as later:
        later.setattr(
            time,
            "monotonic",
            lambda: real_monotonic()
            + (3600 if threading.current_thread() is me else 0),
        )
        pool._queue_jobs(_round(sub, tasks, [2]), handle=None)
        pool._write()  # failed write 1 of a fresh streak
    assert "cpu" not in pool._broken


class _Vanishing(dict):
    """A dict whose entries another thread removes while it is being iterated."""

    def __iter__(self):
        keys = list(super().__iter__())
        for key in keys:
            super().pop(key, None)
        return iter(keys)


def test_ending_streaks_races_no_submit(tmp_path, tasks, pools):
    """A submit that ends a write streak while the follower ends it too neither raises nor leaves a
    job half-registered: every job of the call is open, and none is reported."""
    import threading
    import time

    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub, substrate=Recording())
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    pool._write_failing = _Vanishing(cpu=(0.0, 1, "OSError: Stale file handle"))
    jobs = _round(sub, tasks, [1, 2])
    pool.submit(jobs)  # work arrives at an idle class: its streaks end
    assert all(job in pool._open for job in jobs) and reported == []

    stop, errors = threading.Event(), []

    def follower():
        while not stop.is_set():
            try:
                with pool._lock:
                    pool._write_failing["cpu"] = (0.0, 1, "OSError")
                pool._end_streaks("cpu")
            except Exception as err:  # pragma: no cover - the failure being tested
                errors.append(err)

    thread = threading.Thread(target=follower)
    thread.start()
    try:
        deadline = time.monotonic() + 0.5
        n = 10
        while time.monotonic() < deadline:
            more = _round(sub, tasks, [n, n + 1])
            n += 2
            with pool._lock:  # an idle class again, so the next submit ends its streaks
                pool._open.clear()
                pool._by_key.clear()
            pool._queue_jobs(more, handle=None)
            assert all(job in pool._open for job in more)
    finally:
        stop.set()
        thread.join()
    assert errors == [] and reported == []
    assert "cpu" not in pool._broken


def test_the_deadline_never_breaks_ahead_of_a_due_retry(tmp_path, tasks, pools):
    """With ``poll_s`` longer than ``start_deadline_s``, one refused start does not break the class:
    the backed-off retry runs, and the class breaks only at the failed attempt that completes
    ``start_deadline_failures``."""
    sub = _submitter(tmp_path)
    substrate = RefusesStarts()
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=substrate,
        settings=_settings(poll_s=150.0, start_deadline_s=120.0),
    )
    pool.bind(lambda job, outcome: None)
    pool._queue_jobs(_round(sub, tasks, [1]), handle=None)
    pool._write()
    pool._scale(0.0)  # refused 1
    pool._scale(150.0)  # the retry is due: refused 2
    assert substrate.starts == 2 and "cpu" not in pool._broken
    pool._scale(
        299.0
    )  # long past the deadline, but no retry is due yet: nothing judged
    assert substrate.starts == 2 and "cpu" not in pool._broken
    pool._scale(450.0)  # refused 3: the streak has its failures and its time
    assert substrate.starts == 3 and "cpu" in pool._broken


def test_an_unencodable_job_fails_alone(tmp_path, tasks, pools, monkeypatch):
    """A job its codec cannot encode is ``Failed`` with the error on its own: the jobs beside it are
    written, nothing is held, and the class is not broken."""
    from pydra.engine.runner import Failed
    from pydra.workers.pool.codec import PrivatePickle

    encode = PrivatePickle.encode

    def unpicklable_first(self, job):
        if job.state_index == 0:
            raise TypeError("cannot pickle '_thread.lock' object")
        return encode(self, job)

    monkeypatch.setattr(PrivatePickle, "encode", unpicklable_first)
    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=Recording(),
        settings=_settings(start_deadline_s=0.05, start_deadline_failures=1),
    )
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    pool._queue_jobs(_round(sub, tasks, [1, 2, 3]), handle=None)
    pool._write()
    [(job, outcome)] = reported
    assert job.state_index == 0 and isinstance(outcome, Failed)
    assert "cannot pickle" in str(outcome.error)
    assert sum(len(t["jobs"]) for t in _tickets(pool.root)) == 2
    assert not pool._broken and "cpu" not in pool._waiting


class Flaky:
    """A `LocalSubstrate` whose starts are refused while ``refuse`` is set."""

    def __new__(cls):
        from pydra.workers.pool.substrate import LocalSubstrate, SubstrateError

        class _Flaky(LocalSubstrate):
            refuse = True

            def start(self, worker_class, n):
                if self.refuse:
                    raise SubstrateError("sbatch: error: Socket timed out")
                return super().start(worker_class, n)

        return _Flaky()


@pytest.mark.parametrize("lift", ["release_hold", "retry"])
def test_a_broken_class_recovers(tmp_path, tasks, pools, lift):
    """A class broken by refused starts recovers: after `WarmPool.release_hold`, or on a ``retry``
    of the job it held, the next start is a fresh attempt, and once starts work the job runs.
    """
    sub = _submitter(tmp_path)
    substrate = Flaky()
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=substrate,
        settings=_settings(start_deadline_s=0.2, start_deadline_failures=2),
    )
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    jobs = _round(sub, tasks, [7])
    pool.submit(jobs)
    _wait(lambda: reported, "the job held when the class broke")
    [(_, held)] = reported
    assert isinstance(held, NotRun) and held.held and "Socket timed out" in held.reason
    substrate.refuse = False
    if lift == "release_hold":
        assert pool.release_hold("cpu") == ["cpu"]
    reported.clear()
    pool.submit(jobs)  # the runner's retry hands the held job out again
    _wait(lambda: reported, "the retried job's outcome")
    [(job, outcome)] = reported
    assert job is jobs[0]
    assert not isinstance(outcome, NotRun) and not outcome.errored
    assert "cpu" not in pool._broken


def test_served_result_read_without_the_visibility_wait(
    tmp_path, tasks, pools, monkeypatch
):
    """A ``done`` a taker settled from the store (``served``) is read without the store's visibility
    retry, which only a run's own ``_committed.json`` needs; a plain ``done`` is read settled.
    """
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path, pools, sub, [_class(quantum_jobs=1)], substrate=ExternalSubstrate()
    )
    reads = []

    def recording(job, attempt, settled):
        reads.append((job.task.x, settled))
        return NotRun("read", held=False)

    monkeypatch.setattr(pool, "_result", recording)
    pool.bind(lambda job, outcome: None)
    pool._queue_jobs(_round(sub, tasks, [1, 2]), handle=None)
    pool._poll()
    one, two = sorted(_tickets(pool.root), key=lambda t: t["jobs"][0]["checksum"])
    _settle_ticket(pool.root, "cpu", _digest(one), "done", served=True)
    _settle_ticket(pool.root, "cpu", _digest(two), "done")
    pool._poll()
    assert sorted(settled for _, settled in reads) == [False, True]


def test_content_fault_reported_as_the_jobs_errored_result(tmp_path, tasks, pools):
    """A task that raises is ``failed`` with a content fault; the pool reports the job's own errored
    result (its ``_error.json``), not a bare ``Failed``."""
    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub)
    runner = _drive(sub, tasks.Fails(xs=[1]), pool, check=False)
    ((_, outcome),) = runner.reported
    assert getattr(outcome, "errored", False) is True
    assert runner.errors and "always fails" in "\n".join(runner.errors)


def test_tickets_are_cut_at_a_node_change(tmp_path, tasks, pools):
    """Two nodes of one round on one queue become two tickets, one per node."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    pool.bind(lambda job, outcome: None)
    jobs = _round(sub, tasks, None, task=tasks.TwoNodes(xs=[1, 2]))
    assert len(jobs) == 4
    pool._queue_jobs(jobs, handle=None)
    pool._poll()
    tickets = _tickets(pool.root)
    assert sorted((t["node"], len(t["jobs"])) for t in tickets) == [("A", 2), ("B", 2)]


def test_follower_follows_a_moved_job(tmp_path, tasks, pools):
    """A job moved to another ticket while it is followed (a split, a move up the ladder) is followed
    into that ticket by its newest generation, and reported from there."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(tmp_path, pools, sub, substrate=ExternalSubstrate())
    reported = []
    pool.bind(lambda job, outcome: reported.append((job, outcome)))
    pool._queue_jobs(_round(sub, tasks, [1]), handle=None)
    pool._poll()
    (old,) = _tickets(pool.root)
    with _raw(pool.root) as raw:
        claim = raw.take("cpu", _digest(old))
        (new,) = claim.move([list(claim.ticket.jobs)], "cpu", cause="replaced")
        claim.release()
    pool._poll()  # reads the withdrawal and follows the job into the new ticket
    assert reported == []
    _settle_ticket(pool.root, "cpu", new, fault=NOT_CONTENT)
    pool._poll()
    ((_, outcome),) = reported
    assert outcome.error.fault["type"] == "Probe"


def test_judges_followed_claims_only_when_idle(tmp_path, tasks, pools):
    """A followed ticket of a class this coordinator does not start is judged (its claim tried) only
    once its settled count has not moved for ``stale_after``, at most once per ``stale_after``; while
    the claim is live it is followed, and once it can be taken the job moves to a class this
    coordinator starts."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    first = _warm(
        tmp_path, pools, sub, [_class("cpu-a")], substrate=ExternalSubstrate()
    )
    first.bind(lambda job, outcome: None)
    jobs = _round(sub, tasks, [1])
    first.submit(jobs)
    _wait(lambda: len(_tickets(first.root, "cpu-a")) == 1, "the cpu-a ticket")
    first.close()
    digest = _digest(_tickets(first.root, "cpu-a")[0])
    raw = _raw(first.root)
    holder = raw.take("cpu-a", digest)  # a live claim, beaten by its thread
    try:
        second = _warm(tmp_path, pools, sub, [_class("cpu-b")], substrate=Recording())
        second.bind(lambda job, outcome: None)
        takes = []
        take = second.pool.take

        def counting(queue, digest, **kwargs):
            takes.append(queue)
            return take(queue, digest, **kwargs)

        second.pool.take = counting
        second._queue_jobs(_round(sub, tasks, [1]), handle=None)
        second._poll(now=0.0)  # tries the claim once at resume, and follows
        (followed,) = second._tickets.values()
        assert followed.foreign and len(takes) == 1
        stale = second.settings.lease.stale_after
        followed.moved, followed.judged = 100.0, 0.0
        counts = []
        for now in (101.0, 100.0 + stale, 101.0 + stale):
            second._poll(now=now)
            counts.append(len(takes))
        assert counts == [1, 2, 2]  # idle for stale_after, then not again within it
    finally:
        holder.release()
        raw.close()
    second._poll(now=100.0 + 2 * stale)
    assert len(takes) == 3
    (moved,) = _tickets(second.root, "cpu-b")
    gens = _generations(second.root, sub, jobs[0])
    assert [g["cause"] for g in gens] == ["put", "replaced"]
    assert gens[1]["ticket"] == _digest(moved)


@pytest.mark.skipif(not POSIX, reason="the worker is started with a new session")
def test_claimed_ticket_not_withdrawn(tmp_path, tasks, pools, processes):
    """A ticket of a class the resumed coordinator does not start, under a live claim (its worker is
    running the job), is followed: it is not withdrawn or moved, the job runs once in its holder's
    work directory, and the result is read from there."""
    from pydra.workers.pool.substrate import ExternalSubstrate, LocalSubstrate

    gate = tmp_path / "gate"
    gate.mkdir()
    sub = _submitter(tmp_path)
    first = _warm(
        tmp_path, pools, sub, [_class("cpu-a")], substrate=ExternalSubstrate()
    )
    first.bind(lambda job, outcome: None)
    first.submit(_round(sub, tasks, None, task=tasks.OneGate(where=str(gate))))
    _wait(lambda: len(_tickets(first.root, "cpu-a")) == 1, "the cpu-a ticket")
    first.close()
    _hand_worker(first.root, _class("cpu-a"), processes, tmp_path)
    _wait(lambda: (gate / "first").exists(), "the holder to start the job")
    second = _warm(tmp_path, pools, sub, [_class("cpu-b")], substrate=LocalSubstrate())
    import threading

    box = {}
    driver = threading.Thread(
        target=lambda: box.update(
            runner=_drive(sub, tasks.OneGate(where=str(gate)), second)
        )
    )
    driver.start()
    try:
        _wait(lambda: second._tickets, "the resumed coordinator to follow the ticket")
    finally:
        (gate / "go").touch()
        driver.join(BOUND)
    runner = box["runner"]
    ((_, result),) = runner.reported
    assert result.outputs.out == "first"  # the holder's run, not a second one
    assert _tickets(second.root, "cpu-b") == [] and second.substrate.handles == []
    (ticket,) = _tickets(second.root, "cpu-a")
    with _raw(second.root) as raw:
        outcomes = raw.outcomes("cpu-a", _digest(ticket))
        entry = ticket["jobs"][0]
        gens = raw.generations(entry["checksum"], entry["fingerprint"])
    assert [o["outcome"] for o in outcomes.values()] == ["done"]
    assert [g["cause"] for _, g in gens] == ["put"]


def test_rerun_after_collect_fresh_attempt(tmp_path, tasks, pools):
    """A job run, its ticket collected (``pydracli pool collect``), then run again: the rerun gets the
    next generation and a fresh attempt, never the first one's, so the two runs never share a work
    directory."""
    from pydra.workers.pool.pool import collect_pool

    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    first = _drive(
        sub, tasks.Marks(xs=[1], where=str(marks)), _warm(tmp_path, pools, sub)
    )
    assert first.done
    (ticket,) = _tickets(tmp_path / "pool")
    # the driver sees the job done before the worker's release resolves the ticket
    collected = []

    def collect() -> bool:
        collected.append(collect_pool(tmp_path / "pool", payload_age=0)["tickets"])
        return sum(collected) == 1

    _wait(collect, "the ticket to resolve and be collected")
    assert _tickets(tmp_path / "pool") == []
    again = _drive(
        sub,
        tasks.Marks(xs=[1], where=str(marks)),
        _warm(tmp_path, pools, sub),
        rerun=True,
    )
    assert again.done and len(list(marks.iterdir())) == 2  # ran twice
    entry = ticket["jobs"][0]
    with _raw(tmp_path / "pool") as raw:
        gens = [g for _, g in raw.generations(entry["checksum"], entry["fingerprint"])]
    assert [g["cause"] for g in gens][-1] == "put"
    attempts = {g["attempt"] for g in gens}
    assert len(attempts) == len(gens) and gens[-1]["attempt"].endswith(".1")


HOLD_RELEASE = """
import os

if os.environ.get("PYDRA_TEST_HOLD_RELEASE"):
    import time
    from pathlib import Path

    from pydra.workers.pool import storage

    root = Path(os.environ["PYDRA_TEST_HOLD_RELEASE"])
    release = storage.Claim.release

    def held(self, *args, **kwargs):
        # the release of a claim whose ticket is resolved waits until the worker is told to stop
        if self.resolved():
            deadline = time.monotonic() + 120
            while not list(root.glob("workers/*/stop.json")):
                if time.monotonic() > deadline:
                    break
                time.sleep(0.05)
        return release(self, *args, **kwargs)

    storage.Claim.release = held
"""

# stopping a worker waits this long before it kills; a cap on the drain, not a time it should take
STOP_WAIT = 30.0


@pytest.fixture
def windows_stop(monkeypatch):
    """Stop workers as on Windows: `LocalSubstrate` takes its Windows path, and any signal sent to
    a process kills it at once (TerminateProcess). The signals sent are recorded by pid.
    """
    from pydra.workers.pool import substrate

    monkeypatch.setattr(substrate, "IS_WINDOWS", True)
    signals = []

    def terminate(self, sig):
        signals.append((self.pid, sig))
        os.kill(self.pid, getattr(signal, "SIGKILL", signal.SIGTERM))

    monkeypatch.setattr(subprocess.Popen, "send_signal", terminate)
    return signals


def _only_the_mark_ends(**kwargs):
    """Pool settings under which a worker exits on its stop mark only: never idle or orphaned."""
    return _settings(idle_polls=10**6, orphan_after_s=10**6, **kwargs)


def test_stop_without_signals_lets_the_release_finish(
    tmp_path, tasks, pools, monkeypatch, windows_stop
):
    """A pool closed while its worker is between a ticket's ``done`` outcome and its release lets
    the release finish, so the ticket resolves, and the worker exits on its stop mark without a
    signal."""
    from pydra.workers.pool.pool import collect_pool
    from pydra.workers.pool.substrate import LocalSubstrate

    hold = tmp_path / "hold"
    hold.mkdir()
    (hold / "sitecustomize.py").write_text(HOLD_RELEASE)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(hold), os.environ.get("PYTHONPATH", "")])
    )
    monkeypatch.setenv("PYDRA_TEST_HOLD_RELEASE", str(tmp_path / "pool"))
    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=LocalSubstrate(stop_wait=STOP_WAIT),
        settings=_only_the_mark_ends(),
    )
    assert _drive(sub, tasks.Marks(xs=[1], where=str(marks)), pool).done
    assert collect_pool(tmp_path / "pool", payload_age=0)["tickets"] == 1
    (handle,) = pool.substrate.handles
    assert handle.process.poll() == 0
    assert [sig for pid, sig in windows_stop if pid == handle.process.pid] == []


def test_stop_mark_drains_a_call_longer_than_the_renewal(
    tmp_path, tasks, pools, windows_stop
):
    """A worker inside a call when its pool closes drains on the stop mark as on SIGTERM: it stops
    waiting for the call after ``drain_s`` and releases its claim before ``stop_wait`` kills it,
    though its renewal interval, when its call monitor wakes, is longer than ``stop_wait``.
    """
    from pydra.workers.pool.storage import PoolRoot
    from pydra.workers.pool.substrate import LocalSubstrate

    lease = LeaseSettings(
        stale_after=3 * STOP_WAIT, renew_fraction=0.4, attribute_delay=0.0
    )
    assert lease.renew_interval > STOP_WAIT
    gate = tmp_path / "gate"
    gate.mkdir()
    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=LocalSubstrate(stop_wait=STOP_WAIT),
        settings=_only_the_mark_ends(lease=lease, drain_s=0.5),
    )
    pool.bind(lambda job, outcome: None)
    pool.submit(_round(sub, tasks, None, task=tasks.OneGate(where=str(gate))))
    _wait(lambda: (gate / "first").exists(), "the worker to start the job")
    pool.close()  # the job never ends: only a drain releases its claim
    (handle,) = pool.substrate.handles
    assert handle.process.poll() == 0
    assert [sig for pid, sig in windows_stop if pid == handle.process.pid] == []
    with PoolRoot.open(tmp_path / "pool", lease, create=False, probe=False) as raw:
        claim = raw.claim(
            "cpu"
        )  # a claim left unreleased would be live for stale_after
        assert claim is not None
        claim.release()


def test_stop_wait_shorter_than_the_drain_is_refused(tmp_path, pools):
    """A `LocalSubstrate` whose ``stop_wait`` is shorter than a worker needs to notice its stop
    mark (``poll_s``), drain (``drain_s``) and release (`release_patience`) is refused.
    """
    from pydra.workers.pool.substrate import LocalSubstrate

    sub = _submitter(tmp_path)
    with pytest.raises(ValueError, match="stop_wait"):
        _warm(
            tmp_path,
            pools,
            sub,
            substrate=LocalSubstrate(stop_wait=1.0),
            settings=_settings(drain_s=5.0),
        )


def test_collect_sweeps_the_stop_mark_of_a_worker_that_never_started(tmp_path):
    """Stopping a worker before it started leaves its directory holding only its stop mark.
    Collect removes it once older than ``payload_age`` and than the pool's
    ``max_honoured_stale_after``, the floor that covers a worker still starting; it keeps a
    younger one and a started worker's."""
    from pydra.workers.pool.pool import collect_pool
    from pydra.workers.pool.storage import STOP_MARK, PoolRoot

    never, starting, started = "a" * 32, "e" * 32, "b" * 32
    with PoolRoot.open(tmp_path / "pool", LEASE) as raw:
        for token in (never, starting, started):
            raw.write_worker_record(token, STOP_MARK, {"at": 0})
        raw.write_worker_record(started, "ready.json", {"class": "cpu"})
    workers = tmp_path / "pool" / "workers"
    old = workers.stat().st_mtime - LEASE.max_honoured_stale_after - 1
    for token in (never, started):
        os.utime(workers / token / STOP_MARK, (old, old))
    report = collect_pool(tmp_path / "pool", payload_age=0)
    assert report["workers"] == [never]
    assert not (workers / never).exists()
    assert (workers / starting / STOP_MARK).exists()
    assert (workers / started / STOP_MARK).exists()


def test_local_substrate_needs_the_pools_drain_bound(tmp_path):
    """`LocalSubstrate` checks ``stop_wait`` against the drain bound of the settings the
    attachment carries, and refuses an attachment without them; its default ``stop_wait`` adds
    the shutdown bound to the drain bound."""
    from pydra.workers.pool.substrate import (
        Attachment,
        LocalSubstrate,
        drain_bound,
        shutdown_bound,
    )

    settings = _settings()
    with pytest.raises(ValueError, match="settings"):
        LocalSubstrate().attach(Attachment(tmp_path, tmp_path / "settings.json"))
    local = LocalSubstrate()
    local.attach(Attachment(tmp_path, tmp_path / "s.json", pool_settings=settings))
    assert local.stop_wait == drain_bound(settings) + shutdown_bound(settings)
    with pytest.raises(ValueError, match="stop_wait"):
        LocalSubstrate(stop_wait=drain_bound(settings) / 2).attach(
            Attachment(tmp_path, tmp_path / "s.json", pool_settings=settings)
        )


def test_a_settle_removes_its_job_without_scanning_the_backlog(monkeypatch):
    """`_unwait` costs the same for any number of waiting jobs: it compares no other job."""
    from pydra.workers.pool import pool as pool_module

    compared = []
    monkeypatch.setattr(
        pool_module._Job,
        "__eq__",
        lambda self, other: compared.append(1) or self is other,
    )
    monkeypatch.setattr(pool_module._Job, "__hash__", object.__hash__)
    coordinator = pool_module.WarmPool.__new__(pool_module.WarmPool)
    coordinator._waiting, coordinator._unchecked = {}, {}
    for n in (10, 10_000):
        jobs = [pool_module._Job.__new__(pool_module._Job) for _ in range(n)]
        for j in jobs:
            j.queue = "cpu"
            coordinator._enqueue(j)
        compared.clear()
        coordinator._unwait(jobs[-1])
        assert compared == [], n
        assert len(coordinator._waiting["cpu"]) == n - 1
        coordinator._waiting.clear()


@pytest.mark.parametrize("error, raised", [("corrupt", False), ("bug", True)])
def test_an_unreadable_refusing_ticket_is_logged_and_a_bug_propagates(
    monkeypatch, caplog, error, raised
):
    """`_refused` reads the ticket that refused a job: a corrupt ticket is logged and the job
    placed as new; any other exception is a bug and propagates."""
    from pydra.workers.pool import pool as pool_module
    from pydra.workers.pool.storage import CorruptTicket

    coordinator = pool_module.WarmPool.__new__(pool_module.WarmPool)

    class Storage:
        def _by_label(self, label):
            return type("Q", (), {"name": "cpu"})()

        def read_outcome(self, *args):
            return {"outcome": "withdrawn", "reason": "refused"}

        def ticket(self, queue, digest):
            if error == "corrupt":
                raise CorruptTicket("not the ticket its name digests")
            raise KeyError("a bug")

    coordinator.pool = Storage()
    newest = (1, {"queue": "q", "ticket": "t" * 64, "attempt": "a"})
    if raised:
        with pytest.raises(KeyError):
            coordinator._refused(("c" * 64, "f" * 64), newest)
    else:
        assert coordinator._refused(("c" * 64, "f" * 64), newest) is None
        assert "cannot be read" in caplog.text


def _without_the_run_cache(tmp_path, tasks, pools, store):
    """Run `ForgetThenMany` on a pool over ``store`` through ``Submitter(worker=pool)``: its first
    pool job removes this host's user run cache once the outer job has started, so the later pool
    jobs run where it is absent, as on a worker of another host."""
    from types import SimpleNamespace

    from pydra.utils.general import default_run_cache_root

    pool = _warm(tmp_path, pools, SimpleNamespace(store=store))
    with Submitter(worker=pool) as run:
        task = tasks.ForgetThenMany(where=str(default_run_cache_root()), xs=[1, 2, 3])
        result = run(task, raise_errors=True)
    assert result.outputs.out == [1, 2, 3]
    assert not default_run_cache_root().exists()
    return run


def test_jobs_of_a_submitter_on_a_pool_need_no_user_cache(tmp_path, tasks, pools):
    """A `Submitter` given a pool over a private store and no ``cache_root`` roots its cache at the
    store, so its pool jobs keep their run files where every worker reaches, never in the user cache
    of the submitting host."""
    store = _submitter(tmp_path).store
    run = _without_the_run_cache(tmp_path, tasks, pools, store)
    assert run.cache_root == Path(store.path).resolve()


@pytest.mark.skipif(not POSIX, reason="a shared root is PlatformUnsupported on Windows")
def test_jobs_on_a_shared_store_pool_need_no_user_cache(tmp_path, tasks, pools):
    """On a pool over a shared store, whose root holds no run files, a job with a work directory
    writes nothing in the submitter's cache root, the user run cache of the submitting host.
    """
    from pydra.engine.portable.store import PortableStore

    store = PortableStore(
        tmp_path / "shared",
        trust="shared",
        work_dir=lambda c, f, a: tmp_path / "work" / f"{c}-{a}",
    )
    _without_the_run_cache(tmp_path, tasks, pools, store)
