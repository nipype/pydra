"""Warm environments in the pool worker (RFC 0005 sections 3.1, 4.3, W5b).

A worker runs a shell job whose environment is declared ``warm=True`` through its warm session of that
environment: one container, Apptainer instance or Lmod resolution serves the jobs of many tickets. The
session starts on the first such ticket bound to the union of the ticket's binding roots, restarts once
when a later ticket needs a root it lacks, runs a job cold beyond ``max_mount_roots``, and is closed when
the worker exits. The worker runs the reference each job carries and never resolves one. Warmth is
recorded in the job's pool facts only: the checksum, the slotted context and the stored result are
those of a cold run.

The runtimes are the fakes of ``warm_fakes`` (they run each command on the host, mapping
the container paths back); the worker runs in this process, as in ``test_pool_context``. The O5 cases
(``singularity``, ``lmod``) need a real Apptainer or Lmod and run on a cluster node.
"""

import os
import shutil
import sys
import tempfile
import threading
import time
import typing as ty
from pathlib import Path

import pytest

from pydra.compose import python, shell
from pydra.engine.job import Job
from pydra.engine.lease import LeaseSettings
from pydra.engine.portable.identity import PydraIdentityV1
from pydra.engine.portable.store import PortableStore
from pydra.engine.resident import ResidentKey
from pydra.engine.submitter import Submitter
from pydra.environments import base, docker, lmod, singularity
from pydra.environments.tests.warm_fakes import IMAGE, FakeRuntime

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="the fakes run POSIX commands"
)

LEASE = LeaseSettings(
    stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
)

Cp = shell.define("cp <in_file> <out|out_file>")
COMMITS: list[dict] = []


@python.define
def Inc(x: int) -> int:
    return x + 1


def record_commit(staged, checksum, key, context, digest):
    COMMITS.append({"checksum": checksum, "context": context})


def _sh(script):
    return shell.define("sh")(executable="sh", append_args=["-c", script])


def _input(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """Release every claim a test leaves held, and forget the commits, pass or fail; no warm session
    may be left served."""
    from pydra.engine import resident
    from pydra.workers.pool.storage import Claim

    made = []
    init = Claim.__init__

    def recording(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(Claim, "__init__", recording)
    device = resident.process_device()
    COMMITS.clear()
    yield
    COMMITS.clear()
    resident.set_process_device(device)
    resident.close_process_set()
    for claim in made:
        claim.lease.__exit__(None, None, None)
    assert base._SERVING == []


@pytest.fixture
def roots_below(tmp_path, monkeypatch):
    """A binding root is a directory directly under ``tmp_path``, so that a test can make jobs that
    need distinct roots (as ``test_warm_environments`` does)."""
    (tmp_path / "scratch").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
    real = base.top_root

    def top_root(path):
        path = Path(path).absolute()
        try:
            rel = path.relative_to(tmp_path)
        except ValueError:
            return real(path)
        return tmp_path / rel.parts[0] if rel.parts else tmp_path

    monkeypatch.setattr(base, "top_root", top_root)


def _settings(**kwargs):
    from pydra.workers.pool.settings import PoolSettings

    fields = dict(lease=LEASE, poll_s=0.05, idle_polls=2, timeout_grace=60.0)
    fields.update(kwargs)
    return PoolSettings(**fields)


def _class(name="cpu", **kwargs):
    from pydra.workers.pool.classes import WorkerClass

    fields = dict(name=name, device="cpu", quantum_jobs=64, quantum_s=600.0)
    fields.update(kwargs)
    return WorkerClass(**fields)


def _store(tmp_path: Path, name: str = "store") -> PortableStore:
    root = tmp_path / name
    root.mkdir(mode=0o700, exist_ok=True)
    return PortableStore(
        root,
        trust="private",
        pickle=True,
        identity=PydraIdentityV1(recorded_only={"pool"}),
        on_commit=record_commit,
    )


class Queue:
    """A pool root, a submitting process's submitter in a (warm) environment and the tickets put on
    the pool, for a worker run in this process."""

    def __init__(self, tmp_path: Path, store, environment):
        from pydra.workers.pool.storage import PoolRoot
        from pydra.workers.pool.worker import class_record

        self.pool = PoolRoot.open(tmp_path / "pool", LEASE, probe=False)
        self.submitter = Submitter(
            store=store, worker="debug", environment=environment, telemetry=False
        )
        self.store = store
        self.pool.put_class("cpu", class_record(_class()))

    def jobs(self, tasks, prefix="j", environment=None, resources=None):
        return [
            Job(
                task=t,
                submitter=self.submitter,
                name=f"{prefix}{i}",
                environment=environment,
                resources=resources,
            )
            for i, t in enumerate(tasks)
        ]

    def put(self, jobs, queue="cpu") -> str:
        from pydra.workers.pool.codec import PrivatePickle, write_payload
        from pydra.workers.pool.storage import Ticket, TicketJob

        entries = []
        for job in jobs:
            fingerprint = self.store.identity.fingerprint(job.context)
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
        return self.pool.put(
            queue, Ticket(queue=queue, node="wf/node", jobs=entries, residency=())
        )

    def work(self, cls=None, **kwargs):
        """Run a worker in this process until it is idle; it must exit cleanly, and never resolve an
        environment's reference (`_never_resolve`)."""
        from pydra.workers.pool.worker import Worker

        settings = kwargs.pop("settings", _settings())
        worker = Worker(
            self.pool.path,
            cls or _class(),
            settings=settings,
            device_memory=lambda: None,
            **kwargs,
        )
        with pytest.MonkeyPatch.context() as m:
            _never_resolve(m)
            assert worker.run() == 0
        return worker

    def close(self):
        self.pool.close()
        self.submitter.close()


@pytest.fixture
def queue_of(tmp_path):
    made = []

    def make(store, environment):
        q = Queue(tmp_path, store, environment)
        made.append(q)
        return q

    yield make
    for q in made:
        q.close()


def _until(predicate, bound: float = 60.0) -> bool:
    """Wait at most ``bound`` seconds for ``predicate``; whether it held. A helper thread uses it, so
    that no wait in these tests is unbounded."""
    deadline = time.monotonic() + bound
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def _never_resolve(monkeypatch):
    """Any resolution fails: the worker must run the references the jobs carry."""

    def refuse(self):
        raise AssertionError("a worker resolved an environment reference")

    for cls in (docker.Docker, singularity.Singularity, lmod.Lmod):
        monkeypatch.setattr(cls, "resolve", refuse)


def _env_facts(commit) -> dict:
    (entry,) = [
        e for e in commit["context"]["pool"]["residency"] if e["kind"] == "environment"
    ]
    return entry


# -- the worker runs shell jobs warm -------------------------------------------------------------


def test_shell_task_in_warm_docker_worker_equals_debug(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """Three shell jobs of one ticket run in one started container, one ``docker exec`` each, with the
    reference they carry. Each has the checksum, the slotted context and the stored output of a cold
    ``debug`` run, and a cold ``debug`` lookup afterwards is served from the warm worker's entries.
    """
    fake = FakeRuntime(monkeypatch, tmp_path)
    srcs = [_input(tmp_path / "inputs" / f"in{i}.txt", f"in{i}\n") for i in range(3)]
    tasks = [Cp(in_file=s) for s in srcs]

    # the cold debug run, in a store of its own
    cold_env = docker.Docker(image=IMAGE)
    with Submitter(
        worker="debug", store=_store(tmp_path, "cold-store"), environment=cold_env
    ) as sub:
        cold_jobs = [Job(task=t, submitter=sub, name="c") for t in tasks]
        cold_outputs = [Path(sub(t).outputs.out_file).read_text() for t in tasks]
    assert len(fake.cold) == 3
    COMMITS.clear()

    store = _store(tmp_path)
    q = queue_of(store, docker.Docker(image=IMAGE, warm=True))
    jobs = q.jobs(tasks)
    q.put(jobs)
    calls = len(fake.calls)
    worker = q.work()

    assert len(fake.starts) == 1 and len(fake.execs) == 3
    assert len(fake.cold) == 3  # only the debug run's
    assert worker.sessions == []  # closed at exit
    (started,) = fake.starts
    assert started[-2] == IMAGE  # the carried reference, never resolved here
    assert list(fake.started)[0] in fake.stopped
    # identity: warmth is never keyed, and the pool facts are recorded only
    identity = store.identity
    assert [c["checksum"] for c in COMMITS] == [j.checksum for j in cold_jobs]
    assert [identity.fingerprint(c["context"]) for c in COMMITS] == [
        identity.fingerprint(j.context) for j in cold_jobs
    ]
    assert all("pool" not in j.context for j in jobs + cold_jobs)
    facts = [_env_facts(c) for c in COMMITS]
    assert [f["run"] for f in facts] == ["warm"] * 3
    assert [f["state"] for f in facts] == ["cold", "warm", "warm"]
    assert [f["since_load"] for f in facts] == [0, 1, 2]
    assert all(isinstance(f["load_s"], float) for f in facts)
    key = ResidentKey("environment", "docker", jobs[0].resolved.key).digest()
    assert {f["key"] for f in facts} == {key}
    assert [c["context"]["pool"]["residency"][0]["kind"] for c in COMMITS] == [
        "environment"
    ] * 3
    # served from the warm entries by a cold debug lookup: nothing runs again
    calls = len(fake.calls)
    with Submitter(worker="debug", store=store, environment=cold_env) as sub:
        served = [Path(sub(t).outputs.out_file).read_text() for t in tasks]
    assert served == cold_outputs == ["in0\n", "in1\n", "in2\n"]
    assert len(fake.calls) == calls


def test_new_root_restarts_environment_once(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """The first ticket starts the container with its own roots. A later ticket whose jobs need two
    roots the container lacks restarts it once, bound to the union; a third ticket within those roots
    needs no restart."""
    fake = FakeRuntime(monkeypatch, tmp_path)
    q = queue_of(_store(tmp_path), docker.Docker(image=IMAGE, warm=True))

    def ticket(n, *dirs):
        srcs = [_input(tmp_path / d / f"in{n}.txt", f"{d}{n}\n") for d in dirs]
        jobs = q.jobs([Cp(in_file=s) for s in srcs], prefix=f"t{n}-")
        q.put(jobs)
        return jobs

    ticket(1, "a")
    ticket(2, "b", "c")
    ticket(3, "a", "c")
    worker = q.work()

    assert len(fake.starts) == 2  # one start, one restart for b and c together
    assert len(fake.execs) == 5 and fake.cold == []
    first, second = (
        {h for h, _, m in fake.started[n]["binds"] if m == "ro"} for n in ("id0", "id1")
    )
    assert tmp_path / "a" in first and not {tmp_path / "b", tmp_path / "c"} & first
    assert {tmp_path / d for d in "abc"} <= second
    assert fake.stopped[:1] == ["id0"] and "id1" in fake.stopped
    assert worker.sessions == []
    assert [_env_facts(c)["state"] for c in COMMITS] == [
        "cold",
        "cold",
        "warm",
        "warm",
        "warm",
    ]


def test_beyond_max_mount_roots_runs_cold_in_the_worker(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """A ticket whose union exceeds ``max_mount_roots`` starts nothing for the ticket; each job is
    judged alone, and the one whose roots no start can hold runs through the cold ``execute`` in the
    same worker, with the outputs of a warm run."""
    fake = FakeRuntime(monkeypatch, tmp_path)
    # the store (holding the cache and the scratch) and one input directory are two roots
    q = queue_of(
        _store(tmp_path), docker.Docker(image=IMAGE, warm=True, max_mount_roots=2)
    )
    srcs = [_input(tmp_path / d / "in.txt", f"{d}\n") for d in ("a", "b")]
    jobs = q.jobs([Cp(in_file=s) for s in srcs])
    q.put(jobs)
    q.work()

    assert len(fake.starts) == 1 and len(fake.execs) == 1 and len(fake.cold) == 1
    assert [_env_facts(c)["run"] for c in COMMITS] == ["warm", "cold"]
    with Submitter(
        worker="debug", store=q.store, environment=q.submitter.environment
    ) as sub:
        assert [
            Path(sub(Cp(in_file=s)).outputs.out_file).read_text() for s in srcs
        ] == [
            "a\n",
            "b\n",
        ]


def test_lmod_ticket_resolved_once_in_the_worker(tmp_path, queue_of, monkeypatch):
    """Lmod: the submitting process resolves the requested names once; the worker purges and loads
    the carried full names once for all its jobs, and never resolves."""
    loads = []

    def purge_and_load(names):
        loads.append(list(names))
        env = dict(os.environ, TOOL_VERSION="2.0", LOADEDMODULES="tool/2.0")
        return dict(os.environ), ["tool/2.0"], ["/tree/tool/2.0.lua"], env

    monkeypatch.setenv("MODULESHOME", str(tmp_path / "lmod"))
    monkeypatch.setattr(lmod.Lmod, "_purge_and_load", staticmethod(purge_and_load))
    q = queue_of(_store(tmp_path), lmod.Lmod(modules=["tool"], warm=True))
    tasks = [_sh(f"echo $TOOL_VERSION {i}") for i in range(5)]
    jobs = q.jobs(tasks)
    q.put(jobs[:2])
    q.put(jobs[2:])
    assert loads == [["tool"]]  # the submitting process's one resolution
    q.work()

    assert loads == [["tool"], ["tool/2.0"]]
    assert [_env_facts(c)["since_load"] for c in COMMITS] == [0, 1, 2, 3, 4]
    with Submitter(
        worker="debug", store=q.store, environment=lmod.Lmod(modules=["tool"])
    ) as sub:
        assert [sub(t).outputs.stdout.split() for t in tasks] == [
            ["2.0", str(i)] for i in range(5)
        ]
    assert len(loads) == 3  # the cold lookup resolved; nothing ran again


def test_python_and_cold_shell_jobs_untouched(tmp_path, queue_of, monkeypatch):
    """A shell job whose environment is not declared warm runs through its cold ``execute``: the
    worker opens no session and its pool facts have no environment entry."""
    fake = FakeRuntime(monkeypatch, tmp_path)
    q = queue_of(_store(tmp_path), docker.Docker(image=IMAGE))
    src = _input(tmp_path / "inputs" / "in.txt", "x\n")
    q.put(q.jobs([Cp(in_file=src)]))
    worker = q.work()
    assert fake.starts == [] and len(fake.cold) == 1
    assert worker.served and worker.sessions == []
    assert [e["kind"] for e in COMMITS[0]["context"]["pool"]["residency"]] == []


def test_orphaned_container_reaped_and_own_closed(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """A container a dead worker left on this host is removed at the worker's first warm start, and
    the worker's own container and scratch directory are gone when it exits."""
    fake = FakeRuntime(monkeypatch, tmp_path)
    host, _ = base.owner()
    proc = __import__("subprocess").Popen([sys.executable, "-c", "pass"])
    proc.wait()
    ns = base.pid_namespace() or "-"
    fake.labels = {"orphan": f"{host}:{os.getuid()}:{ns}:{proc.pid}"}
    q = queue_of(_store(tmp_path), docker.Docker(image=IMAGE, warm=True))
    src = _input(tmp_path / "inputs" / "in.txt", "x\n")
    q.put(q.jobs([Cp(in_file=src)]))
    q.work()
    assert "orphan" in fake.stopped
    assert fake.stopped[-1] == list(fake.started)[0]
    assert not list(Path(q.store.path).rglob(".pydra-warm-*"))


def test_drain_stops_the_warm_command(tmp_path, roots_below, queue_of, monkeypatch):
    """On a drain (SIGTERM) whose ``drain_s`` has passed, the command the warm container runs is
    stopped by closing the container; the job is released unsettled and the worker exits cleanly.
    """
    fake = FakeRuntime(monkeypatch, tmp_path)
    fake.exec_delay = SLOW
    returned = []
    execute = fake.execute

    def recording(cmd, *args, **kwargs):
        out = execute(cmd, *args, **kwargs)
        if [str(c) for c in cmd[:2]] == ["docker", "exec"]:
            returned.append(out[0])
        return out

    monkeypatch.setattr(base, "execute", recording)
    q = queue_of(_store(tmp_path), docker.Docker(image=IMAGE, warm=True))
    src = _input(tmp_path / "inputs" / "in.txt", "x\n")
    q.put(q.jobs([Cp(in_file=src)]))
    from pydra.workers.pool.worker import Worker

    worker = Worker(
        q.pool.path,
        _class(),
        settings=_settings(drain_s=0.01),
        device_memory=lambda: None,
    )

    def drain():
        _until(fake.running.is_set)  # bounded: the worker stops either way
        worker.stop.set()
        worker._wake.set()

    threading.Thread(target=drain, daemon=True).start()
    assert worker.run() == 0
    (name,) = fake.started
    assert name in fake.stopped
    for thread in threading.enumerate():
        if thread.name == "pydra-pool-call":
            thread.join(30)
    assert returned == [137]  # the command was killed with its container
    assert COMMITS == []


SLOW = 60.0
"""Seconds a command that must be stopped would run; stopped, it exits 137 instead of 0"""


def _recording_execs(fake, monkeypatch) -> list[tuple[int, float, float]]:
    """``(return code, began, ended)`` of each ``docker exec`` the fake runs, on the monotonic clock."""
    returned: list[tuple[int, float, float]] = []
    execute = fake.execute

    def recording(cmd, *args, **kwargs):
        began = time.monotonic()
        out = execute(cmd, *args, **kwargs)
        if [str(c) for c in cmd[:2]] == ["docker", "exec"]:
            returned.append((out[0], began, time.monotonic()))
        return out

    monkeypatch.setattr(base, "execute", recording)
    return returned


def test_transient_start_failure_is_a_load_fault_then_warm(
    tmp_path, roots_below, queue_of, monkeypatch, caplog
):
    """A warm start that fails is a load fault (RFC 0005 section 4.3 step 3): the ticket's job is
    requeued under a ``restart`` generation charged to no one, and the next claim of it starts the
    container again and runs the job warm. A failed start never makes the environment cold for the
    worker's life."""
    fake = FakeRuntime(monkeypatch, tmp_path)
    start = fake._start
    failed = []

    def flaky(cmd, flag):
        fake.fail_start = not failed
        failed.append(fake.fail_start)
        try:
            return start(cmd, flag)
        finally:
            fake.fail_start = False

    monkeypatch.setattr(fake, "_start", flaky)
    store = _store(tmp_path)
    q = queue_of(store, docker.Docker(image=IMAGE, warm=True))
    src = _input(tmp_path / "inputs" / "in.txt", "x\n")
    (job,) = q.jobs([Cp(in_file=src)])
    q.put([job])
    with caplog.at_level("ERROR", logger="pydra.pool"):
        q.work()

    assert failed == [True, False]  # one failed start, one that worked
    assert len(fake.execs) == 1 and fake.cold == []
    assert any("loading the residency" in r.getMessage() for r in caplog.records)
    gens = [
        r
        for _, r in q.pool.generations(
            job.checksum, store.identity.fingerprint(job.context)
        )
    ]
    assert [g["cause"] for g in gens] == ["put", "restart"]
    assert not any(g.get("charged") for g in gens)
    (commit,) = COMMITS
    assert _env_facts(commit)["run"] == "warm"


def test_unfit_record_code_names_the_warm_environment(tmp_path, queue_of):
    """The code a load runs under names a warm shell job's carried reference (`Code.environment`), so
    a record written under one image stops binding under another; other jobs name none.
    """
    from pydra.workers.pool.calibration import Code, code_of

    src = _input(tmp_path / "inputs" / "in.txt", "x\n")
    q = queue_of(_store(tmp_path), docker.Docker(image=IMAGE, warm=True))
    (warm,) = q.jobs([Cp(in_file=src)])
    (cold,) = q.jobs([Cp(in_file=src)], environment=docker.Docker(image=IMAGE))
    assert code_of(warm) == Code((), warm.resolved.key)
    assert code_of(cold) == Code(())


def test_new_reference_opens_a_fresh_session(tmp_path, queue_of, monkeypatch):
    """A pool that outlives a run: run 2 resolves the same Lmod environment to another default. Its
    jobs get a warm session of their own reference, not a cold run each, and the idle session of the
    old reference is closed when the new one opens."""
    version = ["2.0"]
    events = []

    def purge_and_load(names):
        full = [n if "/" in n else f"{n}/{version[0]}" for n in names]
        events.append(("load", full[0]))
        env = dict(os.environ, TOOL_VERSION=full[0].split("/")[1])
        return dict(os.environ), full, [f"/tree/{full[0]}.lua"], env

    monkeypatch.setenv("MODULESHOME", str(tmp_path / "lmod"))
    monkeypatch.setattr(lmod.Lmod, "_purge_and_load", staticmethod(purge_and_load))
    monkeypatch.setattr(lmod.Lmod, "teardown", lambda self: events.append(("close",)))
    env = lmod.Lmod(modules=["tool"], warm=True)
    q = queue_of(_store(tmp_path), env)
    q.put(q.jobs([_sh(f"echo $TOOL_VERSION {i}") for i in range(2)], "a"))
    version[0] = "3.0"  # the default moved; run 2's submitter resolves again
    run1, q.submitter = q.submitter, Submitter(
        store=q.store, worker="debug", environment=env
    )
    run1.close()
    q.put(q.jobs([_sh(f"echo $TOOL_VERSION {i}") for i in range(2, 4)], "b"))
    events.clear()
    q.work()

    assert events == [
        ("load", "tool/2.0"),
        ("close",),
        ("load", "tool/3.0"),
        ("close",),
    ]
    assert [_env_facts(c)["run"] for c in COMMITS] == ["warm"] * 4


def test_lost_claim_stops_the_warm_command(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """A claim lost while its job's command runs in the warm container: the worker closes the
    container, so the command ends (137) at once rather than after its full run, and the worker,
    still running, serves the next claim in a fresh container."""
    fake = FakeRuntime(monkeypatch, tmp_path)
    fake.exec_delays = [SLOW]  # the first command only; the ones after it are quick
    returned = _recording_execs(fake, monkeypatch)
    q = queue_of(_store(tmp_path), docker.Docker(image=IMAGE, warm=True))
    srcs = [_input(tmp_path / "inputs" / f"in{i}.txt", f"{i}\n") for i in range(2)]
    q.put(q.jobs([Cp(in_file=srcs[0])], "first"))
    q.put(q.jobs([Cp(in_file=srcs[1])], "second"))
    from pydra.workers.pool.worker import Worker

    worker = Worker(
        q.pool.path, _class(), settings=_settings(), device_memory=lambda: None
    )

    def lose():
        _until(fake.running.is_set)  # the command is running; bounded either way
        for token_dir in list(worker._held):
            worker._on_lost(token_dir, "taken over")

    threading.Thread(target=lose, daemon=True).start()
    with pytest.MonkeyPatch.context() as m:
        _never_resolve(m)
        assert worker.run() == 0

    code, _, _ = returned[0]
    assert code == 137  # stopped, not run to its end
    assert len(fake.starts) >= 2  # the next claim got a fresh container
    assert [c for c, _, _ in returned[1:]] and all(c == 0 for c, _, _ in returned[1:])
    assert len(COMMITS) == 2  # both jobs, the lost one run again under a restart


def test_expired_warm_call_acknowledged(tmp_path, roots_below, queue_of, monkeypatch):
    """A warm call past its deadline: the worker stops its command by closing the container and
    acknowledges the expiry at once, so the beater has nothing to kill; the worker recycles.
    """
    from pydra.engine.lease import BeaterClient, LeaseBeater
    from pydra.utils.resources import ComputeResources
    from pydra.workers.pool import worker as worker_module
    from pydra.workers.pool.beater import EXPIRE
    from pydra.workers.pool.worker import EXIT_RECYCLE, Worker

    fake = FakeRuntime(monkeypatch, tmp_path)
    fake.exec_delay = SLOW
    returned = _recording_execs(fake, monkeypatch)
    # never kill this process: an acknowledgement that does not come shows as a late one
    monkeypatch.setattr(
        worker_module,
        "pool_beater",
        lambda grace, **kw: BeaterClient(
            LeaseBeater(on_expire=EXPIRE, kill_after=None), **kw
        ),
    )
    acks = []
    ack = BeaterClient.ack_expired

    def recording(self, token_dir):
        acks.append(time.monotonic())
        return ack(self, token_dir)

    monkeypatch.setattr(BeaterClient, "ack_expired", recording)
    q = queue_of(_store(tmp_path), docker.Docker(image=IMAGE, warm=True))
    src = _input(tmp_path / "inputs" / "in.txt", "x\n")
    q.put(q.jobs([Cp(in_file=src)], resources=ComputeResources(time=1)))
    worker = Worker(
        q.pool.path,
        _class(),
        settings=_settings(timeout_grace=1.0),
        device_memory=lambda: None,
    )
    assert worker.run() == EXIT_RECYCLE
    for thread in threading.enumerate():
        if thread.name == "pydra-pool-call":
            thread.join(30)
    ((code, _, _),) = returned
    assert code == 137  # stopped, not run to its end
    assert acks  # the expiry was acknowledged, so the beater had nothing to kill
    assert COMMITS == []


def test_warm_sessions_bounded_lru(tmp_path, roots_below, queue_of, monkeypatch):
    """At most ``PoolSettings.max_warm_sessions`` warm sessions stay open in a worker: opening one
    more closes the least recently used idle one."""
    fake = FakeRuntime(monkeypatch, tmp_path)
    envs = [docker.Docker(image=f"org/tool@sha256:{c * 64}", warm=True) for c in "234"]
    q = queue_of(_store(tmp_path), envs[0])
    for n, e in enumerate([0, 1, 2, 0]):
        src = _input(tmp_path / "inputs" / f"in{n}.txt", f"{n}\n")
        q.put(q.jobs([Cp(in_file=src)], f"t{n}-", environment=envs[e]))
    q.work(settings=_settings(max_warm_sessions=2))

    assert len(fake.starts) == 4 and fake.cold == []
    # the third environment evicts the first; the first, back, evicts the second
    assert fake.stopped[:2] == ["id0", "id1"]
    assert sorted(fake.stopped) == ["id0", "id1", "id2", "id3"]


def _env_label(job) -> str:
    """The hold label of a warm Docker job's environment: the residency label of its resident key
    alone (the plugin name and the reference the job carries)."""
    import hashlib

    import attrs

    from pydra.utils.digest import jcs

    key = ResidentKey("environment", "docker", job.resolved.key)
    return hashlib.sha256(jcs([attrs.asdict(key)])).hexdigest()


def _broken_image(tmp_path, queue_of, monkeypatch):
    """A queue with a ticket of a warm image whose start always fails, a plain Python ticket, and a
    second ticket of the image, in that order."""
    from pydra.environments import native

    fake = FakeRuntime(monkeypatch, tmp_path, fail_start=True)
    q = queue_of(_store(tmp_path), docker.Docker(image=IMAGE, warm=True))
    srcs = [_input(tmp_path / "inputs" / f"in{i}.txt", f"{i}\n") for i in range(2)]
    (a,) = q.jobs([Cp(in_file=srcs[0])], "a")
    image_a = (a, q.put([a]))
    (p,) = q.jobs([Inc(x=1)], "p", environment=native.Native())
    plain = (p, q.put([p]))
    (c,) = q.jobs([Cp(in_file=srcs[1])], "c")
    image_c = (c, q.put([c]))
    return fake, q, image_a, plain, image_c


def _budget():
    from pydra.workers.pool.faults import FaultBudgets

    return FaultBudgets(infrastructure_replacements=2)


def _outcome(q, digest) -> dict:
    (outcome,) = q.pool.outcomes("cpu", digest).values()
    return outcome


def test_repeated_warm_start_faults_hold_that_environment(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """Warm-start faults count toward a hold whose label names the environment and its resolved
    reference: at the budget the ticket is held under that label, and a later ticket of the same
    image is settled ``held`` without another start."""
    fake, q, (a, da), _, (c, dc) = _broken_image(tmp_path, queue_of, monkeypatch)
    q.work(faults=_budget())

    label = _env_label(a)
    assert label in q.pool.holds("cpu")
    for digest in (da, dc):
        outcome = _outcome(q, digest)
        assert outcome["outcome"] == "held" and outcome["hold"] == label
        assert label in outcome["reason"]
    assert (
        len(fake.starts) == 2
    )  # the budget's two faults, nothing started after the hold


def test_plain_python_ticket_runs_beside_a_held_environment(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """One broken image never holds the class's other work: the plain Python ticket queued behind
    it runs and commits, and the class's empty residency is not held."""
    _, q, (a, _), (p, dp), _ = _broken_image(tmp_path, queue_of, monkeypatch)
    q.work(faults=_budget())

    assert _outcome(q, dp)["outcome"] == "done"
    assert p.checksum in [c["checksum"] for c in COMMITS]
    assert list(q.pool.holds("cpu")) == [_env_label(a)]


def test_coordinator_follows_the_outcomes_hold_label(
    tmp_path, roots_below, monkeypatch
):
    """A `WarmPool` whose worker holds a broken image: the coordinator reads the held outcome's
    ``hold`` label, so a later job of that image is reported held at submission while a plain job and
    a job of another image are put and run. After ``release-hold`` the running coordinator keeps the
    hold for its epoch (the image's next job is reported held, visibly); a new coordinator, a new
    epoch, runs it."""
    from pydra.environments import native
    from pydra.engine.runner import NotRun
    from pydra.workers.pool.pool import WarmPool, release_hold
    from pydra.workers.pool.substrate import ExternalSubstrate
    from pydra.workers.pool.worker import Worker

    broken = IMAGE
    fine = f"org/tool@sha256:{'2' * 64}"
    fake = FakeRuntime(monkeypatch, tmp_path)
    start = fake._start
    starts_fail = [True]

    def flaky(cmd, flag):
        fake.fail_start = starts_fail[0] and broken in cmd
        try:
            return start(cmd, flag)
        finally:
            fake.fail_start = False

    monkeypatch.setattr(fake, "_start", flaky)
    store = _store(tmp_path)
    reported: dict[str, ty.Any] = {}

    def coordinator():
        made = WarmPool(
            tmp_path / "pool",
            [_class()],
            ExternalSubstrate(),
            store=store,
            settings=_settings(),
        )
        made.bind(lambda job, outcome: reported.__setitem__(job.name, outcome))
        return made

    pool = coordinator()
    sub = Submitter(store=store, worker="debug", telemetry=False)
    n = [0]

    def jobs(*kinds):
        made = []
        for kind in kinds:
            n[0] += 1
            name = f"{kind}{n[0]}"
            if kind == "plain":
                task, env = Inc(x=n[0]), native.Native()
            else:
                src = _input(tmp_path / "inputs" / f"{name}.txt", f"{name}\n")
                image = broken if kind == "broken" else fine
                task, env = Cp(in_file=src), docker.Docker(image=image, warm=True)
            made.append(Job(task=task, submitter=sub, name=name, environment=env))
        return made

    def work_until(names):
        deadline = time.monotonic() + 60
        while not set(names) <= set(reported):
            missing = set(names) - set(reported)
            assert time.monotonic() < deadline, f"no outcome for {missing}"
            Worker(
                pool.pool.path,
                _class(),
                settings=_settings(),
                faults=_budget(),
                device_memory=lambda: None,
            ).run()

    def held(name):
        return isinstance(reported[name], NotRun) and reported[name].held

    try:
        first = jobs("broken", "plain", "fine")
        pool.submit(first)
        work_until([j.name for j in first])
        assert held("broken1") and not held("plain2") and not held("fine3")

        later = jobs("broken", "plain", "fine")
        pool.submit(later)
        assert held("broken4")  # at submission: the image is held in this epoch
        work_until(["plain5", "fine6"])
        assert not held("plain5") and not held("fine6")

        starts_fail[0] = False
        assert release_hold(pool.pool.path, "cpu", _env_label(first[0])) == 1
        (again,) = jobs("broken")
        pool.submit([again])
        assert held(again.name)  # the running coordinator keeps the hold for its epoch

        pool.close()
        pool = coordinator()  # a new epoch
        (fresh,) = jobs("broken")
        pool.submit([fresh])
        work_until([fresh.name])
        assert not held(fresh.name)
    finally:
        pool.close()
        sub.close()


def test_coordinator_holds_only_that_environments_jobs(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """The coordinator keys a held outcome by its ``hold`` label: it reports later jobs of that
    environment held, and neither a plain job nor a job of another image."""
    from pydra.workers.pool.pool import WarmPool, _label

    _, q, (a, _), (p, _), _ = _broken_image(tmp_path, queue_of, monkeypatch)
    src = _input(tmp_path / "inputs" / "other.txt", "o\n")
    (other,) = q.jobs(
        [Cp(in_file=src)],
        "o",
        environment=docker.Docker(image=f"org/tool@sha256:{'2' * 64}", warm=True),
    )
    coordinator = WarmPool.__new__(WarmPool)  # only the held-reason bookkeeping
    coordinator._held = {("cpu", _env_label(a)): "held: the image does not start"}
    assert coordinator._held_reason("cpu", a, ()) == "held: the image does not start"
    assert coordinator._held_reason("cpu", p, ()) is None
    assert coordinator._held_reason("cpu", other, ()) is None
    assert coordinator._held_reason("gpu", a, ()) is None
    coordinator._held = {("cpu", _label(())): "residency held"}
    assert coordinator._held_reason("cpu", p, ()) == "residency held"


def test_release_hold_clears_the_environment_hold(
    tmp_path, roots_below, queue_of, monkeypatch
):
    """``release-hold`` of the environment's label lifts the hold: once the image starts again, a
    new ticket of it runs warm."""
    from pydra.workers.pool.pool import release_hold

    fake, q, (a, _), _, _ = _broken_image(tmp_path, queue_of, monkeypatch)
    q.work(faults=_budget())
    label = _env_label(a)
    assert label in q.pool.holds("cpu")

    fake.fail_start = False
    assert release_hold(q.pool.path, "cpu", label) == 1
    assert q.pool.holds("cpu") == {}
    src = _input(tmp_path / "inputs" / "in-new.txt", "new\n")
    (d,) = q.jobs([Cp(in_file=src)], "d")
    dd = q.put([d])
    COMMITS.clear()
    q.work(faults=_budget())
    assert _outcome(q, dd)["outcome"] == "done"
    (commit,) = COMMITS
    assert _env_facts(commit)["run"] == "warm"


# -- real runtimes on a cluster node (``module load apptainer``; Lmod's MODULEPATH) -----------


def _o5_jobs(default: int) -> int:
    return int(os.environ.get("PYDRA_O5_JOBS", default))


@pytest.mark.singularity
@pytest.mark.timeout(3600)
@pytest.mark.skipif(
    not (shutil.which("apptainer") or shutil.which("singularity")),
    reason="no apptainer",
)
def test_warm_apptainer_shell_jobs(tmp_path, queue_of, monkeypatch, record_property):
    """500 shell jobs of one Apptainer image on one worker start one instance; their outputs
    equal a cold ``debug`` run's (sampled), no job sees what another wrote to ``$HOME`` or ``/tmp``,
    and the wall per job is recorded warm against cold."""
    n, sample = _o5_jobs(500), int(os.environ.get("PYDRA_O5_COLD", 20))
    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "apptainer-cache"))
    monkeypatch.setenv("SINGULARITY_CACHEDIR", str(tmp_path / "apptainer-cache"))
    monkeypatch.setenv("PYDRA_IMAGE_CACHE", str(tmp_path / "images"))
    ref = singularity.Singularity(image="docker://alpine", tag="3.20").resolve()
    cold_env = singularity.Singularity(image=ref.run)
    warm_env = singularity.Singularity(image=ref.run, warm=True)
    srcs = [_input(tmp_path / "inputs" / f"in{i}.txt", f"line {i}\n") for i in range(n)]
    tasks = [Cp(in_file=s) for s in srcs]

    began = time.monotonic()
    with Submitter(
        worker="debug", store=_store(tmp_path, "cold"), environment=cold_env
    ) as sub:
        cold = [Path(sub(t).outputs.out_file).read_bytes() for t in tasks[:sample]]
    cold_wall = (time.monotonic() - began) / sample

    starts = []
    real_warm = singularity.Singularity.warm

    def counting(self, *args, **kwargs):
        handle = real_warm(self, *args, **kwargs)
        starts.append(handle)
        return handle

    monkeypatch.setattr(singularity.Singularity, "warm", counting)
    store = _store(tmp_path)
    q = queue_of(store, warm_env)
    jobs = q.jobs(tasks)
    for i in range(0, n, 50):
        q.put(jobs[i : i + 50])
    writers = q.jobs(
        [_sh(f'echo {i} > "$HOME/x"; echo {i} > /tmp/y') for i in range(3)], "w"
    )
    q.put(writers)
    (reader,) = q.jobs(
        [_sh('test -e "$HOME/x" && echo seen-x; test -e /tmp/y && echo seen-y; true')],
        "r",
    )
    q.put([reader])
    COMMITS.clear()
    began = time.monotonic()
    q.work()
    warm_wall = (time.monotonic() - began) / (n + 4)

    assert len(starts) == 1  # one instance for every job
    assert len(COMMITS) == n + 4
    assert all(_env_facts(c)["run"] == "warm" for c in COMMITS)
    with Submitter(worker="debug", store=store, environment=cold_env) as sub:
        warm = [Path(sub(t).outputs.out_file).read_bytes() for t in tasks]
        assert "seen" not in sub(reader.task).outputs.stdout
    assert warm[:sample] == cold
    assert warm == [s.read_bytes() for s in srcs]
    record_property("warm_wall_per_job_s", round(warm_wall, 4))
    record_property("cold_wall_per_job_s", round(cold_wall, 4))
    print(
        f"apptainer: {n} jobs warm {warm_wall * 1000:.0f} ms/job (pool included), "
        f"cold debug {cold_wall * 1000:.0f} ms/job over {sample}"
    )


@pytest.mark.lmod
@pytest.mark.timeout(1800)
@pytest.mark.skipif("MODULEPATH" not in os.environ, reason="modules not available")
def test_lmod_resolved_once(tmp_path, queue_of, monkeypatch):
    """200 shell jobs under an ``Lmod`` environment on one worker: the submitting process
    resolves the names once, the worker loads the carried full names once, and every output is the
    module's."""
    n = _o5_jobs(200)
    modules = tmp_path / "modules" / "tool"
    modules.mkdir(parents=True)
    (modules / "2.0.lua").write_text('setenv("TOOL_VERSION", "2.0")\n')
    monkeypatch.setenv("MODULEPATH", str(tmp_path / "modules"))
    calls = []
    real = lmod.Lmod._purge_and_load.__func__

    def counting(cls, names):
        calls.append(list(names))
        return real(cls, names)

    monkeypatch.setattr(lmod.Lmod, "_purge_and_load", classmethod(counting))
    q = queue_of(_store(tmp_path), lmod.Lmod(modules=["tool"], warm=True))
    tasks = [_sh(f"echo $TOOL_VERSION {i}") for i in range(n)]
    jobs = q.jobs(tasks)
    for i in range(0, n, 50):
        q.put(jobs[i : i + 50])
    assert calls == [["tool"]]
    q.work()
    assert calls == [["tool"], ["tool/2.0"]]  # one resolution per worker
    assert len(COMMITS) == n
    with Submitter(
        worker="debug", store=q.store, environment=lmod.Lmod(modules=["tool"])
    ) as sub:
        outputs = [sub(t).outputs.stdout.split() for t in tasks]
    assert outputs == [["2.0", str(i)] for i in range(n)]
