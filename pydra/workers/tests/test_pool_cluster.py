"""The warm pool on a real Slurm cluster, with GPUs (RFC 0005 section 11, W12).

These tests run only on a cluster: every test is marked ``slurm`` and runs only under the ``slurm``
worker (``--only-worker=slurm``); without it each skips. The GPU tests also skip unless
``PYDRA_TEST_GPU_TYPE`` is set. Every test prints one line,
``VERDICT <test id> PASS|FAIL <short reason>``, then asserts; nothing asserts on wall-clock time.
Run them from a checkout pinned to one commit, inside a Slurm allocation, never on a login node::

    pytest pydra/workers/tests/test_pool_cluster.py --only-worker=slurm -m slurm -s

Environment variables
---------------------
PYDRA_TEST_SHARED_DIR
    Required by every test: a directory on storage that every node of the cluster sees (the pool
    root, the store, the task module and the marks live under it).
PYDRA_TEST_GPU_TYPE
    The GPU type the GPU tests ask for, as Slurm names it (``h100``): workers are submitted with
    ``--gres=gpu:<type>:<n>``. Unset: the GPU tests skip.
PYDRA_TEST_SLURM_PARTITION, PYDRA_TEST_SLURM_ACCOUNT
    The partition and account of every worker job; unset: the cluster's defaults. No ``--qos`` is
    ever passed.
PYDRA_TEST_GPU_PARTITION
    The partition of GPU worker jobs; default ``PYDRA_TEST_SLURM_PARTITION``.
PYDRA_TEST_SLURM_TIME, PYDRA_TEST_SLURM_MEM
    A worker job's time limit in minutes and its memory (``"32G"``); unset: the cluster's defaults.
PYDRA_TEST_ATTRIBUTE_DELAY
    Seconds a client may see a directory change late on the shared storage (default 60, the NFS
    ``acdirmax``); the pool's lease settings are derived from it.
PYDRA_TEST_JOBS
    Jobs of ``test_one_load_n_jobs_gpu`` (default 1000).
PYDRA_TEST_SERVER_PYTHON
    The interpreter of the process-owning provider's child (default this interpreter); point it at
    another environment to serve the model from there.
PYDRA_TEST_TIMEOUT
    Seconds a test may run before pytest-timeout stops it (default 7200).

What the tests need
-------------------
The GPU tests that load a model need ``torch`` (a small ``Linear`` model on the device) or, failing
that, ``cupy`` (a matrix on the device) importable in this interpreter, which the workers share;
with neither they skip and say so. ``test_local_multi_gpu`` and ``test_mistyped_grant_refused``
run their workers on this node through `LocalSubstrate`, so the allocation running pytest must hold
at least two GPUs (the second only for the first test). ``test_claim_lease_cross_node`` and
``test_coordinator_killed_and_resumed_on_another_host`` read ``SLURM_JOB_NODELIST`` and start
their roles with ``srun`` on three distinct nodes of the allocation; with fewer they skip. The other
tests submit their workers with ``sbatch`` and so need ``sbatch`` to work from the node running
pytest. Each verdict line is also appended to ``<PYDRA_TEST_SHARED_DIR>/verdicts.log``.

Loads are counted by files: a provider's ``load`` (and the process provider's child, when it
starts) writes one file per load into a directory of the spec, named ``<Slurm job id>-<pid>-<uuid>``,
so the count is read from any node.
"""

import asyncio
import collections
import contextlib
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import typing as ty
import uuid
from pathlib import Path

import attrs
import pytest

from pydra.engine.lease import LeaseSettings
from pydra.engine.submitter import Submitter

SHARED = os.environ.get("PYDRA_TEST_SHARED_DIR")
GPU_TYPE = os.environ.get("PYDRA_TEST_GPU_TYPE")
PARTITION = os.environ.get("PYDRA_TEST_SLURM_PARTITION")
GPU_PARTITION = os.environ.get("PYDRA_TEST_GPU_PARTITION") or PARTITION
ACCOUNT = os.environ.get("PYDRA_TEST_SLURM_ACCOUNT")
SLURM_TIME = os.environ.get("PYDRA_TEST_SLURM_TIME")
SLURM_MEM = os.environ.get("PYDRA_TEST_SLURM_MEM")
ATTRIBUTE_DELAY = float(os.environ.get("PYDRA_TEST_ATTRIBUTE_DELAY", "60"))
JOBS = int(os.environ.get("PYDRA_TEST_JOBS", "1000"))
SERVER_PYTHON = os.environ.get("PYDRA_TEST_SERVER_PYTHON") or sys.executable
TIMEOUT = int(os.environ.get("PYDRA_TEST_TIMEOUT", "7200"))
POSIX = sys.platform != "win32"

# every cluster test carries @slurm; the module's unit tests of its own helpers do not
slurm_case = pytest.mark.slurm
pytestmark = [
    pytest.mark.timeout(TIMEOUT),
    pytest.mark.skipif(not POSIX, reason="Slurm clusters are POSIX"),
]

gpu = pytest.mark.skipif(
    not GPU_TYPE,
    reason="set PYDRA_TEST_GPU_TYPE (for example h100) to run the GPU tests",
)


def _lease() -> LeaseSettings:
    """Lease settings for the shared storage: stale after ``2 x attribute_delay + 30`` seconds,
    which keeps a live holder's newest mark (two renewals plus the attribute delay) below it.
    """
    stale = 2 * ATTRIBUTE_DELAY + 30
    return LeaseSettings(
        stale_after=stale,
        renew_fraction=0.1,
        attribute_delay=ATTRIBUTE_DELAY,
        timeout=stale + ATTRIBUTE_DELAY + 60,
    )


LEASE = _lease()
BOUND = (
    20 * LEASE.stale_after
)  # seconds any wait on another process may take: a hang bound

TASKS = '''
import json, os, socket, sys, tempfile, time
from pathlib import Path
from uuid import uuid4
from pydra.compose import python, workflow
from pydra.engine.procgroup import ProcessProvider, gpu_memory_by_pid
from pydra.engine.resident import Provided
from pydra.utils.resources import ComputeResources
from pydra.workers.pool.faults import WorkerLost

CAP = "PYDRA_TEST_DEVICE_CAP"  # bytes of device memory a worker may use, set per class


def backend():
    """The library that puts the model on the device: torch, else cupy, else None."""
    import importlib.util

    for name in ("torch", "cupy"):
        if importlib.util.find_spec(name) is not None:
            return name
    return None


def _mark(where):
    # one file per load or start, named by the worker's Slurm job, readable from any node
    job = os.environ.get("SLURM_JOB_ID", "local")
    Path(where, f"{job}-{os.getpid()}-{uuid4().hex}").touch()


class Model:
    """A small model on CUDA device ``index``: torch's Linear, else a cupy matrix. ``ballast``
    bytes more are held on the device; the environment variable CAP caps what may be used."""

    def __init__(self, spec, index):
        self.backend = backend()
        self.width = int(spec.get("width", 1024))
        ballast = int(spec.get("ballast", 0))
        cap = os.environ.get(CAP)
        if self.backend == "torch":
            import torch

            self.device = torch.device("cuda", index)
            if cap:
                total = torch.cuda.get_device_properties(self.device).total_memory
                torch.cuda.set_per_process_memory_fraction(int(cap) / total, self.device)
            torch.manual_seed(int(spec.get("seed", 0)))
            self.layer = torch.nn.Linear(self.width, self.width).to(self.device).eval()
            self.ballast = (
                torch.empty(ballast // 4, dtype=torch.float32, device=self.device)
                if ballast
                else None
            )
        elif self.backend == "cupy":
            import cupy

            cupy.cuda.Device(index).use()
            if cap:
                cupy.get_default_memory_pool().set_limit(size=int(cap))
            # cupy.cuda.memory.OutOfMemoryError propagates as itself: the pool reads it as the
            # device's, by name
            rng = cupy.random.RandomState(int(spec.get("seed", 0)))
            self.layer = rng.standard_normal((self.width, self.width), dtype=cupy.float32)
            self.ballast = cupy.empty(ballast // 4, dtype=cupy.float32) if ballast else None
        else:
            raise RuntimeError("neither torch nor cupy is importable: no way to use the device")

    def __call__(self, x):
        if self.backend == "torch":
            import torch

            with torch.no_grad():
                v = torch.full((1, self.width), float(x), device=self.device)
                return float(self.layer(v).sum().item())
        import cupy

        v = cupy.full((self.width,), float(x), dtype=cupy.float32)
        return float((self.layer @ v).sum())

    def close(self):
        self.layer = self.ballast = None
        if self.backend == "torch":
            import torch

            torch.cuda.empty_cache()
        elif self.backend == "cupy":
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()


class DeviceModel:
    """An in-process provider: a `Model` on the granted device, one file per load."""

    name = "cluster-model"

    def load(self, spec, device):
        _mark(spec["loads"])
        return Model(spec, device.index or 0)

    def close(self, obj):
        obj.close()

    def footprint(self, spec):
        return None

    def measure(self, obj):
        return None

    def version(self):
        return "1"


class _Client:
    """Sends x to the child over its socket and reads the model's answer."""

    def __init__(self, path, closes):
        self.path = path
        self.closes = closes

    def __call__(self, x):
        # a server that is gone is a crash of the resident state, not the job's fault
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(self.path)
                s.sendall(f"{x}\\n".encode())
                data = b""
                while not data.endswith(b"\\n"):
                    chunk = s.recv(4096)
                    if not chunk:
                        raise ConnectionError("the model server closed the connection")
                    data += chunk
        except OSError as err:
            raise WorkerLost(f"the model server stopped: {err}") from err
        return float(data.decode())


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True


class ServerModel(ProcessProvider):
    """A process-owning provider: the child (spec["script"] run by spec["python"]) holds the model
    on the device and answers over a socket. Its close records what was left behind."""

    name = "cluster-server"

    def load(self, spec, device):
        self._socket = os.path.join(tempfile.mkdtemp(prefix="pydra-srv-"), "s")
        return super().load(spec, device)

    def argv(self, spec, device):
        return [spec["python"], spec["script"], json.dumps(spec), self._socket]

    def connect(self, group, spec):
        deadline = time.monotonic() + float(spec.get("start_bound", 900))
        while not os.path.exists(self._socket):
            if not group.leader_running() or time.monotonic() > deadline:
                raise RuntimeError("the model server did not start")
            time.sleep(0.1)
        return _Client(self._socket, spec["closes"])

    def close(self, obj):
        obj.group.members()  # snapshot the descendants
        pids = obj.group.known_pids()
        before = gpu_memory_by_pid(pids)
        super().close(obj)
        after = gpu_memory_by_pid(pids)
        record = {
            "pids": pids,
            "alive_after": [p for p in pids if _alive(p)],
            "device_bytes_before": None if before is None else sum(before.values()),
            "device_bytes_after": None if after is None else sum(after.values()),
        }
        Path(obj.client.closes, f"{os.environ.get('SLURM_JOB_ID', 'local')}-{uuid4().hex}").write_text(
            json.dumps(record)
        )

    def version(self):
        return "1"


@python.define
def Infer(model: Provided[object, "cluster-model"], x: int) -> float:
    return model(x)


@python.define
def Serve(model: Provided[object, "cluster-server"], x: int) -> float:
    return model.client(x)


def queue_snapshot(pool):
    """Each queue of the pool at ``pool`` as its worker sees it now: per ticket its node, write
    time (storage mtime), job count and highest settled count, and the start time of each live
    claim (from the holder record, on the storage clock)."""
    queues = {}
    for q in Path(pool, "queues").iterdir():
        settled = {}
        for mark in (q / "settled").glob("*"):
            digest, _, count = mark.name.rpartition("-")
            settled[digest] = max(settled.get(digest, 0), int(count))
        tickets, claims = {}, {}
        for t in (q / "tickets").glob("*.json"):
            try:
                record = json.loads(t.read_bytes())
                tickets[t.stem] = {
                    "node": record["node"],
                    "written": t.stat().st_mtime,
                    "jobs": len(record["jobs"]),
                    "settled": settled.get(t.stem, 0),
                }
            except (OSError, ValueError):
                continue
        for holder in (q / "claims").glob("*.lease/*/holder.json"):
            try:
                claims[holder.parent.parent.name[: -len(".lease")]] = json.loads(
                    holder.read_bytes()
                )["started"]
            except (OSError, ValueError, KeyError):
                continue
        queues[q.name] = {"tickets": tickets, "claims": claims}
    return queues


@python.define
def Logged(
    model: Provided[object, "cluster-model"], x: int, label: str, order: str, pool: str, delay: float
) -> float:
    # one worker serves both queues: each call records its node and what every queue held
    time.sleep(delay)
    with open(order, "a") as f:
        f.write(json.dumps({"label": label, "x": x, "queues": queue_snapshot(pool)}) + "\\n")
    return model(x)


@workflow.define
def Infers(spec: dict, xs: list, gpu_mem: str = "") -> list:
    need = ComputeResources(gpus=1, gpu_mem=gpu_mem or None)
    return workflow.add(Infer(model=spec).split(x=xs), name="I", resources=need).out


@workflow.define
def Serves(spec: dict, xs: list) -> list:
    need = ComputeResources(gpus=1)
    return workflow.add(Serve(model=spec).split(x=xs), name="S", resources=need).out


@workflow.define(outputs=["a", "b"])
def TwoModels(a: dict, b: dict, xs: list, order: str, pool: str, delay: float):
    need = ComputeResources(gpus=1)
    la = Logged(model=a, label="A", order=order, pool=pool, delay=delay).split(x=xs)
    lb = Logged(model=b, label="B", order=order, pool=pool, delay=delay).split(x=xs)
    ra = workflow.add(la, name="A", resources=need)
    rb = workflow.add(lb, name="B", resources=need)
    return ra.out, rb.out


@python.define
def DeviceHold(x: int, where: str, devices: int) -> str:
    # announce this worker's device, then hold until ``devices`` workers have announced theirs
    d = Path(where)
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "none")
    (d / f"device-{device}").touch()
    deadline = time.monotonic() + 3600
    while len(list(d.glob("device-*"))) < devices:
        if time.monotonic() > deadline:
            raise RuntimeError("the other workers never ran a job")
        time.sleep(0.2)
    return device


@workflow.define
def DeviceHolds(xs: list, where: str, devices: int) -> list:
    need = ComputeResources(gpus=1)
    node = DeviceHold(where=where, devices=devices).split(x=xs)
    return workflow.add(node, name="H", resources=need).out


@python.define
def Echo(x: int) -> int:
    return x


@workflow.define
def GpuEchoes(xs: list) -> list:
    need = ComputeResources(gpus=1)
    return workflow.add(Echo().split(x=xs), name="E", resources=need).out


@python.define
def KillMe(x: int, where: str, k: int) -> int:
    # job k's first execution returns only if the test's kill failed (its worker is killed under
    # it); any other returns at once
    d = Path(where)
    if x == k and not (d / "first").exists():
        (d / "first").touch()
        while not (d / "kill-failed").exists():
            time.sleep(1.0)
    (d / f"done-{x}-{uuid4().hex}").touch()
    return x


@workflow.define
def KillMes(xs: list, where: str, k: int) -> list:
    return workflow.add(KillMe(where=where, k=k).split(x=xs), name="K").out


@python.define
def Quick(x: int, where: str) -> int:
    (Path(where) / f"done-Q-{x}-{uuid4().hex}").touch()
    return x


@python.define
def Gated(x: int, where: str) -> int:
    # marks its start, waits for "go", marks its end
    d = Path(where)
    (d / f"started-G-{x}-{uuid4().hex}").touch()
    while not (d / "go").exists():
        time.sleep(1.0)
    (d / f"done-G-{x}-{uuid4().hex}").touch()
    return x


@workflow.define
def Resume(xs: list, where: str) -> list:
    q = workflow.add(Quick(where=where).split(x=xs), name="Q")
    return workflow.add(Gated(x=q.out, where=where), name="G").out
'''

# The process provider's child: standalone (nothing from pydra), so it can run in another
# environment. It records its start, holds the model on its one visible device and answers x.
SERVER = """
import json, os, socket, sys, uuid
from pathlib import Path

spec, path = json.loads(sys.argv[1]), sys.argv[2]
Path(spec["starts"], f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}-{uuid.uuid4().hex}").touch()
width = int(spec.get("width", 1024))
try:
    import torch

    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    layer = torch.nn.Linear(width, width).to(device).eval()

    def model(x):
        with torch.no_grad():
            return float(layer(torch.full((1, width), x, device=device)).sum().item())

except ImportError:
    import cupy

    cupy.cuda.Device(0).use()
    weight = cupy.random.RandomState(0).standard_normal((width, width), dtype=cupy.float32)

    def model(x):
        return float((weight @ cupy.full((width,), x, dtype=cupy.float32)).sum())

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(path + ".tmp")
os.rename(path + ".tmp", path)
server.listen(16)
while True:
    conn, _ = server.accept()
    with conn:
        data = b""
        while not data.endswith(b"\\n"):
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        if data.endswith(b"\\n"):
            conn.sendall(f"{model(float(data.decode()))!r}\\n".encode())
"""

# A role of the cross-node lease check, run on its own node: A holds and beats past stale_after,
# then SIGKILLs itself holding the lease; B and C try to take it once a second (a fresh lease per
# try, as the pool's one-shot takes do). The one that takes it holds past stale_after, writes
# "ended" and releases only once every other taker has given up, so a second take cannot follow.
LEASE_ROLE = """
import json, os, signal, socket, sys, time
from pathlib import Path
from pydra.engine.lease import DirectoryLease, LeaseSettings, LeaseTimeout
from pydra.utils.storage import storage_now

role, root = sys.argv[1], Path(sys.argv[2])
cfg = json.loads((root / "config.json").read_text())
settings = LeaseSettings(**cfg["lease"])
resource = root / "leases" / "resource"


def log(event, **fields):
    record = {"role": role, "host": socket.gethostname(), "event": event,
              "storage_t": storage_now(root), **fields}
    with open(root / f"events-{role}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\\n")
        f.flush()
        os.fsync(f.fileno())


if role == "A":
    lease = DirectoryLease.at(resource, settings)
    lease.__enter__()
    log("held")
    deadline = time.monotonic() + cfg["hold_s"]
    while time.monotonic() < deadline:
        time.sleep(settings.renew_interval)
        log("beat", owns=lease.prove(), lost=lease.lost)
    log("kill")
    os.kill(os.getpid(), signal.SIGKILL)

others = [r for r in cfg["takers"] if r != role]
log("trying")
bound = time.monotonic() + cfg["bound_s"]
while time.monotonic() < bound:
    if (root / "ended").exists():
        log("gave-up")
        sys.exit(0)
    lease = DirectoryLease.at(resource, settings)
    try:
        delay = lease.try_take()
    except LeaseTimeout:
        delay = 1.0
    if delay is not None:
        time.sleep(1.0)
        continue
    log("took", broke=len(lease.broke))
    end = time.monotonic() + settings.stale_after + 2 * settings.renew_interval
    while time.monotonic() < end:
        time.sleep(settings.renew_interval)
    owned = lease.prove()
    (root / "ended").touch()
    while time.monotonic() < bound and not all(
        any(json.loads(line)["event"] == "gave-up"
            for line in (root / f"events-{o}.jsonl").read_text().splitlines())
        for o in others if (root / f"events-{o}.jsonl").exists()
    ):
        time.sleep(1.0)
    log("released", owns=owned, lost=lease.lost)
    lease.__exit__(None, None, None)
    sys.exit(0)
log("timed-out")
sys.exit(1)
"""

# A coordinator of the resume check, run on its own node. "first" runs the Resume workflow and
# SIGKILLs itself once a Gated job has started (claimed by a worker, held there until "go");
# "second" opens the same pool (the next epoch), runs the same workflow to the end, and records
# the jobs its runner handed to the pool (the rest it read from the store).
COORDINATOR = """
import json, os, signal, sys, threading, time
from pathlib import Path

role, config = sys.argv[1], Path(sys.argv[2])
cfg = json.loads(config.read_text())
import _pydra_cluster_tasks as tasks
from pydra.engine.lease import LeaseSettings
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.pool import WarmPool, _fingerprint
from pydra.workers.pool.settings import PoolSettings
from pydra.workers.pool.storage import job_name
from pydra.workers.pool.substrate import SlurmSubstrate

marks = Path(cfg["marks"])
sub = Submitter(cache_root=cfg["store"], worker="debug")
settings = PoolSettings(lease=LeaseSettings(**cfg["lease"]), **cfg["settings"])
pool = WarmPool(cfg["pool"], [WorkerClass(**cfg["class"])], SlurmSubstrate(), store=sub.store,
                settings=settings)
(marks / f"coordinator-{role}").write_text(json.dumps({"epoch": pool.epoch}))
handed = []
submit = pool.submit


def recording(jobs):
    handed.extend(jobs)
    with open(marks / f"handed-{role}", "a") as f:
        for job in jobs:
            f.write(f"{job.name} {job_name(job.checksum, _fingerprint(job))}\\n")
    return submit(jobs)


pool.submit = recording
# the jobs the runner completes from the store instead of dispatching (a success record, or a
# hit of the store's find): its reuses, counted where they happen
hits = []
read_succeeded = GraphRunner._read_succeeded_jobs


def counting(store, jobs):
    dispatch = read_succeeded(store, jobs)
    kept = {id(job) for job in dispatch}
    hits.extend(job.name for job in jobs if id(job) not in kept)
    return dispatch


GraphRunner._read_succeeded_jobs = staticmethod(counting)
settle_stored = GraphRunner._settle_stored


def settling(self, job, result):
    hits.append(job.name)
    return settle_stored(self, job, result)


GraphRunner._settle_stored = settling
if role == "first":
    def watch():
        while not list(marks.glob("started-G-*")):
            time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=watch, daemon=True).start()
task = tasks.Resume(xs=list(range(cfg["jobs"])), where=str(marks))
out, error = None, None
try:
    with Submitter(worker=pool) as run:
        out = run(task, raise_errors=True).outputs.out
except Exception as exc:
    errors = list(getattr(exc, "errors", None) or [])
    error = (
        f"{len(errors)} job(s) failed; the first: {' '.join(errors[0].split())}"
        if errors
        else f"{type(exc).__name__}: {' '.join(str(exc).split())}"
    )
finally:
    pool.close()
(marks / f"result-{role}.json").write_text(
    json.dumps({"out": out, "error": error, "handed": len(handed), "hits": hits, "epoch": pool.epoch})
)
"""


# -- verdicts ------------------------------------------------------------------------------------


class Verdict:
    """The named checks and figures of one test, reported in its one verdict line."""

    def __init__(self, test_id: str):
        self.test_id = test_id
        self.checks: list[tuple[str, bool]] = []
        self.figures: dict[str, ty.Any] = {}

    def check(self, name: str, ok: ty.Any) -> bool:
        self.checks.append((name, bool(ok)))
        return bool(ok)

    def figure(self, **figures: ty.Any) -> None:
        self.figures.update(figures)


def _report(test_id: str, status: str, reason: str) -> None:
    """Print the verdict line whole (no cut) and keep it in ``<shared>/verdicts.log``."""
    line = " ".join(f"VERDICT {test_id} {status} {reason}".split())
    print(line, flush=True)
    if SHARED:
        with (
            contextlib.suppress(OSError),
            open(Path(SHARED) / "verdicts.log", "a") as f,
        ):
            f.write(line + "\n")


def _reason(v: Verdict, error: str | None = None) -> str:
    """Every check with its result, ``[name=PASS]`` or ``[name=FAIL]``, then the error if any, then
    the figures."""
    checks = " ".join(f"[{name}={'PASS' if ok else 'FAIL'}]" for name, ok in v.checks)
    figures = " ".join(f"{k}={val}" for k, val in v.figures.items())
    parts = [checks or "[no check reached]"]
    if error is not None:
        parts.append(f"error={error}")
    if figures:
        parts.append(figures)
    return "; ".join(parts)


@contextlib.contextmanager
def verdict(test_id: str) -> ty.Iterator[Verdict]:
    """Print ``VERDICT <test id> PASS|FAIL <reason>`` once, naming every check with its result, then
    fail the test unless every check passed. An exception inside is the verdict ``FAIL``, with the
    checks reached and the exception."""
    v = Verdict(test_id)
    try:
        yield v
    except pytest.skip.Exception:
        raise
    except BaseException as exc:
        _report(test_id, "FAIL", _reason(v, f"{type(exc).__name__}: {exc}"))
        raise
    failed = [name for name, ok in v.checks if not ok]
    if failed or not v.checks:
        _report(test_id, "FAIL", _reason(v))
        raise AssertionError(f"{test_id}: failed checks {failed}; {_reason(v)}")
    _report(test_id, "PASS", _reason(v))


# -- fixtures ------------------------------------------------------------------------------------


@pytest.fixture
def slurm(any_worker):
    """Only the ``slurm`` parametrisation runs, on a cluster with shared storage."""
    if any_worker != "slurm":
        pytest.skip("runs only under --only-worker=slurm on a Slurm cluster")
    if not SHARED:
        pytest.skip("set PYDRA_TEST_SHARED_DIR to a directory every node sees")
    if not shutil.which("sbatch"):
        pytest.skip("needs a Slurm scheduler (sbatch on PATH)")


@pytest.fixture(scope="session")
def tasks():
    """The task module and the child's script, on the shared storage, importable here, in the
    workers this process starts and in the ``srun`` steps (through ``PYTHONPATH``)."""
    if not SHARED:
        pytest.skip("set PYDRA_TEST_SHARED_DIR to a directory every node sees")
    where = Path(SHARED) / f"tasks-{uuid.uuid4().hex[:12]}"
    where.mkdir(parents=True)
    (where / "_pydra_cluster_tasks.py").write_text(TASKS)
    (where / "cluster_server.py").write_text(SERVER)
    (where / "cluster_coordinator.py").write_text(COORDINATOR)
    sys.path.insert(0, str(where))
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(where), *filter(None, [old])])
    import _pydra_cluster_tasks

    _pydra_cluster_tasks.WHERE = where
    yield _pydra_cluster_tasks
    sys.path.remove(str(where))
    if old is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = old


@pytest.fixture
def where(request, slurm):
    """A fresh directory of this test on the shared storage."""
    d = Path(SHARED) / f"{request.node.originalname}-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def pools():
    """Every pool a test makes is closed, and any Slurm job named for it is cancelled."""
    made: list = []
    yield made
    for pool in made:
        try:
            pool.close()
        finally:
            _cancel_pool_jobs(pool.root)


def _cancel_pool_jobs(root: Path) -> None:
    """Cancel every queued or running job named for the pool at ``root`` (``pydra-<pool id>-``)."""
    try:
        pool_id = json.loads((Path(root) / "pool.json").read_text())["pool_id"]
    except (OSError, ValueError, KeyError):
        return
    out = subprocess.run(
        ["squeue", "-h", f"--user={os.getuid()}", "-o", "%i|%j"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    ids = [
        line.split("|", 1)[0]
        for line in out.splitlines()
        if "|" in line and line.split("|", 1)[1].startswith(f"pydra-{pool_id}-")
    ]
    if ids:
        subprocess.run(["scancel", *ids], capture_output=True, timeout=120)


def _need_device_library() -> str:
    from importlib.util import find_spec

    for name in ("torch", "cupy"):
        if find_spec(name) is not None:
            return name
    pytest.skip(
        "neither torch nor cupy is importable here (nor, so, in the workers): no way to "
        "put a model on the device; install torch into this environment"
    )


def _need_local_gpus(n: int) -> list[str]:
    """The devices of this allocation (``CUDA_VISIBLE_DEVICES``, else what ``nvidia-smi`` lists)."""
    from pydra.workers.pool.substrate import visible_devices

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [d.strip() for d in visible.split(",") if d.strip()] or visible_devices()
    if len(devices) < n:
        pytest.skip(
            f"needs an allocation holding {n} GPUs on this node; it holds {len(devices)}"
        )
    return devices


def _need_nodes(n: int) -> list[str]:
    """The distinct nodes of this allocation, from ``SLURM_JOB_NODELIST``."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist or not shutil.which("srun"):
        pytest.skip(f"needs a {n}-node Slurm allocation (SLURM_JOB_NODELIST and srun)")
    nodes = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout.split()
    if len(set(nodes)) < n:
        pytest.skip(
            f"needs a {n}-node allocation; SLURM_JOB_NODELIST names {len(set(nodes))}"
        )
    return nodes[:n]


# -- helpers -------------------------------------------------------------------------------------


def _substrate_args(gpu: bool) -> dict[str, str]:
    args = {}
    partition = GPU_PARTITION if gpu else PARTITION
    if partition:
        args["partition"] = partition
    if ACCOUNT:
        args["account"] = ACCOUNT
    return args


def _class(name: str, gpu: bool, **fields):
    from pydra.workers.pool.classes import WorkerClass

    base: dict[str, ty.Any] = dict(name=name, device="cpu", cpus=2, max_workers=1)
    if gpu:
        base.update(device="cuda", gpus=1, gpu_type=GPU_TYPE, cpus=4)
    if SLURM_TIME:
        base["time"] = int(SLURM_TIME) * 60
    if SLURM_MEM:
        base["mem"] = SLURM_MEM
    base["substrate_args"] = _substrate_args(gpu)
    base.update(fields)
    return WorkerClass(**base)


def _settings(**fields):
    from pydra.workers.pool.settings import PoolSettings

    base = dict(lease=LEASE, poll_s=5.0, idle_polls=120)
    base.update(fields)
    return PoolSettings(**base)


def _store(where: Path):
    return Submitter(cache_root=where / "store", worker="debug").store


def _pool(where: Path, pools, classes, substrate=None, **kwargs):
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.substrate import SlurmSubstrate

    kwargs.setdefault("settings", _settings())
    store = kwargs.pop("store", None) or _store(where)
    pool = WarmPool(
        where / "pool",
        classes,
        substrate if substrate is not None else SlurmSubstrate(),
        store=store,
        **kwargs,
    )
    pools.append(pool)
    return pool


def _drive(pool, sub, task):
    """Run ``task``'s nodes on ``pool`` through the asynchronous driver; the pool stays open."""
    from pydra.engine.drivers import drive
    from pydra.engine.runner import GraphRunner

    runner = GraphRunner(task, pool.store, on_error="raise_at_end", submitter=sub)
    asyncio.run(
        drive(
            runner,
            {"pool": pool},
            lambda job: "pool",
            on_error="raise_at_end",
        )
    )
    return runner


def _files(d: Path, prefix: str = "") -> list[str]:
    return sorted(p.name for p in d.iterdir() if p.name.startswith(prefix))


def _by_worker(d: Path) -> dict[str, int]:
    """Loads (or starts) per worker, from the files a provider writes: ``<job>-<pid>-<uuid>``."""
    return dict(collections.Counter(n.rsplit("-", 1)[0] for n in _files(d)))


def _generations(pool_root: Path) -> dict[str, list[dict]]:
    """Every job's follow generations, by job, in order."""
    found: dict[str, list[dict]] = {}
    follow = pool_root / "follow"
    for job in sorted(follow.iterdir()) if follow.exists() else []:
        numbered = sorted((int(p.name), p) for p in job.iterdir() if p.name.isdigit())
        found[job.name] = [json.loads(p.read_bytes()) for _, p in numbered]
    return found


def _losses(pool_root: Path) -> list[Path]:
    return sorted(pool_root.glob("queues/*/losses/*/*"))


def _queue_names(pool_root: Path) -> list[str]:
    return sorted(p.read_text().strip() for p in pool_root.glob("queues/*/name"))


def _ready_marks(pool_root: Path) -> list[dict]:
    return [
        json.loads(p.read_bytes())
        for p in sorted(pool_root.glob("workers/*/ready.json"))
    ]


def _wait(predicate, what: str, bound: float = BOUND) -> None:
    deadline = time.monotonic() + bound
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(1.0)


def _failure(exc: BaseException) -> str:
    """A failed run in one line: how many jobs failed and the first one's error, whole."""
    errors = list(getattr(exc, "errors", None) or [])
    if not errors:
        return f"{type(exc).__name__}: {' '.join(str(exc).split())}"
    return f"{len(errors)} job(s) failed; the first: {' '.join(errors[0].split())}"


def _run(sub, task):
    """``sub(task)``, a failed run raising with its jobs' cause (`_failure`) rather than returning
    a result whose ``outputs`` is None."""
    from pydra.engine.submitter import WorkflowError

    try:
        return sub(task, raise_errors=True)
    except WorkflowError as exc:
        raise AssertionError(f"the run failed: {_failure(exc)}") from exc


def _model_spec(where: Path, name: str = "model", **fields) -> dict:
    loads = where / f"loads-{name}"
    loads.mkdir()
    spec = {"loads": str(loads), "width": 1024, "seed": 0}
    spec.update(fields)
    return spec


def _srun(node: str, argv: list[str], log: Path) -> subprocess.Popen:
    """One ``srun`` step of one task on ``node``, sharing the allocation's resources."""
    out = open(log, "ab")
    try:
        return subprocess.Popen(
            [
                "srun",
                "--nodes=1",
                "--ntasks=1",
                f"--nodelist={node}",
                "--overlap",
                "--kill-on-bad-exit=0",
                *argv,
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        out.close()


# Variables that pass an allocation's sizes and options to srun, sbatch and salloc: a step started
# inside another job with them asks for that job's nodes, CPUs and memory and is refused.
_SLURM_INPUT = ("SLURM_", "SRUN_", "SBATCH_", "SALLOC_")
_SLURM_KEPT = ("SLURM_CONF",)  # where the cluster's configuration is, not a size


def _kill_command(
    job_id: str, token: str, environ: ty.Mapping[str, str]
) -> tuple[list[str], dict[str, str]]:
    """The ``srun`` step that SIGKILLs the worker ``token`` inside its own Slurm job ``job_id``,
    and the environment to run it in: this process's, without the Slurm input variables of the
    allocation running pytest, and with explicit minimal sizes.

    ``scancel --signal=KILL`` is no use here: Slurm turns it into a cancel (SIGTERM, then SIGKILL
    after KillWait), which the worker survives long enough to release its claims. ``[t]oken``
    keeps the pattern from matching this srun's own command line.
    """
    env = {
        k: v
        for k, v in environ.items()
        if k in _SLURM_KEPT or not k.startswith(_SLURM_INPUT)
    }
    argv = [
        "srun",
        f"--jobid={job_id}",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=100M",
        "pkill",
        "-KILL",
        "-f",
        f"[t]oken {token}",
    ]
    return argv, env


def test_kill_command_is_built_clean():
    """Inside an allocation the kill step carries none of its Slurm input variables, only explicit
    minimal sizes in the worker's own job (no Slurm is run)."""
    allocation = {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "SLURM_CONF": "/etc/slurm/slurm.conf",
        "SLURM_JOB_ID": "500",
        "SLURM_NNODES": "3",
        "SLURM_JOB_NUM_NODES": "3",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "65536",
        "SLURM_TRES_PER_TASK": "cpu:8",
        "SRUN_CPUS_PER_TASK": "8",
        "SBATCH_PARTITION": "gpu",
        "SALLOC_ACCOUNT": "lab",
    }
    argv, env = _kill_command("123", "a" * 32, allocation)
    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "SLURM_CONF": "/etc/slurm/slurm.conf",
    }
    assert argv[:7] == [
        "srun",
        "--jobid=123",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=100M",
    ]
    assert argv[7:] == ["pkill", "-KILL", "-f", f"[t]oken {'a' * 32}"]
    assert not any(a.startswith(("--qos", "--partition", "--account")) for a in argv)


def test_a_failed_run_names_its_cause():
    """A run whose jobs failed fails the check with how many failed and the first one's error,
    never later as a None ``outputs`` (no Slurm is run)."""
    from pydra.engine.submitter import WorkflowError

    def sub(task, raise_errors=False):
        if not raise_errors:
            return None  # an errored result: the cause stays in the store
        raise WorkflowError(
            "Workflow job main failed with errors:",
            [
                "Job 'I(0)', Infer(x=0), errored:PoolJobFailed:\n FileNotFoundError: a",
                "Job 'I(1)', Infer(x=1), errored:PoolJobFailed: FileNotFoundError: b",
            ],
        )

    with pytest.raises(AssertionError) as caught:
        _run(sub, object())
    assert str(caught.value) == (
        "the run failed: 2 job(s) failed; the first: "
        "Job 'I(0)', Infer(x=0), errored:PoolJobFailed: FileNotFoundError: a"
    )


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(60)


# -- one worker, one load ------------------------------------------------------------------------


@slurm_case
@gpu
def test_one_load_n_jobs_gpu(where, tasks, pools):
    """One `SlurmSubstrate` worker runs ``PYDRA_TEST_JOBS`` jobs of a provided model: one
    ``sbatch``, one load, every job committed. The calibration's steady per-job time is fitted
    from the warm calls, and the calls after the load are in that steady band rather than the
    cold one: the 95th percentile of their per-job wall is below the first call's (relative to
    the run's own calibration; no wall-clock threshold)."""
    from pydra.workers.pool.calibration import DirectoryCalibration, hardware_of

    _need_device_library()
    with verdict("test_one_load_n_jobs_gpu") as v:
        spec = _model_spec(where)
        cls = _class("gpu", True, max_workers=1)
        pool = _pool(where, pools, [cls])
        substrate = pool.substrate
        with Submitter(worker=pool, providers=[tasks.DeviceModel()]) as sub:
            result = _run(sub, tasks.Infers(spec=spec, xs=list(range(JOBS))))
        out = result.outputs.out
        v.check(
            "every job committed",
            len(out) == JOBS and all(math.isfinite(y) for y in out),
        )
        v.check("one sbatch", len(substrate.handles) == 1)
        loads = _by_worker(Path(spec["loads"]))
        v.check("one load", sum(loads.values()) == 1)
        samples = [
            json.loads(line)
            for p in pool.root.glob("workers/*/samples.jsonl")
            for line in p.read_text().splitlines()
        ]
        cold = [s for s in samples if s["cold"]]
        warm = sorted(
            s["wall_s"] / s["batch_size"]
            for s in samples
            if not s["cold"] and not s["reload"]
        )
        keys = {s["key"] for s in samples}
        ready = _ready_marks(pool.root)
        # the worker files its samples under its class with the device memory it read
        measured = attrs.evolve(cls, gpu_mem=ready[0]["gpu_mem"]) if ready else cls
        times = (
            DirectoryCalibration(pool.root / "calibration").times(
                next(iter(keys)), hardware_of(measured)
            )
            if len(keys) == 1
            else []
        )
        steady = next((t for t in times if t.batch == 1), None)
        v.check("one cold call", len(cold) == 1 and len(warm) == JOBS - 1)
        v.check(
            "calibration fitted from the warm calls",
            steady is not None and steady.n == len(warm),
        )
        p95 = warm[int(0.95 * (len(warm) - 1))] if warm else None
        v.check(
            "warm calls in the steady band",
            bool(cold) and p95 is not None and p95 < cold[0]["wall_s"],
        )
        v.figure(
            jobs=JOBS,
            sbatch=[h.job_id for h in substrate.handles],
            loads_per_worker=loads,
            cold_s=round(cold[0]["wall_s"], 4) if cold else None,
            steady_s=round(steady.per_job_s, 4) if steady else None,
            p95_s=round(p95, 4) if p95 is not None else None,
        )


@slurm_case
@gpu
def test_process_provider_warm_gpu(where, tasks, pools):
    """The same with a process-owning provider whose child serves the model: one child start for
    every job, and its eviction (the worker's idle exit closing it) leaves no process of the
    group and no device memory behind. The coordinator's own idle stop is held back until the
    close is recorded, so its ``scancel`` (whose SIGTERM reaches the child too) cannot race it.
    """
    _need_device_library()
    with verdict("test_process_provider_warm_gpu") as v:
        starts, closes = where / "starts", where / "closes"
        starts.mkdir()
        closes.mkdir()
        spec = {
            "script": str(tasks.WHERE / "cluster_server.py"),
            "python": SERVER_PYTHON,
            "starts": str(starts),
            "closes": str(closes),
            "width": 1024,
        }
        n = min(JOBS, 200)
        pool = _pool(
            where,
            pools,
            [_class("gpu", True, max_workers=1)],
            settings=_settings(idle_polls=6),
        )
        substrate = pool.substrate
        stop, deferred = substrate.stop, []
        substrate.stop = deferred.append  # no scancel until the close record exists
        with Submitter(worker=pool, providers=[tasks.ServerModel()]) as sub:
            result = _run(sub, tasks.Serves(spec=spec, xs=list(range(n))))
        out = result.outputs.out
        v.check(
            "every job committed", len(out) == n and all(math.isfinite(y) for y in out)
        )
        v.check("one sbatch", len(substrate.handles) == 1)
        v.check("one child start", len(_files(starts)) == 1)
        _wait(lambda: _files(closes), "the worker to exit idle and close its provider")
        substrate.stop = stop
        [record] = [json.loads((closes / c).read_text()) for c in _files(closes)]
        v.check("no process left", record["pids"] and not record["alive_after"])
        v.check("child held device memory", (record["device_bytes_before"] or 0) > 0)
        v.check("no device memory left", record["device_bytes_after"] == 0)
        v.figure(
            jobs=n,
            starts=len(_files(starts)),
            deferred_stops=len(deferred),
            close=record,
        )


@slurm_case
@gpu
def test_two_residencies_two_queues(where, tasks, pools):
    """Two models on one class: two routed queues, a worker that yields one queue for the other
    when its quantum ends although that queue still has a waiting ticket, and at most
    ``max_workers`` loads per model.

    Two tickets per queue are written ahead (``max_queued_jobs``); each queue's third is written
    once its first settles, and so is younger than the other queue's waiting ticket. Each call
    snapshots the queues, and at some switch the queue left must still hold a ticket written
    before the claim of the queue taken and not yet settled: the worker passed over waiting work
    of its own queue for older work of the other, which no claim rule but the quantum's does.
    """
    _need_device_library()
    with verdict("test_two_residencies_two_queues") as v:
        quantum, n, max_workers, delay = 4, 16, 1, 2.0
        a, b = _model_spec(where, "a", seed=1), _model_spec(where, "b", seed=2)
        order = where / "order.log"
        pool = _pool(
            where,
            pools,
            [_class("gpu", True, max_workers=max_workers, quantum_jobs=quantum)],
            settings=_settings(poll_s=1.0, max_queued_jobs=2 * quantum),
        )
        with Submitter(worker=pool, providers=[tasks.DeviceModel()]) as sub:
            result = _run(
                sub,
                tasks.TwoModels(
                    a=a,
                    b=b,
                    xs=list(range(n)),
                    order=str(order),
                    pool=str(pool.root),
                    delay=delay,
                ),
            )
        v.check(
            "every job committed", len(result.outputs.a) == n == len(result.outputs.b)
        )
        names = _queue_names(pool.root)
        v.check(
            "two queues", len(names) == 2 and all(q.startswith("gpu#") for q in names)
        )
        calls = [json.loads(line) for line in order.read_text().splitlines()]
        labels = [c["label"] for c in calls]
        runs = [len(list(g)) for _, g in itertools.groupby(labels)]
        yields = [
            i
            for i in range(1, len(calls))
            if labels[i] != labels[i - 1]
            and _left_waiting(calls[i], labels[i - 1], calls[i - 1])
        ]
        v.check("quantum yields with its own queue waiting", yields)
        loads_a, loads_b = _by_worker(Path(a["loads"])), _by_worker(Path(b["loads"]))
        v.check(
            "at most max_workers loads per model",
            0 < sum(loads_a.values()) <= max_workers
            and 0 < sum(loads_b.values()) <= max_workers,
        )
        v.figure(
            queues=len(names),
            runs=runs,
            yields_at=yields,
            loads_a=loads_a,
            loads_b=loads_b,
        )


def _left_waiting(call: dict, left: str, before: dict) -> bool:
    """Whether, at this call (the first after a switch), the queue of node ``left`` held a ticket
    written before the claim the worker now runs under and not yet settled, which the worker's
    node had already listed in ``before``, the last call on ``left``: a ticket the storage showed
    only after the switch made the queue look empty, and leaving it is no yield."""

    def node_of(queue: dict) -> str | None:
        for t in queue["tickets"].values():
            return t["node"].rsplit("/", 1)[-1]
        return None

    queues = list(call["queues"].values())
    taken = [q for q in queues if node_of(q) == call["label"]]
    gone = [q for q in queues if node_of(q) == left]
    if not taken or not gone or not taken[0]["claims"]:
        return False
    claimed = max(taken[0]["claims"].values())
    seen = {
        digest
        for q in before["queues"].values()
        if node_of(q) == left
        for digest in q["tickets"]
    }
    return any(
        t["written"] < claimed and t["settled"] < t["jobs"] and digest in seen
        for digest, t in gone[0]["tickets"].items()
    )


def _snapshot(label: str, tickets: dict, claims: dict) -> dict:
    """A synthetic call record: queue ``qa`` holds node A's tickets, ``qb`` node B's."""
    queues = {"qa": {"tickets": {}, "claims": {}}, "qb": {"tickets": {}, "claims": {}}}
    for digest, (node, written, settled) in tickets.items():
        queue = queues["qa" if node == "A" else "qb"]
        queue["tickets"][digest] = {
            "node": node,
            "written": written,
            "jobs": 4,
            "settled": settled,
        }
    for digest, started in claims.items():
        queues["qa" if digest.startswith("a") else "qb"]["claims"][digest] = started
    return {"label": label, "queues": queues}


def test_left_waiting_needs_the_ticket_seen_before_the_switch():
    """A yield is a switch away from queue A while A held a waiting ticket that the worker's node
    had already listed before the switch. A ticket written before the new claim but first seen
    after the switch (the storage showed it late, so A looked empty) is no yield."""
    before = _snapshot("A", {"a1": ("A", 10.0, 0), "a2": ("A", 11.0, 0)}, {"a1": 12.0})
    after = _snapshot(
        "B",
        {"a1": ("A", 10.0, 4), "a2": ("A", 11.0, 0), "b1": ("B", 10.5, 0)},
        {"b1": 20.0},
    )
    assert _left_waiting(after, "A", before)  # a2 was waiting and already seen: a yield
    late_before = _snapshot("A", {"a1": ("A", 10.0, 0)}, {"a1": 12.0})
    late_after = _snapshot(
        "B",
        {"a1": ("A", 10.0, 4), "a3": ("A", 19.0, 0), "b1": ("B", 10.5, 0)},
        {"b1": 20.0},
    )
    assert not _left_waiting(
        late_after, "A", late_before
    )  # a3 seen only after the switch
    younger = _snapshot("B", {"a1": ("A", 10.0, 4), "a2": ("A", 21.0, 0)}, {"b1": 20.0})
    assert not _left_waiting(younger, "A", before)  # written after the claim
    drained = _snapshot("B", {"a1": ("A", 10.0, 4), "a2": ("A", 11.0, 4)}, {"b1": 20.0})
    assert not _left_waiting(drained, "A", before)  # nothing left waiting


# -- faults --------------------------------------------------------------------------------------


@slurm_case
def test_worker_killed_mid_ticket(where, tasks, pools):
    """SIGKILL of the worker a third of the way through its ticket: the claim is
    taken over once stale by a new worker, under new attempts; every job is committed once and
    the loss is recorded."""
    with verdict("test_worker_killed_mid_ticket") as v:
        n = 12
        k = n // 3
        marks = where / "marks"
        marks.mkdir()
        pool = _pool(
            where, pools, [_class("cpu", False, max_workers=1, quantum_jobs=n)]
        )
        substrate = pool.substrate
        killed: dict[str, ty.Any] = {}
        ended = threading.Event()

        def kill_when_stuck():
            try:
                _wait(
                    lambda: (marks / "first").exists() or ended.is_set(),
                    f"job {k} to start",
                )
                if not (marks / "first").exists():
                    killed["error"] = f"the run ended before job {k} started"
                    return
                [handle] = substrate.handles
                argv, env = _kill_command(handle.job_id, handle.token, os.environ)
                version = subprocess.run(
                    ["srun", "--version"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                shown = (version.stdout or version.stderr).strip().splitlines()
                killed["srun"] = (
                    shown[0] if version.returncode == 0 and shown else "unknown"
                )
                done = subprocess.run(
                    argv, env=env, capture_output=True, text=True, timeout=300
                )
                killed.update(job=handle.job_id, done=len(_files(marks, "done-")))
                if done.returncode:
                    killed["error"] = (
                        f"{' '.join(argv)} exited {done.returncode}: "
                        f"{done.stderr.strip() or done.stdout.strip()}"
                    )
            except (
                BaseException
            ) as exc:  # reported in the verdict, never lost in the thread
                killed["error"] = f"{type(exc).__name__}: {exc}"
            if "error" in killed:
                (marks / "kill-failed").touch()  # job k returns, so the run ends now

        killer = threading.Thread(target=kill_when_stuck, daemon=True)
        killer.start()
        try:
            with Submitter(worker=pool) as sub:
                result = _run(
                    sub, tasks.KillMes(xs=list(range(n)), where=str(marks), k=k)
                )
        finally:
            ended.set()
            killer.join(BOUND)
            v.figure(kill=killed.get("error", "ran"))
        v.check("kill step ran", "job" in killed and "error" not in killed)
        v.check("every job committed", result.outputs.out == list(range(n)))
        done = collections.Counter(
            name.split("-")[1] for name in _files(marks, "done-")
        )
        v.check(
            "each job returned once",
            done == collections.Counter(str(x) for x in range(n)),
        )
        v.check("killed mid-ticket", 0 < killed.get("done", 0) < n)
        generations = _generations(pool.root)
        # a lone open job is placed again under a takeover generation, two or more are split
        moved = [
            job
            for job, gs in generations.items()
            if any(g["cause"] in ("takeover", "split") for g in gs[1:])
        ]
        v.check("taken over under new attempts", moved)
        v.check("loss recorded", _losses(pool.root))
        v.check("a new worker took over", len(substrate.handles) >= 2)
        v.figure(
            killed=killed,
            moved=len(moved),
            losses=len(_losses(pool.root)),
            causes=dict(
                collections.Counter(
                    g["cause"] for gs in generations.values() for g in gs
                )
            ),
            sbatch=[h.job_id for h in substrate.handles],
        )
        if not all(ok for _, ok in v.checks):
            _print_logs(pool.root)


@slurm_case
def test_claim_lease_cross_node(where, slurm):
    """Three roles on three nodes. A holder that
    beats past ``stale_after`` is never taken over; SIGKILLed while holding, it is taken over by
    exactly one of the two takers, after its kill, and no two holds overlap."""
    nodes = _need_nodes(3)
    with verdict("test_claim_lease_cross_node") as v:
        stale = LEASE.stale_after
        (where / "config.json").write_text(
            json.dumps(
                {
                    "lease": LEASE.to_json(),
                    "hold_s": 2 * stale,
                    "bound_s": BOUND,
                    "takers": ["B", "C"],
                }
            )
        )
        script = where / "cluster_lease_role.py"
        script.write_text(LEASE_ROLE)
        roles = {}
        try:
            roles["A"] = _srun(
                nodes[0],
                [sys.executable, str(script), "A", str(where)],
                where / "A.log",
            )
            _wait(lambda: _events(where, "A"), "the holder to take the lease")
            for role, node in (("B", nodes[1]), ("C", nodes[2])):
                roles[role] = _srun(
                    node,
                    [sys.executable, str(script), role, str(where)],
                    where / f"{role}.log",
                )
            for role in ("B", "C"):
                roles[role].wait(BOUND)
            roles["A"].wait(BOUND)
        finally:
            for proc in roles.values():
                _stop(proc)
        events = {r: _events(where, r) for r in ("A", "B", "C")}
        flat = [e for es in events.values() for e in es]
        kill = next((e["storage_t"] for e in events["A"] if e["event"] == "kill"), None)
        took = [e for e in flat if e["event"] == "took"]
        released = [e for e in flat if e["event"] == "released"]
        hosts = sorted({e["host"] for e in flat})
        beats = [e for e in events["A"] if e["event"] == "beat"]
        v.check("three hosts", len(hosts) == 3)
        v.check("holder killed holding", kill is not None)
        v.check(
            "beating holder never taken",
            beats and all(b["owns"] and not b["lost"] for b in beats),
        )
        v.check("exactly one takeover", len(took) == 1 and took[0]["broke"] >= 1)
        v.check(
            "taken after the kill",
            kill is not None and all(t["storage_t"] > kill for t in took),
        )
        v.check(
            "taker kept it to release",
            len(released) == 1 and released[0]["owns"] and not released[0]["lost"],
        )
        v.check("other taker gave up", [e for e in flat if e["event"] == "gave-up"])
        v.figure(
            hosts=len(hosts),
            takes=len(took),
            take_after_kill_s=(
                round(took[0]["storage_t"] - kill, 1)
                if took and kill is not None
                else None
            ),
            stale_after=stale,
        )


@slurm_case
def test_coordinator_killed_and_resumed_on_another_host(where, tasks, slurm):
    """The submitting process is SIGKILLed while its workers hold jobs; a coordinator on a second
    node resumes the run on the same pool: it follows the jobs the first put (none is put twice),
    reads the finished ones from the store, and every job executes once."""
    nodes = _need_nodes(3)
    with verdict("test_coordinator_killed_and_resumed_on_another_host") as v:
        n = 6
        marks = where / "marks"
        marks.mkdir()
        cls = {
            "name": "cpu",
            "device": "cpu",
            "cpus": 2,
            "max_workers": 2,
            "quantum_jobs": 1,
        }
        cls["substrate_args"] = _substrate_args(False)
        if SLURM_TIME:
            cls["time"] = int(SLURM_TIME) * 60
        if SLURM_MEM:
            cls["mem"] = SLURM_MEM
        config = where / "coordinator.json"
        config.write_text(
            json.dumps(
                {
                    "marks": str(marks),
                    "store": str(where / "store"),
                    "pool": str(where / "pool"),
                    "lease": LEASE.to_json(),
                    "settings": {"poll_s": 5.0, "idle_polls": 120},
                    "class": cls,
                    "jobs": n,
                }
            )
        )
        script = str(tasks.WHERE / "cluster_coordinator.py")
        procs = []
        try:
            first = _srun(
                nodes[0],
                [sys.executable, script, "first", str(config)],
                where / "first.log",
            )
            procs.append(first)
            first.wait(BOUND)
            if not v.check(
                "first coordinator killed",
                first.returncode != 0
                and (marks / "coordinator-first").exists()
                and list(marks.glob("started-G-*"))
                and not (marks / "result-first.json").exists(),
            ):
                ended = marks / "result-first.json"
                raise AssertionError(
                    f"the first coordinator exited {first.returncode} unkilled: "
                    + (
                        json.loads(ended.read_text())["error"] or "its run completed"
                        if ended.exists()
                        else f"no result; see {where / 'first.log'}"
                    )
                )
            tickets = {p.stem for p in (where / "pool").glob("queues/*/tickets/*.json")}
            second = _srun(
                nodes[1],
                [sys.executable, script, "second", str(config)],
                where / "second.log",
            )
            procs.append(second)
            _wait(
                lambda: second.poll() is not None
                or any(_node(line) == "G" for line in _lines(marks / "handed-second")),
                "the second coordinator to hand a Gated job to the pool",
            )
            if not any(_node(line) == "G" for line in _lines(marks / "handed-second")):
                ended = marks / "result-second.json"
                raise AssertionError(
                    f"the second coordinator exited {second.returncode} before "
                    "handing a Gated job: "
                    + (
                        json.loads(ended.read_text())["error"] or "no error"
                        if ended.exists()
                        else f"no result; see {where / 'second.log'}"
                    )
                )
            (marks / "go").touch()
            second.wait(BOUND)
        finally:
            for proc in procs:
                _stop(proc)
            _cancel_pool_jobs(where / "pool")
        result = json.loads((marks / "result-second.json").read_text())
        v.check(
            "resumed run complete",
            result["out"] == list(range(n)) and not result["error"],
        )
        executed = collections.Counter(
            "-".join(name.split("-")[1:3]) for name in _files(marks, "done-")
        )
        expected = collections.Counter(f"{node}-{x}" for node in "QG" for x in range(n))
        v.check("every job executed once", executed == expected)
        generations = _generations(where / "pool")
        puts = [sum(g["cause"] == "put" for g in gs) for gs in generations.values()]
        v.check("no job put twice", len(puts) >= n and all(p == 1 for p in puts))
        handed = _lines(marks / "handed-second")
        # reuses are counted where the runner completes a job from the store's records
        hits = [h for h in result["hits"] if _node(f"{h} -") in ("Q", "G")]
        followed = sum(
            1
            for line in handed
            if generations.get(line.split()[1], [{}])[0].get("ticket") in tickets
        )
        v.check("executions + reuses = jobs", len(handed) + len(hits) == 2 * n and hits)
        v.check("followed the first's tickets", followed >= 1)
        firsts = json.loads((marks / "coordinator-first").read_text())
        v.check("second took the next epoch", result["epoch"] > firsts["epoch"])
        v.figure(
            nodes=nodes[:2],
            handed=len(handed),
            reused=len(hits),
            followed=followed,
            first_exit=first.returncode,
            executed=sum(executed.values()),
        )


@slurm_case
@gpu
def test_load_oom_climbs_the_ladder(where, tasks, pools):
    """A class whose usable device memory is capped below the model: its load runs out of memory,
    an `UnfitRecord` is kept, the jobs move to the larger class uncharged and run there, and the
    next placement starts at the larger class."""
    _need_device_library()
    from pydra.engine.job import Job
    from pydra.workers.pool.calibration import DirectoryCalibration
    from pydra.workers.pool.fit import MeasuredFit
    from pydra.workers.pool.placement import fit_request
    from pydra.utils.resources import ComputeResources

    with verdict("test_load_oom_climbs_the_ladder") as v:
        gib = 1024**3
        cap = 2 * gib
        spec = _model_spec(where, ballast=4 * gib)
        args = _substrate_args(True)
        small = _class(
            "small",
            True,
            gpu_mem=cap,
            substrate_args={**args, "export": f"ALL,{tasks.CAP}={cap}"},
        )
        large = _class("large", True)
        calibration = DirectoryCalibration(where / "pool" / "calibration")
        pool = _pool(where, pools, [small, large], fit=MeasuredFit(calibration))
        xs = list(range(8))
        with Submitter(worker=pool, providers=[tasks.DeviceModel()]) as sub:
            result = _run(sub, tasks.Infers(spec=spec, xs=xs, gpu_mem="1G"))
            v.check("every job committed", len(result.outputs.out) == len(xs))
            probe = Job(
                task=tasks.Infer(model=spec, x=len(xs)),
                submitter=sub,
                name="probe",
                resources=ComputeResources(gpus=1, gpu_mem="1G"),
            )
            key = fit_request(probe).key
            records = list(calibration.unfit(key))
            v.check(
                "unfit record kept",
                any(
                    r.worker_class == "small" and r.resource == "device"
                    for r in records
                ),
            )
            generations = [g for gs in _generations(pool.root).values() for g in gs]
            ladder = [g for g in generations if g["cause"] == "ladder"]
            v.check(
                "moved to the larger class",
                ladder and all(g["phase"] == "load" for g in ladder),
            )
            v.check("no job charged", not any(g.get("charged") for g in generations))
            [placed] = pool.place([probe])
            tried = [(t.worker_class, t.verdict) for t in placed.tried]
            v.check(
                "next plan starts at the larger class",
                getattr(placed, "worker_class", None) == "large"
                and tried[:1] == [("small", "unfit")],
            )
        v.figure(
            unfit=[(r.worker_class, r.resource, r.grant) for r in records],
            ladder_moves=len(ladder),
            loads=_by_worker(Path(spec["loads"])),
            tried=tried,
        )


# -- local substrate inside one allocation -------------------------------------------------------


@slurm_case
@gpu
def test_local_multi_gpu(where, tasks, pools):
    """`LocalSubstrate` inside one allocation of two GPUs: one worker per device, given it through
    ``CUDA_VISIBLE_DEVICES``, and both busy at once (every job holds until both devices have run
    one)."""
    from pydra.workers.pool.substrate import LocalSubstrate

    devices = _need_local_gpus(2)[:2]
    with verdict("test_local_multi_gpu") as v:
        marks = where / "marks"
        marks.mkdir()
        substrate = LocalSubstrate(devices=devices)
        cls = _class("gpu", True, max_workers=2, quantum_jobs=1, substrate_args={})
        pool = _pool(
            where,
            pools,
            [cls],
            substrate=substrate,
            settings=_settings(poll_s=1.0, unmeasured_workers=2),
        )
        with Submitter(worker=pool) as sub:
            result = _run(
                sub, tasks.DeviceHolds(xs=list(range(4)), where=str(marks), devices=2)
            )
        v.check(
            "every job committed", sorted(set(result.outputs.out)) == sorted(devices)
        )
        v.check(
            "one worker per device",
            sorted(h.devices for h in substrate.handles)
            == sorted((d,) for d in devices),
        )
        v.check("both busy", _files(marks) == sorted(f"device-{d}" for d in devices))
        ready = _ready_marks(pool.root)
        v.check(
            "each worker sees one GPU of the type",
            len(ready) == 2
            and all(len(r["hardware"]["gpus"]) == 1 for r in ready)
            and all(
                GPU_TYPE.lower() in json.dumps(r["hardware"]["gpus"]).lower()
                for r in ready
            ),
        )
        v.figure(devices=devices, workers=len(substrate.handles))


@slurm_case
@gpu
def test_mistyped_grant_refused(where, tasks, pools):
    """A class naming another GPU type than the allocation's: each worker refuses the hardware it
    finds, the class breaks after its failed starts, and the jobs are ``NotRun(held=True)``, never
    failed."""
    from pydra.workers.pool.faults import FaultBudgets
    from pydra.workers.pool.substrate import LocalSubstrate
    from pydra.workers.pool.worker import EXIT_HARDWARE

    devices = _need_local_gpus(1)
    with verdict("test_mistyped_grant_refused") as v:
        wrong = "a100" if GPU_TYPE.lower() != "a100" else "h100"
        substrate = LocalSubstrate(devices=devices[:1])
        cls = _class("gpu", True, gpu_type=wrong, substrate_args={})
        pool = _pool(
            where,
            pools,
            [cls],
            substrate=substrate,
            faults=FaultBudgets(start_failures=2),
            settings=_settings(poll_s=1.0),
        )
        sub = Submitter(worker=pool)
        runner = _drive(pool, sub, tasks.GpuEchoes(xs=list(range(4))))
        held = [o for _, o in runner.not_run] if runner.not_run else []
        v.check("failed starts", len(substrate.handles) >= 2)
        v.check(
            "workers refused the hardware",
            substrate.handles
            and all(h.process.wait(BOUND) == EXIT_HARDWARE for h in substrate.handles),
        )
        v.check(
            "every job held",
            len(held) == 4 and all(getattr(o, "held", False) for o in held),
        )
        v.check(
            "no job failed", not runner.errors and not getattr(runner, "failed", None)
        )
        v.figure(
            wrong=wrong,
            starts=len(substrate.handles),
            held=len(held),
            exits=[h.process.returncode for h in substrate.handles],
        )


# -- the pool absent -----------------------------------------------------------------------------


PLAIN = """
import json, sys
from pathlib import Path
from pydra.compose import python, workflow
from pydra.engine.submitter import Submitter


@python.define
def Add(x: int) -> int:
    return x + 1


@workflow.define
def Adds(xs: list) -> list:
    return workflow.add(Add().split(x=xs), name="A").out


cache, args = sys.argv[1], sys.argv[2]
with Submitter(worker="slurm", cache_root=cache, sbatch_args=args) as sub:
    result = sub(Adds(xs=[1, 2, 3]))
print(json.dumps({
    "out": result.outputs.out,
    "pool_modules": sorted(m for m in sys.modules if m.startswith("pydra.workers.pool")),
}))
"""


@slurm_case
def test_plain_workflow_untouched_slurm(where, slurm):
    """A workflow on the plain Slurm worker, with no pool: it runs, imports nothing of
    ``pydra.workers.pool`` and leaves no pool file in the test's directory, which holds its cache
    and its user cache root (``PYDRA_CACHE_ROOT``)."""
    with verdict("test_plain_workflow_untouched_slurm") as v:
        script = where / "plain.py"
        script.write_text(PLAIN)
        cache = where / "cache"
        args = " ".join(f"--{k}={val}" for k, val in _substrate_args(False).items())
        before = sorted(p.relative_to(where) for p in where.rglob("*"))
        from pydra.utils.general import CACHE_ROOT_ENV_VAR

        env = {**os.environ, CACHE_ROOT_ENV_VAR: str(where / "cache" / "user")}
        proc = subprocess.run(
            [sys.executable, str(script), str(cache), args],
            capture_output=True,
            text=True,
            timeout=BOUND,
            cwd=where,
            env=env,
        )
        v.check("ran", proc.returncode == 0)
        report = (
            json.loads(proc.stdout.strip().splitlines()[-1])
            if proc.returncode == 0
            else {}
        )
        v.check("outputs", report.get("out") == [2, 3, 4])
        v.check("no pool import", report.get("pool_modules") == [])
        after = sorted(p.relative_to(where) for p in where.rglob("*"))
        new = [p for p in after if p not in before and p.parts[0] != "cache"]
        pool_files = [
            p
            for p in after
            if p.name in ("pool.json", "payloads", "queues", "follow", "coordinator")
        ]
        v.check("no new files outside the cache", new == [])
        v.check("no pool files", pool_files == [])
        v.figure(
            returncode=proc.returncode,
            new=[str(p) for p in new][:5],
            stderr=proc.stderr.strip().splitlines()[-1:] if proc.returncode else [],
        )


# -- small helpers used above ---------------------------------------------------------------------


def _events(where: Path, role: str) -> list[dict]:
    return [json.loads(line) for line in _lines(where / f"events-{role}.jsonl")]


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def _node(handed: str) -> str:
    """The node of a line the coordinator wrote for each job it handed to the pool."""
    return handed.split()[0].rsplit("/", 1)[-1].split("[", 1)[0]


def _print_logs(pool_root: Path, lines: int = 40) -> None:
    """The tail of each worker's log, for a failed verdict (shown by pytest on failure)."""
    for log in sorted((pool_root / "private" / "logs").glob("*.log")):
        print(f"--- {log.name}")
        print("\n".join(log.read_text(errors="replace").splitlines()[-lines:]))
