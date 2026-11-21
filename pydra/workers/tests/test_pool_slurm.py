"""The Slurm substrate (RFC 0005 section 4.5, W8), with ``sbatch``, ``squeue`` and ``scancel`` faked.

Ported from senselab's ``substrates_test.py``: the per-user CPU cap bounds the workers started, a
class becomes one ``sbatch`` line, failed starts back off and stop, a class whose workers never start
holds its jobs, job names are scoped to the pool and its epoch, and worker logs go under the pool's
private directory. The fakes are shell scripts on ``PATH`` that keep their queue in a directory.
Nothing is timed: backoff and ``attribute_delay`` run on fake clocks, and the only real wait is the
marked test that runs on the Slurm CI container.
"""

import asyncio
import contextlib
import json
import os
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from pydra.engine.lease import LeaseSettings
from pydra.engine.runner import GraphRunner, NotRun
from pydra.engine.submitter import Submitter
from pydra.workers import slurm
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.faults import FaultBudgets
from pydra.workers.pool.settings import PoolSettings
from pydra.workers.pool.substrate import (
    Attachment,
    SlurmSubstrate,
    SubstrateError,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the fake Slurm commands are shell scripts"
)

LEASE = LeaseSettings(
    stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
)
BOUND = 120.0  # seconds any wait on another process may take before the test fails
POOL_ID = "0123456789ab"

# sbatch: records its arguments, queues a job named by --job-name with --comment, prints its id.
# Files in $FAKE_SLURM change it: `refuse` (exit 1), `vanish` (the job is never listed), `state`
# (the state the job is queued in, default PENDING),
# `hang-after-queue` and `hang-before-queue` (each holds a count of calls to hang past the timeout).
SBATCH = """#!/bin/sh
d="$FAKE_SLURM"
n=$(cat "$d/next" 2>/dev/null || echo 100)
echo $((n + 1)) > "$d/next"
for a in "$@"; do printf '%s\\n' "$a" >> "$d/sbatch.log"; done
echo "--" >> "$d/sbatch.log"
[ -f "$d/refuse" ] && { echo "sbatch: error: invalid partition specified" >&2; exit 1; }
take() {
  [ -f "$d/$1" ] || return 1
  left=$(cat "$d/$1")
  [ "$left" -gt 0 ] || return 1
  echo $((left - 1)) > "$d/$1"
  return 0
}
take hang-before-queue && { echo $$ >> "$d/hung"; exec sleep 300; }
name=""; comment=""
for a in "$@"; do
  case "$a" in
    --job-name=*) name="${a#--job-name=}";;
    --comment=*) comment="${a#--comment=}";;
  esac
done
mkdir -p "$d/jobs"
state=$(cat "$d/state" 2>/dev/null || echo PENDING)
[ -f "$d/vanish" ] || echo "$state|$name|$comment" > "$d/jobs/$n"
take hang-after-queue && { echo $$ >> "$d/hung"; exec sleep 300; }
echo "$n"
"""
# squeue: one row per queued job, `id|state|name|comment`, filtered by --name=; an ended job is
# listed only with --states=all, as Slurm's default lists pending and running jobs; logs its
# arguments.
SQUEUE = """#!/bin/sh
d="$FAKE_SLURM"
echo "$@" >> "$d/squeue.log"
[ -f "$d/squeue-fails" ] && { echo "slurm_load_jobs error: Socket timed out" >&2; exit 1; }
want=""; all=""
for a in "$@"; do
  case "$a" in --name=*) want="${a#--name=}";; --states=all) all=1;; esac
done
for j in "$d"/jobs/*; do
  [ -f "$j" ] || continue
  IFS='|' read -r state name comment < "$j"
  [ -n "$want" ] && [ "$name" != "$want" ] && continue
  case "$state" in PENDING|RUNNING|COMPLETING|REQUEUE_HOLD|REQUEUE_FED) ;; *) [ -n "$all" ] || continue;; esac
  echo "$(basename "$j")|$state|$name|$comment"
done
exit 0
"""
# scancel: removes the jobs from the queue, unless `linger` exists (a job still listed while it
# stops, as during Slurm's KillWait)
SCANCEL = """#!/bin/sh
for j in "$@"; do
  [ -f "$FAKE_SLURM/linger" ] || rm -f "$FAKE_SLURM/jobs/$j"
  echo "$j" >> "$FAKE_SLURM/cancelled"
done
"""


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """Fake ``sbatch``, ``squeue`` and ``scancel`` first on ``PATH``, their state in a directory, and
    a fresh process-wide queue listing; any ``sbatch`` left hanging is killed at the end.
    """
    bin_dir, state = tmp_path / "bin", tmp_path / "slurm"
    bin_dir.mkdir()
    state.mkdir()
    for name, body in (("sbatch", SBATCH), ("squeue", SQUEUE), ("scancel", SCANCEL)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(stat.S_IRWXU)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM", str(state))
    monkeypatch.setattr(slurm, "_SHARED_LISTING", slurm.QueueListing())
    yield state
    hung = _hung(state)
    for pid in hung:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    assert not [
        pid for pid in hung if _running(pid)
    ], "a hung fake sbatch outlived its test"


def _hung(state: Path) -> list[int]:
    """The pids of this test's fake sbatch calls that hung (each records its own)."""
    where = state / "hung"
    return [int(p) for p in where.read_text().split()] if where.exists() else []


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a killed child not yet reaped by its parent is a zombie: gone for this purpose
    out = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return bool(out) and not out.startswith("Z")


def _submitted(state: Path) -> list[list[str]]:
    log = state / "sbatch.log"
    calls: list[list[str]] = [[]]
    for line in log.read_text().splitlines() if log.exists() else []:
        calls.append([]) if line == "--" else calls[-1].append(line)
    return [c for c in calls if c]


def _jobs(state: Path) -> list[str]:
    where = state / "jobs"
    return sorted(p.name for p in where.iterdir()) if where.exists() else []


def _squeues(state: Path) -> list[str]:
    log = state / "squeue.log"
    return log.read_text().splitlines() if log.exists() else []


class FakeClock:
    """A monotonic clock that moves only when told, and a ``sleep`` that records and moves it."""

    def __init__(self, now: float = 1000.0):
        self.now = now
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _cls(name="cpu", **kwargs) -> WorkerClass:
    fields = dict(name=name, device="cpu", cpus=4, max_workers=32)
    fields.update(kwargs)
    return WorkerClass(**fields)


GPU = dict(
    device="cuda",
    gpus=2,
    gpu_type="h100",
    cpus=8,
    mem="64G",
    time=7200,
    substrate_args={"partition": "gpuq", "account": "lab"},
)


def _substrate(
    tmp_path, *, epoch=1, pool_id=POOL_ID, clock=None, root=None, classes=(), **kwargs
) -> SlurmSubstrate:
    clock = clock or FakeClock()
    kwargs.setdefault("attribute_delay", 60.0)
    kwargs.setdefault("submit_timeout", 5.0)
    kwargs.setdefault("max_age", 0.0)
    kwargs.setdefault("listing", slurm.QueueListing(clock=clock))
    sub = SlurmSubstrate(clock=clock, sleep=clock.sleep, **kwargs)
    root = root or tmp_path / "pool"
    (root / "private").mkdir(parents=True, exist_ok=True, mode=0o700)
    sub.attach(
        Attachment(
            root,
            root / "private" / "settings.json",
            pool_id=pool_id,
            epoch=epoch,
            classes=tuple(classes),
        )
    )
    return sub


# -- the sbatch line of a class ------------------------------------------------------------------


def test_class_to_sbatch_args(tmp_path, fake_slurm):
    """One ``sbatch`` per worker: typed ``--gres``, CPUs, memory, time and the class's own
    arguments, a scoped name, the class as comment, and ``--wrap`` running the worker command.
    """
    sub = _substrate(tmp_path, python="/opt/env/bin/python")
    handles = sub.start(_cls("gpu", **GPU), 2)
    assert [h.job_id for h in handles] == ["100", "101"]
    first, second = _submitted(fake_slurm)
    assert first[0] == "--parsable"
    assert {
        "--gres=gpu:h100:2",
        "--cpus-per-task=8",
        "--mem=65536M",
        "--time=120",
        "--partition=gpuq",
        "--account=lab",
        "--comment=gpu",
        "--no-requeue",
    } <= set(first)
    assert not any(a.startswith("--qos") for a in first)
    assert first[-2] == "--wrap"
    token = handles[0].token
    assert len(token) == 32 and int(token, 16) >= 0
    root = tmp_path / "pool"
    assert first[-1] == (
        f"/opt/env/bin/python -m pydra.workers.pool.worker --root {root} --class gpu "
        f"--settings {root / 'private' / 'settings.json'} --substrate --token {token}"
    )
    assert handles[0].token != handles[1].token and second[-1] != first[-1]
    assert [h.worker_class for h in handles] == ["gpu", "gpu"]
    assert [h.cpus for h in handles] == [8, 8]


def test_untyped_and_cpu_classes(tmp_path, fake_slurm):
    """A cuda class with no type asks for ``gpu:<n>`` (at least one); a cpu class for no GPU and
    leaves unset sizes to the site's defaults."""
    sub = _substrate(tmp_path)
    sub.start(WorkerClass(name="any-gpu", device="cuda"), 1)
    sub.start(WorkerClass(name="small", device="cpu"), 1)
    gpu, cpu = _submitted(fake_slurm)
    assert "--gres=gpu:1" in gpu
    assert not any(
        a.startswith(("--gres", "--cpus-per-task", "--mem", "--time")) for a in cpu
    )


def test_providers_and_codec_reach_the_worker_command(tmp_path, fake_slurm):
    sub = SlurmSubstrate(max_age=0.0)
    root = tmp_path / "pool"
    sub.attach(
        Attachment(
            root,
            root / "s.json",
            providers=("m:P", "n:Q"),
            codec="c:Codec",
            pool_id=POOL_ID,
            epoch=1,
        )
    )
    sub.start(_cls(), 1)
    [args] = _submitted(fake_slurm)
    assert args[-1].endswith("--provider m:P --provider n:Q --codec c:Codec")


@pytest.mark.parametrize(
    "key",
    [
        "qos",
        "--qos",
        "gres",
        "output",
        "job-name",
        "wrap",
        "clusters",
        "M",
        "gpus-per-node",
        "gpus-per-task",
        "tres-per-task",
        "cpus-per-gpu",
        "mem-per-gpu",
        "requeue",
        "no-requeue",
    ],
)
def test_qos_and_substrate_owned_options_refused(tmp_path, fake_slurm, key):
    """``--qos`` is never passed, and options the substrate sets are not the class's: a class naming
    one cannot start (placement skips it) and starting it submits nothing."""
    sub = _substrate(tmp_path)
    cls = _cls(substrate_args={key: "x"})
    why = sub.can_start(cls)
    assert isinstance(why, str) and key.lstrip("-") in why
    with pytest.raises(ValueError, match=key.lstrip("-")):
        sub.start(cls, 1)
    assert _submitted(fake_slurm) == []


def test_start_before_attach_refused(tmp_path, fake_slurm):
    with pytest.raises(RuntimeError, match="attach"):
        SlurmSubstrate().start(_cls(), 1)


# -- the per-user CPU cap ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cap", "asked", "started"),
    [(10, 5, 2), (None, 5, 5), (100, 5, 5), (3, 1, 0)],
    ids=["cap", "no-cap", "cap-above", "cap-below-one"],
)
def test_per_user_cpu_cap(tmp_path, fake_slurm, cap, asked, started):
    """Workers start while their CPUs (``cls.cpus`` each) fit under ``per_user_cpu_cap``; with the
    default (None) nothing caps them; none start while the live ones fill the cap."""
    sub = _substrate(tmp_path, per_user_cpu_cap=cap)
    assert len(sub.start(_cls(), asked)) == started
    assert len(_submitted(fake_slurm)) == started
    if started < asked:
        assert sub.start(_cls(), asked) == []
        assert "cap" in sub.can_start(_cls())
    assert SlurmSubstrate().per_user_cpu_cap is None


def test_cap_counts_jobs_listed_and_just_submitted(tmp_path, fake_slurm):
    """The cap counts the live jobs of the last listing and those submitted since it was taken."""
    clock = FakeClock()
    sub = _substrate(tmp_path, per_user_cpu_cap=8, clock=clock, max_age=30.0)
    assert sub.live() == []  # the listing every later call this cycle reuses
    assert len(sub.start(_cls(), 1)) == 1
    assert len(sub.start(_cls(), 1)) == 1
    assert sub.can_start(_cls()) != True  # noqa: E712
    assert len(_squeues(fake_slurm)) == 1


# -- names, listing and stopping -----------------------------------------------------------------


def test_job_names_scoped_to_pool_epoch_and_token(tmp_path, fake_slurm):
    """A job is named ``pydra-<pool id>-e<epoch>-<token>``; ``live`` lists this pool's jobs of this
    epoch only, from one ``squeue --user`` (never ``--me``)."""
    mine = _substrate(tmp_path, epoch=2)
    started = mine.start(_cls(), 2)
    other_pool = _substrate(tmp_path, pool_id="ba9876543210", root=tmp_path / "other")
    other_pool.start(_cls(), 1)
    earlier = _substrate(tmp_path, epoch=1)
    earlier.start(_cls(), 1)
    names = [a for c in _submitted(fake_slurm) for a in c if a.startswith("--job-name")]
    assert names[0] == f"--job-name=pydra-{POOL_ID}-e2-{started[0].token}"
    assert names[2].startswith("--job-name=pydra-ba9876543210-e1-")
    assert [h.job_id for h in mine.live()] == [h.job_id for h in started]
    assert mine.live()[0] is started[0]  # the same handle every listing
    uid = os.getuid()
    assert all(
        f"--user={uid}" in line and "--me" not in line for line in _squeues(fake_slurm)
    )


def test_a_job_that_leaves_the_queue_is_gone_with_its_log(tmp_path, fake_slurm):
    """A worker's job leaving the queue drops it from ``live``; its cause names the job and the last
    line of its log."""
    clock = FakeClock()
    sub = _substrate(tmp_path, clock=clock)
    [h] = sub.start(_cls(), 1)
    clock.advance(1.0)
    assert sub.live() == [h] and h.cause is None
    h.log.write_text("starting\nRuntimeError: no GPU\n")
    (fake_slurm / "jobs" / h.job_id).unlink()
    assert sub.live() == []
    assert h.job_id in h.cause and "RuntimeError: no GPU" in h.cause


def test_a_job_submitted_after_the_listing_is_live(tmp_path, fake_slurm):
    """A cached listing taken before a submission returned does not count the job as gone."""
    clock = FakeClock()
    sub = _substrate(tmp_path, clock=clock, max_age=30.0)
    assert sub.live() == []
    clock.advance(1.0)
    [h] = sub.start(_cls(), 1)
    (
        fake_slurm / "jobs" / h.job_id
    ).unlink()  # already ended: the next fresh listing shows it
    assert sub.live() == [h]
    clock.advance(31.0)
    assert sub.live() == []


def test_stop_cancels_the_jobs(tmp_path, fake_slurm):
    sub = _substrate(tmp_path)
    handles = sub.start(_cls(), 3)
    sub.stop(handles[:2])
    assert (fake_slurm / "cancelled").read_text().split() == [
        h.job_id for h in handles[:2]
    ]
    assert sub.live() == [handles[2]]
    sub.stop([])
    assert (fake_slurm / "cancelled").read_text().split() == [
        h.job_id for h in handles[:2]
    ]


def test_a_stopped_job_still_listed_is_not_adopted_again(tmp_path, fake_slurm):
    """A job ``stop`` cancelled that Slurm still lists while it stops is neither live nor adopted
    as an unrecorded worker; once it leaves the queue it is forgotten."""
    sub = _substrate(tmp_path)
    [h] = sub.start(_cls(), 1)
    (fake_slurm / "linger").touch()
    sub.stop([h])
    assert sub.live() == []
    assert sub.live() == []
    assert sub.handles == [h]
    (fake_slurm / "jobs" / h.job_id).unlink()
    assert sub.live() == [] and sub.handles == [h]


def test_an_unknown_job_of_this_epoch_is_listed_as_surplus(tmp_path, fake_slurm):
    """A job of this pool and epoch the substrate did not record (a submission that landed late)
    is listed, so scaling sees it as a surplus and stops it."""
    sub = _substrate(tmp_path, epoch=3)
    (fake_slurm / "jobs").mkdir()
    token = "f" * 32
    (fake_slurm / "jobs" / "77").write_text(f"PENDING|pydra-{POOL_ID}-e3-{token}|cpu\n")
    [late] = sub.live()
    assert (late.job_id, late.token, late.worker_class) == ("77", token, "cpu")
    assert sub.live() == [late]


def test_logs_under_the_pools_private_directory(tmp_path, fake_slurm):
    """A worker's output goes to ``<pool>/private/logs/<token>.log``, made owner-only before the
    first submission."""
    sub = _substrate(tmp_path)
    [h] = sub.start(_cls(), 1)
    [args] = _submitted(fake_slurm)
    logs = tmp_path / "pool" / "private" / "logs"
    assert f"--output={logs / (h.token + '.log')}" in args and h.log == logs / (
        h.token + ".log"
    )
    assert logs.stat().st_mode & 0o777 == 0o700


# -- failed submissions --------------------------------------------------------------------------


def test_a_refused_submission_raises_its_message(tmp_path, fake_slurm):
    (fake_slurm / "refuse").touch()
    sub = _substrate(tmp_path)
    with pytest.raises(SubstrateError, match="invalid partition"):
        sub.start(_cls(), 1)
    assert sub.live() == []


def test_an_sbatch_that_cannot_run_is_a_substrate_error(
    tmp_path, fake_slurm, monkeypatch
):
    (tmp_path / "bin" / "sbatch").chmod(stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    with pytest.raises(SubstrateError, match="could not run"):
        _substrate(tmp_path).start(_cls(), 1)


def test_a_start_aborted_by_a_refusal_keeps_the_jobs_it_submitted(tmp_path, fake_slurm):
    """The workers submitted before a refused ``sbatch`` are recorded and carried by the error."""
    sub = _substrate(tmp_path)
    real = sub._sbatch
    calls = []

    def second_refused(argv):
        calls.append(argv)
        if len(calls) == 2:
            (fake_slurm / "refuse").touch()
        return real(argv)

    sub._sbatch = second_refused
    with pytest.raises(SubstrateError) as failed:
        sub.start(_cls(), 3)
    [first] = failed.value.started
    assert sub.live() == [first]


def test_timed_out_submission_that_landed_not_doubled(tmp_path, fake_slurm):
    """An ``sbatch`` that queues its job and then hangs past the timeout is found by its name and
    never submitted again; nothing waits for ``attribute_delay``."""
    (fake_slurm / "hang-after-queue").write_text("1")
    clock = FakeClock()
    sub = _substrate(tmp_path, clock=clock, submit_timeout=1.0)
    [h] = sub.start(_cls(), 1)
    assert _jobs(fake_slurm) == [h.job_id] and len(_submitted(fake_slurm)) == 1
    assert any(
        f"--name=pydra-{POOL_ID}-e1-{h.token}" in q for q in _squeues(fake_slurm)
    )
    assert clock.slept == []
    assert sub.live() == [h]


def test_timed_out_submission_not_found_resubmitted_once_after_attribute_delay(
    tmp_path, fake_slurm
):
    """A timed-out ``sbatch`` whose name is not found, looked up again after ``attribute_delay``,
    is submitted once more, under a new token; a job the first lands later is a surplus of this
    epoch that ``live`` lists."""
    (fake_slurm / "hang-before-queue").write_text("1")
    clock = FakeClock()
    sub = _substrate(tmp_path, clock=clock, submit_timeout=1.0, attribute_delay=60.0)
    [h] = sub.start(_cls(), 1)
    first, second = _submitted(fake_slurm)
    assert clock.slept == [60.0]
    lookups = [q for q in _squeues(fake_slurm) if "--name=" in q]
    assert len(lookups) == 2 and lookups[0] == lookups[1]
    assert f"--job-name=pydra-{POOL_ID}-e1-{h.token}" in second
    assert f"--job-name=pydra-{POOL_ID}-e1-{h.token}" not in first
    late = next(a for a in first if a.startswith("--job-name=")).split("=", 1)[1]
    (fake_slurm / "jobs" / "999").write_text(f"PENDING|{late}|cpu\n")
    assert sorted(x.job_id for x in sub.live()) == sorted([h.job_id, "999"])


def test_timed_out_twice_raises(tmp_path, fake_slurm):
    """A resubmission that times out too, and is not found, is a substrate error: two ``sbatch``
    calls, never a third."""
    (fake_slurm / "hang-before-queue").write_text("2")
    clock = FakeClock()
    sub = _substrate(tmp_path, clock=clock, submit_timeout=1.0)
    with pytest.raises(SubstrateError, match="timed out"):
        sub.start(_cls(), 1)
    assert len(_submitted(fake_slurm)) == 2


def test_a_timed_out_submission_that_already_ended_is_found(tmp_path, fake_slurm):
    """The lookup of a timed-out submission sees jobs in every state: one that already failed is
    the worker (a failed start), never submitted again."""
    (fake_slurm / "hang-after-queue").write_text("1")
    (fake_slurm / "state").write_text("FAILED")
    clock = FakeClock()
    sub = _substrate(tmp_path, clock=clock, submit_timeout=1.0)
    [h] = sub.start(_cls(), 1)
    assert len(_submitted(fake_slurm)) == 1 and _jobs(fake_slurm) == [h.job_id]
    clock.advance(1.0)
    assert sub.live() == [] and h.job_id in h.cause


def test_a_failed_lookup_is_not_read_as_not_found(tmp_path, fake_slurm):
    """When the lookup of a timed-out submission fails, its outcome is unknown: nothing is
    submitted again."""
    (fake_slurm / "hang-before-queue").write_text("1")
    (fake_slurm / "squeue-fails").touch()
    sub = _substrate(tmp_path, submit_timeout=1.0)
    with pytest.raises(SubstrateError, match="lookup"):
        sub.start(_cls(), 1)
    assert len(_submitted(fake_slurm)) == 1


# -- one squeue per cycle ------------------------------------------------------------------------


def test_squeue_shared_with_slurm_worker(tmp_path, fake_slurm, monkeypatch):
    """The substrate and a `SlurmWorker` in one process share one ``squeue --user`` per poll cycle:
    whichever polls first in a cycle lists the queue, the other reads that listing."""
    clock = FakeClock()
    monkeypatch.setattr(slurm, "_SHARED_LISTING", slurm.QueueListing(clock=clock))
    sub = _substrate(tmp_path, clock=clock, max_age=5.0, listing=None)
    worker = slurm.SlurmWorker(poll_delay=5)
    [h] = sub.start(_cls(), 1)

    async def worker_poll():
        return await worker._squeue_all()

    assert sub.live() == [h]
    assert set(asyncio.run(worker_poll()).rows) == {h.job_id}
    assert len(_squeues(fake_slurm)) == 1
    clock.advance(6.0)
    assert set(asyncio.run(worker_poll()).rows) == {h.job_id}
    assert sub.live() == [h] and sub.can_start(_cls()) is True
    assert len(_squeues(fake_slurm)) == 2


def test_slurm_worker_never_reads_a_listing_older_than_its_job(
    tmp_path, fake_slurm, monkeypatch
):
    """A shared listing taken before a job was submitted does not say the job left the queue: the
    worker waits for a newer listing rather than asking ``sacct`` too early."""
    clock = FakeClock()
    listing = slurm.QueueListing(clock=clock)
    monkeypatch.setattr(slurm, "_SHARED_LISTING", listing)
    listing.read(
        0.0, 5.0
    )  # another reader's listing of the queue before the submission
    clock.advance(1.0)
    job_id = subprocess.run(
        ["sbatch", "--parsable", "--job-name=x", "--wrap", "true"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    worker = slurm.SlurmWorker(poll_delay=5)

    async def poll():
        return await worker._poll_job(job_id, since=clock())

    clock.advance(1.0)  # the old listing is still within the worker's poll interval
    assert asyncio.run(poll()) is False
    assert len(_squeues(fake_slurm)) == 1


class ScriptedQueue:
    """Stands in for `SlurmWorker`'s queue listing and ``sacct``/``scontrol``: each listing is the
    next scripted ``(rows, taken)``, and every call is recorded in order."""

    def __init__(self, clock, listings, outcomes):
        self.clock = clock
        self.listings = list(listings)
        self.outcomes = list(outcomes)
        self.events: list[str] = []

    async def snapshot(self):
        rows, taken = self.listings.pop(0)
        self.events.append(f"listing@{taken}")
        return slurm.QueueSnapshot(rows, taken)

    async def verify(self, jobid):
        self.events.append("sacct")
        return self.outcomes.pop(0)

    async def command(self, *cmd, hide_display=False, strip=False):
        assert cmd[:2] == ("scontrol", "requeue"), cmd
        self.events.append(f"scontrol@{self.clock()}")
        self.clock.advance(
            1.0
        )  # the requeue takes a second; listings may be taken meanwhile
        return 0, "", ""


def _scripted_worker(monkeypatch, clock, listings, outcomes):
    queue = ScriptedQueue(clock, listings, outcomes)
    monkeypatch.setattr(slurm, "_SHARED_LISTING", slurm.QueueListing(clock=clock))
    monkeypatch.setattr(slurm.base, "read_and_display_async", queue.command)
    monkeypatch.setattr(
        slurm.SlurmWorker, "_queue_snapshot", lambda self: queue.snapshot()
    )
    monkeypatch.setattr(
        slurm.SlurmWorker, "_verify_exit_code", lambda self, jobid: queue.verify(jobid)
    )
    return slurm.SlurmWorker(poll_delay=0), queue


def _run_wait(worker, tmp_path, jobid="101"):
    job = types.SimpleNamespace(lockfile=tmp_path / "job.lock")

    async def go():
        return await asyncio.wait_for(worker._wait(job, jobid, "err"), BOUND)

    return asyncio.run(go())


def test_wait_ignores_a_listing_taken_before_its_job(tmp_path, monkeypatch):
    """`_wait` reads a listing taken before the job was submitted as no news: ``sacct`` is asked
    only after a listing taken since shows the job gone."""
    clock = FakeClock(1000.0)
    worker, queue = _scripted_worker(
        monkeypatch, clock, [({}, 999.0), ({}, 1000.0)], [True]
    )
    assert _run_wait(worker, tmp_path) is True
    assert queue.events == ["listing@999.0", "listing@1000.0", "sacct"]


def test_wait_ignores_a_listing_taken_during_its_requeue(tmp_path, monkeypatch):
    """After a requeue, a listing taken while ``scontrol requeue`` was in flight is no news: the job
    is not read as gone and requeued twice."""
    clock = FakeClock(1000.0)
    listings = [
        ({}, 1000.0),  # the job ended: sacct says NODE_FAIL, so it is requeued at 1000
        ({}, 1000.5),  # taken while scontrol ran (1000 to 1001): the job is not listed
        ({"101": ("PENDING", "", "")}, 1001.0),
        ({}, 1002.0),
    ]
    worker, queue = _scripted_worker(monkeypatch, clock, listings, ["NODE_FAIL", True])
    assert _run_wait(worker, tmp_path) is True
    assert queue.events == [
        "listing@1000.0",
        "sacct",
        "scontrol@1000.0",
        "listing@1000.5",
        "listing@1001.0",
        "listing@1002.0",
        "sacct",
    ]


SLOW_SQUEUE = """#!/bin/sh
d="$FAKE_SLURM"
touch "$d/squeue-started"
i=0
while [ ! -f "$d/squeue-go" ] && [ $i -lt 600 ]; do sleep 0.05; i=$((i + 1)); done
exit 0
"""


def test_event_loop_not_blocked_by_a_listing_in_flight(tmp_path, fake_slurm):
    """While another thread's ``squeue`` runs, the event loop's `aread` returns the listing it
    already has instead of waiting for that ``squeue`` to finish."""
    (tmp_path / "bin" / "squeue").write_text(SLOW_SQUEUE)
    listing = slurm.QueueListing()
    listing._keep(slurm.QueueSnapshot({"7": ("RUNNING", "n", "c")}, listing.clock()))
    finished = threading.Event()

    def other_reader():
        try:
            listing.read(0.0, 10.0)
        except RuntimeError:
            pass
        finished.set()

    thread = threading.Thread(target=other_reader)
    thread.start()
    try:
        deadline = time.monotonic() + BOUND
        while not (fake_slurm / "squeue-started").exists():
            assert (
                time.monotonic() < deadline
            ), "the other reader's squeue never started"
            time.sleep(0.01)

        async def loop_read():
            return await listing.aread(max_age=float("inf"))

        snapshot = asyncio.run(loop_read())
        assert set(snapshot.rows) == {"7"}
        assert not finished.is_set(), "aread waited for the other reader's squeue"
    finally:
        (fake_slurm / "squeue-go").touch()
        thread.join(BOUND)


# -- through the pool ----------------------------------------------------------------------------

TASKS = """
from pydra.compose import python, workflow
from pydra.utils.resources import ComputeResources


@python.define
def Echo(x: int) -> int:
    return x


@workflow.define
def Many(xs: list) -> list:
    return workflow.add(Echo().split(x=xs), name="E").out


@workflow.define
def GpuMany(xs: list) -> list:
    gpu = ComputeResources(gpus=1, gpu_type="h100")
    return workflow.add(Echo().split(x=xs), name="E", resources=gpu).out
"""


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    """The task module, importable here and in workers started from here."""
    where = tmp_path_factory.mktemp("pool_slurm_tasks")
    (where / "_pydra_pool_slurm_tasks.py").write_text(TASKS)
    sys.path.insert(0, str(where))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(where), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    import _pydra_pool_slurm_tasks

    return _pydra_pool_slurm_tasks


@pytest.fixture
def pools():
    """Every pool a test makes is closed (which cancels the jobs its substrate lists)."""
    made = []
    yield made
    for pool in made:
        pool.close()


def _pool(tmp_path, pools, substrate, sub, **kwargs):
    from pydra.workers.pool.pool import WarmPool

    kwargs.setdefault("settings", PoolSettings(lease=LEASE, poll_s=0.05))
    classes = kwargs.pop("classes", [_cls(max_workers=2)])
    pool = WarmPool(
        tmp_path / "pool",
        classes,
        substrate,
        store=sub.store,
        probe=False,
        **kwargs,
    )
    pools.append(pool)
    pool.bind(lambda job, outcome: None)
    return pool


def _queue_round(pool, sub, tasks, xs):
    runner = GraphRunner(
        tasks.Many(xs=xs), sub.store, on_error="raise_at_end", submitter=sub
    )
    pool._queue_jobs(runner.ready().dispatch, handle=None)
    pool._write()


def test_pool_names_jobs_by_its_epoch_and_reads_its_settings(
    tmp_path, fake_slurm, tasks, pools
):
    """A pool attaches its id, its coordinator epoch and its settings: jobs are named by them,
    ``attribute_delay`` is the pool's lease setting and listings live for one poll."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    lease = LeaseSettings(
        stale_after=60.0, renew_fraction=0.4, attribute_delay=7.0, timeout=70.0
    )
    substrate = SlurmSubstrate()
    pool = _pool(
        tmp_path, pools, substrate, sub, settings=PoolSettings(lease=lease, poll_s=3.0)
    )
    assert substrate.attribute_delay == 7.0 and substrate.max_age == 3.0
    _queue_round(pool, sub, tasks, [1])
    pool._scale(100.0)
    [args] = _submitted(fake_slurm)
    name = next(a for a in args if a.startswith("--job-name="))
    assert name.startswith(f"--job-name=pydra-{pool.pool.pool_id}-e{pool.epoch}-")
    pool.close()
    assert len((fake_slurm / "cancelled").read_text().split()) == 1


def test_failed_starts_back_off_and_stop(tmp_path, fake_slurm, tasks, pools):
    """Jobs that leave the queue without a ready mark are failed starts: the next start of the class
    waits ``poll_s``, doubling, on the follower's clock, and after ``start_failures`` the class is
    broken and starts no more."""
    (fake_slurm / "vanish").touch()
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    clock = FakeClock()
    substrate = SlurmSubstrate(
        clock=clock, max_age=0.0, listing=slurm.QueueListing(clock=clock)
    )
    pool = _pool(
        tmp_path,
        pools,
        substrate,
        sub,
        faults=FaultBudgets(start_failures=3),
        settings=PoolSettings(lease=LEASE, poll_s=0.5),  # exact in binary
    )
    _queue_round(pool, sub, tasks, [1])
    poll = pool.settings.poll_s
    starts = []
    for step in (0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 20, 40):
        clock.now = 1000.0 + step * poll
        pool._scale(100.0 + step * poll)
        starts.append(len(_submitted(fake_slurm)))
    # a start, seen gone by the next listing; poll_s after that failure the second start; seen
    # gone, 2 x poll_s later the third; its failure breaks the class
    assert starts == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    assert "cpu" in pool._broken and "3 consecutive" in pool._broken["cpu"]
    assert "left the Slurm queue" in pool._broken["cpu"]


def test_refused_submissions_are_failed_starts(tmp_path, fake_slurm, tasks, pools):
    """An ``sbatch`` the scheduler refuses is a failed start of its class, with the scheduler's
    message as the cause: it backs off and breaks the class, never retried every poll.
    """
    (fake_slurm / "refuse").touch()
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(tmp_path, pools, substrate, sub, faults=FaultBudgets(start_failures=2))
    _queue_round(pool, sub, tasks, [1])
    poll = pool.settings.poll_s
    for step in (0, 0, 0.5, 1, 1, 3, 5):
        pool._scale(100.0 + step * poll)
    assert len(_submitted(fake_slurm)) == 2
    assert "invalid partition" in pool._broken["cpu"]


def test_a_failed_listing_is_not_read_as_failed_starts(
    tmp_path, fake_slurm, tasks, pools
):
    """A poll whose ``squeue`` fails skips scaling: workers started are not counted gone."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(tmp_path, pools, substrate, sub, faults=FaultBudgets(start_failures=1))
    _queue_round(pool, sub, tasks, [1])
    pool._scale(100.0)
    assert len(_submitted(fake_slurm)) == 1
    (fake_slurm / "squeue-fails").touch()
    for step in range(5):
        pool._scale(101.0 + step)
    assert pool._failures["cpu"] == 0 and "cpu" not in pool._broken
    pool.close()  # with squeue down, close cancels the workers this coordinator started
    assert (fake_slurm / "cancelled").read_text().split() == ["100"]


def test_workers_that_never_start_hold_their_jobs(tmp_path, fake_slurm, tasks, pools):
    """A class whose jobs never reach a ready mark is broken after ``start_failures`` and its jobs
    are ``NotRun(held=True)``, naming the class, the count and the latest cause."""
    (fake_slurm / "vanish").touch()
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    clock = FakeClock()
    substrate = SlurmSubstrate(
        clock=clock, max_age=0.0, listing=slurm.QueueListing(clock=clock)
    )
    pool = _pool(tmp_path, pools, substrate, sub, faults=FaultBudgets(start_failures=2))
    reported = []
    pool.bind(lambda job, outcome: reported.append(outcome))
    _queue_round(pool, sub, tasks, [1, 2])
    poll = pool.settings.poll_s
    for step in range(12):
        clock.now = 1000.0 + step * poll
        pool._poll(100.0 + step * poll)
    assert len(reported) == 2
    for outcome in reported:
        assert isinstance(outcome, NotRun) and outcome.held
        assert "'cpu'" in outcome.reason and "2 consecutive" in outcome.reason
        assert "left the Slurm queue" in outcome.reason


def test_a_lasting_listing_failure_holds_the_run(tmp_path, fake_slurm, tasks, pools):
    """A ``squeue`` that keeps failing is skipped poll by poll, and once ``listing_failures`` polls
    in a row have failed and ``stale_after`` has passed since the last good listing, a class with
    work and no ready worker is broken: its jobs are ``NotRun(held=True)`` naming the error, not
    left waiting for ever."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(
        tmp_path, pools, substrate, sub, faults=FaultBudgets(listing_failures=3)
    )
    reported = []
    pool.bind(lambda job, outcome: reported.append(outcome))
    _queue_round(pool, sub, tasks, [1])
    (
        fake_slurm / "vanish"
    ).touch()  # the worker started never reaches the queue or a ready mark
    pool._poll(100.0)  # the last good listing: one start
    (fake_slurm / "squeue-fails").touch()
    stale = LEASE.stale_after  # 4 s
    for now in (
        100.5,
        101.0,
        101.5,
        102.0,
        103.0,
    ):  # the count is reached; the floor is not
        pool._poll(now)
    assert reported == [] and "cpu" not in pool._broken
    pool._poll(100.0 + stale)
    [outcome] = reported
    assert isinstance(outcome, NotRun) and outcome.held
    assert "times in a row" in outcome.reason and "Socket timed out" in outcome.reason
    assert len(_submitted(fake_slurm)) == 1


def _worker_lease(root: Path, token: str):
    """A worker's lease as its beater leaves it (``workers/<token>/lease/<hex>/beat``), and a
    function that renews it: each call moves the beat's mtime one second on."""
    beat = root / "workers" / token / "lease" / ("b" * 32) / "beat"
    beat.parent.mkdir(parents=True, exist_ok=True)
    beat.write_text("")
    when = {"t": time.time() + 1000.0}  # newer than the directories just made

    def renew():
        when["t"] += 1.0
        os.utime(beat, (when["t"], when["t"]))

    renew()
    return renew


def test_a_class_that_settled_once_then_stalled_breaks(
    tmp_path, fake_slurm, tasks, pools
):
    """A settle exempts its class for ``stale_after`` only: a class that settled one job early in
    the outage and then stalled breaks once the window, the count and the floor have passed.
    """
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(
        tmp_path, pools, substrate, sub, faults=FaultBudgets(listing_failures=2)
    )
    _queue_round(pool, sub, tasks, [1, 2, 3])
    (fake_slurm / "vanish").touch()
    pool._poll(100.0)
    (fake_slurm / "squeue-fails").touch()
    [digest] = [t.digest for t in pool._tickets.values()]
    settled = pool.pool.settled
    pool.pool.settled = lambda queue: {**settled(queue), digest: 1}  # one job, at 101
    pool.pool.outcomes = lambda queue, digest, expect, within: {}
    for now in (101.0, 102.0, 103.0, 104.0):
        pool._poll(now)
    assert "cpu" not in pool._broken  # within stale_after of the settle at 101
    for now in (105.0, 106.0):
        pool._poll(now)
    assert "cpu" in pool._broken


def test_a_long_gap_alone_does_not_break_a_class(tmp_path, fake_slurm, tasks, pools):
    """The time floor is not enough by itself: fewer than ``listing_failures`` failed polls, however
    far apart, break nothing."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(
        tmp_path, pools, substrate, sub, faults=FaultBudgets(listing_failures=3)
    )
    _queue_round(pool, sub, tasks, [1])
    (fake_slurm / "vanish").touch()
    pool._poll(100.0)
    (fake_slurm / "squeue-fails").touch()
    for now in (200.0, 300.0):  # far past stale_after, two failures of three
        pool._poll(now)
    assert "cpu" not in pool._broken
    pool._poll(400.0)
    assert "cpu" in pool._broken


def test_a_worker_ready_during_the_outage_protects_its_class(
    tmp_path, fake_slurm, tasks, pools
):
    """A ready mark a started worker writes while ``squeue`` is down is read from storage, so the
    class is not charged for the outage."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(
        tmp_path, pools, substrate, sub, faults=FaultBudgets(listing_failures=2)
    )
    _queue_round(pool, sub, tasks, [1])
    pool._scale(100.0)
    [token] = list(pool._started)
    (fake_slurm / "squeue-fails").touch()
    mark = tmp_path / "pool" / "workers" / token / "ready.json"
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(json.dumps({"token": token}))
    renew = _worker_lease(tmp_path / "pool", token)
    for step in range(10):
        renew()
        pool._poll(101.0 + step)
    assert "cpu" not in pool._broken


def test_a_listing_outage_does_not_break_a_class_with_a_ready_worker(
    tmp_path, fake_slurm, tasks, pools
):
    """While ``squeue`` fails, a class whose worker is ready is making progress through storage:
    the failures are not counted against it, however long they last."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(tmp_path, pools, substrate, sub)
    reported = []
    pool.bind(lambda job, outcome: reported.append(outcome))
    _queue_round(pool, sub, tasks, [1, 2])
    pool._scale(100.0)
    [token] = list(pool._started)
    pool._ready.add(token)  # the worker is up and working
    renew = _worker_lease(tmp_path / "pool", token)
    (fake_slurm / "squeue-fails").touch()
    # 7 is the default budget; 30 polls span far past stale_after
    for step in range(30):
        renew()  # its beater keeps renewing its lease
        pool._poll(101.0 + step)
    assert "cpu" not in pool._broken and reported == []


def test_a_ready_worker_that_is_gone_no_longer_protects_its_class(
    tmp_path, fake_slurm, tasks, pools
):
    """A worker that marked ready and then ended while ``squeue`` was down (its lease no longer
    renewed) stops exempting its class: once its lease has not moved for ``stale_after``, the
    failures count, and the class breaks at the count and the time floor."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(
        tmp_path, pools, substrate, sub, faults=FaultBudgets(listing_failures=3)
    )
    reported = []
    pool.bind(lambda job, outcome: reported.append(outcome))
    _queue_round(pool, sub, tasks, [1])
    pool._poll(100.0)  # the last good listing: the worker is started
    [token] = list(pool._started)
    pool._ready.add(token)
    _worker_lease(tmp_path / "pool", token)  # beaten once, then never again
    (fake_slurm / "jobs" / pool._started[token].job_id).unlink()  # the job ended
    (fake_slurm / "squeue-fails").touch()
    for now in (101.0, 102.0, 103.0, 104.0, 105.0, 106.0):
        pool._poll(now)
    # the lease, first seen at 101, is fresh until 105; three failures from then on
    assert "cpu" not in pool._broken and reported == []
    pool._poll(107.0)
    [outcome] = reported
    assert outcome.held and "times in a row" in outcome.reason


def test_a_ready_worker_without_a_lease_does_not_protect_its_class(
    tmp_path, fake_slurm, tasks, pools
):
    """A ready worker whose lease is gone (released as it exited) is not live: its class breaks as
    soon as the count and the time floor are reached."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(
        tmp_path, pools, substrate, sub, faults=FaultBudgets(listing_failures=3)
    )
    _queue_round(pool, sub, tasks, [1])
    pool._poll(100.0)
    [token] = list(pool._started)
    pool._ready.add(token)  # it marked ready, then exited and released its lease
    (fake_slurm / "squeue-fails").touch()
    for now in (101.0, 102.0, 103.0):
        pool._poll(now)
    assert "cpu" not in pool._broken  # the floor (stale_after = 4 s) is not reached
    pool._poll(104.0)
    assert "cpu" in pool._broken


def test_a_class_that_settles_tickets_is_not_broken_by_a_listing_outage(
    tmp_path, fake_slurm, tasks, pools
):
    """A class one of whose tickets settled since the last good listing is not charged for the
    outage either: a worker is running its work."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(
        tmp_path, pools, substrate, sub, faults=FaultBudgets(listing_failures=2)
    )
    _queue_round(pool, sub, tasks, [1, 2, 3])
    (
        fake_slurm / "vanish"
    ).touch()  # no ready mark: only settled tickets show the progress
    pool._poll(100.0)
    (fake_slurm / "squeue-fails").touch()
    [digest] = [t.digest for t in pool._tickets.values()]
    moved = {"n": 0}
    settled = pool.pool.settled

    def settling(queue):
        found = settled(queue)
        moved["n"] += 1
        return {**found, digest: moved["n"]}  # a worker settles one more job each poll

    pool.pool.settled = settling
    pool.pool.outcomes = lambda queue, digest, expect, within: {}
    for step in range(10):
        pool._poll(101.0 + step)
    assert "cpu" not in pool._broken


def test_a_good_listing_resets_the_listing_failures(tmp_path, fake_slurm, tasks, pools):
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0)
    pool = _pool(
        tmp_path, pools, substrate, sub, faults=FaultBudgets(listing_failures=2)
    )
    _queue_round(pool, sub, tasks, [1])
    for step in range(6):
        failing = fake_slurm / "squeue-fails"
        failing.touch() if step % 2 == 0 else failing.unlink()
        pool._scale(100.0 + step)
    assert "cpu" not in pool._broken


def test_a_transient_listing_failure_at_submit_does_not_move_a_job(
    tmp_path, fake_slurm, tasks, pools
):
    """With a CPU cap, a ``squeue`` failure while a job is placed leaves it on its class: the
    class is not read as unable to start, so the job is neither Unplaceable nor moved.
    """
    from pydra.workers.pool.placement import Placement

    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate(max_age=0.0, per_user_cpu_cap=16)
    pool = _pool(
        tmp_path,
        pools,
        substrate,
        sub,
        classes=[_cls(max_workers=2), _cls("big", cpus=16, max_workers=1)],
    )
    runner = GraphRunner(
        tasks.Many(xs=[1]), sub.store, on_error="raise_at_end", submitter=sub
    )
    jobs = runner.ready().dispatch
    [control] = pool.place(jobs)  # with the queue listed
    assert isinstance(control, Placement)
    (fake_slurm / "squeue-fails").touch()
    [placement] = pool.place(jobs)
    assert isinstance(placement, Placement)
    assert placement.worker_class == control.worker_class


def test_earlier_epoch_jobs_adopted_and_pending_ones_cancelled(tmp_path, fake_slurm):
    """A coordinator lists this pool's jobs of earlier epochs as its own unrecorded workers, so
    they count and can be stopped; at its first listing it cancels those still pending.
    """
    (fake_slurm / "state").write_text("RUNNING")
    first = _substrate(tmp_path, epoch=1)
    running = first.start(_cls("gpu", **GPU), 2)
    (fake_slurm / "state").write_text("PENDING")
    [pending] = first.start(_cls("gpu", **GPU), 1)
    later = _substrate(
        tmp_path, epoch=3, root=tmp_path / "pool", classes=[_cls("gpu", **GPU)]
    )
    other = _substrate(tmp_path, epoch=1, pool_id="ba9876543210", root=tmp_path / "o")
    other.start(_cls(), 1)
    live = later.live()
    assert sorted(h.job_id for h in live) == sorted(h.job_id for h in running)
    assert all(h.worker_class == "gpu" for h in live)
    assert (fake_slurm / "cancelled").read_text().split() == [pending.job_id]
    assert later.live() == live
    assert sum(h.cpus for h in live) == 16  # the class's CPUs, from the pool's classes


@pytest.mark.parametrize("held", ["PENDING", "REQUEUE_HOLD", "REQUEUE_FED"])
def test_earlier_epoch_jobs_held_before_starting_are_cancelled(
    tmp_path, fake_slurm, held
):
    """The sweep cancels every earlier-epoch job that never started, a held or requeue-held one as
    much as a plain pending one; a running one is adopted."""
    (fake_slurm / "state").write_text("RUNNING")
    first = _substrate(tmp_path, epoch=1)
    [running] = first.start(_cls(), 1)
    (fake_slurm / "state").write_text(held)
    [waiting] = first.start(_cls(), 1)
    later = _substrate(tmp_path, epoch=2, root=tmp_path / "pool")
    assert [h.job_id for h in later.live()] == [running.job_id]
    assert (fake_slurm / "cancelled").read_text().split() == [waiting.job_id]


def test_restarted_coordinator_never_exceeds_max_workers(
    tmp_path, fake_slurm, tasks, pools
):
    """After a coordinator restart the new one counts the old epoch's running GPU workers: the
    class never has more than ``max_workers`` jobs in the queue."""
    from pydra.workers.pool.pool import WarmPool

    (fake_slurm / "state").write_text("RUNNING")
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    gpu = _cls("gpu", **{**GPU, "max_workers": 2, "quantum_jobs": 1})

    def coordinator():
        pool = WarmPool(
            tmp_path / "pool",
            [gpu],
            SlurmSubstrate(max_age=0.0),
            store=sub.store,
            probe=False,
            settings=PoolSettings(lease=LEASE, poll_s=0.05, unmeasured_workers=4),
        )
        pools.append(pool)
        pool.bind(lambda job, outcome: None)
        runner = GraphRunner(
            tasks.GpuMany(xs=list(range(6))),
            sub.store,
            on_error="raise_at_end",
            submitter=sub,
        )
        pool._queue_jobs(runner.ready().dispatch, handle=None)
        pool._write()
        return pool

    first = coordinator()
    for step in range(4):
        first._scale(100.0 + step)
    assert len(_jobs(fake_slurm)) == 2
    # the coordinator dies: its lease ends, its workers keep running
    first._closed = True
    first._lease.__exit__(None, None, None)
    first.pool.close()
    second = coordinator()
    assert second.epoch == first.epoch + 1
    for step in range(6):
        second._scale(200.0 + step)
        assert len(_jobs(fake_slurm)) <= 2
    assert len(_submitted(fake_slurm)) == 2


# -- a real scheduler ----------------------------------------------------------------------------


@pytest.mark.slurm
@pytest.mark.skipif(not shutil.which("sbatch"), reason="needs a Slurm scheduler")
def test_slurm_substrate_real(tmp_path, tasks):
    """On a real Slurm cluster a pool starts its worker with one ``sbatch``, the worker runs the
    round and its log is under the pool's private directory; closing the pool cancels it.
    """
    from pydra.engine.drivers import drive
    from pydra.workers.pool.pool import WarmPool

    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    substrate = SlurmSubstrate()
    pool = WarmPool(
        tmp_path / "pool",
        [_cls(cpus=1, max_workers=1)],
        substrate,
        store=sub.store,
        settings=PoolSettings(lease=LEASE, poll_s=0.5, idle_polls=600),
    )
    try:
        runner = GraphRunner(
            tasks.Many(xs=[1, 2, 3]), sub.store, on_error="raise_at_end", submitter=sub
        )
        asyncio.run(
            asyncio.wait_for(
                drive(
                    runner,
                    {"pool": pool},
                    lambda job: "pool",
                    on_error="raise_at_end",
                    audit=sub.audit,
                    max_concurrent=float("inf"),
                ),
                BOUND * 3,
            )
        )
        assert not runner.errors and not runner.not_run
        assert [h.job_id for h in substrate.handles]
        logs = tmp_path / "pool" / "private" / "logs"
        assert [p.name for p in logs.iterdir()] == [
            f"{h.token}.log" for h in substrate.handles
        ]
    finally:
        pool.close()
    deadline = time.monotonic() + BOUND
    while _ours(substrate):
        assert time.monotonic() < deadline, "the pool's Slurm job outlived close"
        time.sleep(1.0)


def _ours(substrate) -> list[str]:
    rows = substrate._listing.read(0.0, 60.0).rows
    return [j for j, (_, name, _) in rows.items() if name.startswith(substrate.prefix)]
