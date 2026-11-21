# Proposal 0005: warm worker pools, resident state and GPU-fit placement

Status: proposed, for maintainer discussion; revision 4. Branch `rfc/warm-worker-pools`, off
`enh/execution-extensions` at `529d2efa`. Line references are to `529d2efa` unless another commit is
named; `enh/execution-extensions` has since moved to `68090364`, which still has `prove()` and no
public `try_take()`. RFC 0002 is #930 at `6a9ececb`, RFC 0003 #931 at `36546b9f`, RFC 0004 #932 at
`91b8795d`; their section numbers are those revisions'. Code that exists only on the integration branch
of the RFC 0002–0004 implementation is cited at `d7b0988f` and marked "integration" (its lease is
`pydra/engine/lease.py`); RFC items are named as the RFCs name them (RFC 0002 R1–R8, RFC 0003 P1–P8,
RFC 0004 I0–I9), proposal 0001's items carry the prefix "0001" (0001 A1, 0001 F1), and the
implementation plan's sub-tasks (P4c, P9, I6a, I6b) are marked "plan task". senselab is cited at
`sensein/senselab` `2.x` @ `4472cfbb`; `rt/` is `packages/senselab-core/src/senselab_core/runtime/`
and `ex/` is `rt/execution/`. Revision 2 answered the architecture review of revision 1; revision 3
answered the verification of revision 2 (protocol gaps N1–N7, table gaps D1–D3); revision 4 answers the
check of revision 3 (I1–I4, M1–M6). §13 maps each item to the text that resolves it.

This proposal replaces proposal 0001's I (`pydra-workers-pool`, 0001 §3.7) and the provider and
batching parts of H (0001 §3.6). It answers RFC 0002 §5's question about run identity: it belongs to
the pool, not to the `Executor` protocol (§5.3). It does not reopen RFC 0002 §3.2's "Not adopted"
list: a warm pool is an `Executor` that takes pydra jobs.

## 1. Motivation

pydra runs a job in one of three ways: in the caller's process (`debug`), in a process of a local
pool (`cf`), or in a fresh process on a cluster (`slurm`, `sge`). Only the first keeps anything
between jobs, and it runs one job at a time on one machine. Everything a job loads before it can do
its work is loaded again for every job:

| Problem | Evidence |
|---|---|
| A cluster job is a process, an import of pydra and its task's libraries, and a load of whatever the task needs, per job | `SlurmWorker._prepare_runscripts` writes one `_job.pklz` per job and a script that calls `load_and_run` on it (`workers/slurm.py:122-131`; `engine/job.py:723`). An array (RFC 0002 R5) starts one such process per element |
| A model load costs more than the jobs that use it | measured through senselab's subprocess worker on one H100 (§9): loads of 8.9 to 26.5 s, warm per-job times of 0.013 to 8.36 s over five units, and a first request after a load of 7.8 to 30 times the steady time |
| A container is started per command | `Docker.execute` builds `docker run ...` for every job (`environments/docker.py:41-42`); `Singularity.execute` runs `singularity exec` per job (`environments/singularity.py:21-23`). No measurement of the cost exists yet; W3's benchmark is the first (§11) |
| A module environment is resolved per command | `Lmod.execute` runs `lmod python load <modules>` and parses its output for every job (`environments/lmod.py:32-41`) |
| `Environment` has no lifetime | `setup` and `teardown` are empty, with a `TODO` (`environments/base.py:28-32`, `:50-51`) |
| A resource request names hardware but not fit | `ComputeResources(cpus, mem, gpus, gpu_type, time, tags)` (`utils/resources.py:37-63`) maps to `sbatch` arguments and routing tags. Nothing says whether a task's working set fits a GPU class, or where to go when it does not |
| A scheduler sees one job per element | a 10⁴-element GPU node is 10⁴ Slurm array tasks, each waiting in the queue, each loading the model. senselab measured 426 s from submission to the first commit of a one-worker GPU run under a busy partition (§9), paid once per worker in a pool and once per element in an array |
| A cluster job that dies leaves nothing to take over | a pydra Slurm job either writes its result or is requeued by Slurm (0001 A1). There is no lease a second process could judge |

**What pydra users gain:**
- A shell or container workflow over many inputs (a FreeSurfer, FSL or ANTs step per subject, a
  container tool per file) runs on a few long-lived workers that keep the container, the Apptainer
  instance or the resolved module environment up, instead of one scheduler job and one container start
  per input.
- A Python workflow that calls a model, a large lookup table or a GPU context loads it once per worker,
  in the worker's process or in a process the worker keeps (an interpreter in a locked environment, a
  server).
- GPU tasks are placed on a class whose memory fits them, from declared figures or from measurements,
  and move to a larger class when a load runs out of memory, instead of failing.
- On Slurm, the scheduler sees one job per worker, typed by GPU (`--gres=gpu:<type>:<n>`), and the
  workers pull work. Per-job scheduler latency, array limits and `MaxSubmitPU` stop scaling with the
  number of elements.
- A worker that dies loses at most the jobs it held; another worker takes them over, under a new
  attempt, once its lease is stale, and the run completes.
- Plain workflows that do not opt in change in nothing (§2.3).

**One motivating user.** senselab 2.x (sensein/senselab#617) built all of this on its side of RFC
0002's seam: pool workers with a claim loop (`ex/pool_worker.py:519-554`), a count-bound LRU of worker
processes (`WorkerSet`, `ex/workers.py:265-320`), its heavy backends running in isolated environments
as a `WorkerClient` subprocess (`ex/workers.py:330-341`, `rt/worker_client.py`), queues named
`<class>#<model identity>` (`ex/pool.py:536-541`), placement by calibrated fit (`rt/fit.py:156-196`),
one `sbatch` per worker with typed `--gres` (`ex/substrates.py:556-579`), and a three-class fault
model (`ex/pool.py:818-857`, its ADR-009). Its owner has decided that this machinery moves into pydra
now, before senselab's first pin of the RFC 0002–0004 stack, and that senselab adopts pydra's design
with no compatibility layer (§10.2). The design below generalises that implementation and its
measurements (§9). No pydra API names senselab, audio or a model family; senselab becomes one client.

## 2. Goals, non-goals, compatibility

### 2.1 Goals

- Long-lived workers that keep **resident state** across jobs: provided objects, in the worker's
  process or behind a process the provider owns (a model, a GPU context, an interpreter in an isolated
  environment, a server), and warm environments (a running container, an Apptainer instance, a
  resolved module environment).
- Routing by **resident key**: a job names the resident state it needs; it is queued with jobs that
  need the same state; a worker prefers the queue whose state it holds.
- **Placement and fit**: a declared request per task, worker classes per substrate, a pluggable fit
  judgement that can use measurements, a ladder of larger classes, and admission that refuses what fits
  nowhere.
- **Backends**: one host with several GPUs, and Slurm with one `sbatch` per worker. Workers pull
  through a protocol on shared storage, consistent with RFC 0003 (results through the store,
  `_committed.json`), with leases, heartbeats and recovery.
- **Faults typed at the source** and mapped onto RFC 0002's outcomes.
- **Identity untouched**: resident state never enters a checksum, a lookup context or an output.

**What the protocol guarantees.** At most one committed result per key (RFC 0003 §5.1's first writer
wins; the claim proof before `publish` only narrows the window in which a lost holder publishes, §4.3);
never two executions in one work directory, because every execution start in a pool, reruns included,
runs under an attempt allocated by an exclusive create and never reused (§5.5); no job left unclaimed while a worker of its
class lives. It does not guarantee exactly-once execution: a job whose holder is lost may run again
under a new attempt (senselab ADR-003, "Not guaranteed: exactly-once execution").

### 2.2 Non-goals

- A general distributed scheduler. Workers pull from directories on a shared filesystem; there is no
  daemon, no message broker and no network service.
- Resident state shared across workers (one inference server consumed by many workers). A provider may
  own a server private to its worker (§3.1); senselab's per-host service registry (`ex/services.py`)
  stays inside its provider for now, and a later RFC moves it into pydra when a second client needs it.
- Pools shared by several users (§5.2, §12).
- Kubernetes and cloud substrates. The substrate protocol (§4.5) admits them; none ships here.

### 2.3 Compatibility

- **A run that does not use `WarmPool` is unchanged**: `import pydra` does not import
  `pydra.workers.pool` (tested by `sys.modules`, W6), no new file is written to a cache, and
  `Submitter`'s defaults, checksums, the store layout and every worker are as today.
- **New optional fields** default to today's behaviour: `ComputeResources.gpu_mem` and
  `ComputeResources.residency` (§3.2), the field attribute `provided=` (§3.1), `Environment.warm()`
  returning `None` (§3.1).
- **A provided field outside a pool** is resolved by an in-process `ResidentSet` under `debug`, so a
  task that declares one runs everywhere (§3.1).
- **Checksums:** a provided field's value is its *spec* (a model id and revision, a path), keyed as
  any input value is. The loaded object is never hashed (§8).

## 3. Declarations

### 3.1 Resident state: provided fields and warm environments

A **resident key** names state a worker can hold between jobs:

```python
@attrs.frozen
class ResidentKey:
    kind: Literal["provider", "environment"]
    name: str          # the provider's name, or the environment's plugin name
    spec: str          # the spec's RFC 0004 §6.2 digest ("json@1:…"), or the environment's resolved reference
    def digest(self) -> str: ...   # sha256 of the canonical JSON of the three fields
```

`spec` is the digest RFC 0004 §6.2 gives the input value, so the identity record and the residency
digest derive from one encoding and cannot drift.

**Providers** (0001 §3.6, made concrete). An input field may be marked provided:

```python
@python.define
def embed(audio: Path, model: Provided[ModelSpec, "hf-model"]) -> np.ndarray: ...
# Provided[T, name] is Annotated[T, ProvidedBy(name)]

class Provider(Protocol):
    name: ClassVar[str]
    def load(self, spec: JSON, device: Device) -> object: ...     # an object or a client of a process it starts
    def close(self, obj: object) -> None: ...                     # stops what load started, and waits
    def footprint(self, spec: JSON) -> Footprint | None: ...      # declared bytes, host and device
    def measure(self, obj: object) -> Footprint | None: ...       # what the loaded state holds now
    def version(self) -> str: ...                                 # the loader's code: binds calibration, recorded in context
```

- The field's value is the spec (a JSON value). `TypeParser` validates and the checksum hashes it as
  `T`, like any input. The task function receives the loaded object (or the client). `field(provided=
  "<name>")` is the long spelling of the same marker.
- Before running a job, the executing process replaces each provided field's value by
  `resident_set.get(ResidentKey("provider", name, digest(spec)))`, which loads on a miss.
- **A provider may own a process.** `load` may start an interpreter in a locked environment, a worker
  speaking a wire protocol, or a server, and return a client to it. Then:
  - `close` stops the process group and waits for every member to be gone, so a restart does not race
    the dying process for GPU memory (senselab R-295);
  - the resident's memory is what `measure` reports (by PID through `nvidia-smi
    --query-compute-apps`, or as the process reports it); the worker reads device-wide memory only as
    the co-tenancy baseline (§4.3);
  - the worker takes its cgroup OOM baseline after each provider start, so a child's OOM kill is
    attributed to the call that ran it (§6).
  This is how senselab's isolated backends stay warm: its `WorkerSet` of `WorkerClient` subprocesses
  becomes one provider, and its `host` kind an in-process one (§10.2). It is also the residency
  primitive for an interpreter in another environment; 0001 G2's `Venv.warm()` later reuses it.
- A `ResidentSet` holds loaded state in LRU order under a host and a device byte budget. State in use
  by a running job is never evicted. Eviction calls `close`. It is new: senselab's `WorkerSet` is
  count-bound with no byte budget (`ex/workers.py:311-320`), and its byte-budget `ModelCache` has no
  busy guard (`senselab_worker/model_cache.py:107-110`). Under `debug` the `Submitter` holds one
  `ResidentSet` for its lifetime, so a notebook that calls the same task twice loads once. Under `cf`
  each process has its own, unrouted.
- Providers are registered by entry point (`pydra.providers`) or passed as
  `Submitter(providers=[...])` / `WarmPool(providers=[...])`. An unregistered name is a wiring error at
  hand-out.
- **A provider object is read-only to jobs.** A job that mutates it (switches a model's mode, consumes
  a random state, appends to a cache) makes later jobs depend on earlier ones. The sampled cold re-check
  (§4.4, `recheck_fraction`) measures this; pydra cannot prevent it.

**Warm environments.** `Environment` gains one optional method:

```python
class Environment:
    def warm(self, roots: Sequence[Path], device: Device) -> WarmEnvironment | None:
        """A handle that runs many jobs in one started environment; None: not warmable."""

class WarmEnvironment(Protocol):
    key: ResidentKey                          # kind="environment", spec = the resolved reference
    roots: frozenset[Path]                    # the binding roots it was started with
    def covers(self, job: Job) -> bool: ...   # the job's bindings fall inside `roots`
    def execute(self, job: Job) -> dict[str, Any]: ...   # as Environment.execute
    def close(self) -> None: ...
```

| Environment | `warm()` | Each job | Defaults when `warm=True` |
|---|---|---|---|
| `Native` | `None` | as today | — |
| `Docker` | `docker run -d --init --user <uid>:<gid>` of the resolved reference (RFC 0004 §5.3) with `roots` bound and a sleeper as entrypoint; `--gpus` with the worker's `CUDA_VISIBLE_DEVICES` on a `cuda` class | `docker exec -w <work dir> -e TMPDIR=<work dir>/tmp -e HOME=<work dir>/home <id> <command>` | non-root user; per-job `TMPDIR` and `HOME`; `--read-only --tmpfs /tmp` under `Docker(read_only=True)` |
| `Singularity` / Apptainer | `apptainer instance start -c --no-home` (`--nv` and the worker's `CUDA_VISIBLE_DEVICES` on a `cuda` class) with `roots` bound | `apptainer exec --env TMPDIR=…,HOME=… instance://<token> <command>` | contained; no home; per-job environment |
| `Lmod` | resolves `module load <full names>` once (RFC 0004 §5.3) and keeps the resulting environment mapping | the command runs with that mapping | no process is kept; the saving is the resolution (`environments/lmod.py:33`) |
| `Venv` (0001 G2) | through a process-owning provider (above) | a call over that process's protocol | after G2 |

- **Binding roots come from the jobs.** `place()` computes each job's bindings
  (`Container.get_bindings`, `environments/base.py:162-243`) and reduces them to top-level roots; a
  ticket carries the union for its jobs (§5.1). The worker starts the warm environment with that union
  plus `WorkerClass.mounts`, the store root and the work root, and restarts it once when a later ticket
  needs a root it lacks, up to `max_mount_roots` (default 16, unmeasured); beyond it, the job runs
  through the environment's cold `execute` in the same worker. `covers` stays as the per-job check.
  Correctness never depends on warmth.
- **Isolation.** The defaults in the table remove the two writable places a job reaches without trying
  (`$HOME`, `/tmp`), and the non-root user keeps outputs owned by the submitting user, which RFC 0003's
  owner-only rules and `files_from`'s reflink copy need; pydra's cold `Docker.execute` runs as the
  image's default user with `-v` and `-w` only (`environments/docker.py:35-61`). A job's output equals
  the cold run's by construction only when nothing a job writes elsewhere in the container's writable
  layers is read by a later job. There is no periodic restart by default; `max_jobs: int | None` on the
  environment is an opt-in restart for tools known to leak. The sampled cold re-check (§4.4) is the
  measuring tool.
- **Opt-in per environment.** `Docker(..., warm=True)`. A container tool that keeps state inside the
  container between invocations (a license daemon, a cache in its own writable path) would make one
  job's output depend on another's, so warmth is the author's statement that it does not.
- `setup` and `teardown` (`environments/base.py:31`, `:50`) are called by `WarmEnvironment` start and
  close; their `TODO` is closed by this.

**A job's residency** is the tuple of its resident keys, ordered: its environment's key when the
environment has `warm=True`, then one key per provided field. Its **residency digest** is the sha256 of
the canonical list of their digests, or none for a job with no resident state.

### 3.2 Resource requests and worker classes

`ComputeResources` (`utils/resources.py:37-63`) gains two optional fields:

```python
gpu_mem: str | None = None        # device memory per GPU the task's working set needs, "40G"
residency: Literal["route", "ignore"] = "route"   # "ignore": run on any worker of the class, unrouted
```

Neither enters a checksum; both are scheduler metadata like the existing fields. `residency="ignore"`
is new (senselab has no equivalent): the escape hatch for a provided field whose load is cheap.

A **worker class** is what a substrate can start:

```python
@attrs.frozen
class WorkerClass:
    name: str                              # "gpu-h100", "cpu-large"
    device: Literal["cpu", "cuda", "mps"]
    gpus: int = 0
    gpu_type: str | None = None            # matched against ComputeResources.gpu_type
    gpu_mem: int | None = None             # bytes per GPU; None: unknown at placement (§3.3)
    cpus: int | None = None
    mem: int | None = None                 # host bytes
    time: int | None = None                # seconds per worker job
    max_workers: int = 1
    workers_per_device: int = 1            # co-tenancy; the fit divides device memory by it
    mounts: tuple[Path, ...] = ()          # roots every warm environment of the class binds
    substrate_args: Mapping[str, str] = {} # partition, account, qos, constraint: the substrate's own
    quantum_jobs: int = 64                 # most jobs one claim gathers (§4.2); unmeasured
    quantum_s: float = 300.0               # most seconds a worker stays on one queue; unmeasured
```

Classes are data, loaded from a mapping or a YAML file by `WorkerClass.load`. pydra ships no class
table and no default fitted to any cluster. `pydracli pool classes --probe` prints what one host offers
(devices, memory, CPUs) as a starting point and fills `gpu_mem`. `workers_per_device` is pydra's
extension: senselab's `Local` gives each CUDA worker an exclusive device (`ex/substrates.py:359-362`),
and its co-tenancy appears only in sizing. `quantum_jobs` 64 differs from senselab's `quantum_items`
100; neither value was measured.

### 3.3 Placement and fit

**Placement** maps a job to a class and a queue, in the submitting process, before anything is
written:

```python
def place(job: Job, classes: Sequence[WorkerClass], fit: Fit, can_start: CanStart) -> Placement | Unplaceable
@attrs.frozen
class Placement:
    worker_class: str
    queue: str                  # "<class>" or "<class>#<residency digest>"
    tried: tuple[ClassTried, ...]
    batch: int                  # most jobs of this residency one call takes (§4.4)
    roots: frozenset[Path]      # the job's binding roots (§3.1)
```

1. **Candidates.** Classes whose `device`, `gpus`, `cpus` and `mem` satisfy the request, and whose
   `gpu_type` equals the request's when the request names one; `tags` filter further when a class lists
   `tags`. A GPU request never falls back to CPU silently; no candidate is `Unplaceable`.
2. **Order.** Candidates by per-device memory (then host memory), least first; classes with
   `gpu_mem=None` come last, after every class whose memory is known, ordered among themselves by host
   memory. When the request names
   a `gpu_type` this is senselab's `larger_class` order (`rt/fit.py:30-48`); when it names none, the
   ladder climbs across GPU types (an A100-40 class, then A100-80, then H100). Calibration stays per
   hardware class (`Calibration.memory(key, hardware)`), so crossing types changes nothing else. A class
   the substrate says it cannot start (`can_start(cls)` returns a reason, §4.5) is skipped and recorded
   in `tried`.
3. **Judge and climb.** `fit.judge(request, cls)` from the first candidate up; on `"too small"` or
   `"unfit"`, the next, until one `"fits"` or `"unknown"`. The walk is senselab's `choose`
   (`rt/fit.py:173-196`).
4. **Unplaceable.** A job that fits no class is reported `Failed(Unplaceable(...))` at submit, with
   every class tried and the numbers in the message. It is a plan outcome: no worker started, no
   environment charged (senselab ADR-009 amendment 2026-11-12).

```python
Verdict = Literal["fits", "too small", "unfit", "unknown"]   # senselab's "grant unknown" is "unknown"

class Fit(Protocol):
    def judge(self, request: FitRequest, cls: WorkerClass) -> ClassTried: ...
    def batch(self, request: FitRequest, cls: WorkerClass, units: Sequence[float | None]) -> int: ...

@attrs.frozen
class FitRequest:
    residency: tuple[ResidentKey, ...]
    resources: ComputeResources
    units: float | None          # the job's size in the task's own units (§4.4), when it declares one
```

- **Time is part of fit.** A request whose `ComputeResources.time` exceeds the class's `time` is
  `"unfit"` for that class, with both numbers, under every `Fit`.
- **A class with `gpu_mem=None`** is judged `"unknown"` under every `Fit`, which places, because the
  submitting process cannot read a device it does not have; it sorts last in the ladder (step 2), so a
  class with known memory that fits is always preferred. The worker reads the device at start (§4.3),
  fills `gpu_mem`, records it in its ready mark, and re-judges each ticket it claims. A ticket whose
  declared need exceeds the grant is refused by that worker: under the claim lease it writes each
  unsettled job `withdrawn` with reason `refused` and the memory it read (§5.5), and releases the claim
  unrun. The coordinator's follower sees the refusal and re-places those jobs from the next class up
  under fresh attempts, uncharged; later placements on that class use the memory read in any ready mark
  of the class instead of `"unknown"` (§6).
- **`DeclaredFit`** (the default) judges from declared figures: `gpu_mem` (or the providers'
  `footprint`) against the class's per-device memory over `workers_per_device`, `mem` against host
  memory, `time` against `time`. No figure gives `"unknown"`. `batch` returns 1.
- **`MeasuredFit(calibration, headroom=0.9)`** is senselab's rule (`rt/fit.py:156-170`): a class is
  `"unfit"` when a load of the residency ran out of memory in a grant at least as large; `"too small"`
  when `resident + m0 > headroom × grant`; otherwise `"fits"`. `resident`, `m0` and `m1` are a
  `MemoryFit` (`rt/calibration.py:238-243`): `m0` and `m1` are a Theil–Sen fit of peak memory beyond
  `resident` against padded units over warm requests per stratum (`rt/probe.py:359-386`). `batch`
  returns `max(1, floor((headroom × share − resident − m0) / (m1 × length)))`, where `share` is the
  class's per-device memory over `workers_per_device` and `length` the stratum's padded units
  (`rt/probe.py:394-402`), and, when per-job times at several batch sizes are recorded, the fastest
  batch at or below that bound. `headroom` 0.9 is senselab's site value, not a pydra default fitted to
  anything.
- **`Calibration`** is where measurements live:

```python
class Calibration(Protocol):
    def memory(self, key: str, hardware: HardwareClass) -> MemoryFit | None: ...
    def unfit(self, key: str) -> Sequence[UnfitRecord]: ...
    def times(self, key: str, hardware: HardwareClass) -> Sequence[TimeRecord]: ...
    def parity(self, key: str) -> Sequence[ParityRecord]: ...
    def record(self, record: Sample | UnfitRecord | ParityRecord) -> None: ...
```

  `key` is the residency digest. `DirectoryCalibration(root)` ships: JSON records per key and hardware
  class, written by workers (§4.3), readable by later runs. `WarmPool(calibration=None)` uses
  `DirectoryCalibration(root / "calibration")`, so a pool without a client-managed store still learns.
  An `UnfitRecord` names the residency, the resource that ran out (`"device"` or `"host"`), the grant,
  and the code that loaded it: the providers' `version()` and the environment's resolved reference. A
  change to either gives a record that no longer binds, as senselab's `current_unfit` does
  (`rt/fit.py:146-150`). A `ParityRecord` is the result of one sampled cold re-check (§4.4): the
  residency, the batch sizes compared, and whether the output digests were equal. A client that
  measures deliberately (senselab's probe) writes the same records.
- **Admission and backpressure.** Placement runs at `WarmPool.submit` for every job. Placed jobs are
  written as tickets up to `max_queued_jobs` per queue (default 4 × `quantum_jobs` × `max_workers`,
  unmeasured); the rest wait in the submitting process and are written as tickets settle. `submit`
  never blocks and never refuses a placeable job. `plan()` (RFC 0002 R8) shows each node's placement
  (§7.4).

## 4. The executor

### 4.1 `WarmPool`

```python
class WarmPool:                                   # an RFC 0002 §3.2 Executor
    batches = True                                # an instance attribute, as in the integrated protocol
    checks_store: ClassVar[bool] = True           # workers run jobs through Job.run
    inline: ClassVar[bool] = False

    def __init__(self, root: Path, classes: Sequence[WorkerClass], substrate: Substrate, *,
                 store: ResultStore, fit: Fit = DeclaredFit(), calibration: Calibration | None = None,
                 providers: Sequence[Provider] = (), faults: FaultBudgets = FaultBudgets(),
                 classify: Classifier = default_classifier, settings: PoolSettings = PoolSettings(),
                 codec: JobCodec = PrivatePickle()): ...
    def bind(self, report: Callable[[Job, Outcome], None]) -> None: ...
    def submit(self, jobs: Sequence[Job]) -> Handle: ...
    def cancel(self, handle: Handle) -> None: ...
    def close(self) -> None: ...
    def place(self, jobs: Sequence[Job]) -> list[Placement | Unplaceable]: ...   # for plan(), §7.4
```

- `Submitter(worker=WarmPool(...))`, or `Submitter(workers={"gpu": WarmPool(...), "local": "cf"},
  router=...)` with 0001 F2's router; or an external driver (RFC 0002 §3.5) that calls `bind` and
  `submit` itself.
- `root` is the pool directory (§5.1), on storage every worker and the submitting process see. It is
  per pool, not per run: a resubmitted run finds its predecessor's tickets (§5.5), and runs may use one
  pool one after another.
- **One live coordinator per pool.** At construction `WarmPool` lists `coordinator/` and waits while
  another coordinator's lease is live, logging the live one once; `PoolSettings.coordinator_wait`
  (default `None`, wait without bound) caps the wait and then raises `PoolBusy` naming it. A second run
  that must proceed at the same time uses its own root. Two coordinators that start on a free pool at
  once both take an epoch; the epoch fence (§5.3) stops the older one, whose run ends `RunIncomplete`
  with reason `PoolFenced` and can be resubmitted.
- **Lifecycle.** `close` writes the coordinator's `ended` mark. A coordinator that still holds the newest
  epoch then stops the substrate-started workers of the pool that `substrate.live()` lists, including those a dead predecessor started; workers started by
  hand (`ExternalSubstrate`) are left running. A fenced coordinator (`PoolFenced`) stops nothing, since the
  workers serve the newer one. `pydracli pool collect` removes ended coordinators' directories, settled
  tickets, and the payloads no unsettled ticket names (§5.2); the work directories of attempts that were
  taken over or withdrawn are the store's `collect`'s (RFC 0003 §9). It never removes a job's highest follow generation (§5.1).

### 4.2 Queues, tickets and routing

- **A queue** is named `<class>` for jobs with no residency (or `residency="ignore"`), and
  `<class>#<residency digest>` otherwise. This is senselab's `_queue_name` (`ex/pool.py:536-541`)
  with the model identity generalised to a residency.
- **A ticket** is one queue item: up to `quantum_jobs` jobs of one node and one queue, cut from one
  `submit` call in node order. It is a JSON document (§5.1) naming each job's checksum, lookup
  fingerprint, attempt, payload and binding roots, never a pickle. A whole round of a 100-element node
  becomes two tickets of 50 on one queue, claimed by warm workers, not 100 scheduler jobs.
- **A job is in at most one live ticket.** Before writing a ticket, the pool looks the job's
  `(checksum, fingerprint)` up in its follow index (§5.1, §5.5): the job's newest follow generation names
  the ticket that holds it. A job held by a live ticket is followed, not put again. senselab found the alternative on the cluster: a resumed coordinator re-cut unfinished jobs
  into new items while the old ones were still queued, and 61 jobs ran twice (R-273, §9). senselab's
  mechanism is `Pool._live_items`/`_follow` (`ex/pool.py:395-447`), tested by `pool_follow_test.py`.
- **Routing.** A worker serves the queues of its class. It claims from the queue whose oldest waiting
  ticket is oldest, preferring a queue whose residency it holds while its quantum lasts
  (`quantum_jobs` jobs or `quantum_s` seconds), then takes the oldest work again
  (`ex/pool_worker.py:519-554`, `:610-631`). On a queue it has not served, it claims one ticket, so a
  load falls on one ticket; once warm, it gathers single-job tickets up to its quantum.
- **The bound this gives.** With at least one live worker of the class, a ticket waits at most
  (residencies with waiting work on the class × `quantum_s`) plus one load. With N idle workers and one
  new cold queue, up to N workers load it (one ticket each); the scaler (§4.5) does not reserve workers
  per queue. No other fairness guard exists, in senselab or here.
- **Eviction.** Before loading a residency, the worker's `ResidentSet` evicts idle resident state,
  least recently used first, until the measured or declared footprint fits; state in use is never
  evicted. A resident model is a cache, not a reservation (senselab `execution.md` §3.4): a worker never
  refuses work because it holds something.

### 4.3 The worker

`pydracli pool worker --root <pool> --class <name>` (or `python -m pydra.workers.pool.worker`) runs
one loop:

1. **Start.** Read `pool.json`; refuse to start (`PoolSettingsMismatch`, exit code 3) when its
   `LeaseSettings` (`stale_after`, `renew_fraction`, `attribute_delay`, `timeout`) differ in any field
   from the worker's own, because a worker and its coordinator that judge staleness by different bounds
   disagree about who is alive (senselab's `LeaseMismatch` compares its lease section the same way,
   `ex/epoch.py:61`; R-277 is the defect it closed). The fifth field, `max_wait`, is left out on
   purpose: it caps how long one process waits for a lease, not when a lease is stale. Before importing
   any provider, start the **beater** (§5.3), a helper subprocess that beats every lease the worker
   holds; the worker takes its leases without an in-process beat thread. Take the worker lease and hand
   it to the beater. Read the
   hardware as a `HardwareClass` (device kind, product name, memory, compute capability, granted CPUs);
   refuse to start (`WrongHardware`) when it does not match the class's `gpu_type` or `gpus`, so a
   mis-typed grant costs one failed start, not a queue of failed jobs; fill the class's `gpu_mem` when
   it is `None`. Write the ready mark.
2. **Claim** a ticket (§4.2) under a claim lease (§5.3), hand the claim to the beater and wait for its
   acknowledgement, then re-judge the ticket against the read hardware (§3.3). Before each job the worker
   reads the job's newest follow generation. It takes the job's attempt from that generation, never from
   the ticket file (which is never rewritten after a takeover). It skips the job, writing nothing, when that
   generation names another ticket or the job has a `withdrawn` or `expired` outcome there (§5.5).
   Otherwise it creates the attempt's `started` mark by exclusive create. If the mark exists, the attempt
   has already run once (a release after a stopped call, a content retry), so the worker first writes the
   next generation (`cause: "restart"`) naming this ticket and starts that attempt instead (§5.5).
3. **Load** the ticket's residency through the `ResidentSet`: warm environments by `warm(roots,
   device)` (restarting one whose roots do not cover the ticket, §3.1), providers by `load(spec,
   device)`. A failure here is a load fault (§6): it never charges a job.
4. **Run** the ticket's jobs, one call per batch (§4.4), each through `Job.run(rerun)` in this process
   with `chdir=False` (0001 H0, merged), under the job's pool attempt (§5.5). `Job.run` looks the job
   up, runs it in its work directory and, under the pool, **proves the claim lease immediately before
   `publish`** through the slot RFC 0003 P4 gives its de-duplication lease (plan task P4c), as
   `PickleStore.commit` already proves its lease (`lease_still_held`, integration `store.py:361-367`,
   `:590`). The proof and the rename are not one atomic step, and a holder learns of a loss late
   (§5.3), so the proof narrows the window in which a lost holder publishes but does not close it; the
   guarantee of one committed result is the store's first writer wins (RFC 0003 §5.1). Before the call
   the worker writes the call record `workers/<token>/calls/<seq>.json`, naming every job and attempt in
   the call, by exclusive create, and sends it with the call's deadline to the beater (§5.3, "Job
   timeout"). It writes `<seq>.done` when the call returns. A loss is charged as a crash only to the jobs
   of a call that has a record and no `done` (§5.5). Python tasks run in-process with
   provided fields bound; shell tasks run through the warm environment when it `covers` the job,
   otherwise through the environment's cold `execute`.
5. **Settle each job as it finishes**: its measurement first, then its outcome record, then the
   ticket's settled mark (§5.1). A ticket of 64 jobs is not settled once at the end; a worker that stops
   after job 40 leaves 40 settled (senselab T24-2, §9).
6. **Prove** the claim lease every renewal interval from the loop while a call runs, and act on the
   beater's `lost` message as soon as it arrives (§5.3). A claim whose lease was lost is abandoned: the
   call in flight is stopped when it is a subprocess, a provider's process or a warm environment's
   command; an in-process Python call that cannot be stopped finishes, and its `publish` is refused by
   the failed proof of step 4 unless it wins the race the proof narrows. Its outcome is not written.
7. **Exit** after `idle_polls` polls of `poll_s` with nothing claimable; on SIGTERM after settling
   finished jobs and releasing the rest; and, for a worker a substrate started, when no coordinator
   lease of the pool has been live for `orphan_after_s`. A worker started by hand (`ExternalSubstrate`)
   exits on idle only, so one that attached before any coordinator is not ended by the absence of one.
   A worker also exits (recycles) after a call's deadline passed (§5.3, "Job timeout") and when its
   beater dies: it takes no new claim, lets the call in flight finish under the proof of step 4, settles
   what finished and exits, so its unbeaten claims go stale and are taken over.
   Resident state is closed on exit: containers and providers' processes are stopped and their process
   groups waited for (R-295).

**What a worker records** (`Sample`, per settled job or batch): wall seconds, claim seconds, batch size
and padded units, host baseline/resident/peak bytes, device bytes (device-wide, read once per claim on a
CUDA class, as the co-tenancy baseline; a process-owning provider's own figure comes from `measure`),
load seconds, whether this job's residency was cold or warm (`since_load`), and any `reload` (§4.4).
These go to the `Calibration` and to the job's execute context (§8). The fields are senselab's `Sample`
(`rt/sizing.py:362-382`) without the audio ones.

### 4.4 Whole-round batching

Two levels, both opt-in:

- **A ticket is a batch of jobs into one warm worker.** This needs nothing of the task. It amortises the
  claim, the lease, the load and the per-ticket overhead. senselab measured, through its subprocess
  worker, a per-item overhead of 0.098 s plus 7.29 ms per added job (§9). The overhead of this
  proposal's mechanism (an in-process call, a JSON ticket and a payload file per job) is unmeasured;
  W6's `bench_pool_overhead` measures it (§11).
- **A batched call** runs several jobs in one invocation. A Python task declares
  `@python.define(batched=embed_many, units=duration)`, where `embed_many(audio: list[Path], model) ->
  list[T | BaseException]` takes the per-job inputs as lists and returns, per job, its outputs or the
  exception that job raised. The worker groups a ticket's jobs into calls of at most `Placement.batch`,
  ordered by `units` (a job's size in its own units, such as seconds of audio or voxels), so a call pads
  little. Each job's result is published under its own checksum, or its exception is that job's content
  fault (§6), which is never bisected. A shell task cannot be batched this way; its tickets still
  amortise the environment.
- **Bisection.** A batched call that raises as a whole is bisected only on the resource and crash
  triggers (§6): split in halves, each sent again, until the failing job is alone. A half that completed
  keeps its results. Any other raise from the whole call fails the jobs of that call. This is senselab's
  one bisection engine (`ex/job.py:121-142`, `:291-363`; its ADR-009 amendment 2026-11-11).
- **Reload after a resource fault.** After a resource fault at a call, the worker closes and reloads the
  ticket's residency before running the halves (uncharged, recorded as `reload` in the sample): an
  in-process CUDA out-of-memory leaves a context the next call cannot trust, and a process-owning
  provider restarts its process, as senselab's `WorkerSet` replaces a client that failed a request
  (`worker_set_test.py`).
- **A batched call must not change a result.** A model that pads a batch to its longest item can give
  different outputs for a short item batched with long ones than alone. senselab encodes such items
  alone in its parity references with 1.x (`benchmarks.md` amendment 2026-11-03), and recorded "repeat
  and batch differed by 0" for its own measures (`config-derivations.md:379`); this is the rationale,
  not a found defect. The batched implementation is the author's statement that results equal the
  unbatched ones; the batch size is recorded in the context and never keyed (§8).
- **The sampled cold re-check.** `PoolSettings.recheck_fraction` (default 0): a sampled job is run
  again on a fresh residency, unbatched, through RFC 0003's `Verify` (§7.1 of that RFC), and its output
  digests compared with the committed entry's. The comparison is written to the calibration store as a
  `ParityRecord`, so a batched implementation's or a warm environment's equality claim is measured over
  time; a mismatch is also a content fault of the *task*, reported once per residency through the audit.
  senselab turns it on in calibration runs.

### 4.5 Substrates and scaling

```python
class Substrate(Protocol):
    def start(self, cls: WorkerClass, n: int) -> list[WorkerHandle]: ...   # never blocks on a queue
    def live(self) -> list[WorkerHandle]: ...          # one listing per call: pending and running
    def stop(self, workers: Sequence[WorkerHandle]) -> None: ...
    def can_start(self, cls: WorkerClass) -> bool | str: ...   # True, or why not
```

- **`can_start`** answers whether one more worker of a class may start. The default reads
  `per_user_cpu_cap` (`cls.cpus` × live workers against it); a site overrides it for GPU GRES caps
  (Slurm `MaxTRESPU` by `gres/gpu`), submit caps (`MaxSubmitPU`) and anything else. Placement skips a
  class that can never start (§3.3); scaling stops at the first `False`.
- **`LocalSubstrate(devices=None)`** starts worker processes on this host under the submitting
  process. A `cuda` class's worker gets devices through `CUDA_VISIBLE_DEVICES` (from `nvidia-smi` when
  not given), at most `workers_per_device` per device, so a host with four GPUs runs four one-GPU
  workers or two two-GPU workers. senselab's `Local` (`ex/substrates.py:357-420`), with co-tenancy
  added.
- **`SlurmSubstrate(...)`** submits **one `sbatch` per worker**, never per job: `--gres=gpu:<type>:<n>`,
  `--cpus-per-task`, `--mem`, `--time` and the class's `substrate_args`, with `--wrap` running the worker
  command for that class, a job name scoped to the pool, its coordinator epoch and a token, and logs
  under the pool's private directory. A submission that times out is followed by a lookup of its name;
  when the name is found, it is not submitted again, and when it is not, it is submitted once more only
  after `attribute_delay`. senselab's version resubmits after a failed lookup and records the duplicate
  worker as an open defect (R-286, `ex/substrates.py:603-641`); this rule narrows it. It does not close
  it: `attribute_delay` bounds NFS attribute caching, not slurmctld, so a timed-out `sbatch` can still
  land later. A duplicate worker is harmless: it is a surplus, and scaling stops it. `live()` reads one
  `squeue --user` per poll cycle, shared with `SlurmWorker`'s poller (`workers/slurm.py:457-527`
  integration; 0001 A2) so a run with both polls once.
- **`ExternalSubstrate`** starts nothing: workers are started by hand or by another system and attach by
  pointing at the root.
- **Scaling.** Each poll the pool computes, per queue, the workers its waiting jobs want:
  `min(ceil(jobs × s_per_job / target_wall_s), ceil(jobs / quantum_jobs), cls.max_workers)` from the
  slowest measured per-job time in the calibration, or `unmeasured_workers` (default 1) when none is
  measured; summed per class, and started while `can_start(cls)` is `True`. A surplus is stopped, pending
  workers first, running ones only once their queues have no unsettled ticket. This is senselab's
  `workers_wanted` and `scale` (`ex/pool.py:511-534`, `ex/substrates.py:884`).
- **Failed starts.** A worker that leaves `live()` without a ready mark is a failed start of its class.
  The next start of the class backs off, doubling; after `start_failures` consecutive failures (default
  7, pydra's choice; senselab reuses `infrastructure_replacements`, `ex/substrates.py:15-17`) the class is
  broken and its jobs are `NotRun(held=True)` naming the class, the count, the latest cause and the class
  values (§6). senselab saw this on the cluster when four coordinators filled a partition's per-user
  submit limit (§9).

## 5. The worker protocol over shared storage

### 5.1 Layout

```
<root>/                                 0700 (private pool)
  pool.json                             header: "pool@1", pool id (random), the full LeaseSettings, codec, owner uid, probe results
  coordinator/<epoch>/                  one per coordinator, taken by mkdir; lease, attempt, ended, faults.json (§6)
  classes/<name>.json                   the WorkerClass as JCS
  queues/<label>/                       label = sha256 of the queue name
    name                                the queue name, written once
    tickets/<digest>.json               a ticket, JCS; digest = sha256 of its canonical JSON; first writer wins
    segments/<seq>/slot-<n>             index: an O_EXCL regular file whose content is the ticket digest
    segments/<seq>/jobs-<n>-<count>     the ticket's job count, so a listing gives the queue's depth
    claims/<digest>.lease/              the claim's DirectoryLease (a directory, beats inside)
    losses/<digest>/<n>                 a takeover of this ticket (§5.5); O_EXCL
    outcome/<digest>/<job>-<attempt>.json  one per job and attempt: done | failed | held | withdrawn | expired
    settled/<digest>-<count>            an O_EXCL empty file each time the ticket's outcome count reaches <count>
    held-<epoch>-<label>                a hold of a residency on this queue (§6)
    faults/<epoch>-<label>/<n>          an infrastructure fault of a residency in an epoch (§6); O_EXCL
  follow/<checksum>-<fingerprint>/<g>   generation g of the job: an O_EXCL regular file naming its ticket and attempt
  follow/<checksum>-<fingerprint>/<g>.started  O_EXCL: attempt <g> was started, by which worker token
  workers/<token>/                      lease/, ready.json, samples.jsonl, calls/<seq>.json and <seq>.done (O_EXCL)
  payloads/<sha256>                     a job's payload (§5.2), read by every attempt of the job
  calibration/                          the default DirectoryCalibration
  private/                              Slurm logs; 0700
```

The queue directories are senselab's `queue.py` layout (`ex/queue.py:1-40`) with items renamed tickets,
outcomes per job and attempt (§4.3 step 5), probe marks left to clients, and three changes:

- **No symlinks.** senselab's slots and `slotted/` were symlinks (`ex/queue.py:576`, `:596`); here a slot
  and a follow entry are `O_EXCL` regular files holding a digest. senselab's ADR-003 (amendment
  2026-11-03) moved leases from links to `mkdir` because SMB, NTFS and some FUSE mounts have no symlink or
  `O_EXCL` link; the pool must run on every root the store supports.
- **Settled marks.** The follower lists `settled/` of each queue once per `poll_s` (default 5 s on
  `LocalSubstrate`, 30 s on `SlurmSubstrate`; both unmeasured) and reads `outcome/<digest>/` only for
  tickets whose count moved. Listing one directory per ticket per poll would be 10⁴ `getdents` per poll
  at 10⁶ jobs on NFS. A writer that has just written an outcome counts the files in `outcome/<digest>/`
  and creates `settled/<digest>-<count>`, then removes `settled/<digest>-<count - 1>`, so the
  directory holds one mark per ticket. Two writers can compute the same count (a loser finishing a job
  during a takeover, and the taker), and an existing mark makes the create fail. On `EEXIST` the writer
  counts again and creates the mark for the new count, repeating until it has created a mark, or found
  one, whose count includes its own outcome. The last count is therefore always written.
- **Follow generations.** The follow index is a directory per job, `follow/<checksum>-<fingerprint>/`,
  holding numbered generations. A generation is an `O_EXCL` regular file, never modified or removed while
  the pool lives (`collect` removes a settled job's generations except the highest, which stays as the durable
  record that keeps numbering from restarting at 0): `{"ticket": <digest>, "queue":
  <label>, "attempt": <attempt>, "cause": "put" | "replaced" | "split" | "takeover" | "restart" | "ladder",
  "phase": "call" | "load" | null, "prior": <attempt> | null, "charged": <bool>, "uuid": …}`. The newest generation names the ticket that holds the job and the
  attempt it runs under; moving a job writes generation `g+1`, and nothing is ever removed and recreated.
  `g` is allocated as one more than the highest listed; on `EEXIST` the writer lists again and takes the
  next number. The job's attempt is `<pool id>.<g>` (§5.5), so the allocation that orders the index is
  also the one that makes attempts unique.
- **Outcome records are exclusive creates; the first one wins.** A worker whose `done` meets an existing
  `expired` (§5.3) writes nothing more for that job; its published result, if any, is found by the
  taker's lookup (§5.5).
- **Every write is synced.** Through `StorageDir` (RFC 0003 P1): descriptor-relative, `O_NOFOLLOW`,
  staged, file and directory fsynced before the link, never modified in place. senselab's adversarial
  review found unsynced exclusive writes in its own queue and run directory (R-290), and its outcome link
  had no directory fsync (`ex/queue.py:180-186`). An outcome write that meets `EEXIST` after a lost reply
  reads the record back and accepts it when the record's uuid is its own (`ex/queue.py:168-186`).

What the protocol must survive on NFS, and W4's fakes simulate: the attribute cache (directory mtimes up
to `acdirmax`, 60 s by default, late; covered by `attribute_delay`), lost replies to `link` and `mkdir`
(read back by uuid), `ESTALE` on a directory another client removed (treated as absent), and the 30 s
polling floor measured on the cluster (§9).

A ticket:

```json
{"format": "pool-ticket@1", "queue": "gpu-h100#3f…", "node": "wf/embed",
 "residency": [{"kind": "provider", "name": "hf-model", "spec": "json@1:…"}],
 "batch": 16, "roots": ["/data/study-a"], "ladder": ["gpu-h100-80", "gpu-b200"],
 "jobs": [{"checksum": "…", "fingerprint": "…", "attempt": "9c41e07a5b2d.0", "runner_attempt": 0,
           "payload": {"codec": "private-pickle@1", "path": "<root>/payloads/<sha256>", "sha256": "…"},
           "units": 4.6}]}
```

A job's `attempt` is the pool attempt of the generation that put it in this ticket (§5.5);
`ladder` is the rest of the job's placement order, computed by the coordinator's `Fit` at submit (§3.3); a
claim holder moves a job up by it without judging (§5.5). `runner_attempt` is the attempt the `GraphRunner` assigned (RFC 0002 R7), carried for reporting only. A
fresh runner may start its attempts at 0 again, which is why the pool never uses the runner's attempt as
a work directory's address.

### 5.2 Payloads and trust

A worker must never unpickle a file that another user can write (RFC 0003 §6.4; senselab ADR-007, which
puts every substrate's pickles under a 0700 private directory).

- **`PrivatePickle` (default codec).** The job is cloudpickled once to `<root>/payloads/<sha256>`, under
  the pool's 0700 root, named by the sha256 the ticket carries. The worker reads it only after the
  ownership walk of RFC 0003 §6.4 passes for `payloads/` (owned by the worker's user, no component
  writable by others) and the digest matches. This is the trust `_job.pklz` already has under the Slurm
  worker (`workers/slurm.py:122-125`), made explicit and checked. The payload is outside every run's work
  directory: a run writes its own `_job.pklz` there and the store empties the directory once it publishes
  (RFC 0003 §5.4), so a payload there would be gone before a restart or a taker reads it. Every attempt of
  the job (a restart, a taker, a half-ticket, a re-placed ticket, §5.5) reads the same payload and runs
  it under its own attempt. The store's `collect` never touches `payloads/`; the pool's `collect` removes
  a payload once no unsettled ticket names it and it is older than the newest ticket that could (a
  payload is written before the ticket that names it).
- **`JobCodec`** is pluggable: `encode(job) -> (bytes, media_type)`, `decode(bytes, store) -> Job`. A
  client whose jobs are import references (0001 G2) with inputs in RFC 0004 §6.2's canonical typed
  encoding can carry them as JSON in the ticket and need no pickle.
- **A pool is single-user.** Workers run as the submitting user (its `sbatch` jobs or its processes). The
  root is 0700; `pool.json` records the owner's uid and a worker of another uid refuses it. A group pool
  needs, first, a codec-only payload path and RFC 0003 §6.4's ownership rules for every file a worker
  reads (§12).
- **No host names on shared paths.** A worker's token and its host are recorded as digests of a per-boot
  random value (senselab's `host_token()`, ADR-007 amendment), never a hostname.

### 5.3 Leases, heartbeats and liveness

Every exclusion in a pool is pydra's `DirectoryLease` (#924, #928), judged on the storage clock. There is
**no host-bound lock** in the protocol: no `flock`, no `SoftFileLock`. senselab lost a resume to one: a
coordinator killed on one host left a lock only its own host could break, and a resume on another host
waited until its 6 h limit (T24-9, §9). A test (W4) fails if a pool module takes one.

- **A named lease (L0).** The integrated `DirectoryLease` is job-shaped, built from `(checksum,
  cache_dir, lockfile)` (integration `lease.py:348-354`). L0 adds `DirectoryLease.at(lockfile: Path,
  settings: LeaseSettings)`, a lease with no checksum, in `engine/lease.py`; the pool's worker, claim and
  coordinator leases use it. There is no public takeover call: a stale lease is taken over through
  `try_take()` (integration `lease.py:774-790`), and a holder checks its own with `prove()` (`:378`).
- **Cross-process beating (L0).** The integrated lease keeps its token in the object (`_token`,
  integration `lease.py:357`), starts an in-process `_Heartbeat` thread when it is taken
  (`lease.py:263-299`, started at `:456-458`), and records the taker's pid in `holder.json`. The worker
  cannot use that thread (a GIL-bound call would starve it), so L0 adds:
  - `DirectoryLease.at(lockfile, settings, *, beat: Literal["thread", "external"] = "thread")`. With
    `"external"`, `try_take()` and `__enter__` take the lease without starting `_Heartbeat`, and the
    property `token_dir` gives the holder's token directory. `prove()` is unchanged: it makes and removes
    a `proof-<uuid>` directory, which never collides with a beat.
  - `LeaseBeater(on_expire=None, kill_after=None)`, a loop in its own process that beats any number of token directories by the same
    rule as `_Heartbeat` (`beat-<n+1>` made, `beat-<n>` removed; `ENOENT`/`ESTALE` is a loss; a failure
    lasting `stale_after` is a loss), counting `n` per directory. It is the only beater of each directory
    it holds, so no two beaters race one `beat-<n+1>`. `on_expire(token_dir, payload)` is an importable
    callable run in the beater process when a deadline passes. `kill_after` is the holder's policy, not
    L0's: when set, the beater sends `SIGKILL` to the holder's pid if no `ack_expired` arrives within
    that many seconds of `expired`. L0's default is never to kill.
  - `BeaterClient`, the holder's end of a pipe to it. `hold(token_dir, settings)`:
    the beater starts beating and acknowledges; the worker runs nothing under a claim before the
    acknowledgement. `release(token_dir)`: the beater stops beating that directory and acknowledges, and
    only then does the holder release (rename the token directory away), so a release never reads as a
    loss. `deadline(token_dir, at | None, payload=None)`: sets or clears the directory's expiry and the JSON
    payload handed to `on_expire` (job timeout, below). `ack_expired(token_dir)`: the holder has stopped
    the expired call. `ping(timeout)`: the beater acknowledges and does nothing else, so a holder finds a
    hung beater while no other request is due; one unanswered ping is a miss (a holder whose call
    held the GIL reads the answer late), and `PING_MISSES` (3) consecutive misses are the beater's
    death. From the beater: `lost(token_dir, reason)` when a beat finds the directory gone or
    failing, and `expired(token_dir)` when a deadline passes, after `on_expire` has run and beating of
    that directory has stopped. Any other request not acknowledged within `renew_interval` is treated
    as the beater's death (a beater wedged on a hard mount must not block the holder while its claims go
    stale). A reader thread in the holder turns `lost` into the
    lease's lost state, so its next `prove()` answers `False` without a storage round trip; a call that
    holds the GIL delays only that bookkeeping, since `prove()` before `publish` asks the storage anyway.
  - A holder whose pipe reaches end of file, or whose acknowledgement times out, has lost its beater: it
    takes no new claim and recycles (§4.3 step 7).
  The pool's one-shot takes (claim, cancel, move) use a fresh `DirectoryLease` per attempt, because
  `try_take()` keeps the bounds of a wait between calls and raises `LeaseTimeout` once they pass
  (integration `lease.py:774-790`); L0 adds no other take call.
  The worker runs no in-process beat thread while it has a beater: every lease it takes is taken with
  `beat="external"`. The coordinator, which runs no task code, beats its own lease with the thread.
- **Settings, whole.** `pool.json` records the full `LeaseSettings` (`stale_after`, `renew_fraction`,
  `attribute_delay`, `timeout`; integration `lease.py:147`), whose beat-bound inequalities the class
  already checks, and every process compares all four (§4.3 step 1). The fifth field, `max_wait`
  (integration `lease.py:207`), is excluded on purpose: it caps a waiter, not a staleness bound, and may
  differ between processes. `attribute_delay` defaults to `ATTRIBUTE_DELAY` (60 s, integration
  `lease.py:34`).
- **`pydracli pool probe <root>`** runs when a pool is created and on demand: it measures the directory
  mtime delay between two processes and checks the primitives the protocol needs (`O_EXCL` create,
  `mkdir`, rename, fsync of a directory), as senselab's `probe_capabilities` does, records the result in
  `pool.json`, and refuses a root that lacks a primitive or whose measured delay exceeds
  `attribute_delay`.
- **The beater.** The claim and worker leases beat from a `LeaseBeater` subprocess the worker starts
  before it imports any provider, through `BeaterClient` (above). It exits when the worker's pipe closes.
  A call that holds the GIL, or a process stalled in a driver call, cannot miss a beat; with
  `MISSED_BEATS = 1` (integration `lease.py:47`) a missed second beat would be a lost claim and a takeover
  of a job still running. On macOS, where processes start by `spawn`, the beater is a second interpreter
  start of about 0.5 s per worker; that cost is accepted. The worker's loop still calls `prove()` every
  renewal interval (§4.3 step 6), so a worker whose claim was taken over learns it from the storage. A
  proof makes and removes a directory in the claim's token directory, which moves the mtime staleness
  is judged by, so it renews the claim. The worker therefore proves only while its beater answers: it
  pings the beater (`BeaterClient.ping`) every renewal interval while a call runs, so a hung beater is
  found as a dead one is. Once its beater is dead it stops proving, and it enforces the call's
  deadline itself: past it, it writes the call's `expired` outcomes (`pool_expire`) and recycles, so
  the claim goes stale and is taken over (§13, W6 amendment).
- **Job timeout.** Because the beater keeps a wedged call's claim alive, the pool bounds a call
  explicitly. Before each call the worker sends `deadline(claim, now + T + timeout_grace, call record)`, where `T` is
  the sum of the call's jobs' `ComputeResources.time`, or `PoolSettings.call_timeout` for a job that
  declares none, and `timeout_grace` is a `PoolSettings` field (default 60 s, unmeasured); it clears the
  deadline when the call returns. With neither figure there is no deadline, and the bound is the worker's
  own time limit. The pool's beater is `LeaseBeater(on_expire=pool_expire, kill_after=timeout_grace)`.
  When a deadline passes, the beater stops beating that claim and runs `pool_expire`, which writes each job
  in the call record an `expired` outcome under its attempt (first writer wins against a late `done`,
  §5.1), then sends `expired`. A worker that can stop the call (a subprocess, a provider's process, a warm
  environment's command) stops it, sends `ack_expired` and recycles (§4.3 step 7), since in-process state
  after an abandoned call is not trusted. A worker that does not acknowledge within `timeout_grace` (an
  in-process call holding the GIL, or a call stuck in a driver) is killed by the beater, which then exits.
  Either way the claim goes stale and is taken over under fresh attempts (§5.5). The taker looks the job up
  in the store first (a committed result settles `done`) and only then treats an `expired` prior attempt
  as a resource fault at a call (§6), never as a crash. Jobs the worker held that were not in the call are
  re-placed uncharged. A timeout moves only to a ladder class whose `time` is larger or whose `gpu_type`
  differs; otherwise it fails at once. A hang therefore costs at most `(resource_moves + 1) × (T + 2 ×
  timeout_grace + stale_after)`, plus log₂ n such waits when it was a batched call of n jobs, and each wait
  kills a warm worker.
- **Worker lease** `workers/<token>/lease/`: held for the worker's life. The pool reads it to count live
  workers and to tell a slow worker from a dead one.
- **Claim lease** `claims/<digest>.lease/`: one per claimed ticket. Taking a claim uses `try_take()`, so a
  worker never blocks on a held ticket; it moves on.
- **Coordinator lease** `coordinator/<epoch>/lease/`: held by the submitting process. One coordinator is
  live on a pool at a time (§4.1): a new one waits until no other coordinator lease is live, then takes
  epoch `n+1` by `mkdir`. A coordinator that sees a newer epoch stops writing tickets and ends its run
  `RunIncomplete` (`PoolFenced`). Since a waiting coordinator takes an epoch only once its predecessor's
  lease is stale or released, the fence acts only on two coordinators that started on a free pool at once,
  or on one whose lease went stale while it lived (a fence for efficiency, not correctness: results are
  idempotent, RFC 0003 §5.1, and attempts never collide, §5.5). This is senselab's epoch
  (`ex/epoch.py`) and answers RFC 0002 §5's question: the run identity that fences a restarted coordinator
  belongs to the pool. The epoch is taken after every fallible step of `WarmPool` construction (senselab
  R-283, §9).
- **Liveness without a daemon.** Nothing pings a worker. A claim whose lease is stale is taken over by the
  next worker that tries it; a holder learns it lost its claim at its next proof, which can be up to one
  renewal interval plus `attribute_delay` late (senselab ADR-003 amendment 2026-11-06, "Not
  guaranteed"). §5.5 makes that lateness harmless.

### 5.4 Results through the store

The pool directory never holds a result.

- A worker runs each job through `Job.run`, which proves the claim and publishes through the store (RFC
  0003 §5.1, §10): under `PortableStore` the entry is renamed into place, `_committed.json` is written to
  the attempt's work directory, and `on_commit` (provenance) runs before the rename. Only then does the
  worker write the job's outcome record (`done`, naming the attempt), so an outcome is never visible
  before its result and its provenance (senselab T24-10: the last claim's provenance, written after its
  commit, was lost to the stop; §9).
- A failed job writes `_error.json` to its attempt's work directory (RFC 0003 §11) and an outcome record
  `failed` with its fault record (§6). Nothing is stored.
- The submitting process follows settled marks (§5.1) and reads each `done` job's result through
  `Job.result()`: by the `_committed.json` of the attempt the outcome names, then by the entry at the
  job's own address, after RFC 0002 §3.3's visibility retry. It reports the `Result`, carrying
  `committed`, to the driver. For each job the follower reads the outcome in the ticket that the job's
  newest follow generation names (§5.1), and reports only a final outcome: `done`, `failed`, `held`, or
  `withdrawn` with reason `"cancel"`. A `withdrawn` outcome with reason `"refused"` makes the follower
  re-place the job (§3.3); one with reason `"replaced"` or `"split"` is followed into the ticket it names;
  an `expired` outcome waits for the taker (§5.3). That read path does not exist yet: `Job.result()` reads through `lookup`
  (integration `job.py:675-728`) and `PortableStore` marks `work_dir` accepted and not yet used
  (integration `portable/store.py:302-305`); RFC 0003 P4 (plan task P4c) and P8 add it, which is why W4
  and W5 depend on them.
- Under `PickleStore`, `Job.run` commits into the slot under the job's lease as today and `Job.result()`
  reads through `lookup`; the pool works, but warm facts are kept out of the context (§8).

### 5.5 Recovery

- **Every execution has a fresh attempt.** A job's pool attempt is `<pool id>.<g>`, where `g` is the
  follow generation (§5.1) that put it where it runs, allocated by an exclusive create and never reused;
  the pool id is a random value written once to `pool.json`, so two pools over one store cannot collide.
  Every placement writes a generation: the first put (`cause: "put"`), a re-placement (`"replaced"`), a
  half of a split ticket (`"split"`), a takeover (`"takeover"`) and a move up the ladder (`"ladder"`).
  Every start of an attempt that already started writes one too (`"restart"`): the claimant creates the
  attempt's `started` mark before running it, and finding the mark present it writes generation `g+1`
  first (§4.3 step 2). That covers a release after a stopped call (SIGTERM), a content retry under
  `content_attempts`, and any rerun of a job whose attempt has run. `collect` keeps each job's highest
  generation, so a rerun after `collect` continues from it. Two executions of one job may happen; two in
  one work directory never do, because no attempt is started twice. The runner's own attempt
  is carried as `runner_attempt` and never addresses a work directory, since a resubmitted run's fresh
  `GraphRunner` may number from 0 again.
- **A job leaves a ticket only by `withdrawn`.** A `withdrawn` outcome is written for a job under its
  current attempt (the attempt of its newest generation naming that ticket) by a process that holds the ticket's claim lease, and carries `reason` (`"cancel"`,
  `"replaced"`, `"split"` or `"refused"`) and, when the job moved, `by`: the ticket that now holds it. A
  process that does not hold the claim takes it with `try_take()`, which succeeds only on a ticket that is
  unclaimed or whose claim is stale; a ticket with a live claim is never withdrawn from under its holder.
  A worker skips a job that has a `withdrawn` outcome, or whose newest generation names another ticket
  (§4.3 step 2), and writes nothing for it.
- **Moving a job keeps R-273 closed.** Every move writes, in this order: (1) the new ticket file, with no
  slot yet; (2) for each moved job, generation `g+1` naming the new ticket and the job's fresh attempt;
  (3) the new ticket's slot; (4) the `withdrawn` outcomes on the old ticket; (5) the release of the old
  claim. A first put writes the ticket, then generation 0, then the slot. At every instant, then, each
  job's newest generation names a ticket that holds it: the old one until (2), the new one after it. A
  coordinator decides whether to put a job by reading that newest generation; it puts the job again only
  when the job has no generation or the newest names a ticket in which the job has a final outcome, or a
  settled ticket `collect` has removed. No
  entry is removed or rewritten while the pool lives, so there is no window in which a live job looks
  unheld and is re-cut (R-273). A process that dies between (2) and (3) leaves a ticket without a slot,
  which no worker can claim; the next process that reaches the old ticket (a taker, or a resuming
  coordinator through `try_take()`) finds the newest generation naming a ticket without a slot and writes
  the slot. A duplicate slot is harmless: a ticket is claimed by its digest's lease.
- **A takeover is a new attempt.** A worker that takes over a stale claim writes `losses/<digest>/<n>`.
  For a single-job ticket it first looks the job up in the store (a committed result settles `done`), then
  writes the job's next generation (`"takeover"`, `prior` the lost attempt, `charged` by the rule below)
  naming the same ticket and runs the job under that attempt in a new work directory (RFC
  0003 §5.4 addresses it by attempt), with a new outcome record naming it. A multi-job ticket is split
  instead (below). The loser's work directory is left alone and removed by `collect`. If the loser is
  still running, its next generation check skips its remaining jobs, and its next proof fails and refuses
  its `publish` (§4.3 steps 2, 4 and 6); that narrows the window in which it can publish, and the store's
  first writer wins whatever it publishes (§2.1). Jobs the loser already published are found by the
  taker's `Job.run` lookup and settle as hits.
- **Isolating a job that kills its worker.** A taker of a multi-job ticket splits its jobs without a final
  outcome into two half-tickets by the move order above: each job gets a fresh generation (`"split"`,
  `prior` the lost attempt), so a half's jobs never share an attempt with the loser, with the other half,
  or with any later split of a half, however deep the splitting goes. The taker then releases and claims
  like any worker. **Crash charging follows the call record** (§4.3 step 4). The taker reads the lost
  attempt's `started` mark for the worker token, and that worker's newest call record without `done`. A
  generation is `charged` only when the lost attempt was the only job of that in-flight call and has no
  `expired` outcome. A job that was in the call with others is split further, uncharged; a job the worker
  had gathered or claimed but not yet started, or that was outside the in-flight call (a neighbour of a
  timed-out call), is re-placed uncharged. A job's crash count is the number of its generations with
  `charged` true; at `crashes_charged` (default 2, as senselab's
  profiles) the job is `Failed(JobCrashedWorker)`, a content fault of that job. A job that completes on a
  fresh worker is not charged; its survived losses are recorded. A prior attempt with an `expired` outcome
  was a timeout (§5.3), handled as a resource fault at a call (§6).
- **Who moves a job up the ladder.** The process holding the job's claim: the worker after an
  out-of-memory call or a load fault, the taker after an `expired` attempt (store lookup first). It moves
  the job to the next class in the ticket's `ladder` by the five-step order, `cause: "ladder"` with
  `phase` set, under a fresh generation, using no `Fit` of its own. `resource_moves` is counted from the
  job's generations: those with `cause: "ladder"` and `phase: "call"`. Load moves are bounded by the
  ladder's height and not charged. With the ladder or the budget spent, the holder writes `failed`
  (`ResourceExhausted`).
- **The coordinator dies.** Workers keep running claimed and queued tickets and publish to the store.
  Substrate-started workers exit after `orphan_after_s` with no live coordinator lease (§4.3 step 7). A
  resubmitted run is a new `GraphRunner` over the same store (RFC 0002 §3.1 "Resume"), whose coordinator
  waits until its predecessor's lease is stale (§4.1): jobs whose results were published are settled by
  `find` in `ready()` or by `Job.run`'s lookup; a job whose newest generation names a ticket with a live
  claim is **followed** (no new ticket; the result is read from the job's address, RFC 0003 §7.2),
  whatever its class, because its holder is running. A job whose ticket is unclaimed or has a stale claim
  is followed when the ticket's queue belongs to a class the resumed coordinator starts, or its substrate is
  `ExternalSubstrate`; otherwise the coordinator takes the ticket's claim by `try_take()`, moves the job to
  a ticket of a class it starts (`withdrawn`, reason `"replaced"`, under a fresh generation), and releases.
  A followed ticket of a class nobody starts therefore cannot hang a run, and a claimed ticket is never
  withdrawn under a running holder. A followed claim that later goes stale is handled the same way. The
  follower judges a followed claim's staleness only for a ticket whose settled count has not moved for
  `stale_after`, and at most once per `stale_after`, so its cost stays one listing per queue per poll. On
  resume, before the follow rule, a job whose newest generation names a ticket where it has a `refused`
  withdrawal is re-placed (§3.3). senselab's resume chain (start, SIGKILL of the coordinator, resume on another host)
  completed 1000 of 1000 with at most one execution per key under the follow rule (§9).
- **A worker is preempted.** Slurm's SIGTERM is the worker's SIGTERM: finished jobs are settled, the rest
  released. A SIGKILL is a death.
- **Storage faults** (an `EIO` on the pool directory, a store that refuses a write) are infrastructure
  faults (§6). A corrupt ticket file is `failed` for its jobs as a content fault of the ticket; a corrupt
  outcome record is re-read, then treated as absent, never raised out of the follower (senselab R-278).

## 6. Faults and outcomes

Faults are typed where they happen, by a classifier, and each class has one effect. The table maps them
to RFC 0002 §3.1's outcomes. Budgets are `FaultBudgets`; every default is pydra's choice, and where
senselab's profiles differ the table says so (`rt/data/profiles/laptop.yaml:111-115`,
`slurm.yaml:128-132`).

| Fault | Typed by | Effect inside the pool | Outcome to the runner |
|---|---|---|---|
| content: the task raised, a command exited non-zero, an exception element of a batched call | the classifier's default for exceptions from the task's own code | retried until `content_attempts` (default 1, no retry, as pydra does today; senselab's default is 2), each retry under a `"restart"` generation (§5.5) | errored `Result`, or `Failed` when nothing ran |
| resource at a call: out of memory, a job timeout | name table (`MemoryError`, `torch.OutOfMemoryError` by qualified name, no import), the worker's own cgroup OOM event; a timeout is an `expired` outcome written by the beater when a call's deadline passes (§5.3), acted on by the taker after a store lookup | the residency is reloaded (§4.4); a batched call is bisected; a lone job is moved up the ticket's `ladder` by the claim holder (§5.5) at most `resource_moves` times, counted from its `"ladder"` generations (default 2; senselab's field is `resource_replacements`, default 0), uncharged; a timeout moves only to a class with a larger `time` or another `gpu_type` | `Failed(ResourceExhausted)` once the ladder or the budget ends |
| resource at a load | a load raising the above | an `UnfitRecord` is written for the residency and the class; the ticket's jobs move up the ladder, uncharged; a load is never an item | `Failed(Unplaceable)` when no larger class exists |
| infrastructure: a load that failed otherwise, a broken environment, a lost service, a storage error | anything raised by the pool or an environment's start, and every unclassified exception there (flagged `gap`) | requeued on the same class, uncharged; after `infrastructure_replacements` failures of one residency in the current coordinator epoch (default 3; senselab's default is 6) the residency is held on its queue (`held-<epoch>-<label>`) | `NotRun(held=True)` for each of its jobs, reason naming the residency and the latest cause |
| hardware below a ticket's need: a `gpu_mem=None` class whose worker reads less device memory than the ticket declares | the claiming worker's re-judgement (§3.3, `WrongHardware`) | the worker writes `withdrawn` (reason `"refused"`, the memory read) and releases unrun; the coordinator's follower re-places the jobs from the next class up under fresh attempts, uncharged; later placements use the read memory | `Failed(Unplaceable)` when no larger class exists; otherwise none |
| worker lost mid-ticket | a stale claim lease | taken over under fresh attempts (§5.5); a multi-job ticket split into halves; a lone job lost `crashes_charged` times | `Failed(JobCrashedWorker)` (content) |
| workers of a class never start | missing ready marks | backoff; after `start_failures` (default 7) the class is broken | `NotRun(held=True)` for its queued jobs |
| cancel, `fail_fast` | the driver | the coordinator takes each unsettled ticket's claim by `try_take()`; an unclaimed or stale ticket gets `withdrawn` (reason `"cancel"`) for its unsettled jobs; a ticket with a live claim is left to its holder, whose results stay in the store for a later run | `NotRun(held=False)` for every cancelled job, at once |
| policy refusal raised by a task or hook | the task | none | `Failed` per job |

- **A held job is not a failure.** Readers stay blocked, the run ends `RunIncomplete` (RFC 0002 §3.1), and
  a later run (a new coordinator epoch) runs the held jobs; a hold counts only for its own epoch or a newer
  one, so a stale coordinator can never hold a newer one's work.
- **A fault's class survives every wrapper.** A load failure that carries a resource record stays a
  resource fault; a wrapper that re-raises keeps the original's record. A record attached to an
  exception (`faults.attach`), or to any exception of its `__cause__`/`__context__` chain, is the fault
  whatever classifier is in use, and the default classifier looks for out of memory and `WorkerLost`
  along that chain. senselab found a load-time OOM
  turned into an infrastructure fault by a wrapper, which held a healthy environment instead of moving to a
  larger class (R-287, §9).
- **One bisection engine** for local and pool execution (§4.4), with two triggers: a resource fault, and a
  crash (`WorkerLost`, a protocol error, a fatal raised by a request rather than a load). A load is never
  split or retried as a request. An exception element of a batched return is never a trigger. Each
  trigger reloads the residency first, and a reload is a load (a fault there is the residency's, never
  the call's). A lone job's crash inside a live worker is run again under a fresh `"restart"` attempt,
  uncharged; lost `crashes_charged` times in a row it is `JobCrashedWorker`. A crash whose jobs all
  completed afterwards was survived: it counts as an infrastructure fault of the residency, charged to
  no job (senselab's `_survived`), so an environment that crashes intermittently is held.
- **The pool's machinery is not the job.** Only the task's own exception and a load's are classified;
  an error of the pool's machinery around a call (a storage error starting the attempt or writing the
  call record) is an infrastructure fault of the residency, and a `KeyboardInterrupt` or `SystemExit`
  from it is not typed at all.
- **OOM attribution is per worker.** A SIGKILL is a resource fault only when the worker's own cgroup
  recorded an OOM kill after the worker's latest baseline, taken at start and after each provider start;
  an OOM in a sibling process of the same Slurm job does not turn the next `scancel` into an OOM (senselab
  R-294). On Slurm the worker's cgroup is its step's, whose `memory.events` `oom_kill` is readable.
- **The classifier is a hook**: `classify(exc, phase: Literal["load", "call", "pool"]) -> FaultRecord`.
  The default treats an exception from the task's own code as content (pydra's behaviour today) and an
  unclassified exception from the pool's machinery as infrastructure with `gap=True`, reported once per
  type. A client with a stricter rule (senselab's: every unclassified exception is infrastructure) passes
  its own.

## 7. Composition with the `GraphRunner`

### 7.1 `ready()`, `complete()` and the drivers

- `WarmPool` is an RFC 0002 §3.2 executor with `batches = True`: the asynchronous driver hands it every job
  of a round routed to it, in node order, and it turns them into tickets per queue (§4.2). Each job is
  reported exactly once, as `Result`, `Failed` or `NotRun`, from the follower thread through the bound
  callable.
- `checks_store = True`: workers run `Job.run`, which looks up and publishes. The runner may use
  `lookup_before_dispatch` or not; a hit found by a worker settles as a hit either way.
- **No limit for a batching executor.** RFC 0002 at `6a9ececb` applies `limit` by whole node groups and
  states that a driver of a `batches = True` executor passes no cap (RFC 0002 §3.1, `ready()`). This
  proposal asks that R7's drivers enforce it, not only document it: `drive()` passes no `limit` to a round
  routed to a `batches = True` executor. A pool caps itself (workers, quantum, `max_queued_jobs`).
- **Same-checksum twins** are folded by the runner (RFC 0002 §3.1 at `6a9ececb`, "Sharing is the
  runner's") before they reach the pool; the pool's follow index (§5.5) folds what remains across runs.
- **`retry(job)`** of a `NotRun(held=True)` job hands it out again; the pool puts it on its queue only once
  the hold's epoch has passed or the hold is lifted (`pydracli pool release-hold`).

### 7.2 External drivers

A system that drives the runner itself (RFC 0002 §3.5) uses `WarmPool` as it would any executor:
`bind(report)`, `submit(rnd.dispatch)`, and `runner.complete` on each report on its own thread
(`GraphRunner` is single-threaded, so reports are marshalled to the driver's thread, as the asynchronous
driver does). Nothing in the pool depends on `Submitter`.

### 7.3 Outcomes and provenance

- Every executed job's provenance is written in the worker by the store's `on_commit` (RFC 0003 §7),
  before the entry is visible. `Round.settled` gives the runner's `reused`, `absent` and `workflow-stored`
  records (RFC 0002 §3.4) unchanged.
- The pool's own events (claimed, taken over under a new attempt, held, moved up the ladder, unfit
  recorded, reloaded, re-check result) go to the audit hook as messages with the job's checksum and
  attempt, so a client can log every job's path through the pool without reading the pool directory.

### 7.4 `plan()`

`place` is not part of RFC 0002's `Executor` protocol, and `plan()` is the runner's (R8). This proposal
asks R8 to name one optional executor method, `place(jobs) -> list[Placement | Unplaceable]`, found with
`getattr` as `read_succeeded` is. When the executor a node is routed to has it, each `PlanNode` carries
the placement: class, queue, every class tried with its verdict and numbers, and the batch. An
unplaceable node shows as such before anything runs.

## 8. Identity

Warm state must never reach a key, a lookup context or an output.

- **Keys.** A job's checksum is computed in the submitting process (RFC 0002 §3.4) from its inputs and
  identity record (RFC 0004 §3.1). A provided field's spec is an input value, digested once by RFC 0004
  §6.2, and `ResidentKey.spec` is that digest (§3.1). An environment's resolved reference is already keyed
  for shell tasks (RFC 0004 §5.3). The pool adds nothing: no worker, class, batch size or cold or warm flag
  enters a key. Placement never changes a checksum (senselab pins this: a plan's hash is
  placement-independent, `fit_test.py`, §9).
- **Device is not keyed.** A result computed on CPU and one computed on CUDA share a slot, as today: the
  environment and hardware are recorded, not keyed (RFC 0004 §3.2). The reuse policy decides whether to
  serve across them.
- **Lookup context.** Built once in the submitting process and carried with the job (RFC 0003 §4); the pool
  knows nothing of it.
- **Execute context.** The worker adds one section, `pool`, to `context(job, "execute")`:

  ```json
  {"pool": {"scheme": "pool-facts@1", "worker": "<token>", "host": "<host token>",
            "class": "gpu-h100", "attempt": "9c41e07a5b2d.3",
            "hardware": {"device": "cuda", "name": "…", "memory_mib": 81559,
                         "compute_capability": "9.0", "cpus": 8},
            "residency": [{"key": "<digest>", "kind": "provider", "name": "hf-model",
                           "version": "<provider version>", "state": "warm", "since_load": 37, "load_s": 14.6}],
            "batch": {"size": 16, "index": 3}, "losses_survived": 1}}
  ```

  `version` is the provider's, and for an environment the entry carries its resolved reference, so a
  result is traceable to the code that loaded its model. The provider's module enters the key only when
  the task's code reaches it (RFC 0004 §4.1).
- **Recorded-only.** RFC 0003 publishes an entry under the execute context's fingerprint, which excludes the
  identity's `recorded_only` fields (RFC 0003 §4). `pool` must therefore be recorded-only, or every worker
  and every batch size would split one result into many slots. Two parts:
  - **W11a**: the `pool` section and the audit messages; `WarmPool` refuses a store whose identity does not
    list `pool` in `recorded_only` (`PoolFactsWouldSplitSlots`). A client identity (senselab's, under RFC
    0003 P9's adoption path, plan task P9) lists it itself. Under `pydra-identity@1` roots and
    `PickleStore`, whose slot is a hash of the whole context, the facts go to the audit record and
    `Result.runtime` only.
  - **W11b**: `pydra-identity@2` lists `pool` in `recorded_only` from its first release; it ships with RFC
    0004 I8 so the header changes once. `recorded_only` is empty today (integration `identity.py:74`).
- **Outputs.** A result computed cold equals one computed warm, and one computed in a batch equals one
  computed alone. This is the task author's statement for batched calls and warm environments (§3.1,
  §4.4); the sampled re-check measures it; the pool's tests pin it for pydra's own tasks (§11).
- **Context cost.** RFC 0004 I7 resolves tools per job in the submitting process. A worker never recomputes
  a lookup context; it computes the `pool` section from state it already holds.

## 9. Prior evidence

What this proposal reuses from senselab's execution layer, with its source. `R-*` are rows of senselab's
`specs/20261030-140022-senselab-2x-api/review-register.md` and its adversarial reviews; `T24-*` are its
Phase 3 Task 24 fixes; job numbers are ORCD scheduler ids recorded in its `reviews/phase3-exit-evidence.md`.
Timings were taken on shared nodes and are upper bounds. **Every figure below was measured through
senselab's mechanism**: a host process driving a backend over its wire protocol in a subprocess worker,
items as JSON plan fragments. This proposal's mechanism (in-process or process-owning providers, a JSON
ticket and a payload file per job) is expected to cost no more but is unmeasured; W3's and W6's benchmarks
measure it and their results are added here (§11).

**Measured.**
- **Model loads and the first request** (`reviews/t20r10-evidence.md`, one H100, 1000 items per unit):
  load 14.6 s (a Whisper-class ASR), 8.9 s (speaker embeddings), 26.5 s (diarization), 23.5 s (a
  transformer PII tagger, 64.5 GB resident), 0.05 s (a second ASR whose first request loads). Warm per-job
  worker time 0.013 s (speaker embeddings), 0.038 s (diarization), 0.34 s (ASR), 1.74 s (second ASR),
  8.36 s (PII tagger): 0.013 to 8.36 s over five units. The first request after a load took 7.8 to 30.3
  times the steady prediction; requests from the second on were within the steady band. → §1, §4.2.
- **Batching** (same source): a batch of 19 was 2.3 times faster per job than a batch of 1 for the ASR
  unit, with device peak 1.69 → about 3 GB (6.4 MB per padded second); speaker embeddings 1.6 to 2.3 times;
  diarization at most 10 % faster, and slower at a batch of 90 on the longest stratum (padding 2.3, peak
  9.05 GB). → §4.4: batching is per task and sized by measurement, not a default.
- **Per-item overhead** (`reviews/phase3-exit-evidence.md`, R-253, job 28564898): 0.098 s for a one-job
  item, rising 7.29 ± 0.46 ms per added job; 0.10 to 0.26 s across every measured unit. → §4.4.
- **Pool against in-process** (Step 2): 1000 CPU items in 291 s in-process and 80.7 s on a two-worker local
  pool. This is the pool's amortisation of per-item overhead against an in-process baseline that carried it,
  not a scaling figure. → §1.
- **Cold start, separately from throughput.** Step 4 (job 28580776): 426 s from start to the first commit of
  a one-worker H100 run (worker queue wait, environment and load), nothing committed before. Step 3 (job
  28578691): 1000 GPU items committed in 1329 s by one H100 worker. → §1, §4.5 (cold start is per worker).
- **A 30 s window after calibration** at 100 items was scheduling latency (worker start and a 30 s poll), not
  per-item cost (senselab Phase 3 ledger, T20R). → §4.5, §5.1 (`poll_s` and scaling are measured apart from
  per-job cost).
- **Resident memory, `m0`, `m1`** per unit on H100, and the fit rule `resident + m0 > 0.9 × grant`
  (`calibration-probe.md`, ADR-009 amendment 2026-11-12, `rt/fit.py`, `rt/probe.py:359-402`). → §3.3.

**Behaviour proven on the cluster.**
- **Abrupt stop and resume** (Step 4, jobs 29047613, 29047616, 29055900): interrupted at 335 of 1000,
  resumed on a different host; executions 663, reuses 337, no key executed twice, no item claimed twice;
  provenance 9/9. → §5.5.
- **Cross-host leases** (job 29047619): lease and queue checks on three hosts over NFS, overlapping holds and
  lost items. → §5.3; ported as `test_claim_lease_cross_node` (W12).
- **Load OOM on a typed GPU** (job 29047615): the load-OOM path records `unfit` and the next placement avoids
  the class (`pool_unfit_test.py`). → §3.3, §6.
- **Typed grants and QOS**: four coordinators filled a partition's per-user submit limit of 4; every worker
  `sbatch` was refused; the class was broken after 7 starts with a message naming the profile values (job
  28564896). GPU workers on a preemptable partition sat in the scheduler for hours; one worker per class on
  a capped partition was the workaround. → §4.5 (`can_start`; failed starts hold, they do not fail jobs;
  latency is unbounded when no worker is granted).

**Failure modes found and fixed, which the design and tests start from.**
- T24-9: a `SoftFileLock` left by a SIGKILLed coordinator could be broken only on its own host; a resume
  elsewhere waited until its 6 h limit. → §5.3, W4.
- R-273 / T24-2: a resumed coordinator re-cut unfinished jobs into new items while old items were queued; 61
  jobs ran twice. Fix: follow the live item that holds a job; settle each request's items before the next.
  → §4.2, §4.3 step 5, §5.5.
- T24-2: items settled before their measurement was recorded lost their measurements. → §4.3 step 5.
- T24-10: the last claim's provenance was written after its commit and lost at stop. → §5.4.
- R-277: the stale bound that judged claims was the process's, not the run's; senselab's `LeaseMismatch`
  closed it for the lease section. → §4.3 step 1, §5.3 (all four settings).
- R-283: the epoch was taken before the fallible steps of start. → §5.3.
- R-286 (open in senselab): a Slurm submit retry could double a worker. → §4.5 narrows it.
- R-287, R-289, R-292: fault classes lost through wrappers, hand-typed failures, two server paths
  disagreeing. → §6.
- R-288: two bisection engines, neither attributing a crash to its item. → §4.4, §6.
- R-290, R-285: unsynced exclusive writes; a short write that poisoned a log. → §5.1.
- R-294: cgroup-wide OOM attribution. R-295: a stop that did not wait for its process group. → §3.1, §4.3,
  §6.

**Rationale, not a found defect.** senselab encodes speaker-embedding items alone in its parity references
with 1.x, because 1.x zero-pads a batch to its longest item (`benchmarks.md` amendment 2026-11-03); its own
measures recorded "repeat and batch differed by 0" (`config-derivations.md:379`). → §4.4, §8.

**Tests reused as specifications** (senselab `packages/senselab-core/tests/runtime/`, 144 tests across 14
files at `4472cfbb`): `execution/queue_test.py` (oldest-first, idempotent put, one outcome, expired claim
reclaimed, double claim commits once, lost reply on the outcome link, claim cost flat in queue size, never
pickled), `execution/pool_worker_test.py` (idle exit, quantum yield, gather once warm, resident memory out
of per-item samples, device memory once per claim, failing batch bisected, settle before the next request,
reclaimed logged, claim renewed during a long item, SIGTERM releases, OOM kill is resource, a broken
environment costs one load per claim, a lost claim abandons its request), `execution/worker_set_test.py`
(bounded LRU, busy never evicted, evicted process closed, failed client replaced),
`execution/substrates_test.py` (QOS cap, class to `sbatch` args, one CUDA worker per device, failed starts
back off and stop, workers that never start hold their jobs, scoped job names, logs under the private
directory), `execution/pool_follow_test.py`, `execution/abrupt_stop_test.py` (20 boundaries × 2 host
shapes), `execution/resume_test.py`, `execution/pool_unfit_test.py`, `execution/bisection_engine_test.py`,
`execution/fault_class_test.py`, `fit_test.py`, `locks_test.py` and `probe_lease_cross_node_test.py`. W
items port them (§11).

## 10. Migration

### 10.1 In pydra

- **Unchanged:** every existing worker, `Submitter`'s defaults, checksums, store layouts, `debug` semantics,
  environments' cold `execute`.
- **New and opt-in:** `DirectoryLease.at` with `beat="external"`, `LeaseBeater` and `BeaterClient` (L0); `pydra.workers.pool` (`WarmPool`, `WorkerClass`,
  `LocalSubstrate`, `SlurmSubstrate`, `ExternalSubstrate`, `DeclaredFit`, `MeasuredFit`,
  `DirectoryCalibration`, `FaultBudgets`, `PoolSettings`, `PrivatePickle`, `JobCodec`); `pydracli pool
  {worker, classes, probe, status, release-hold, collect}`; `Provider`, `Provided`, `field(provided=)`,
  `ResidentSet`; `Environment.warm()`, `WarmEnvironment`; `ComputeResources.gpu_mem` and `.residency`;
  `python.define(batched=, units=)`.
- **Asked of RFC 0002 R7/R8:** `drive()` passes no `limit` to a `batches = True` executor (§7.1); R8 names
  the optional `place` method (§7.4).
- **Superseded proposals:** 0001's I and the provider and batching parts of H. 0001 §3.6's prerequisite H0
  (`chdir=False`) has merged and is what lets one process run many jobs.

### 10.2 senselab

senselab 2.x has no compatibility requirement: its executor, pool and worker APIs, its keys and its store
may change to adopt this design, and it may pin pydra by commit or vendor it before a release. It switches
in one pull request, with no period in which both implementations run.

**Deletes** (`rt/`, `ex/`): the pool controller and scaler (`ex/pool.py`), the claim loop
(`ex/pool_worker.py`), the queue (`ex/queue.py`), the epoch (`ex/epoch.py`), the substrates and fleet
(`ex/substrates.py` `Local`, `Slurm`, `Fleet`, `scale`), the `WorkerSet` checkout and eviction
(`ex/workers.py`), the placement walk (`rt/fit.py` `judge`, `choose`, `larger_class`), the bisection engine
(`ex/job.py`), its `Executor` protocol (`admit`, `release`, `fragment_size`, `bind_epoch`, `PlanFragment`,
`ResultRef` handles) with `ex/in_process.py`, and the runner bridge the earlier architecture review lists
(`ExecutorWorker`, `Follower`, `serve_shared`, `_upstream`). Their tests leave with them, after their cases
are ported into pydra's (§9, §11).

**Keeps, as a client of pydra:**
- its backends as pydra tasks whose model is a provided field;
- **one process-owning provider, `senselab-worker@3`**, which starts a `WorkerClient` subprocess in a
  backend's locked environment and speaks wire protocol v3 to it (`ex/workers.py:330-341`,
  `rt/worker_client.py`), with `measure` by the child's PID and `close` stopping its process group; and an
  in-process provider for its `host` kind (`rt/host_worker.py`, `ModelCache`). Models are loaded by a
  resolved 40-hex revision (its pinning rule). Its server backends are the same provider owning a server
  private to the worker; its per-host `ServiceRegistry` (`ex/services.py`) stays inside that provider until
  a later RFC moves service sharing into pydra;
- its calibration probe (deliberate strata runs, batch sizing per stratum), writing pydra
  `DirectoryCalibration` records, and `MeasuredFit` with its site headroom; `recheck_fraction` on in
  calibration runs;
- its profiles, loaded into `WorkerClass` lists, `FaultBudgets` (with its own defaults: `content_attempts`
  2, `infrastructure_replacements` 6, `resource_moves` 0) and `PoolSettings`; its `can_start` for the
  cluster's GRES and submit caps;
- its fault classifier (every unclassified exception is infrastructure) as the `classify` hook;
- its store identity and entry format over `PortableStore` (plan task P9), listing `pool` as recorded-only;
  its licence gate in `Reuse.verdict`; its PROV mapping in `on_commit`, reading the `pool` section;
- its run directory and event log (its ADR-008), fed from the driver's reports and the audit messages.

**Switch-over.** After the senselab-needed W items (§11) are integrated and S1 (its repin after R7) has
landed: senselab's driver builds `WarmPool(root=<run>/pool, ...)` from its profile, routes model nodes to
it and the rest to `debug` or `cf`, and deletes the modules above. Keys may change once (a store recompute
is acceptable before 2.0.0). Proof: its ported ADR tests, then the GPU cases of W12, then its own Phase 3
exit steps 2 to 5 rerun at the pin.

## 11. Delivery

| PR | Contents | Depends on | senselab-needed |
|---|---|---|---|
| L0 | `DirectoryLease.at(lockfile, settings, beat=)` in `engine/lease.py`, with `beat="external"` taking without a heartbeat thread and exposing `token_dir`; `LeaseBeater(on_expire, kill_after)` (a process that beats many token directories, with deadlines and payloads) and `BeaterClient` (`hold`, `release`, `deadline(token_dir, at, payload)`, `ack_expired`; `lost`, `expired` back; acknowledgement timeout of `renew_interval` = beater death) (§5.3); `LeaseSettings` serialised for `pool.json` and compared on its four staleness fields, `max_wait` excluded. Tests: a named lease takes, beats, goes stale and is taken over through `try_take`; two settings differing only in `attribute_delay` compare unequal, two differing only in `max_wait` compare equal; `test_external_lease_starts_no_thread`; `test_lease_beaten_from_another_process` (taken in process A with `beat="external"`, beaten by a `LeaseBeater` in process B, proved in A beyond 3 × `stale_after` under fast settings); `test_beater_reports_loss` (the token directory removed by a taker: B sends `lost`, A's `prove()` is `False`); `test_release_is_not_a_loss` (A releases after B's acknowledgement; no `lost`); `test_deadline_stops_beating` (B stops at the deadline, sends `expired`, and a third process takes the lease over by `try_take` after `stale_after`); `test_beater_eof` (B killed: A sees end of file); `test_deadline_payload_reaches_callback` (the payload sent with `deadline` is what `on_expire` receives, before `expired` is sent); `test_ack_timeout_is_beater_death` (a beater that stops answering: `hold` times out after `renew_interval` and A recycles); `test_kill_after_unacknowledged_expiry` (with `kill_after` set, no `ack_expired` kills the holder; unset, it never does) | #928 (merged), RFC 0002 R6 (`try_take`) | yes |
| W1 | `WorkerClass` and its loader; `ComputeResources.gpu_mem`, `.residency`; `place()`, the ladder (across GPU types when unnamed), `Fit`, `DeclaredFit` with `time`, `ClassTried`, `Unplaceable`, `CanStart`. Tests: oversized job placed on the larger class; `test_ladder_crosses_gpu_type_when_unnamed`; the ladder stays on a named type; `test_time_exceeds_class_is_unfit`; `gpu_mem=None` places as unknown; fits nothing → `Unplaceable` naming every class and number; GPU request never placed on CPU; a class `can_start` refuses is skipped and recorded; placement leaves the checksum unchanged | 0001 F1 (merged) | yes |
| W2 | `ResidentKey` over RFC 0004 §6.2 digests; `Provider`, `Provided[T, name]` as `Annotated`, `field(provided=)`; `ResidentSet` (LRU, host and device budgets, busy never evicted, `close` on evict); providers under `debug`. Tests: provided spec keyed and object not; the residency digest and the identity record use one digest; eviction order; a busy resident survives a tighter budget; `debug` loads once across two `Submitter` calls; unregistered provider is a wiring error | 0001 H0 (merged), RFC 0004 I0 | yes |
| W2b | Process-owning providers: `load` that starts a process, `close` that stops its process group and waits, `measure` by PID, the cgroup baseline after each start. Tests: a provider whose `load` spawns a child; eviction kills the whole group and waits; a child's OOM is attributed to its call; a stopped provider's GPU memory is free before the next load | W2 | yes |
| W3 | `Environment.warm(roots, device)` and `WarmEnvironment` for `Docker`, `Singularity`, `Lmod`; binding roots from `get_bindings`; restart once per new root up to `max_mount_roots`; isolation defaults; GPU flags at start; `max_jobs`; `setup`/`teardown` called; `bench_warm_environment` (`docker run` against `docker exec`, `apptainer exec` against an instance, 200 commands each). Tests (fakes for `docker`, `apptainer`, `lmod`): one `docker run -d`, N `docker exec`; one `module load` resolution for N commands; job A writes `$HOME/x` and `/tmp/y` and job B in the same instance sees neither; outputs owned by the submitting user; a job whose root was not mounted triggers one restart, not a cold run; beyond `max_mount_roots` the job runs cold; warm and cold outputs equal; `Native` returns `None`. Real-daemon Docker tests on the ubuntu runner behind the existing marker; Apptainer on ORCD | RFC 0004 I6 (plan tasks I6a, I6b), W2 | no |
| W4 | Pool storage: `pool.json`, queues, tickets with roots, `O_EXCL` slot files, follow generations (`follow/<ck>-<fp>/<g>`) and pool attempts `<pool id>.<g>`, the move order (§5.5), claim leases on L0, per-job-and-attempt outcome records with uuid read-back and `withdrawn` reasons, settled marks with recount on `EEXIST`, holds, `losses/`; `PrivatePickle` with the ownership walk and digest, `JobCodec`; `pydracli pool probe`. Tests ported from senselab's `queue_test.py`; a counting fake `StorageDir` asserts every index, outcome and settle write syncs file and directory; a lost `link` reply is read back as this write's; `ESTALE` reads as absent; no symlink is made; a pickle in a group-writable directory is refused; a ticket never contains a pickle; `probe` refuses a root without `O_EXCL` create; a lint test that no pool module takes a host-bound lock; `test_settle_count_race` (two writers computing one count: both outcomes covered by a written mark); `test_generation_allocation_race` (two writers of one job's next generation: distinct numbers, distinct attempts); `test_move_never_unheld` (a move killed after each of its five steps: a coordinator reading the index always finds a ticket holding the job, and puts nothing twice); `test_split_of_split_attempts_distinct`; `test_withdraw_needs_claim` (`try_take` on a live claim fails and nothing is written); `test_outcome_first_writer_wins` (`expired` then `done` for one attempt: `expired` stands); `test_collect_keeps_highest_generation`; tickets carry `ladder` | L0, RFC 0003 P1, RFC 0003 P4 (plan task P4c) | yes |
| W5 | The worker without warm environments: the beater subprocess over L0's `BeaterClient` (leases taken with `beat="external"`, `lost` handling, end of file recycles), call deadlines and recycling on `expired` (§5.3), the generation and `withdrawn` check before each job, claim loop (oldest first, warm preference, quantum, gather once warm), hardware check and `gpu_mem` fill, ready mark, proofs, `Job.run` with `chdir=False` under the pool attempt with the claim proved before `publish`, per-job settle after measurement, `Sample`s, SIGTERM drain, idle and orphan exit, process-group stop; `pydracli pool worker`. Tests ported from `pool_worker_test.py`; `test_gil_bound_call_keeps_claim` (a task holding the GIL for 3 × the renewal interval under fast settings keeps its claim); a settings mismatch in any of the four fields refuses to start; a mis-typed GPU grant fails the start; `test_wedged_call_times_out` (an in-process call that never returns past its deadline: `expired` outcomes, the worker killed, the job taken over under a fresh attempt and typed a resource fault); `test_worker_skips_withdrawn_and_superseded`; `test_released_started_job_gets_fresh_attempt` (SIGTERM during a subprocess call, release, reclaim: a `"restart"` generation and a new work directory); a content retry runs under a new attempt; the call record is written before the call and `done` after; `test_no_heartbeat_thread_in_worker`; a `WrongHardware` re-judgement writes `withdrawn` (reason `refused`) and runs nothing | W2, W2b, W4, RFC 0003 P8 | yes |
| W5b | Warm environments in the worker: shell tasks through `WarmEnvironment`, restarts on new roots, the cold fallback. Tests: a shell task in a warm Docker worker equals `debug`; a ticket with a new root restarts the environment once | W3, W5 | no |
| W6 | `WarmPool` and `LocalSubstrate`: placement at submit, tickets per queue, one live coordinator per pool (`coordinator_wait`, `PoolBusy`, `PoolFenced`), follow (live claims always; unclaimed or stale tickets only for classes this coordinator starts, otherwise moved under `try_take`), the follower over settled marks and newest generations, re-placement after a `refused` withdrawal, read-back through `Job.result()`, outcome mapping, `cancel` → `NotRun`, `max_queued_jobs`, scaling with `can_start`, one worker per GPU, failed starts, lifecycle rules; `pydracli pool {status, release-hold, collect}`; `bench_pool_overhead` (10⁴ trivial jobs on `LocalSubstrate`: per-job wall and coordinator write time per job, `PrivatePickle` against a JSON codec). Tests: a 100-element round becomes tickets of at most `quantum_jobs`; resubmission follows live tickets and runs nothing twice; an unclaimed ticket of a class the new coordinator does not start is moved to a class it starts under fresh attempts; `test_claimed_ticket_not_withdrawn` (a live claim of a class the new coordinator does not start is followed, and its holder's work directory is never shared); `test_resumed_runner_attempt_zero_no_collision` (a fresh runner numbering from 0 gets distinct work directories); `test_rerun_after_collect_fresh_attempt`; a fenced coordinator's `close` stops no worker; a `refused` job whose coordinator died is re-placed on resume; followed claims are judged only for tickets idle for `stale_after`; a `refused` ticket is re-placed one class up; `test_job_outlives_stale_after` (a job sleeps past `stale_after` with its beater stopped: takeover under a new attempt, two distinct work directories, one entry, the loser's `publish` refused); `test_follower_cost_10k_jobs` (one listing per queue per poll, by a counting fake); a pool whose workers never start holds its jobs (`RunIncomplete`); a second coordinator on a live pool waits, and with `coordinator_wait` raises `PoolBusy`; two coordinators racing for a free pool: the older is fenced (`RunIncomplete`, `PoolFenced`) and nothing runs twice; cancel withdraws only unclaimed or stale tickets; a hand-started worker survives the absence of a coordinator; `test_plain_workflow_untouched` (`"pydra.workers.pool" not in sys.modules` after `import pydra` and a `cf` run; no new files) | W5, RFC 0002 R7 | yes |
| W7 | Faults: `classify` and the default table, `FaultBudgets`, one bisection engine with resource and crash triggers, reload after a resource fault, ladder moves, crash counting and `JobCrashedWorker`, holds per residency and epoch, timeouts from `expired` outcomes (bisected when batched, the ladder for a lone job), half-tickets with fresh generations. Tests ported from `bisection_engine_test.py`, `fault_class_test.py`, `pool_backend_oom_test.py`; `test_resource_fault_reloads_residency`; a load OOM moves up and charges nothing; a class survives a wrapper; a sibling's OOM does not type a `scancel`; a process-owning provider restarts its process after a resource fault; an `expired` prior attempt is a resource fault, not a crash; `test_timeout_kill_spares_gathered_jobs` (a worker holding three gathered claims killed by one call's timeout: only that call's job is typed, the other two are re-placed uncharged); `test_expired_with_committed_result_settles_done` (a call that publishes during the grace period: the taker's lookup settles `done`, no ladder move); `test_resource_moves_counted_from_generations` (a resumed coordinator and a second taker see the same count); a timeout does not move to a class with the same `time` and `gpu_type` | W6, W2b | yes |
| W8 | `SlurmSubstrate`: one `sbatch` per worker with typed `--gres`, scoped names, a timed-out submission looked up before any retry, the shared `squeue` poller, logs under `private/`, `can_start` with a CPU cap default. Tests ported from `substrates_test.py` with a fake `sbatch`/`squeue`; a submission that times out and lands is not doubled; marked Slurm tests in the existing Slurm CI container | W6, RFC 0002 R5 | yes |
| W9 | Calibration: `Sample`, `UnfitRecord`, `TimeRecord`, `ParityRecord`, `DirectoryCalibration` (default under the root), `MeasuredFit` (`judge`, `batch` by the stated rule); `recheck_fraction` through `Verify`, writing `ParityRecord`s; `place` for `plan()`. Tests ported from `fit_test.py`, `pool_unfit_test.py`; an unfit record bound to a provider version stops binding after a version change; a re-check mismatch writes a parity record and one audit message; `plan()` shows the classes tried | W7, RFC 0002 R8, RFC 0003 P5 (`Verify`) | yes |
| W10 | Batched calls: `python.define(batched=, units=)` returning `list[T \| BaseException]`, grouping by units, bisection through W7. Tests: a batched call's results equal the unbatched ones for pydra's own test tasks; `test_content_fault_in_batched_call_fails_one_job`; one crashing item isolated in log₂ n calls; a half that completed keeps its results | W7, W2 | later |
| W11a | The `pool` context section (with provider `version` and the attempt), audit messages, `PoolFactsWouldSplitSlots`. Tests: cold and warm results share one slot under an identity that lists `pool`; two workers' results share one slot; a store whose identity keys `pool` is refused; under `@1` and `PickleStore` the facts reach only the audit | W5, RFC 0003 P5 (`on_commit`) | yes |
| W11b | `pool` in `pydra-identity@2`'s `recorded_only`. Test: a default `@2` root accepts a `WarmPool` without configuration | RFC 0004 I1, ships with I8 | no |
| W12 | ORCD suite and documentation (below) | W2b, W8, W9, W11a; shell cases W5b | GPU cases yes; shell and Lmod cases no |

**Order.** L0, W1 and W2 can start at once when based on the integration branch (`d7b0988f`), which carries
RFC 0002 R6 (`try_take`, which L0 needs) and RFC 0004 I0 (which W2 needs). Based on
`enh/execution-extensions` (`68090364`), W1 can start at once, L0 waits for R6 and W2 for I0; `prove()`
is already there. W4 follows L0, RFC 0003 P1 and RFC 0003 P4 (plan task P4c); W5 follows W2, W2b, W4 and
RFC 0003 P8 (§5.4). Neither P4c nor P8 is on the integration branch yet, so both must be scheduled before
senselab's switch-over. senselab's path is L0, W1, W2, W2b, W4, W5, W6, W7, W8, W9, W11a, then W10 later;
none of it waits for RFC 0004 I6 or I8.
W3 and W5b wait for RFC 0004 I6 (plan tasks I6a, I6b); W11b rides the I8 roll-up so the `@2` header carries
`pool` from its first release. RFC 0002 R7 (`NotRun`, `attempt`, `RunIncomplete`, and the no-limit rule of
§7.1) gates W6; R8 gates W9's `plan()` part. senselab's switch-over (§10.2) waits for S1 and the
senselab-needed set only. Every PR has tests on `debug` and `cf` where they apply, none changes an existing
test's expectation, and each benchmark's result is added to §9 in its PR.

**ORCD suite (W12).** Submitted jobs from a commit-pinned checkout, typed `--gres=gpu:h100:<n>`, never run on
a login node; each reports a verdict line per check.
- `test_one_load_n_jobs_h100`: one `SlurmSubstrate` worker, 1000 jobs of a provided-model task; asserts one
  `sbatch`, one load (the provider's load counter), every job committed, and per-job wall after the load
  within the calibration's band.
- `test_process_provider_warm_h100`: the same with a process-owning provider (a model served by a child in a
  separate locked environment); asserts one child start and that eviction leaves no process and no device
  memory behind.
- `test_two_residencies_two_queues`: two models on one class; asserts two queues, the quantum yield, and at
  most `max_workers` loads per model.
- `test_worker_killed_mid_ticket`: `scancel --signal=KILL` of a worker after a third of the jobs; asserts
  takeover after `stale_after` under new attempts, every job committed once, losses recorded.
- `test_claim_lease_cross_node`: ported from senselab's `probe_lease_cross_node_test.py`; three nodes, the
  holder SIGKILLed, exactly one takeover.
- `test_coordinator_killed_and_resumed_on_another_host`: SIGKILL of the submitting process, resume from a
  second node; asserts follow (no job put twice), executions + reuses = jobs.
- `test_load_oom_climbs_the_ladder`: a class whose usable device memory is capped (by
  `torch.cuda.set_per_process_memory_fraction`) below the model; asserts an `UnfitRecord`, the move to the
  larger class, no job charged, and that the next `plan()` starts at the larger class.
- `test_local_multi_gpu`: `LocalSubstrate` inside one allocation of two H100s; asserts one worker per device by
  `CUDA_VISIBLE_DEVICES` and both busy.
- `test_mistyped_grant_refused`: a class naming `gpu_type="a100"` on an H100 grant; asserts failed starts, then
  `NotRun(held=True)` and no job failed.
- `test_warm_apptainer_shell_jobs` (pydra-only): 500 shell jobs of one Apptainer image (a CPU tool) on one
  worker; asserts one instance start, outputs equal a cold `debug` run, no write visible across jobs, and
  records wall per job warm against cold (`bench_warm_environment` on ORCD).
- `test_lmod_resolved_once` (pydra-only): 200 shell jobs under an `Lmod` environment; one resolution per worker.
- `test_plain_workflow_untouched_slurm`: the existing Slurm tests with the pool absent; no
  `pydra.workers.pool` import, no new files.

**Documentation.** A how-to for a shell workflow on a warm Apptainer pool, one for a GPU model on Slurm, one
for writing a process-owning provider, and the worker protocol as a reference page.

## 12. Decisions and open questions

**Decided in this revision** (from the architecture review; the owner ruled on the three marked ‡):

| Question | Decision |
|---|---|
| Home | Core, `pydra.workers.pool`, imported only when named (tested by `sys.modules`). The core-type hooks cannot be a plugin; the Slurm poller and the lease are core. |
| `provided=` | The field attribute, with `Provided[T, name]` as an `Annotated` marker. The spec is an input (keyed), `debug` resolves it in-process, and `TypeParser` sees `T`. Per-node binding would put a keyed value outside the task's inputs. |
| Warm-environment isolation | No periodic restart by default. Defaults: fresh work directory, per-job `TMPDIR` and `HOME`, non-root user, `--containall`/`--no-home` for Apptainer, `--read-only` when `read_only=True`; `max_jobs` as an opt-in restart for tools known to leak (§3.1). |
| Cold re-check ‡ | Yes, `PoolSettings.recheck_fraction` (default 0) through RFC 0003 `Verify`; each result is written to the calibration store as a `ParityRecord`, and a mismatch is a content fault of the task reported once through audit (§4.4). |
| Content retries | `content_attempts=1`; retries opt-in through `FaultBudgets`. senselab's default is 2. |
| Services ‡ | Outside this RFC. A provider may own a server private to its worker. senselab's per-host `ServiceRegistry` stays inside its provider; a later RFC moves service sharing into pydra when a second client appears. |
| Group pools | Single-user now. Preconditions recorded: a codec-only payload path and RFC 0003 §6.4's ownership rules for every file a worker reads. |
| Calibration home | Passed explicitly; `None` means `DirectoryCalibration(root / "calibration")`. |
| CPU cap | `Substrate.can_start(cls) -> bool \| str`, with a default reading `per_user_cpu_cap`; sites override it for GRES and submit caps. |
| `max_concurrent` | The drivers pass no limit to a `batches = True` executor: one line in RFC 0002 R7, enforced, not only documented. |
| Venv residency | Not waiting for G2. The process-owning provider is the residency primitive for an interpreter in an environment; G2's `Venv.warm()` later reuses it. |
| Beater on macOS ‡ | The helper-subprocess beater is accepted under `spawn` (about 0.5 s per worker start). |

**Decided in revision 3** (from the verification of revision 2; the owner ruled on each):

| Question | Decision |
|---|---|
| N1 Withdrawal of claimed tickets | Withdrawal, on resume and on cancel, takes the ticket's claim by `try_take()`, so only an unclaimed or stale ticket is withdrawn; a ticket with a live claim is followed (resume) or left to its holder (cancel). A re-placed job always gets a fresh attempt; workers skip `withdrawn` jobs (§5.5, §6). |
| N2 Attempts after a loss; follow repointing | Every placement, half-tickets included, allocates the job's next follow generation by exclusive create, and the attempt is `<pool id>.<g>`, so attempts cannot collide however deep a split goes; `<attempt>-l<n>` is withdrawn. Follow entries are never modified or removed: a move writes generation `g+1` before the old ticket's `withdrawn`, which keeps R-273 closed (§5.1, §5.5). |
| N3 Cross-process lease | L0 adds `beat="external"`, `token_dir`, `LeaseBeater` and `BeaterClient` (`hold`, `release` with acknowledgement, `deadline`; `lost`, `expired` back). The worker runs no in-process beat thread while it has a beater; a `lost` marks the lease lost and abandons the claim; a beater's end of file recycles the worker (§5.3, L0). |
| N4 Job timeout | The beater stops beating a claim whose call deadline (declared time, or `call_timeout`, plus `timeout_grace`) has passed, writes `expired` outcomes and tells the worker, which stops the call and recycles, or is killed after `timeout_grace`; the takeover runs as a new attempt and types the timeout as a resource fault (§5.3, §6). |
| N5 Settle-mark race | On `EEXIST` for `settled/<digest>-<count>`, recount and retry until a mark covering the writer's outcome exists (§5.1). |
| N6 Wording | §4.5 "narrows R-286"; the committed result comes from the store's first writer wins, and the proof only narrows a lost holder's window (§2.1, §4.3, §5.5). |
| N7 Shared pools | One live coordinator per pool, with the epoch fence kept; a second live run waits (`coordinator_wait`, `PoolBusy`) or uses its own pool (§4.1, §5.3). |
| D1–D3 Dependencies and order | W2b is a dependency of W5, W7 and W12; the Order paragraph names the base (integration branch for L0 and W2), P1 and P4c for W4, P8 for W5 (§11). |
| `max_wait` | Excluded from the settings comparison on purpose: it caps a waiter, not staleness (§4.3, §5.3). |
| `gpu_mem=None` | Judged `"unknown"` and sorted last in the ladder; a `WrongHardware` refusal is re-placed by the coordinator from the next class up, with its own §6 row (§3.3, §6). |

**Decided in revision 4** (from the check of revision 3; the owner ruled on each):

| Question | Decision |
|---|---|
| I1 Reruns of a started job | Every start of an attempt that already started writes a `"restart"` generation first, detected by an exclusive `started` mark per attempt: a SIGTERM release after a stopped call, a content retry, any rerun. `collect` keeps each job's highest generation, so numbering never restarts (§4.3 step 2, §5.1, §5.5). |
| I2 Crash charging | The worker writes a call record naming the call's jobs before invoking it (and gives it to the beater). A loss charges a crash only to the job that was alone in the in-flight call; jobs gathered or claimed but idle, or outside a timed-out call, are re-placed uncharged (§4.3 step 4, §5.5). |
| I3 `expired` against a committed result | Outcome records are exclusive creates and the first wins. The taker looks the job up in the store before any ladder move. The claim holder performs the move by the five-step order from the ticket's `ladder`, and `resource_moves` is counted from `"ladder"` generations (§5.1, §5.5, §6). |
| I4 L0's part of a timeout | `deadline(token_dir, at, payload)`, an `on_expire(token_dir, payload)` callback in the beater, `ack_expired`, and `kill_after` as the holder's policy (L0 never kills by default). A `hold`/`release` acknowledgement timeout of `renew_interval` is beater death (§5.3, L0). |
| M1 Attempt source | A claimant takes each job's attempt from its newest generation, never from the ticket file (§4.3 step 2, §5.5). |
| M2 Fenced `close` | Only a coordinator holding the newest epoch stops workers (§4.1). |
| M3 Follower cost | Followed claims are judged only for tickets idle for `stale_after`, at most once per `stale_after` (§5.5). |
| M4 Timeout on the ladder | A timeout moves only to a class with a larger `time` or another `gpu_type`; the worst-case cost of a hang is stated (§5.3, §6). |
| M5 `refused` on resume | Re-placed before the follow rule applies (§5.5). |
| M6 One-shot takes | A fresh `DirectoryLease` per take; L0 adds no `take_once()` (§5.3). |

**Open for maintainers:**
- Should `Provided[T, name]` also accept a provider class instead of a name, so a task package can ship its
  provider beside the task?
- `quantum_jobs`, `quantum_s`, `poll_s`, `max_queued_jobs` and `max_mount_roots` are unmeasured. Should their
  defaults wait for W6's and W3's benchmarks, or ship as stated and be revised?
- Should `pydracli pool probe` refuse a root whose measured mtime delay exceeds `attribute_delay`, as
  proposed, or raise `attribute_delay` in `pool.json` to the measured value?
- Is `<pool id>.<g>` the right spelling of a pool attempt, or should RFC 0003 §5.4 define attempts allocated
  outside the runner?
- `timeout_grace` (60 s) is unmeasured, and so is the cost of one exclusive generation create per job per
  placement; W6's `bench_pool_overhead` measures the latter.

## 13. Changes in revisions

### Amendment to revision 4 (implementation of W7)

| Item | Change |
|---|---|
| Infrastructure count | §5.1 layout, §6: a residency's infrastructure faults in an epoch (survived crashes among them) are counted by O_EXCL records `faults/<epoch>-<label>/<n>` on its queue; at `infrastructure_replacements` the worker writes `held-<epoch>-<label>`, settles the faulting ticket's open jobs `held` (reason naming the residency, the count, the epoch and the latest cause), and settles any ticket of that residency it had already gathered `held` without loading or running it. A failed load is one fault of the residency, not one per job. A worker on a pool with no coordinator epoch holds at epoch 1. The coordinator, on reading a `held` outcome, reports every other job **of that residency** it follows or places on that queue `NotRun(held=True)` for its epoch, with no extra listing per poll; jobs of other residencies sharing the queue (`residency="ignore"`, §4.2) are not held, and a worker judges holds per ticket, not per queue. |
| `start_failures` | A `FaultBudgets` field (default 7), no longer a `PoolSettings` field. |
| Passing budgets and classifier | §5.1 layout: each coordinator writes its `FaultBudgets` and its classifier as a `module:qualname` reference to `coordinator/<epoch>/faults.json`, once; every worker, whoever started it (`ExternalSubstrate` included), reads the newest epoch's at start. The coordinator resolves the reference before writing it and refuses one that does not round-trip (a `partial`, a bound method, a lambda, a nested function). A worker that cannot import it, or finds it not callable, exits non-zero before its ready mark, so `start_failures` breaks the class and its jobs are held; it never falls back to the default. A running worker reads the file again whenever the newest epoch changes. |
| Worker-local holds | A worker remembers a hold it wrote, with its epoch, only while that epoch is the newest and the hold mark stands: a new run and `pydracli pool release-hold` both clear it. A hold on a routed queue (`<class>#<digest>`, one residency) skips the queue without a walk. |
| Crash in a live worker | §6 "One bisection engine": a lone crash is retried under a fresh, uncharged `restart` attempt; the engine counts the job's lone losses in a row (capped by its own count, whatever a caller's count says), and a survived crash is the residency's infrastructure fault. Generations are charged only by takeovers (§5.5). |
| Load out of memory | Moves up the ladder at `phase: "load"`, uncharged, as §5.5 says; the `UnfitRecord` §6 names is written from W9, which adds the record type. |

### Amendment to revision 4 (implementation of W6)

| Item | Change |
|---|---|
| A proof renews a claim | Found implementing W6, fixed in W5. `prove()` moves the claim's token directory's mtime, so a worker whose beater had died kept the claim of its call alive by proving it, and a call that never returned held its claim forever with no deadline, since deadlines are the beater's. §5.3 "The beater": the worker pings its beater every renewal interval while a call runs (L0 adds `BeaterClient.ping`), proves only while the beater answers, and with a dead beater writes the call's `expired` outcomes itself past the deadline and recycles. W6's `test_job_outlives_stale_after` stops the beater alone, as its row says; W5 adds a test of a dead or hung beater under a call that never returns. |
| Submitter wiring and pool lifetime | Deferred from W6 to W11a, which owns the `pool` execute context and a pool that outlives a run. Until then a `WarmPool` is used as an external driver uses it (§7.2): `drive` closes its executors at the end of a run, and a sole coordinator's `close` stops its substrate's workers, so warm state does not survive from one run to the next. W11a decides who owns a pool's lifetime (the `Submitter`, a `drive` that does not close executors it was handed, or a `close` that leaves workers to exit on idle) and wires `Submitter(worker=WarmPool(...))`. |
| `pydracli pool collect` | §4.1: it also removes tickets resolved more than its age ago, with their settled marks, outcomes and losses; a ticket's slot stays in its segment's index. The writer of a settled mark removes the ticket's previous mark (§5.1), so within a run a queue's `settled/` holds one mark per ticket, which bounds the follower's listing at 10⁶ jobs; collect bounds it across runs. A lower mark left by a race (two writers' counts skipping one) stays until collect. |

### Amendment to revision 4 (implementation of W5)

| Item | Change |
|---|---|
| Payload location | §5.2, §5.1 layout, the ticket example and §4.1 Lifecycle: payloads live at `<root>/payloads/<sha256>`, written once, read by every attempt, removed only by the pool's `collect`; a run's work directory never holds one, since the store empties it after publishing. |
| Attempts outside the runner | `<pool id>.<g>` is unique per job only; RFC 0003 §5.4 states that a `work_dir` function receives such attempts and must use the job's address, and the default layout qualifies them by it. |

### Revision 4

Answers the check of revision 3 (`afac5394`).

| Item | How it is addressed |
|---|---|
| I1 Attempt per execution | §2.1; §4.3 step 2 (`started` mark, `"restart"` generation); §5.1 layout, generation `cause`, `collect` keeps the highest; §5.5 "Every execution has a fresh attempt"; §6 content row; §4.1 Lifecycle; W5 `test_released_started_job_gets_fresh_attempt`, W6 `test_rerun_after_collect_fresh_attempt`, W4 `test_collect_keeps_highest_generation` |
| I2 Crash charging | §4.3 step 4 (call record); §5.1 `workers/<token>/calls/`; generation `charged`; §5.5 "Crash charging follows the call record"; §5.3 "Job timeout"; W7 `test_timeout_kill_spares_gathered_jobs` |
| I3 `expired` and the ladder | §5.1 "Outcome records are exclusive creates"; ticket `ladder`; §5.3 "Job timeout" (store lookup first); §5.5 takeover and "Who moves a job up the ladder"; §6 resource row; W4 `test_outcome_first_writer_wins`, W7 tests |
| I4 L0 timeout hook | §5.3 `LeaseBeater(on_expire, kill_after)`, `deadline(token_dir, at, payload)`, `ack_expired`, acknowledgement timeout; L0 row with `test_deadline_payload_reaches_callback`, `test_ack_timeout_is_beater_death`, `test_kill_after_unacknowledged_expiry` |
| M1 | §4.3 step 2; §5.5 "A job leaves a ticket only by `withdrawn`" |
| M2 | §4.1 Lifecycle; W6 test |
| M3 | §5.5 "The coordinator dies"; W6 test |
| M4 | §5.3 "Job timeout" (arithmetic, class condition); §6; W7 test |
| M5 | §5.5 "The coordinator dies"; W6 test |
| M6 | §5.3 "Cross-process beating (L0)" |

### Revision 3

Answers the verification of revision 2 (`875de66a`).

| Item | How it is addressed |
|---|---|
| N1 Withdrawing a claimed ticket | §5.5 "A job leaves a ticket only by `withdrawn`", "The coordinator dies"; §6 cancel row; §4.3 step 2; W5, W6 tests |
| N2 Half-ticket attempts; follow repointing | §5.1 "Follow generations" and layout; §5.5 "Every execution has a fresh attempt", "Moving a job keeps R-273 closed", "Isolating a job"; ticket `attempt`/`runner_attempt`; §8 example; W4 tests |
| N3 Cross-process lease | §5.3 "Cross-process beating (L0)", "The beater"; §4.3 steps 1, 2, 6, 7; L0 API and tests; W5 |
| N4 Wedged call | §5.3 "Job timeout"; §4.3 steps 4, 7; §5.5 crash count excludes `expired`; §6 resource row; W5 `test_wedged_call_times_out`, W7 |
| N5 Settle count race | §5.1 "Settled marks"; W4 `test_settle_count_race` |
| N6 Overclaims | §2.1; §4.3 steps 4 and 6; §4.5 and §9 R-286 "narrows"; §5.5 |
| N7 Shared pools against the fence | §4.1 "One live coordinator per pool", Lifecycle; §5.3 coordinator lease; W6 tests |
| D1 W2b | W5, W7, W12 depend on W2b |
| D2, D3 Order | §11 Order paragraph (base per item; P1, P4c for W4; P8 for W5) |
| Ids | header naming sentence (RFC 0002 R1–R8, "0001" prefix); 0001 A1 (§1), 0001 F2 (§4.1), 0001 F1 and 0001 H0 (§11); "B-4" dropped (§9) |
| `max_wait` | §4.3 step 1, §5.3 Settings; L0 test |
| `gpu_mem=None` in the ladder; `WrongHardware` re-placement | §3.3 step 2 and "A class with `gpu_mem=None`"; §5.4 follower; §6 row |
| Code check | `try_take()` is integration only (`d7b0988f`, `lease.py:774`); `prove()` exists at `529d2efa` and `68090364` as well; the integrated lease is `pydra/engine/lease.py` (header, §11 Order) |

### Revision 2

Answers the architecture review of revision 1 (`ef0c27d9`).

| Recommendation | How it is addressed |
|---|---|
| 1. A provider may own a process | §3.1 Providers (`load` may start a process, `close` stops its group and waits, `measure`, cgroup baseline after each start); §2.1; §10.2 (`senselab-worker@3` provider, `host` kind in-process); §12 Venv residency; W2b |
| 2. Takeover is a new attempt; proof before publish | §5.5 (`<attempt>-l<n>`, distinct work directory, outcome names the attempt); §4.3 steps 4 and 6 (claim proved before `publish` through the P4c de-duplication slot); §2.1 guarantee restated; W6 `test_job_outlives_stale_after` |
| 3. Named lease, whole settings, probe, beater subprocess | §5.3 (L0 `DirectoryLease.at`, `pool.json` carries and compares all four `LeaseSettings`, `attribute_delay` 60 s, `pydracli pool probe`, beater forked before providers load); §4.3 step 1; L0; W5 `test_gil_bound_call_keeps_claim` |
| 4. Mounts from the jobs; container isolation | §3.1 (binding roots from `get_bindings`, ticket union, one restart per new root, `max_mount_roots`, isolation defaults, GPU flags at start); §5.1 ticket `roots`; W3 tests |
| 5. Settled marks, no symlinks, counted fsyncs | §5.1 (`settled/<digest>-<count>`, `poll_s`, `O_EXCL` regular files, synced writes, uuid read-back, NFS behaviours); W4 tests; W6 `test_follower_cost_10k_jobs` |
| 6. Per-item batched outcomes; reload after a resource fault; budget attribution | §4.4 (`list[T \| BaseException]`, reload), §6 (defaults marked as pydra's with senselab's values); W7, W10 tests |
| 7. Ladder across types; `gpu_mem` unknown at placement; time in fit | §3.3 steps 1–2, "Time is part of fit", "A class with `gpu_mem=None`"; §4.3 step 2; W1 tests |
| 8. One spec digest; loader in context; `Annotated`; device not keyed | §3.1 `ResidentKey.spec`, `Provided[T, name]`; §8 (residency `version`, device not keyed) |
| 9. Lifecycle rules | §4.1 Lifecycle; §4.3 step 7; §5.5 follow only for started classes |
| 10. Evidence labels; benchmarks | §9 header and Measured list (426 s and 1329 s split; 0.013–8.36 s over five units; pool-vs-in-process; rationale entry); W3 `bench_warm_environment`, W6 `bench_pool_overhead` |
| 11. W decomposition | §11 (L0, W2b, W5/W5b, W11a/W11b; CLI commands assigned; senselab-needed column; order) |
| 12. Missing tests | §11 W1, W3, W5, W6, W7, W12 (cross-node claim lease, process provider, `sys.modules`) |
| 13. Record the §12 answers | §12 "Decided in this revision", including the three owner rulings |

| Citation in revision 1 | Corrected to |
|---|---|
| RFC 0002 at `1956706e`; §7.1 "the architecture review recommends" node-group `limit` | RFC 0002 at `6a9ececb`, which has node-granular `limit` and runner-side sharing (§7.1) |
| P4c, P9 as RFC 0003 items; I6a, I6b as RFC 0004 items | plan tasks of RFC 0003 P4 and of RFC 0004 I6; header and §11 |
| §5.2 "RFC 0004 §6.2; 0001 G2's import-reference jobs" | import-reference jobs are 0001 G2; RFC 0004 §6.2 is the canonical typed encoding of their inputs |
| `rt/sizing.py:362-382` as the source of `m0`/`m1` | `rt/probe.py:359-386` (Theil–Sen), `MemoryFit` at `rt/calibration.py:238-243`; `Sample` stays at `rt/sizing.py` (§3.3, §4.3) |
| `ex/workers.py:265-320` as an LRU under byte budgets | a count-bound LRU of worker processes; the byte LRU is `ModelCache`, without a busy guard; `ResidentSet` is new (§3.1) |
| `infrastructure_replacements` default 3 as senselab's | 3 is pydra's choice; senselab's is 6 (§6) |
| `resource_moves` default 2 as senselab's | senselab's field is `resource_replacements`, default 0 (§6) |
| `content_attempts` "as pydra today" | 1 is pydra's behaviour; senselab's default is 2 (§6) |
| R-286 "a submission is made once; on a timeout it looks the name up" | senselab resubmits after a failed lookup and R-286 is open; §4.5 states pydra's rule as the fix |
| "cold start at 426 s ... 1000 GPU items in 1329 s" as one run | two jobs: 28580776 (Step 4) and 28578691 (Step 3) (§9) |
| warm per-job 0.013–0.34 s | 0.013–8.36 s over five units (§1, §9) |
| speaker-embedding batches "changed short items' vectors" | a parity note against 1.x; "repeat and batch differed by 0" (§4.4, §9) |
| follow index as senselab's `pool_follow` | `Pool._live_items`/`_follow` (`ex/pool.py:395-447`), tested by `pool_follow_test.py` (§4.2) |
| `Executor.batches: ClassVar[bool]` | an instance attribute in the integrated protocol (§4.1) |
| `Job.result()` "by `_committed.json`, then by the entry" as existing | not yet in code; added by RFC 0003 P4 (plan task P4c) and P8 (§5.4) |
| `DirectoryLease` "takeover" as an API | no public takeover; reached through `try_take()`; `prove()` exists (§5.3) |
| `Local` with `workers_per_device` co-tenancy | senselab's `Local` gives each CUDA worker an exclusive device; co-tenancy is pydra's extension (§3.2, §4.5) |
| `PoolSettingsMismatch` as senselab R-277 | senselab's `LeaseMismatch` (`ex/epoch.py:61`) compares the lease section; R-277 is the defect it closed (§4.3, §9) |
| `environments/base.py:162` for `get_bindings` | `environments/base.py:162-243` (§3.1) |

### Revision 1

Initial draft (`ef0c27d9`). No review findings.
