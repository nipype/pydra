# Proposal 0005: warm worker pools, resident state and GPU-fit placement

Status: proposed, for maintainer discussion; revision 1. Branch `rfc/warm-worker-pools`, off
`enh/execution-extensions` at `529d2efa`. Line references are to `529d2efa` unless another commit is
named. RFC 0002 is #930 at `1956706e`, RFC 0003 #931 at `36546b9f`, RFC 0004 #932 at `91b8795d`;
their section numbers are those revisions'. Code that exists only on the RFC 0002–0004
implementation branches (the `GraphRunner`, the drivers, `PortableStore`) is cited by RFC section,
not by line. senselab is cited at `sensein/senselab` `2.x` @ `4472cfbb`; its runtime paths are
abbreviated `rt/` for `packages/senselab-core/src/senselab_core/runtime/` and `ex/` for
`rt/execution/`.

This proposal replaces proposal 0001's I (`pydra-workers-pool`, 0001 §3.7) and the provider half
and batching half of H (0001 §3.6). It answers RFC 0002 §5's question about run identity (it belongs
to the pool, not to the `Executor` protocol, §5.3). It does not reopen RFC 0002 §3.2's "Not adopted"
list: a warm pool is an `Executor` that takes pydra jobs.

## 1. Motivation

pydra runs a job in one of three ways: in the caller's process (`debug`), in a process of a local
pool (`cf`), or in a fresh process on a cluster (`slurm`, `sge`). Only the first keeps anything
between jobs, and it runs one job at a time on one machine. Everything that a job loads before it can
do its work is therefore loaded again for every job:

| Problem | Evidence |
|---|---|
| A cluster job is a process, an import of pydra and its task's libraries, and a load of whatever the task needs, per job | `SlurmWorker._prepare_runscripts` writes one `_job.pklz` per job and a script that calls `load_and_run` on it (`workers/slurm.py:122-131`; `engine/job.py:723`). An array (RFC 0002 R5) starts one such process per element |
| A model load costs more than the jobs that use it | measured on one H100 (§9): loads of 8.9 to 26.5 s for four speech models, against warm per-job times of 0.013 to 0.34 s; the first request after a load took 7.8 to 30 times the steady time |
| A container is started per command | `Docker.execute` builds `docker run ...` for every job (`environments/docker.py:41-42`); `Singularity.execute` runs `singularity exec` per job (`environments/singularity.py:21-23`) |
| A module environment is resolved per command | `Lmod.execute` runs `lmod python load <modules>` and parses its output for every job (`environments/lmod.py:32-41`) |
| `Environment` has no lifetime | `setup` and `teardown` are empty, with a `TODO` (`environments/base.py:28-32`, `:50-51`) |
| A resource request names hardware but not fit | `ComputeResources(cpus, mem, gpus, gpu_type, time, tags)` (`utils/resources.py:37-63`) maps to `sbatch` arguments and routing tags. Nothing says whether a task's working set fits a GPU class, or where to go when it does not |
| A scheduler sees one job per element | a 10⁴-element GPU node is 10⁴ Slurm array tasks, each waiting in the queue, each loading the model. senselab measured a worker's cold start at 426 s under a busy partition (§9), paid once per worker in a pool and once per element in an array |
| A cluster job that dies leaves nothing to take over | a pydra Slurm job either writes its result or is requeued by Slurm (A1). There is no lease a second process could judge |

**What pydra users gain:**
- A shell or container workflow over many inputs (a FreeSurfer, FSL or ANTs step per subject, a
  container tool per file) runs on a few long-lived workers that keep the container, the Apptainer
  instance or the resolved module environment up, instead of one scheduler job and one container start
  per input.
- A Python workflow that calls a model, a large lookup table or a GPU context loads it once per worker.
- GPU tasks are placed on a class whose memory fits them, from declared figures or from measurements,
  and move to a larger class when a load runs out of memory, instead of failing.
- On Slurm, the scheduler sees one job per worker, typed by GPU (`--gres=gpu:<type>:<n>`), and the
  workers pull work. Per-job scheduler latency, array limits and `MaxSubmitPU` stop scaling with the
  number of elements.
- A worker that dies loses at most the jobs it held; another worker takes them over once its lease is
  stale, and the run completes.
- Plain workflows that do not opt in change in nothing (§2.3).

**One motivating user.** senselab 2.x (sensein/senselab#617) built all of this on its side of RFC
0002's seam: pool workers with a claim loop (`ex/pool_worker.py:519-554`), resident models in an LRU
`WorkerSet` (`ex/workers.py:265-320`), queues named `<class>#<model identity>`
(`ex/pool.py:536-541`), placement by calibrated fit (`rt/fit.py:156-196`), one `sbatch` per worker
with typed `--gres` (`ex/substrates.py:556-579`), and a three-class fault model (`ex/pool.py:818-857`,
its ADR-009). Its owner has decided that this machinery moves into pydra now, before senselab's first
pin of the RFC 0002–0004 stack, and that senselab adopts pydra's design with no compatibility layer
(§10.2). The design below generalises that implementation and its measurements (§9). No pydra API
names senselab, audio or a model family; senselab becomes one client.

## 2. Goals, non-goals, compatibility

### 2.1 Goals

- Long-lived workers that keep **resident state** across jobs: provided Python objects (a model, a
  GPU context, an index), and warm environments (a running container, an Apptainer instance, a
  resolved module environment, later a virtual-environment interpreter).
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

### 2.2 Non-goals

- A general distributed scheduler. Workers pull from directories on a shared filesystem; there is no
  daemon, no message broker and no network service.
- Services shared across workers (one inference server consumed by many workers). senselab has them
  (`ex/services.py`); they are an open question (§12).
- Pools shared by several users. A pool belongs to the user who submits it, as a pydra cache does
  today (§5.2).
- Kubernetes and cloud substrates. The substrate protocol (§4.5) admits them; none ships here.

### 2.3 Compatibility

- **A run that does not use `WarmPool` is unchanged**: no new import at `import pydra`, no new file
  in a cache, no change to `Submitter`'s defaults, checksums, the store layout or any worker. The
  pool lives in `pydra.workers.pool` and is imported when it is named.
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
    spec: str          # canonical JSON (RFC 8785) of the spec, or the environment's resolved reference
    def digest(self) -> str: ...   # sha256 of the canonical JSON of the three fields
```

**Providers** (0001 §3.6, made concrete). An input field may be marked `provided="<name>"`:

```python
@python.define
def embed(audio: Path, model: Provided[ModelSpec] = field(provided="hf-model")) -> np.ndarray: ...

class Provider(Protocol):
    name: ClassVar[str]
    def load(self, spec: JSON, device: Device) -> object: ...     # may raise; see §6
    def close(self, obj: object) -> None: ...
    def footprint(self, spec: JSON) -> Footprint | None: ...      # declared bytes, host and device
    def version(self) -> str: ...                                 # binds calibration records (§3.3)
```

- The field's value in the task is the spec (a JSON value). The checksum hashes that value like any
  input. `Provided[T]` is `T` to the type checker of the task's caller and to `TypeParser`; the
  function receives the loaded object.
- Before running a job, the executing process replaces each provided field's value by
  `resident_set.get(ResidentKey("provider", name, canonical(spec)))`, which loads on a miss.
- A `ResidentSet` holds loaded objects in LRU order under a host and a device byte budget. An object
  in use by a running job is never evicted. Eviction calls `close`. Under `debug`, the `Submitter`
  holds one `ResidentSet` for its lifetime, so a notebook that calls the same task twice loads once.
  Under `cf` each process has its own, so residency there is per process and unrouted.
- Providers are registered by entry point (`pydra.providers`) or passed as
  `Submitter(providers=[...])` / `WarmPool(providers=[...])`. An unregistered name is a wiring error at
  hand-out.
- **A provider object is read-only to jobs.** A job that mutates it (switches a model's mode, consumes
  a random state, appends to a cache) makes later jobs depend on earlier ones. pydra cannot detect
  this; the provider's documentation must state it, and §12 asks whether a sampled cold re-check
  belongs in pydra.

**Warm environments.** `Environment` gains one optional method:

```python
class Environment:
    def warm(self, mounts: Sequence[Path]) -> WarmEnvironment | None:
        """A handle that runs many jobs in one started environment; None: not warmable."""

class WarmEnvironment(Protocol):
    key: ResidentKey                          # kind="environment", spec = the resolved reference
    def covers(self, job: Job) -> bool: ...   # its bindings fall inside `mounts`
    def execute(self, job: Job) -> dict[str, Any]: ...   # as Environment.execute
    def close(self) -> None: ...
```

| Environment | `warm()` | Each job | Notes |
|---|---|---|---|
| `Native` | `None` | as today | nothing to keep |
| `Docker` | `docker run -d --init` of the resolved reference (RFC 0004 §5.3), with `mounts` bound and the image's entrypoint replaced by a sleeper | `docker exec -w <work dir> <id> <command>` | a fresh work directory per job; `--read-only` with a tmpfs `/tmp` when `Docker(read_only=True)` |
| `Singularity` / Apptainer | `apptainer instance start` of the resolved reference with `mounts` bound | `apptainer exec instance://<name> <command>` | the instance name is a token, never a path |
| `Lmod` | resolves `module load <full names>` once (RFC 0004 §5.3) and keeps the resulting environment mapping | the command runs with that mapping | no process is kept; the saving is the resolution (`environments/lmod.py:33`) |
| `Venv` (0001 G2) | a long-lived interpreter in the environment, speaking the worker protocol of G2 | a call over that protocol | after G2; out of this proposal's delivery (§11) |

- **Mounts are fixed when the environment starts.** A warm container cannot gain a bind mount. A
  pool's `mounts` are the store root, the work-directory root and the roots listed in the worker class
  (§3.2). A job whose bindings (`Container.get_bindings`, `environments/base.py:162`) fall outside them
  runs through the environment's cold `execute`, in the same worker. Correctness never depends on
  warmth.
- **Opt-in per environment.** `Docker(..., warm=True)`. A container tool that keeps state inside the
  container between invocations (a license daemon, a cache under `$HOME`) would make one job's output
  depend on another's, so warmth is the author's statement that it does not.
- `setup` and `teardown` (`environments/base.py:31`, `:50`) are called by `WarmEnvironment` start and
  close; their `TODO` is closed by this.

**A job's residency** is the tuple of its resident keys, ordered: its environment's key when the
environment is warmable and `warm=True`, then one key per provided field. Its **residency digest** is
the sha256 of the canonical list of their digests, or none for a job with no resident state.

### 3.2 Resource requests and worker classes

`ComputeResources` (`utils/resources.py:37-63`) gains two optional fields:

```python
gpu_mem: str | None = None        # device memory per GPU the task's working set needs, "40G"
residency: Literal["route", "ignore"] = "route"   # "ignore": run on any worker of the class, unrouted
```

Neither enters a checksum; both are scheduler metadata like the existing fields.

A **worker class** is what a substrate can start:

```python
@attrs.frozen
class WorkerClass:
    name: str                              # "gpu-h100", "cpu-large"
    device: Literal["cpu", "cuda", "mps"]
    gpus: int = 0
    gpu_type: str | None = None            # matched against ComputeResources.gpu_type
    gpu_mem: int | None = None             # bytes per GPU; read from the device when None (§4.3)
    cpus: int | None = None
    mem: int | None = None                 # host bytes
    time: int | None = None                # seconds per worker job
    max_workers: int = 1
    workers_per_device: int = 1            # co-tenancy; the fit divides device memory by it
    mounts: tuple[Path, ...] = ()          # extra roots warm environments bind (§3.1)
    substrate_args: Mapping[str, str] = {} # partition, account, qos, constraint: the substrate's own
    quantum_jobs: int = 64                 # most jobs one claim gathers (§4.3)
    quantum_s: float = 300.0               # most seconds a worker stays on one queue
```

Classes are data, loaded from a mapping or a YAML file by `WorkerClass.load`. pydra ships no class
table and no default fitted to any cluster; a site writes its own. `pydracli pool classes --probe`
prints what one host offers (devices, memory, CPUs) as a starting point.

### 3.3 Placement and fit

**Placement** maps a job to a class and a queue, in the submitting process, before anything is
written:

```python
def place(job: Job, classes: Sequence[WorkerClass], fit: Fit) -> Placement | Unplaceable
@attrs.frozen
class Placement:
    worker_class: str
    queue: str                  # "<class>" or "<class>#<residency digest>"
    tried: tuple[ClassTried, ...]
    batch: int                  # most jobs of this residency one call takes (§4.4)
```

1. **Candidates.** Classes whose `device`, `gpus`, `gpu_type`, `cpus` and `mem` satisfy the request;
   `tags` filter further when a class lists `tags`. A GPU request never falls back to CPU silently;
   no candidate is `Unplaceable`.
2. **Start.** The candidate with the least device memory (then host memory), as senselab's
   `larger_class` orders them (`rt/fit.py:31-52`), skipping a class whose `cpus` exceed the
   substrate's per-user CPU cap, which could never start a worker.
3. **Judge and climb.** `fit.judge(request, cls)` for the start class; on `"too small"` or `"unfit"`,
   the next larger class of the same device and GPU type, until one `"fits"` or `"unknown"`. The walk
   is `choose` (`rt/fit.py:173-196`), unchanged.
4. **Unplaceable.** A job that fits no class is reported `Failed(Unplaceable(...))` at submit, with
   every class tried and the numbers in the message. It is a plan outcome: no worker started, no
   environment charged.

```python
Verdict = Literal["fits", "too small", "unfit", "unknown"]

class Fit(Protocol):
    def judge(self, request: FitRequest, cls: WorkerClass) -> ClassTried: ...
    def batch(self, request: FitRequest, cls: WorkerClass, units: Sequence[float | None]) -> int: ...

@attrs.frozen
class FitRequest:
    residency: tuple[ResidentKey, ...]
    resources: ComputeResources
    units: float | None          # the job's size in the task's own units (§4.4), when it declares one
```

- **`DeclaredFit`** (the default) judges from declared figures only: `gpu_mem` (or the providers'
  `footprint`) against the class's per-device memory over `workers_per_device`, `mem` against the
  class's host memory. No figure gives `"unknown"`, which places. `batch` returns 1.
- **`MeasuredFit(calibration, headroom=0.9)`** is senselab's rule (`rt/fit.py:156-170`): a class is
  `"unfit"` when a load of the residency ran out of memory in a grant at least as large; `"too small"`
  when `resident + m0 > headroom × grant`; otherwise `"fits"`. `resident` is the measured bytes the
  residency holds after its load on that hardware; `m0` is the fixed peak beyond it. `batch` returns
  the largest `B` with `resident + m0 + m1·Σ padded units ≤ headroom × grant` and, when per-job times
  at several `B` are recorded, the fastest such `B` (§4.4). `headroom` is a parameter; 0.9 is
  senselab's site value, not a pydra default fitted to anything.
- **`Calibration`** is where measurements live:

```python
class Calibration(Protocol):
    def memory(self, key: str, hardware: HardwareClass) -> MemoryFit | None: ...
    def unfit(self, key: str) -> Sequence[UnfitRecord]: ...
    def times(self, key: str, hardware: HardwareClass) -> Sequence[TimeRecord]: ...
    def record(self, sample: Sample) -> None: ...
```

  `key` is the residency digest. `DirectoryCalibration(root)` ships: JSON records per key and
  hardware class, written by workers (§4.3), readable by other runs. An `UnfitRecord` names the
  residency, the resource that ran out (`"device"` or `"host"`), the grant, and the code that loaded
  it: the providers' `version()` and the environment's resolved reference. A change to either gives a
  record that no longer binds, as senselab's `current_unfit` does (`rt/fit.py:146-150`). A client
  that measures deliberately (senselab's probe) writes the same records.
- **Admission and backpressure.** Placement runs at `WarmPool.submit` for every job. Placed jobs are
  written as tickets up to `max_queued_jobs` per queue (default 4 × `quantum_jobs` × `max_workers`);
  the rest wait in the submitting process and are written as tickets settle. `submit` never blocks
  and never refuses a placeable job. `plan()` (RFC 0002 R8) shows each node's placement (§7.4).

## 4. The executor

### 4.1 `WarmPool`

```python
class WarmPool:                                   # an RFC 0002 §3.2 Executor
    batches: ClassVar[bool] = True
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
  router=...)` with F2's router; or an external driver (RFC 0002 §3.5) that calls `bind` and
  `submit` itself.
- `root` is the pool directory (§5.1), on storage every worker and the submitting process see. It is
  per pool, not per run: two runs may share one pool, and a resubmitted run finds its predecessor's
  tickets (§5.5).
- `close` writes the coordinator's `ended` mark, stops the substrate's workers it started, and leaves
  the directory for `pydracli pool collect`.

### 4.2 Queues, tickets and routing

- **A queue** is named `<class>` for jobs with no residency (or `residency="ignore"`), and
  `<class>#<residency digest>` otherwise. This is senselab's `_queue_name` (`ex/pool.py:536-541`)
  with the model identity generalised to a residency.
- **A ticket** is one queue item: up to `Placement.batch × k` jobs of one node and one queue, cut from
  one `submit` call in node order, at most `quantum_jobs`. It is a JSON document (§5.1) naming each
  job's checksum, lookup fingerprint, attempt and payload, never a pickle. A whole round of a
  100-element node therefore becomes two tickets of 50 on one queue, claimed by warm workers, not 100
  scheduler jobs.
- **A job is in at most one live ticket.** Before writing a ticket, the pool looks the job's
  `(checksum, fingerprint)` up in the pool's follow index (§5.5); a job already held by a live ticket
  is followed, not put again. senselab found the alternative on the cluster: a resumed coordinator
  re-cut unfinished jobs into new tickets while the old ones were still queued, and 61 jobs ran twice
  (§9, R-273).
- **Routing.** A worker serves the queues of its class. It claims from the queue whose oldest waiting
  ticket is oldest, preferring a queue whose residency it holds while its quantum lasts
  (`quantum_jobs` jobs or `quantum_s` seconds), then yields to the oldest work again
  (`ex/pool_worker.py:519-554`, `:610-631`). On a queue it has not served, it claims one ticket, so a
  load falls on one ticket; once warm, it gathers single-job tickets up to its quantum.
- **Eviction.** Before loading a residency, the worker's `ResidentSet` evicts idle resident state, least
  recently used first, until the measured or declared footprint fits the worker's budget; state in use
  is never evicted. A resident model is a cache, not a reservation: a worker never refuses work because
  it holds something.

### 4.3 The worker

`pydracli pool worker --root <pool> --class <name>` (or `python -m pydra.workers.pool.worker`) runs
one loop:

1. **Start.** Read `pool.json`; refuse to start (`PoolSettingsMismatch`, exit code 3) when the lease
   settings there differ from the worker's own, because a worker and its coordinator that judge
   staleness by different bounds disagree about who is alive (senselab R-277, §9). Take the worker
   lease (§5.3). Read the hardware: device kind, product name, memory, compute capability, granted
   CPUs, as a `HardwareClass`; refuse to start (`WrongHardware`) when it does not match the class's
   `gpu_type` or `gpus`, so a mis-typed grant costs one failed start, not a queue of failed jobs.
   Write the ready mark.
2. **Claim** a ticket (§4.2), under a claim lease (§5.3).
3. **Load** the ticket's residency through the `ResidentSet`: warm environments by `warm(mounts)`,
   providers by `load(spec, device)`. A failure here is a load fault (§6): it never charges a job.
4. **Run** the ticket's jobs, one call per batch (§4.4), each through `Job.run(rerun)` in this process
   with `chdir=False` (0001 H0, merged), which looks the job up, runs it in its work directory and
   publishes (RFC 0003 §10). Python tasks run in-process with provided fields bound; shell tasks run
   through the warm environment when it `covers` the job, otherwise through the environment's cold
   `execute`.
5. **Settle each job as it finishes**: its measurement first, then its outcome record. A ticket of 64
   jobs is not settled once at the end; a worker that stops after job 40 leaves 40 settled
   (senselab T24-2, §9).
6. **Prove** the claim lease every renewal interval while a call runs. A claim whose lease was lost is
   abandoned: the call in flight is stopped when it is a subprocess or a warm environment's command,
   allowed to finish when it is in-process Python, and in both cases its outcome is not written.
7. **Exit** after `idle_polls` polls of `poll_s` with nothing claimable, on SIGTERM after settling
   finished jobs and releasing the rest, or when no coordinator lease of the pool has been live for
   `orphan_after_s`. Resident state is closed on exit; a container is stopped and its process group
   waited for, so an immediate restart does not race the dying process for GPU memory (senselab R-295).

**What a worker records** (`Sample`, per settled job or batch): wall seconds, claim seconds, batch
size and padded units, host baseline/resident/peak bytes, device baseline/after-load/peak bytes
(read once per claim on a CUDA class), the load seconds and whether this job's residency was cold or
warm (`since_load`). These go to the `Calibration` when one is given, and to the job's execute
context (§8). The measures are senselab's (`rt/sizing.py:362-382`), without the audio fields.

### 4.4 Whole-round batching

Two levels, both opt-in:

- **A ticket is a batch of jobs into one warm worker.** This needs nothing of the task. It amortises the
  claim, the lease, the load and the per-ticket overhead. senselab measured a per-item overhead of
  0.10 to 0.26 s and 7.3 ms per added job (§9), so a ticket of one job pays that per job.
- **A batched call** runs several jobs in one invocation. A Python task declares
  `@python.define(batched=embed_many)`, where `embed_many(audio: list[Path], model) -> list[...]`
  takes the per-job inputs as lists and returns one output set per job. The worker groups a ticket's
  jobs into calls of at most `Placement.batch`, ordered by `units` when the task declares
  `units=<callable>` (a job's size in its own units, such as seconds of audio or voxels), so a call pads
  little. Each job's result is published under its own checksum. A shell task cannot be batched this
  way; its tickets still amortise the environment.
- **Bisection.** A batched call that raises a resource fault, or crashes its process, is split in
  halves, each sent again, until the failing job is alone (§6). A half that completed keeps its results.
  This is senselab's one bisection engine (`ex/job.py`, its ADR-009 amendment 2026-11-11), which
  replaced two engines that disagreed.
- **A batched call must not change a result.** A model that pads a batch to its longest item can give
  different outputs for a short item batched with long ones than alone; senselab hit exactly this
  with speaker embeddings and encodes such items alone (§9). The batched implementation is the
  author's statement that results equal the unbatched ones; the batch size is recorded in the context
  and never keyed (§8), and §12 asks whether pydra should offer a sampled equality check.

### 4.5 Substrates and scaling

```python
class Substrate(Protocol):
    def start(self, cls: WorkerClass, n: int) -> list[WorkerHandle]: ...   # never blocks on a queue
    def live(self) -> list[WorkerHandle]: ...          # one listing per call: pending and running
    def stop(self, workers: Sequence[WorkerHandle]) -> None: ...
    per_user_cpu_cap: int | None                         # the site's cap; None: unknown
```

- **`LocalSubstrate(devices=None)`** starts worker processes on this host under the submitting
  process. A `cuda` class's worker gets devices of its own through `CUDA_VISIBLE_DEVICES` (devices from
  `nvidia-smi` when not given), at most `workers_per_device` per device, so a host with four GPUs runs
  four one-GPU workers, or two two-GPU workers. senselab's `Local` (`ex/substrates.py:357-420`).
- **`SlurmSubstrate(...)`** submits **one `sbatch` per worker**, never per job: `--gres=gpu:<type>:<n>`,
  `--cpus-per-task`, `--mem`, `--time` and the class's `substrate_args`, with `--wrap` running the
  worker command for that class, a job name scoped to the pool, its coordinator epoch and a token, and
  logs under the pool's private directory. A submission is made once; on a timeout it looks the name up
  before trying again, so a late-landing submission is not doubled (senselab R-286). `live()` reads
  one `squeue --user` per poll cycle, shared with `SlurmWorker`'s poller (0001 A2) so a run with both
  polls once.
- **`ExternalSubstrate`** starts nothing: workers are started by hand or by another system and
  attach by pointing at the root.
- **Scaling.** Each poll the pool computes, per queue, the workers its waiting jobs want:
  `min(ceil(jobs × s_per_job / target_wall_s), ceil(jobs / quantum_jobs), cls.max_workers)` from the
  slowest measured per-job time in the calibration, or `unmeasured_workers` (default 1) when none is
  measured; summed per class and capped by `per_user_cpu_cap // cls.cpus`. A surplus is stopped,
  pending workers first, running ones only once their queues have no unsettled ticket. This is
  senselab's `workers_wanted` and `scale` (`ex/pool.py:511-534`, `ex/substrates.py:884`). A site whose
  array `%C` limits concurrency differently states it through `per_user_cpu_cap`.
- **Failed starts.** A worker that leaves `live()` without a ready mark is a failed start of its class.
  The next start of the class backs off, doubling; after `start_failures` consecutive failures
  (default 7) the class is broken and its jobs are `NotRun(held=True)` naming the class, the count, the
  latest cause and the class values (§6). senselab saw this on the cluster when four coordinators
  filled a partition's per-user submit limit (§9).

## 5. The worker protocol over shared storage

### 5.1 Layout

```
<root>/                               0700 (private pool)
  pool.json                           header: "pool@1", lease settings, codec, created by
  coordinator/<epoch>/                one per coordinator, taken by mkdir; lease, attempt, ended
  classes/<name>.json                 the WorkerClass as JCS
  queues/<label>/                     label = sha256 of the queue name
    name                              the queue name, written once
    tickets/<digest>.json             a ticket, JCS; digest = sha256 of its canonical JSON; first writer wins
    segments/<seq>/slot-<n>           index: a symlink to <digest>, made by O_EXCL symlink, never followed
    segments/<seq>/jobs-<n>-<count>   the ticket's job count, so a listing gives the queue's depth
    claims/<digest>.lease             the claim's DirectoryLease
    outcome/<digest>/<job index>.json one per job: done | failed | held | withdrawn; first writer wins
    losses/<digest>/<n>               a claim of this ticket found stale and taken over (§5.5)
    held-<epoch>-<label>              a hold of a residency on this queue (§6)
  follow/<checksum>-<fingerprint>     a symlink to the live ticket holding the job (§5.5)
  workers/<token>/                    lease, ready.json, samples.jsonl
  private/                            Slurm logs, payloads outside work dirs; 0700
```

The queue directories are senselab's `queue.py` layout (`ex/queue.py:1-40`) with items renamed
tickets, outcomes split per job (§4.3 step 5) and probe marks left to clients. A ticket:

```json
{"format": "pool-ticket@1", "queue": "gpu-h100#3f…", "node": "wf/embed",
 "residency": [{"kind": "provider", "name": "hf-model", "spec": "{\"id\":\"…\",\"revision\":\"<40-hex>\"}"}],
 "batch": 16,
 "jobs": [{"checksum": "…", "fingerprint": "…", "attempt": "…",
           "payload": {"codec": "private-pickle@1", "path": "<work dir>/_job.pklz", "sha256": "…"},
           "units": 4.6}]}
```

Every write follows RFC 0003 §9's rules through `StorageDir` (P1): descriptor-relative, `O_NOFOLLOW`,
staged and fsynced before the link, never modified in place. senselab's adversarial review found seven
hand-rolled writes that skipped the fsync; every one became a durability defect (R-290, §9).

### 5.2 Payloads and trust

A worker must never unpickle a file that another user can write (RFC 0003 §6.4; senselab ADR-007,
which puts every substrate's pickles under a 0700 private directory).

- **`PrivatePickle` (default codec).** The job is cloudpickled to `_job.pklz` in its work directory
  (`work_dir(checksum, fingerprint, attempt)`, RFC 0003 §5.4), which the store creates 0700. The ticket
  carries the file's sha256. The worker reads it only after the ownership walk of RFC 0003 §6.4
  passes for the work directory (owned by the worker's user, no component writable by others) and the
  digest matches. This is the trust `_job.pklz` already has under the Slurm worker today
  (`workers/slurm.py:122-125`), made explicit and checked.
- **`JobCodec`** is pluggable: `encode(job) -> (bytes, media_type)`, `decode(bytes, store) -> Job`. A
  client whose jobs are import references and canonically encoded inputs (RFC 0004 §6.2; 0001 G2's
  import-reference jobs) can carry them as JSON in the ticket and need no pickle.
- **A pool is single-user.** Workers run as the submitting user (they are its `sbatch` jobs or its
  processes). The pool root is 0700; `pool.json` records the owner's uid and a worker of another uid
  refuses it. Group pools are an open question (§12).
- **No host names on shared paths.** A worker's token and its host are recorded as digests of a
  per-boot random value (senselab's `host_token()`, ADR-007 amendment), never a hostname.

### 5.3 Leases, heartbeats and liveness

Every exclusion in a pool is pydra's `DirectoryLease` (#924, #928), judged on the storage clock, stale
after `stale_after` (default `STALE_AFTER`, `engine/lease.py:27`) with heartbeats at the lease's
fraction of it. There is **no host-bound lock** anywhere in the protocol: no `flock`, no `SoftFileLock`.
senselab lost a resume to one: a coordinator killed on one host left a lock that only its own host could
break, and a resume on another host waited forever (T24-9, §9). A test (§11, W4) fails if any pool
module takes one.

- **Worker lease** `workers/<token>/lease`: held for the worker's life. The pool reads it to count live
  workers and to tell a slow worker from a dead one.
- **Claim lease** `claims/<digest>.lease`: one per claimed ticket, renewed by the worker's heartbeat
  thread and proved by its loop every renewal interval (§4.3 step 6). Taking a claim uses
  `Lease.try_take()` (RFC 0002 §3.1 R6), so a worker never blocks on a held ticket; it moves on.
- **Coordinator lease** `coordinator/<epoch>/lease`: held by the submitting process. A newer
  coordinator takes epoch `n+1` by `mkdir`; a coordinator that sees a newer epoch stops writing tickets
  (a fence for efficiency, not correctness: results are idempotent, RFC 0003 §5.1). This is senselab's
  epoch (`ex/epoch.py`) and answers RFC 0002 §5's question: the run identity that fences a restarted
  coordinator belongs to the pool, not to the `Executor` protocol. The epoch is taken after every
  fallible step of `WarmPool` construction, not before (senselab R-283, §9).
- **Liveness without a daemon.** Nothing pings a worker. A claim whose lease is stale is taken over by
  the next worker that tries it (`DirectoryLease` takeover); a worker that lost its claim learns so at
  its next proof.

### 5.4 Results through the store

The pool directory never holds a result.

- A worker runs each job through `Job.run`, which publishes through the store (RFC 0003 §5.1, §10):
  under `PortableStore` the entry is renamed into place, `_committed.json` is written to the work
  directory, and `on_commit` (provenance) runs before the rename. Only then does the worker write the
  job's outcome record (`done`), so an outcome is never visible before its result and its provenance
  (senselab T24-10: the last claim's provenance, written after its commit, was lost to the stop; §9).
- A failed job writes `_error.json` to its work directory (RFC 0003 §11) and an outcome record
  `failed` with its fault record (§6). Nothing is stored.
- The submitting process follows outcome records (one listing of `outcome/` per queue per poll) and
  reads each `done` job's result through `Job.result()`: by `_committed.json`, then by the entry at the
  job's own address, after RFC 0002 §3.3's visibility retry. It reports the `Result`, carrying
  `committed`, to the driver.
- Under `PickleStore`, `Job.run` commits into the slot under the job's lease as today and `Job.result()`
  reads through `lookup`; the pool works, but warm facts are kept out of the context (§8).

### 5.5 Recovery

- **A worker dies mid-ticket.** Its claim lease goes stale. The next claimer takes it over, writes
  `losses/<digest>/<n>`, and runs the jobs that have no outcome record. Jobs already published are
  found by `Job.run`'s lookup and settle as hits, so nothing runs twice that finished. After a loss, a
  multi-job ticket is claimed in halves (written as two tickets that the follow index points at), so a
  job that kills its worker is isolated. A job that is alone and has been lost `crashes_charged` times
  (default 2) on fresh workers is `Failed(JobCrashedWorker)`, a content fault of that job. A job that
  completes on a fresh worker is not charged; its survived losses are recorded.
- **The coordinator dies.** Workers keep running claimed and queued tickets and publish to the store.
  They exit after `orphan_after_s` with no live coordinator lease. A resubmitted run is a new
  `GraphRunner` over the same store (RFC 0002 §3.1 "Resume"): jobs whose results were published are
  settled by `find` in `ready()` or by `Job.run`'s lookup; jobs still held by a live ticket are
  **followed** through `follow/` (they get no new ticket, and the result is read from the job's address,
  RFC 0003 §7.2); only the rest are put again. senselab's resume chain (start, SIGKILL of the
  coordinator, resume on another host) completed 1000 of 1000 with at most one execution per key under
  exactly this rule (§9).
- **A worker is preempted.** Slurm's SIGTERM is the worker's SIGTERM: finished jobs are settled, the
  rest released. A SIGKILL is a death.
- **Storage faults** (an `EIO` on the pool directory, a store that refuses a write) are
  infrastructure faults (§6). A corrupt ticket file is `failed` for its jobs as a content fault of the
  ticket; a corrupt outcome record is re-read, then treated as absent, never raised out of the
  follower (senselab R-278).

## 6. Faults and outcomes

Faults are typed where they happen, by a classifier, and each class has one effect. The table maps
them to RFC 0002 §3.1's outcomes; budgets are `FaultBudgets` (all integers, all overridable).

| Fault | Typed by | Effect inside the pool | Outcome to the runner |
|---|---|---|---|
| content: the task raised, a command exited non-zero | the classifier's default for exceptions from the task's own code | retried until `content_attempts` (default 1: none, as pydra today) | errored `Result`, or `Failed` when nothing ran |
| resource at a call: out of memory, a job timeout | name table (`MemoryError`, `torch.OutOfMemoryError` by qualified name, no import), the worker's own cgroup OOM event | a batched call is bisected (§4.4); a lone job moves up the class ladder (§3.3), at most `resource_moves` (default 2) times, uncharged | `Failed(ResourceExhausted)` once the ladder or the budget ends |
| resource at a load | a load raising the above | an `UnfitRecord` is written for the residency and the class; the ticket's jobs move up the ladder, uncharged; a load is never an item | `Failed(Unplaceable)` when no larger class exists |
| infrastructure: a load that failed otherwise, a broken environment, a lost service, a storage error | anything raised by the pool or an environment's start, and every unclassified exception there (flagged `gap`) | requeued on the same class, uncharged; after `infrastructure_replacements` (default 3) failures of one residency in the current coordinator epoch, the residency is held on its queue (`held-<epoch>-<label>`) | `NotRun(held=True)` for each of its jobs, reason naming the residency and the latest cause |
| worker lost mid-ticket | a stale claim lease | taken over; halves; a lone job lost `crashes_charged` times | `Failed(JobCrashedWorker)` (content) |
| workers of a class never start | missing ready marks | backoff; after `start_failures` the class is broken | `NotRun(held=True)` for its queued jobs |
| cancel, `fail_fast` | the driver | unclaimed tickets get `withdrawn` outcomes; claimed ones are abandoned | `NotRun(held=False)` |
| policy refusal raised by a task or hook | the task | none | `Failed` per job |

- **A held job is not a failure.** Readers stay blocked, the run ends `RunIncomplete` (RFC 0002 §3.1),
  and a later run (a new coordinator epoch) runs the held jobs; a hold counts only for its own epoch or
  a newer one, so a stale coordinator can never hold a newer one's work.
- **A fault's class survives every wrapper.** A load failure that carries a resource record stays a
  resource fault; a wrapper that re-raises keeps the original's record. senselab found a load-time OOM
  turned into an infrastructure fault by a wrapper, which held a healthy environment instead of moving
  to a larger class (R-287, §9).
- **One bisection engine** for local and pool execution (§4.4), with two triggers: a resource fault, and
  a crash (`WorkerLost`, a protocol error, a fatal raised by a request rather than a load). A load is
  never split or retried as a request.
- **OOM attribution is per worker.** A SIGKILL is a resource fault only when the worker's own cgroup
  recorded an OOM kill after the worker's baseline; an OOM in a sibling process of the same Slurm job
  does not turn the next `scancel` into an OOM (senselab R-294).
- **The classifier is a hook**: `classify(exc, phase: Literal["load", "call", "pool"]) -> FaultRecord`.
  The default treats an exception from the task's own code as content (pydra's behaviour today) and an
  unclassified exception from the pool's machinery as infrastructure with `gap=True`, reported once per
  type. A client with a stricter rule (senselab's: every unclassified exception is infrastructure)
  passes its own.

## 7. Composition with the `GraphRunner`

### 7.1 `ready()`, `complete()` and the drivers

- `WarmPool` is an RFC 0002 §3.2 executor with `batches = True`: the asynchronous driver hands it every
  job of a round routed to it, in node order, and it turns them into tickets per queue (§4.2). Each job
  is reported exactly once, as `Result`, `Failed` or `NotRun`, from the follower thread through the
  bound callable.
- `checks_store = True`: workers run `Job.run`, which looks up and publishes. The runner may use
  `lookup_before_dispatch` or not; a hit found by a worker settles as a hit either way.
- **`max_concurrent`.** RFC 0002's `ready(limit)` caps jobs handed out per call. A pool caps itself (by
  workers, quantum and `max_queued_jobs`), so it should be driven with `max_concurrent = inf`; with a
  cap, the runner should apply it by whole node groups (the architecture review behind this proposal
  recommends this for R7/R8), otherwise one 100-element node arrives as 25 rounds of 4 and is cut into
  25 tickets.
- **Same-checksum twins** are folded by the runner (RFC 0002 §3.1, R7) before they reach the pool; the
  pool's follow index (§5.5) folds what remains across runs.
- **`retry(job)`** of a `NotRun(held=True)` job hands it out again; the pool puts it on its queue only
  once the hold's epoch has passed or the hold is lifted (`pydracli pool release-hold`).

### 7.2 External drivers

A system that drives the runner itself (RFC 0002 §3.5) uses `WarmPool` as it would any executor:
`bind(report)`, `submit(rnd.dispatch)`, and `runner.complete` on each report from its own thread
(`GraphRunner` is single-threaded, so reports are marshalled to the driver's thread). Nothing in the
pool depends on `Submitter`.

### 7.3 Outcomes and provenance

- Every executed job's provenance is written in the worker by the store's `on_commit` (RFC 0003 §7),
  before the entry is visible. `Round.settled` gives the runner's `reused`, `absent` and
  `workflow-stored` records (RFC 0002 §3.4) unchanged.
- The pool's own events (claimed, lost and taken over, held, moved up the ladder, unfit recorded) go to
  the audit hook as messages with the job's checksum and attempt, so a client can log every job's path
  through the pool without reading the pool directory.

### 7.4 `plan()`

When the executor a node is routed to has `place`, RFC 0002 R8's `plan()` calls it and each
`PlanNode` carries the placement: class, queue, every class tried with its verdict and numbers, and the
batch. An unplaceable node shows as such before anything runs.

## 8. Identity

Warm state must never reach a key, a lookup context or an output.

- **Keys.** A job's checksum is computed in the submitting process (RFC 0002 §3.4) from its inputs and
  identity record (RFC 0004 §3.1). A provided field's spec is an input value; an environment's resolved
  reference is already keyed for shell tasks (RFC 0004 §5.3). The pool adds nothing: no worker id, no
  class, no batch size, no cold or warm flag enters a key. Placement never changes a checksum
  (senselab pins this: a plan's hash is placement-independent, §9).
- **Lookup context.** Built once in the submitting process and carried with the job (RFC 0003 §4); the
  pool knows nothing of it.
- **Execute context.** The worker adds one section, `pool`, to `context(job, "execute")`:

  ```json
  {"pool": {"scheme": "pool-facts@1", "worker": "<token>", "host": "<host token>",
            "class": "gpu-h100", "hardware": {"device": "cuda", "name": "…", "memory_mib": 81559,
            "compute_capability": "9.0", "cpus": 8},
            "residency": [{"key": "<digest>", "state": "warm", "since_load": 37, "load_s": 14.6}],
            "batch": {"size": 16, "index": 3}, "losses_survived": 0}}
  ```

  RFC 0003 publishes an entry under the execute context's fingerprint, which excludes the identity's
  `recorded_only` fields (RFC 0003 §4). `pool` must therefore be recorded-only, or every worker and
  every batch size would split one result into many slots. `pydra-identity@2` lists `pool` in
  `recorded_only` from its first release (W11, which ships with RFC 0004 I8 so the header changes
  once). `WarmPool` refuses a store whose identity keys `pool` (`PoolFactsWouldSplitSlots`). Under
  `pydra-identity@1` roots and `PickleStore`, whose slot is a hash of the whole context, the facts go
  to the audit record and `Result.runtime` only.
- **Outputs.** A result computed cold equals one computed warm, and one computed in a batch equals one
  computed alone; this is the task author's statement for batched calls and warm environments (§3.1,
  §4.4) and the pool's tests pin it for pydra's own (§11).
- **Context cost.** RFC 0004 I7 resolves tools per job in the submitting process. A worker never
  recomputes a lookup context; it computes the `pool` section from state it already holds.

## 9. Prior evidence

What this proposal reuses from senselab's execution layer, with its source. Ids `R-*` are rows of
senselab's `specs/20261030-140022-senselab-2x-api/review-register.md` and its adversarial reviews;
`T24-*` are its Phase 3 Task 24 fixes; job numbers are ORCD scheduler ids recorded in its
`reviews/phase3-exit-evidence.md`. Timings were taken on shared nodes and are upper bounds.

**Measured.**
- **Model loads and the first request** (`reviews/t20r10-evidence.md`, one H100, 1000 items per unit):
  load 14.6 s (Whisper-class ASR), 8.9 s (speaker embeddings), 26.5 s (diarization), 23.5 s (a
  transformer PII tagger, 64.5 GB resident); warm per-job worker time 0.013 to 0.34 s for the audio
  units. The first request after a load took 7.8 to 30.3 times the steady prediction; requests from the
  second on were within the steady band. → §1, §4.2 (one load per ticket on a cold queue).
- **Batching** (same source): a batch of 19 was 2.3 times faster per job than a batch of 1 for the ASR
  unit, with device peak 1.69 → about 3 GB (6.4 MB per padded second); speaker embeddings 1.6 to 2.3
  times; diarization at most 10 % faster, and slower at a batch of 90 on the longest stratum (padding
  2.3, peak 9.05 GB). → §4.4: batching is per task and sized by measurement, not a default.
- **Per-item overhead** (`reviews/phase3-exit-evidence.md`, R-253, job 28564898): 0.098 s for a one-job
  item, rising 7.29 ± 0.46 ms per added job; 0.10 to 0.26 s across every measured unit. → §4.4.
- **Pool against in-process**: 1000 CPU items in 291 s in-process, 80.7 s on a two-worker local pool;
  1000 GPU items committed in 1329 s by one H100 worker, after a cold start of 426 s in which nothing
  committed (Steps 2 to 4). → §1, §4.5 (cold start is per worker, so it is paid once).
- **A 30 s window after calibration** at 100 items was scheduling latency (worker start and a 30 s
  poll), not per-item cost (phase-3 ledger, T20R). → §4.5: scaling and `poll_s` are measured separately
  from per-job cost.
- **Resident memory, `m0`, `m1`** per unit on H100 and the fit rule `resident + m0 > 0.9 × grant`
  (`calibration-probe.md`, ADR-009 amendment 2026-11-12, `rt/fit.py`). → §3.3 `MeasuredFit`.

**Behaviour proven on the cluster.**
- **Abrupt stop and resume** (Steps 4, jobs 29047613/29047616/29055900): interrupted at 335 of 1000,
  resumed on a different host; executions 663, reuses 337, no key executed twice, no item claimed
  twice; provenance 9/9. → §5.5.
- **Cross-host leases** (job 29047619): lease and queue checks on three hosts over NFS, overlapping
  holds and lost items. → §5.3.
- **Load OOM on a typed GPU** (job 29047615): the load-OOM path records `unfit` and the next placement
  avoids the class (`pool_unfit_test.py`). → §3.3, §6.
- **Typed grants and QOS**: four coordinators filled a partition's per-user submit limit of 4; every
  worker `sbatch` was refused; the class was broken after 7 starts with a message naming the profile
  values (job 28564896). GPU workers on a preemptable partition sat in the scheduler for hours; one
  worker per class on a capped partition was the workaround. → §4.5 (failed starts hold, they do not
  fail jobs; latency is unbounded when no worker is granted).

**Failure modes found and fixed, which the design and tests start from.**
- T24-9: a `SoftFileLock` left by a SIGKILLed coordinator could be broken only on its own host; a resume
  elsewhere waited for 6 h. Fix: no host-bound lock; a lint test. → §5.3, W4.
- R-273 / T24-2: a resumed coordinator re-cut unfinished jobs into new items while old items were
  queued; 61 jobs ran twice. Fix: follow the live item that holds a job; settle each request's items
  before the next. → §4.2, §4.3 step 5, §5.5.
- T24-2 (B-4): items settled before their measurement was recorded lost their measurements. → §4.3
  step 5 (measurement, then outcome).
- T24-10: the last claim's provenance was written after its commit and lost at stop. → §5.4.
- R-277: the stale bound that judged claims was the process's, not the run's. → §4.3 step 1.
- R-283: the epoch was taken before the fallible steps of start. → §5.3.
- R-286: Slurm submit retries could double a worker. → §4.5.
- R-287, R-289, R-292: fault classes lost through wrappers, hand-typed failures, two server paths
  disagreeing. → §6.
- R-288: two bisection engines, neither attributing a crash to its item. → §4.4, §6.
- R-290, R-285: unsynced exclusive writes; a short write that poisoned a log. → §5.1.
- R-294: cgroup-wide OOM attribution. R-295: a stop that did not wait for its process group. → §4.3, §6.
- Speaker-embedding batches zero-padded to the longest item changed short items' vectors (benchmarks
  amendment 2026-11-03). → §4.4, §8.

**Tests reused as specifications** (senselab `packages/senselab-core/tests/runtime/`):
`execution/queue_test.py` (oldest-first, idempotent put, one outcome, expired claim reclaimed, double
claim commits once, lost reply on the outcome link, claim cost flat in queue size, never pickled),
`execution/pool_worker_test.py` (idle exit, quantum yield, gather once warm, resident memory out of
per-item samples, device memory once per claim, failing batch bisected, settle before the next request,
reclaimed logged, claim renewed during a long item, SIGTERM releases, OOM kill is resource, a broken
environment costs one load per claim, a lost claim abandons its request), `execution/worker_set_test.py`
(bounded LRU, busy never evicted, evicted process closed, failed client replaced),
`execution/substrates_test.py` (QOS cap, class to `sbatch` args, one CUDA worker per device, failed
starts back off and stop, workers that never start hold their jobs, scoped job names, logs under the
private directory), `execution/pool_follow_test.py`, `execution/abrupt_stop_test.py` (20 boundaries ×
2 host shapes), `execution/resume_test.py`, `execution/pool_unfit_test.py`, `fit_test.py`,
`locks_test.py` and `probe_lease_cross_node_test.py`. W-items port them (§11).

## 10. Migration

### 10.1 In pydra

- **Unchanged:** every existing worker, `Submitter`'s defaults, checksums, store layouts, `debug`
  semantics, environments' cold `execute`.
- **New and opt-in:** `pydra.workers.pool` (`WarmPool`, `WorkerClass`, `LocalSubstrate`,
  `SlurmSubstrate`, `ExternalSubstrate`, `DeclaredFit`, `MeasuredFit`, `DirectoryCalibration`,
  `FaultBudgets`, `PoolSettings`, `PrivatePickle`, `JobCodec`); `pydracli pool {worker,classes,status,
  release-hold,collect}`; `Provider`, `Provided`, `field(provided=)`, `ResidentSet`;
  `Environment.warm()`, `WarmEnvironment`; `ComputeResources.gpu_mem` and `.residency`;
  `python.define(batched=, units=)`.
- **Superseded proposals:** 0001's I and the provider and batching parts of H. 0001 §3.6's
  prerequisite H0 (`chdir=False`) has merged and is what lets one process run many jobs.

### 10.2 senselab

senselab 2.x has no compatibility requirement: its executor, pool and worker APIs, its keys and its
store may change to adopt this design, and it may pin pydra by commit or vendor it before a release. It
switches in one pull request, with no period in which both implementations run.

**Deletes** (`rt/`, `ex/`): the pool controller and scaler (`ex/pool.py`), the claim loop
(`ex/pool_worker.py`), the queue (`ex/queue.py`), the epoch (`ex/epoch.py`), the substrates and fleet
(`ex/substrates.py` `Local`, `Slurm`, `Fleet`, `scale`), the `WorkerSet` and `ModelCache`
(`ex/workers.py`, `rt/host_worker.py`), the placement walk (`rt/fit.py` `judge`, `choose`,
`larger_class`), the bisection engine (`ex/job.py`), its `Executor` protocol (`admit`, `release`,
`fragment_size`, `bind_epoch`, `PlanFragment`, `ResultRef` handles) with `ex/in_process.py`, and the
runner bridge the architecture review lists (`ExecutorWorker`, `Follower`, `serve_shared`, `_upstream`).
Their tests leave with them, after their cases are ported into pydra's (§9, §11).

**Keeps, as a client of pydra:**
- backends as pydra tasks whose model is a `provided=` field, and providers that load a model by a
  resolved 40-hex revision (its pinning rule), with `footprint` and `version`;
- its subprocess-venv and server environments as pydra `Environment`s with `warm()` (the venv one after
  G2);
- its calibration probe (deliberate strata runs, batch sizing per stratum) writing pydra
  `DirectoryCalibration` records, and `MeasuredFit` with its site headroom;
- its profiles, loaded into `WorkerClass` lists, `FaultBudgets` and `PoolSettings`;
- its fault classifier (every unclassified exception is infrastructure) as the `classify` hook;
- its store identity and entry format over `PortableStore` (RFC 0003 P9), with `pool` recorded-only;
  its licence gate in `Reuse.verdict`; its PROV mapping in `on_commit`, reading the `pool` section;
- its run directory and event log (its ADR-008), fed from the driver's reports and the audit messages.

**Switch-over.** After W1–W11 are integrated: senselab's driver builds `WarmPool(root=<run>/pool, ...)`
from its profile, routes GPU and model nodes to it and the rest to `debug` or `cf`, and deletes the
modules above. Keys may change once (a store recompute is acceptable before 2.0.0). Proof: its ported
ADR tests, then the ORCD steps of §11 (W12), then its own Phase 3 exit steps 2 to 5 rerun at the pin.

## 11. Delivery

| PR | Contents | Depends on |
|---|---|---|
| W1 | `WorkerClass` and its loader; `ComputeResources.gpu_mem`, `.residency`; `place()`, the ladder, `Fit`, `DeclaredFit`, `ClassTried`, `Unplaceable`. Tests: oversized job placed on the larger class; fits nothing → `Unplaceable` naming every class and number; GPU request never placed on CPU; class above the CPU cap skipped; placement leaves the checksum unchanged | F1 (merged) |
| W2 | `ResidentKey`; `Provider`, `Provided`, `field(provided=)`; `ResidentSet` (LRU, host and device budgets, busy never evicted, `close` on evict); providers under `debug`. Tests: provided spec keyed and object not; two specs two keys; eviction order; a busy resident survives a tighter budget; `debug` loads once across two `Submitter` calls; unregistered provider is a wiring error | H0 (merged), RFC 0004 I0 |
| W3 | `Environment.warm()` and `WarmEnvironment` for `Docker`, `Singularity`, `Lmod`; `covers` and the cold fallback; `setup`/`teardown` called. Tests (fakes for `docker`, `apptainer`, `lmod`): one `docker run -d`, N `docker exec`; one `module load` resolution for N commands; a binding outside the mounts runs cold in the same worker; warm and cold outputs equal; `Native` returns `None`; marked tests against a real daemon and a real Apptainer | RFC 0004 I6a, I6b (resolved references), W2 |
| W4 | Pool storage: `pool.json`, queues, tickets, segments, claim leases with `try_take`, per-job outcome records, holds, the follow index, `losses/`; `PrivatePickle` with the ownership walk and digest, `JobCodec`. Tests ported from senselab's `queue_test.py`; a pickle in a group-writable directory refused; a ticket never contains a pickle; a lint test that no pool module takes a host-bound lock | P1, P4c, RFC 0002 R6 (`try_take`), #928 |
| W5 | The worker: claim loop (oldest first, warm preference, quantum, gather once warm), hardware check, ready mark, heartbeat and proofs, per-job settle after measurement, `Job.run` with `chdir=False`, warm environments for shell tasks, `Sample`s, SIGTERM drain, idle and orphan exit, process-group stop; `pydracli pool worker`. Tests ported from `pool_worker_test.py`; a shell task in a warm Docker worker equals `debug`; a settings mismatch refuses to start | W2, W3, W4, RFC 0003 P8 |
| W6 | `WarmPool` and `LocalSubstrate`: placement at submit, tickets per queue, follow, the follower and read-back through `Job.result()`, outcome mapping, `cancel` → `NotRun`, `max_queued_jobs`, scaling, one worker per GPU, failed starts. Tests: a 100-element round becomes tickets of at most `quantum_jobs`; resubmission follows live tickets and runs nothing twice; a pool whose workers never start holds its jobs (`RunIncomplete`); twins folded by the runner reach the pool once | W5, RFC 0002 R7 |
| W7 | Faults: `classify` and the default table, `FaultBudgets`, one bisection engine with resource and crash triggers, ladder moves, crash counting and `JobCrashedWorker`, holds per residency and epoch. Tests ported from `bisection_engine_test.py`, `fault_class_test.py`, `pool_backend_oom_test.py`; a load OOM moves up and charges nothing; a class survives a wrapper; a sibling's OOM does not type a `scancel` | W6 |
| W8 | `SlurmSubstrate`: one `sbatch` per worker with typed `--gres`, scoped names, one-shot submission with name recovery, the shared `squeue` poller, logs under `private/`, the per-user CPU cap. Tests ported from `substrates_test.py` with a fake `sbatch`/`squeue`; marked Slurm tests in the existing Slurm CI container | W6, RFC 0002 R5 |
| W9 | Calibration: `Sample`, `UnfitRecord`, `TimeRecord`, `DirectoryCalibration`, `MeasuredFit` (`judge`, `batch`); `place` for `plan()`. Tests ported from `fit_test.py`, `pool_unfit_test.py`; an unfit record bound to a provider version stops binding after a version change; `plan()` shows the classes tried | W7, RFC 0002 R8 |
| W10 | Batched calls: `python.define(batched=, units=)`, grouping by units, bisection through W7. Tests: a batched call's results equal the unbatched ones for pydra's own test tasks; one crashing item isolated in log₂ n calls; a half that completed keeps its results | W7, W2 |
| W11 | Identity and provenance: `pool` in `pydra-identity@2`'s `recorded_only`; `PoolFactsWouldSplitSlots`; the `pool` section; audit messages. Tests: cold and warm results share one slot; two workers' results share one slot; under `@1` and `PickleStore` the facts reach only the audit | RFC 0004 I1, ships with I8; RFC 0003 P5 (`on_commit`) |
| W12 | ORCD suite and documentation (below) | W8, W9, W3 |

W1, W2 and W4 can start at once (W4 after P4c). W3 waits for RFC 0004's I6a/I6b; W11 rides the I8
roll-up so the `@2` header carries `pool` from its first release. RFC 0002 R7 (`NotRun`, `attempt`,
`RunIncomplete`) gates W6; R8 gates W9's `plan()` part. RFC 0003 P4c (work directories, `attempt`,
`_committed.json`) gates W4 and P8 (reporting `Committed`) gates W5. senselab's switch-over (§10.2)
follows its S1 repin (after R7) and P9, and needs W1–W11. Every PR has tests on `debug` and `cf` where
they apply, and none changes an existing test's expectation.

**ORCD suite (W12).** Submitted jobs from a commit-pinned checkout, typed `--gres=gpu:h100:<n>`, never
run on a login node; each reports a verdict line per check.
- `test_one_load_n_jobs_h100`: one `SlurmSubstrate` worker, 1000 jobs of a provided-model task;
  asserts one `sbatch`, one load (the provider's load counter), every job committed, and per-job wall
  after the load within the calibration's band.
- `test_two_residencies_two_queues`: two models on one class; asserts two queues, the quantum yield, and
  at most `max_workers` loads per model.
- `test_worker_killed_mid_ticket`: `scancel --signal=KILL` of a worker after a third of the jobs; asserts
  takeover after `stale_after`, every job committed once, losses recorded.
- `test_coordinator_killed_and_resumed_on_another_host`: SIGKILL of the submitting process, resume from a
  second node; asserts follow (no job put twice), executions + reuses = jobs.
- `test_load_oom_climbs_the_ladder`: a class whose usable device memory is capped (by
  `torch.cuda.set_per_process_memory_fraction`) below the model; asserts an `UnfitRecord`, the move to
  the larger class, no job charged, and that the next `plan()` starts at the larger class.
- `test_local_multi_gpu`: `LocalSubstrate` inside one allocation of two H100s; asserts one worker per
  device by `CUDA_VISIBLE_DEVICES` and both busy.
- `test_mistyped_grant_refused`: a class naming `gpu_type="a100"` on an H100 grant; asserts failed starts,
  then `NotRun(held=True)` and no job failed.
- `test_warm_apptainer_shell_jobs`: 500 shell jobs of one Apptainer image (a CPU tool) on one worker;
  asserts one instance start, outputs equal a cold `debug` run, and records wall per job warm against
  cold so the saving is measured rather than asserted.
- `test_lmod_resolved_once`: 200 shell jobs under an `Lmod` environment; one resolution per worker.
- `test_plain_workflow_untouched`: the existing Slurm tests with the pool absent; no `pydra.workers.pool`
  import, no new files.

**Documentation.** A how-to for a shell workflow on a warm Apptainer pool, one for a GPU model on
Slurm, and the worker protocol as a reference page.

## 12. Open questions for maintainers

- **Home.** `pydra.workers.pool` in core (proposed, because routing, `ComputeResources` and the
  environments change with it), or a `pydra-workers-pool` plugin package as 0001 §3.7 said?
- **`provided=`.** Is a field attribute with `Provided[T]` the right spelling, or should providers be
  bound per node (`workflow.add(task, provide={"model": spec})`), keeping task signatures plain?
- **Warm environments' isolation.** Is "fresh work directory per job, optional read-only root" enough
  for warm containers, or should pydra restart a warm container every N jobs by default?
- **A cold re-check.** Should the pool offer a sampled check that recomputes a fraction of jobs on fresh
  resident state (and in unbatched calls) and compares output digests, reusing RFC 0003's `Verify`?
- **Content retries.** pydra today does not retry a failed job. Is `content_attempts=1` the right
  default, with retries opt-in?
- **Services.** Should one resident state be consumable by many workers (an inference server, a
  database), as senselab's `svc` classes are, or does that stay outside pydra?
- **Group pools.** Should a pool be shareable within a group, with codec-only (no pickle) payloads, or
  stay single-user?
- **Calibration home.** Should `DirectoryCalibration` live beside the store (one per root) or be passed
  explicitly (proposed)?
- **The CPU cap.** Is a single `per_user_cpu_cap` enough to express site limits, or should a substrate
  take a callable that answers "can I start one more of this class"?
- **`max_concurrent` with batching executors.** Should RFC 0002's drivers pass no limit to an executor
  with `batches = True`, rather than relying on documentation?
- **Venv residency.** Should the warm `Venv` environment wait for G2, or should this proposal define the
  in-environment worker protocol itself?

## 13. Changes in revisions

Revision 1. No review yet.

| Finding | How it is addressed |
|---|---|
