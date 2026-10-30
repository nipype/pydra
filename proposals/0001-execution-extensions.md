# Proposal 0001: execution extensions for large, heterogeneous workloads

Status: draft, for discussion. Branch: `enh/execution-extensions`, off `main` at `c5081a6`
(1.0a12). Changes land in 1.0 alphas as they are ready.

## 1. Motivation

Pydra's graph model (typed tasks, lazy fields, split/combine, nested workflows, content-hash
caching) fits pipelines that are large and heterogeneous:
- 10⁴–10⁶ items;
- some steps on GPUs holding a large model;
- some steps needing a different Python or library stack;
- running on a laptop and on Slurm clusters with shared network filesystems.

Users with these workloads (neuroimaging at scale, and senselab for behavioural signals) currently
work around pydra's execution layer rather than through it.

The following problems were measured on `c5081a6`. Scripts are in senselab
`specs/20261030-140022-senselab-2x-api/bench/`. A loaded laptop was used, so treat timings as upper
bounds.

| Problem | Evidence |
|---|---|
| One failed element blocks all downstream work in a split | 8-way split, 1 failure: the 7 healthy elements never run their downstream node. Barrier semantics in `engine/submitter.py` (`get_runnable_tasks`). |
| Per-item overhead dominates small tasks | about 4–5 ms per trivial item cold and 2–3 ms per cached item; about 34 KB of pickles and 3.3 files per item; `cf` no faster than `debug`, with about 45% of time spent pickling |
| The cache is not safe on shared filesystems | run-long `SoftFileLock` left stale by SIGKILL (#778); non-atomic writes; cloudpickle results tie a cache to one Python and are unsafe to share across users |
| A Submitter has exactly one worker | `engine/submitter.py:160`; there is no way to send GPU nodes to Slurm and CPU nodes to local processes in one run |
| Python tasks ignore environments | `compose/python.py` `_run` never consults an environment (#840); environments do not enter the checksum (#826) |
| No per-task resources | only worker-wide `sbatch_args` |
| Slurm worker | one `sbatch` per job; a `squeue` poll per job every second; unbounded requeue, including a user's own `scancel`; crashes when `--error` is passed in `sbatch_args` (`workers/slurm.py:118` calls `.replace` on `None`) |
| No long-lived state across jobs | a loaded model survives only by accident in `cf` processes |
| No plan before execution | the static graph prints, but splits and expected cache hits are not shown |

## 2. Goals and non-goals

**Goals**
- Per-element dataflow.
- Cheap per-item execution.
- A shared-filesystem-safe, Python-version-neutral cache.
- Several workers per run, with routing per node.
- Python tasks in environments, with environment identity in the checksum.
- Per-task resources.
- Long-lived worker-scoped resources with batching.
- A robust Slurm worker and a pool worker.
- Provenance records.
- A printable plan.
- In the long term, a graph runner separable from execution.

**Non-goals**
- Changing the task definition language or lazy-field semantics.
- Making containers mandatory.
- Multi-node model parallelism.

**Compatibility.** Each change keeps current behaviour by default. Where a default must change (the
cache layout, hashes that gain environment identity), the change is a one-time recompute,
announced in the alpha's release notes.

## 3. Proposals

Each proposal gives the API sketch, where it attaches, the risk, and its tests. They are numbered
in delivery order: 1–5 are the foundation, 6–11 build on them, and 12 is the end state.

### 3.1 Per-element dataflow and keep-going
- **API:** `Submitter(on_error="stop" | "continue")`, default `"stop"` for now. `Result.errors`
  gives a summary of failed elements with their tracebacks.
- **How:**
  - Enable the per-index readiness path in `NodeExecution.get_runnable_tasks`. A downstream
    element becomes runnable when *its* upstream indices succeed.
  - An errored element marks only its own descendants unrunnable.
  - This needs the partial-state-from-lazy-fields refactor that the FIXME in `engine/state.py`
    names.
- **Risk:** high. It touches splitter/combiner index bookkeeping.
- **Tests:**
  - partial failure in split, nested split, combine after partial failure, outer and scalar
    splitters;
  - `stop` keeps today's behaviour.

### 3.2 Cheap submission path
- **How:**
  - Do not re-pickle tasks for in-process workers (`Job.__getstate__`).
  - Write `_job.pklz` lazily, only for remote workers or crash dumps.
  - Memoise the checksum per job.
  - Check cache existence with one `scandir` per cache root per scheduling round.
- **Target:** under 0.5 ms per item cold and under 0.1 ms per cached item for trivial tasks.
  A benchmark script is added to `benchmark.py`.
- **Risk:** medium. `pydracli crash --rerun` depends on `_job.pklz`; keep it for failures.

### 3.3 Atomic cache with leases
- **How:**
  - Run each job in `cache_root/tmp/<uuid>` and `rename` into `<checksum>/` on success. The first
    writer wins; later writers discard their copy.
  - Replace the run-long `SoftFileLock` with a lease file (`owner`, `host`, `pid`, `heartbeat`)
    renewed periodically. Stale leases are taken over after a TTL.
  - The protocol is the one senselab's `work_pool` measured on NFS v4.2: 2000/2000 items, no
    double or lost runs.
- **Risk:** medium. Tasks that record their own output paths during the run need path rewriting
  after the rename. `copyfile` outputs are relocated as part of the commit.
- **Tests:**
  - two processes racing on one checksum;
  - SIGKILL mid-run, then a rerun takes over the lease;
  - a read-only shared cache.
- Closes #778.

### 3.4 `ResultStore` and a portable result format
- **API:**
  ```python
  class ResultStore(Protocol):
      def lookup(self, checksum: str) -> Result | None: ...
      def begin(self, checksum: str) -> Staging: ...          # tmp dir + lease
      def commit(self, staging: Staging, result: Result) -> Result: ...
  ```
  `Submitter(result_store=...)`. The default is `PickleStore`, which behaves as today.
  `PortableStore` writes JSON for scalars and attrs records, `.npy` for arrays, files as files,
  and falls back to pickle only for types with no registered serialiser. Serialisers are
  registered per type.
- **Cache layout** versioned independently of pydra (`tasks-v1/…`). Upgrading pydra does not
  invalidate a cache unless the layout version changes (#825).
- **How:** all reads and writes go through four places today (`Job.checksum`, `cache_dir`,
  `save`, `load_result`). The store slots in there.
- **Risk:** medium. Every result reader is affected.

### 3.5 Several workers per run, with routing
- **API:**
  ```python
  sub = Submitter(worker={"default": "cf", "gpu": SlurmWorker(partition="gpu"), "pool": PoolWorker(...)})
  wf.add(SegmentModel(...), worker="gpu")
  ```
  Alternatively, a `router(job) -> worker_name` callable, for example routing from
  `ComputeResources` (3.6).
- **How:** a router called at the two dispatch sites in `engine/submitter.py`. Nested workflows
  inherit the routing unless they override it.
- **Risk:** low to medium.

### 3.6 `ComputeResources`
- **API:** `ComputeResources(cpus, mem, gpus, gpu_class, time, extra)` set on a task definition
  (`@python.define(resources=...)`) and overridable per node (`wf.add(..., resources=...)`).
  Workers map it to their own terms: sbatch args, local slots, a pool resource class.
- **Never part of the checksum.**
- **Risk:** low. Refs #227, #830.

### 3.7 Environment identity and Python tasks in environments
- **API:**
  - `Environment.checksum() -> bytes`, folded into `Job.checksum`. For example: a Docker image
    digest, a Singularity image hash, or a venv's lock hash.
  - A new `Venv(python=..., lock=...)` environment, or `Venv(path=...)` for an existing one.
- **Python tasks in an environment:**
  - The function is shipped as an **import reference** (`module:function`) with encoded inputs
    (the `PortableStore` serialisers), not as a cloudpickle, so the env may run a different Python
    version.
  - A small stdlib-only runner inside the env imports the function, runs it, and writes a portable
    result.
  - The runner is pluggable, so projects can supply their own. senselab's worker runtime is one.
- **Risk:** medium. Hashes change for tasks that use environments (a one-time recompute).
- Closes #826 and #840.

### 3.8 Worker-scoped resources and batching
- **API:**
  ```python
  @python.define
  def Transcribe(audio: Audio, model: Resource[AsrModel] = resource("asr", key=...)) -> Transcript: ...

  sub = Submitter(worker=..., resources={"asr": provider})   # provider: key -> loaded object, LRU, budget
  ```
  - `resource(...)` fields are excluded from the hash, except for their `key`, which is part of
    identity.
  - Workers keep providers alive across jobs.
  - A task may declare `batch=BatchSpec(max_items, max_wait_ms)` and a `run_batch` function. A
    batching worker then groups ready jobs with the same resource key into one call.
  - `Worker.run` is async, so batching needs no submitter change.
- **Risk:** medium. Process-wide `os.chdir` must be removed from the job path first, so that
  several jobs can share one worker process.

### 3.9 Pool worker and Slurm fixes
- **Slurm fixes:**
  - the `--error` crash;
  - a requeue budget, and no requeue of a user `CANCELLED`;
  - one `squeue --me` per poll cycle covering all jobs;
  - `sbatch --array` for split states.
- **`PoolWorker`:**
  - Publishes jobs to a lease queue on a shared filesystem, with one queue per resource class
    (and resource key, for 3.8 affinity).
  - Pool processes (`pydra pool work <dir> --classes ...`) run on any substrate (local processes,
    Slurm jobs, Kubernetes jobs) and claim, run and commit jobs through the `ResultStore`.
  - A controller scales pool processes from queue depth within caps.
  - The queue holds items and the scheduler holds workers, so a cluster sees a few long jobs
    instead of millions of short ones.
- **Packaging:** the pool can ship as `pydra-workers-pool` to keep core small.
- **Risk:** low for the fixes; medium for the pool.

### 3.10 Provenance sink
- **API:** `Submitter(prov_sink=JsonlProvSink(path))`. One record per job from the
  `post_run_task` hook (which fires on errors too).
- **Record contents:**
  - `used`: input checksums;
  - `generated`: output ids;
  - the environment checksum;
  - host;
  - times;
  - status.
- The record shape follows W3C PROV and maps to BIDS BEP028.
- The existing network fetch of `context.jsonld` is removed. Refs #798, #566.
- **Risk:** low.

### 3.11 `Submitter.plan()`
- **API:** `Submitter.plan(task) -> Plan`, listing nodes, splits and their sizes, the worker and
  resources for each node, and expected cache hits. Nodes inside workflows built at run time are
  marked *deferred*. `print(plan)` gives a readable table.
- **Risk:** low.

### 3.12 Graph runner and executor split (end state)
- **What:** split the Submitter into a **graph runner** (`ready() -> jobs`,
  `complete(job, result)`, `plan()`) and an **executor interface** that the current workers
  implement.
- **Why:** an external system (a queue, an orchestrator) can then drive pydra's graph semantics
  without pydra's execution loop, and the workers become ordinary executors.
- Proposed after 3.4 exists, because job expansion currently needs upstream results from pydra's
  cache.

## 4. Delivery

| PR | Contents | Depends on |
|---|---|---|
| A | 3.9 Slurm fixes (small, independent) | — |
| B | 3.2 cheap submission | — |
| C | 3.3 atomic cache + leases | — |
| D | 3.4 `ResultStore` + `PortableStore` + versioned layout | C |
| E | 3.1 per-element dataflow + keep-going | — (State refactor) |
| F | 3.5 routing + 3.6 resources | — |
| G | 3.7 env checksum + `Venv` runner | D |
| H | 3.8 resources + batching | F, D |
| I | 3.9 pool worker | C, D, F |
| J | 3.10 provenance sink, 3.11 plan | — |
| K | 3.12 runner/executor split | D, I |

Each PR comes with tests on `debug` and `cf`, plus Slurm where relevant (`--only-worker=slurm`
on a cluster). Each merges into `main` and ships in the next 1.0 alpha. Downstream projects can
track a branch through `uv` git sources until then.

## 5. Open questions for maintainers

- Should `on_error="continue"` become the default in 1.0?
- Should the pool worker live in core or in `pydra-workers-pool`?
- Should `PortableStore` become the default once it is stable?
- Is the naming of `resource(...)` and `Resource[...]` acceptable alongside `ComputeResources`?
