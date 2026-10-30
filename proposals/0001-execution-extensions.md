# Proposal 0001: execution extensions for large, heterogeneous workloads

Status: proposed, for maintainer discussion. Branch `enh/execution-extensions`, off `main` at
`c5081a6` (1.0a12).

## 1. Motivation

Pydra's graph model (typed tasks, lazy fields, split/combine, nested workflows, content-hash
caching) is general-purpose. These proposals widen the workloads its execution layer serves, from
neuroimaging to other scientific domains:
- **neuroimaging:** a per-subject split over thousands of subjects, with GPU segmentation on a
  cluster partition and CPU post-processing locally;
- **behavioural and physiological signals:** a corpus of 10⁵ audio, video or wearable recordings,
  where a GPU model serves many items and some steps need a different Python or library stack;
- **other domains** with the same shape, such as genomics per-sample pipelines and ML evaluation
  sweeps: many small items, mixed CPU/GPU steps, shared cluster storage.

Today such users work around pydra's execution layer rather than through it. senselab
(sensein/senselab#617) is one motivating user. Nipype 2 and pydra-tasks users get the same
benefits, and the defaults stay unchanged for them (§2).

The following was measured on `c5081a6`. The scripts are in `benchmarks/proposal_0001/`. The machine
was loaded, so treat timings as upper bounds.

| Problem | Evidence |
|---|---|
| One failed element blocks all downstream elements of a split | `pydra_partial_failure.py`: 8-way split, 1 failure, and the 7 healthy elements never run the next node. `engine/submitter.py:853` (`if True:  # self.node.splitter:`) marks the whole downstream node unrunnable; the per-index path exists but is disabled (FIXME at `submitter.py:850`) |
| Per-item overhead dominates small tasks | `pydra_overhead.py`: about 4–5 ms per trivial item cold and 2–3 ms per cached item; about 34 KB and 3.3 files per item; `cf` no faster than `debug`, with about 45% of the time spent pickling. Under `cf` the job is pickled (`workers/cf.py:75`) with the task pickled again inside it (`Job.__getstate__`, `engine/job.py:185`); `_job.pklz` is written on every run (`job.py:311`) |
| The cache is not safe when shared | run-long `SoftFileLock` (`job.py:331`); rmtree then rewrite (`job.py:307-308`); `_result.pklz` written in place (`engine/result.py:219`). `clean_stale_locks` (`submitter.py:107,162-166,529-535`) unlinks any lock older than this run's start, which on a shared cache can remove another submitter's live lock (#778). Cloudpickle results tie a cache to one Python version |
| A Submitter has exactly one worker | `submitter.py:160` |
| Python tasks ignore environments | `compose/python.py:228-233` never consults `job.environment` (shell tasks do, at `compose/shell/task.py:259`); the checksum is task-only (`compose/base/task.py:476`) (#826, #840) |
| Resources are per worker, not per task | `sbatch_args` is worker-wide. The SGE worker does read a per-job `qsub_args` (`workers/sge.py:111,119`), which is the precedent to generalise |
| Slurm worker | one `squeue -j` per job per second (`workers/slurm.py:29,145`); requeue is unbounded and includes a user's `CANCELLED` (`slurm.py:128-139`; `--no-requeue` opts out); `--error` in `sbatch_args` crashes it (`slurm.py:104,118`), as `_verify_exit_code` would too (`slurm.py:168`) |
| No long-lived state across jobs | expensive objects, such as a loaded neural network, a template or atlas held in memory, or a reference index, are rebuilt per job |
| No plan before execution | the static graph prints, but splits and expected cache hits are not shown |

## 2. Goals, non-goals, compatibility

**Goals:**
- per-element dataflow;
- cheap per-item execution;
- a cache that is safe to share, with an optional Python-version-neutral result format;
- several workers per run;
- Python tasks in environments, with environment identity in the checksum;
- per-task resources;
- long-lived worker-scoped objects with batching;
- a robust Slurm worker;
- a printable plan.

**Non-goals:**
- Changing the task definition language or lazy-field semantics.
- Making containers mandatory.
- Multi-node model parallelism.

**Defaults that do not change:**
- `PickleStore`;
- the cache layout until a major version;
- the `debug` default worker;
- Slurm requeue on `PREEMPTED`/`TIMEOUT`;
- error semantics under `debug`;
- the `os.chdir` behaviour for Python tasks that write relative paths;
- checksums of tasks that use no environment, or `Native`/`Lmod`.

Anything new is opt-in until maintainers decide otherwise.

## 3. Proposals

### 3.1 Per-element dataflow and keep-going
- **Behaviour:** a downstream element becomes runnable when its own upstream indices succeed. An
  errored element marks only its own descendants unrunnable.
- **API:** `Submitter(on_error="raise_at_end" | "fail_fast")`.
  - `raise_at_end` is today's async-worker behaviour (`submitter.py:446-476`): independent nodes
    keep running, and errors are raised together at the end.
  - `fail_fast` stops at the first error.
  - The per-element change applies under both modes.
  - The error summary is attached to the raised workflow error and kept on the Submitter, not on
    the per-job `Result`.
- **Two steps:**
  1. Enable the per-index path where the downstream splitter does not re-split an already split
     state.
  2. The partial-`State` refactor that the FIXME names.
- **Risk:** high. The tests need the full splitter/combiner matrix (outer, scalar, nested, combine
  after partial failure) and `debug` parity.

### 3.2 Cheap submission path
- **Step 1:** extend `benchmark.py` with the scripts above, so maintainers can reproduce the
  numbers.
- **Step 2:**
  - Avoid re-pickling the task inside the job for in-process workers.
  - Write `_job.pklz` only for remote workers and on failure, which keeps
    `pydracli crash --rerun` working; announce that it is gone on success.
  - Check cache existence with one `scandir` per cache root per scheduling round.
  - Avoid recomputing `Task._hash` (`task.py:471`). The job checksum is already memoised
    (`job.py:219-222`).
- **Target:** under 0.5 ms per item cold and under 0.1 ms per cached item for trivial tasks.

### 3.3 Result store interface (first), then safe commits
- **Step 1:** a `ResultStore` interface that touches the four call sites (`Job.checksum`,
  `cache_dir`, `save`, `load_result`), with a behaviour-identical `PickleStore`. No behaviour
  changes.
  ```python
  class ResultStore(Protocol):
      def lookup(self, checksum: str) -> Result | None: ...
      def acquire(self, checksum: str) -> Lease: ...   # replaces the run-long SoftFileLock
      def commit(self, lease: Lease, result: Result) -> None: ...
  ```
- **Step 2: safe commits inside the store.**
  - The job runs in `<checksum>/` under a **lease** (owner, host, pid, heartbeat, TTL takeover)
    instead of a run-long soft lock.
  - The result is committed by an atomic write-then-rename of the result file.
  - Shell tasks resolve `output_file_template` and copy destinations against `cache_dir`
    (`job.py:280-288`), and `can_resume` needs a stable directory. Running in place and
    committing atomically keeps both working. This is lower risk than running in a temporary
    directory and renaming it.
  - `clean_stale_locks` becomes lease expiry, so it never removes a live holder's lock.
- **Step 3: `PortableStore`** (opt-in). JSON for scalars and attrs records, `.npy` for arrays,
  files as files, and pickle only for types without a registered serialiser. It uses a versioned
  layout (`tasks-v<N>/`) (#825).
- **Tests:**
  - two processes racing on one checksum;
  - SIGKILL mid-run, then the rerun takes over the lease;
  - a read-only shared cache;
  - `debug` and `cf` parity.

### 3.4 Per-task resources and worker routing
- **`ComputeResources`:**
  - `ComputeResources(cpus, mem, gpus, gpu_type, time, tags)` is set by the define machinery in
    `compose/base`, so it works for `python.define` and `shell.define` alike.
  - It can be overridden per node with `workflow.add(task, resources=...)`, beside the existing
    `environment=` and `hooks=` (`compose/workflow.py:235`).
  - It is never part of the checksum.
  - The Slurm and SGE workers map it to `sbatch` and `qsub` arguments, generalising SGE's per-job
    `qsub_args`.
- **Routing (a separate PR):**
  - `Submitter(workers={"local": "cf", "gpu": SlurmWorker(...)}, router=...)`.
  - The router maps a job's resources or tags to a worker name, and is called at the three
    dispatch sites (`submitter.py:292-295`, `:328`, `:435-441`).
  - Workflow definitions name no worker, so reusable task packages stay worker-agnostic.

### 3.5 Environment identity, and Python tasks in environments
- **Step 1: identity (independent; closes #826).**
  - Add `Environment.checksum()`, folded into the job checksum.
  - `Native` and `Lmod` contribute nothing, so existing caches keep their hashes.
  - `Docker` and `Singularity` accept a user-supplied `identity=`. That avoids contacting a
    registry or hashing a multi-GB image at hash time; a resolved digest is used when available.
- **Step 2: Python tasks in environments** (closes #840; depends on 3.3, step 3).
  - A `Venv(path=... | python=..., lock=...)` environment.
  - When the environment's interpreter version matches, the task is cloudpickled as today.
  - Otherwise it is shipped by import reference with `PortableStore`-encoded inputs. Functions
    defined in `__main__`, notebooks or closures then raise a clear error asking for an
    importable function.
  - A small stdlib runner inside the environment executes the task. The runner is pluggable.
  - Could ship as `pydra-environments-venv`, using the existing namespace mechanism
    (`environments/base.py:52-65`).

### 3.6 Worker-provided objects and batching
- **Field attribute:** a new `Arg` attribute `provided=<name>` (following the hash-exclusion
  precedent of `container_path`, `task.py:494`). It marks an input as supplied by the worker. Only
  the provider's declared key enters the checksum.
- **Provider registry:** `Submitter(providers={"segmenter": provider, "template": provider})`. A provider maps a key to an
  object, with LRU eviction and a memory budget, and workers keep providers alive across jobs.
  "Provided" avoids a clash with `ComputeResources`.
- **Batching:**
  - A task may declare a batched implementation.
  - A batching worker groups ready jobs that share a provider key.
  - Each batch returns a result per item, and a failing batch is bisected so one item cannot fail
    its batch-mates.
- **Prerequisite (its own PR):** an option to run Python tasks without `os.chdir` (`job.py:338`),
  so several jobs can share one process. Default unchanged.
- **Packaging:** ships with the pool worker (3.7) as a plugin.

### 3.7 Pool worker (plugin) and Slurm fixes (core)
- **Slurm, three PRs:**
  - **A0:** the `--error` crash, in both places, with a test.
  - **A1:** a requeue budget, and no requeue of a user's `CANCELLED`.
  - **A2:** one `squeue --me` per poll cycle for all jobs, and `sbatch --array` for split states,
    reusing the tuple path at `slurm.py:49-62`.
- **`pydra-workers-pool` (plugin):**
  - Jobs go to a lease queue on a shared filesystem.
  - Pool processes on any substrate (local, Slurm, Kubernetes) claim, run and commit through the
    `ResultStore`.
  - A controller scales the pool processes from queue depth within caps.
  - The scheduler then sees a few long jobs instead of one job per element.

### 3.8 Provenance through the existing audit and messengers
- Extend `Audit` and the messengers (`engine/audit.py`, `utils/messenger.py`) to emit one record
  per job: used checksums, generated outputs, environment checksum (3.5), host, times and status.
- It uses a JSON-lines messenger and needs no new system.
- It depends on 3.5 step 1 and on 3.3 step 1.

### 3.9 `Submitter.plan()`
- Lists nodes, split sizes, resources and worker per node, and expected cache hits.
- Nodes inside workflows built at run time are marked deferred.
- Depends on 3.4 and on the cache-existence scan in 3.2.

### 3.11 Declared task identity and context-aware cache reuse
- **Problem.** Function hashing uses the AST, falling back to bytecode. It misses changes in called
  functions (#777), generic objects hash brittlely (#776), and the same function can hash
  differently across platforms or Python versions. A hash can show exactness but cannot measure
  distance.
- **Declared identity.** A task may declare a `version` (`@python.define(version="2")`). The
  checksum then uses that version instead of the function's code. CI in the defining project
  guards it: changing the source without bumping the version fails. Types can register their own
  byte representation, extending `__bytes_repr__`, so domain objects hash portably.
- **Execution context.** Each cache entry records the context that produced it (environment
  checksum from 3.5, platform, Python version) beside the result, outside the checksum.
- **Reuse policy.** `Submitter(reuse="exact" | "any")`, with `exact` as the default and today's
  behaviour. `"exact"` reuses only entries from the same context; `"any"` reuses across contexts
  and marks the result as such. Downstream projects can plug in richer policies, such as measured
  equivalence classes.
- **Risk:** low to medium. It is opt-in, and the default keeps today's checksums.

### 3.10 Graph runner and executor split: a separate RFC
- Separating `ready()`, `complete()` and `plan()` from execution would let external orchestrators
  drive pydra's graph.
- It depends on 3.1 (per-element readiness) and 3.3. It will be written up as its own RFC once
  those land.

## 4. Delivery

| PR | Contents | Depends on |
|---|---|---|
| A0 | Slurm `--error` fix | — |
| A1 | Slurm requeue budget, `CANCELLED` | — |
| B1 | benchmarks in `benchmark.py` | — |
| B2 | cheap submission | B1 |
| D1 | `ResultStore` interface + `PickleStore` (no behaviour change) | — |
| C | leases + atomic commit inside the store | D1 |
| G1 | `Environment.checksum()` | — |
| E1 | per-element readiness, no re-split case | — |
| E2 | partial `State` refactor | E1 |
| F1 | `ComputeResources` (+ Slurm/SGE mapping) | A1 |
| F2 | worker routing | F1 |
| A2 | Slurm batched polling + arrays | A1 |
| D2 | `PortableStore` + versioned layout | D1 |
| G2 | `Venv` environment + runner | D2, G1 |
| H0 | Python tasks without `chdir` (opt-in) | — |
| H | provided objects + batching (plugin) | H0, F1, D1 |
| I | `pydra-workers-pool` | C, D2, F2 |
| J1 | provenance via messengers | G1, D1 |
| J2 | `plan()` | F2, B2 |
| K1 | declared task identity + context-aware reuse | G1, D1 |

Each PR has tests on `debug` and `cf`, and on Slurm where relevant. PRs merge to `main` when the
maintainers accept them and ship in the next 1.0 alpha. Downstream projects can track this branch
through `uv` git sources in the meantime.

## 5. Open questions

- Which `on_error` mode should be the 1.0 default?
- Should `PortableStore` become the default at the next major version?
- Are `provided=` and `ComputeResources(tags=...)` acceptable names?
- Which plugins live under `nipype/` and which in separate repositories?
