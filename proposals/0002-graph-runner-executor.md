# Proposal 0002: a graph runner and an executor in place of one Submitter

Status: proposed, for maintainer discussion. Branch `rfc/graph-runner-executor`, off
`enh/execution-extensions` at `6315ad7e`. This is the separate RFC that proposal 0001 §3.10 names.
Line references are to `6315ad7e` unless a PR is named.

## 1. Motivation

`Submitter` does two jobs. It works out which jobs can run: it constructs workflows, prepares
`State`, resolves lazy fields, tracks per-element readiness and applies `on_error`
(`engine/submitter.py:548-750`, `NodeExecution` at `:889-1262`). It also runs them: it picks a
worker, awaits one coroutine per job and collects the futures (`:700-720`, `:383-400`,
`:819-833`). The two are interleaved in one loop of about 160 lines in `expand_workflow_async`
(`:591-750`), with a second copy for synchronous workers in `expand_workflow` (`:548-589`).

The interleaving has costs that do not depend on any one user:

| Problem | Evidence |
|---|---|
| A worker sees one job at a time, so a worker that wants the ready set rebuilds it | `Worker.run(job)` is awaited per job (`submitter.py:713-716`). The Slurm worker collects jobs until the current event-loop tick ends to form an array (`workers/slurm.py:329-367`). The SGE worker sleeps `collect_jobs_delay` for the same purpose (`workers/sge.py:274-278`). Both groupings depend on asyncio scheduling, not on the graph |
| Completion is inferred from the store, not reported | `NodeExecution.update_status` re-reads each queued or running job's result through `Job.done` (`submitter.py:1061-1096`, `job.py:548-562`). When the store is slow to show a result, the loop retries for ten seconds and then raises "Something has gone wrong" (`submitter.py:621-699`) |
| Jobs are handed out repeatedly and deduplicated after the fact | a node returns every queued job on every round (`submitter.py:1234`); the loop filters them through `finished`, `futured` and `sharing` (`:608-614`, `:700-719`) |
| No other loop can drive the graph | readiness lives in `NodeExecution` and is only advanced from inside `expand_workflow_async`. An external orchestrator, a notebook stepping a workflow, or a test of readiness without processes, has no entry point |
| No plan | proposal 0001 §3.9 (`plan()`, J2) needs readiness and cache state without execution; today they cannot be separated |
| Batching (0001 §3.6, H) and the pool (§3.7, I) need whole ready sets | the per-job `run` gives them none |

**What pydra users gain from the split:**
- Slurm and SGE arrays formed from the jobs that became ready together, deterministically.
- A batching worker (H) and a pool worker (I) that receive ready sets, not single jobs.
- `plan()` (J2) as a method of the runner, without running anything.
- Readiness and error propagation testable without workers or processes.
- External drivers (a workflow manager, a service, a notebook) that call `ready()` and
  `complete()` and run jobs their own way, while pydra keeps the graph semantics.
- One loop instead of two (`expand_workflow` and `expand_workflow_async`).

**One motivating user.** senselab 2.x (sensein/senselab#617) compiles its pipelines to a pydra
workflow, supplies each node's checksum through `checksum=` (0001 §3.12), and runs the jobs on its
own executors behind an `Executor` protocol (`submit`, `status`, `cancel`, `poll`, plus
`admit`/`release` of a plan's input values and `bind_epoch`; senselab `2.x` at `46608622`,
`packages/senselab-core/src/senselab_core/runtime/execution/executor.py:355-389`). Without a seam
it subclasses `Worker` and rebuilds the ready set by collecting jobs until `loop.call_soon` fires
(`runtime/execution/runner.py:548-574`), recovers its own fragment from each job's resolved inputs
(`:734`), resends jobs that pydra folded into another of the same checksum (`:404`), and keeps an
in-memory store only to answer pydra's `lookup` (`:118`). Its "Route B" plan is this split, with its
executors then generalised into pydra's batching (H) and pool (I) workers. This RFC is written so
that none of that is senselab-specific; the rest of it does not refer to senselab.

## 2. The first question: can `State` and the graph express routing, conditional nodes, loops and typed absent values?

A pipeline system layered on pydra wants, per node: typed ports, nesting, fan-out and gather,
routing on a recorded predicate, bounded loops with distinct ids per round, and a typed absent value
that is not a failure. This section answers from the code which of these pydra has.

### 2.1 What pydra has

- **Fan-out and gather.** `State` holds a scalar or outer splitter, a combiner, the container depth
  per input, and links to upstream states (`engine/state.py:82-117`, `other_states` at `:424-445`,
  `_connect_splitters` at `:478-526`). `prepare_states` builds indices and values once the inputs are
  known (`:849-877`). Split and combine over records map onto this directly.
- **Nesting.** A workflow task is a node. Its job expands in the same loop (`submitter.py:710-712`,
  `compose/workflow.py:392-394`) and is never routed to a worker (`submitter.py:389-390`).
- **Per-element readiness (E1, landed).** An element is released when the upstream elements it
  reads have succeeded and is marked unrunnable when one has failed (`submitter.py:1223-1234`,
  `_has_failed`/`_has_succeeded` at `:1264-1276`).
- **Data-dependent construction.** A nested workflow is constructed when its job runs
  (`submitter.py:602`). Its inputs are resolved by then, and every input that is not lazy is passed
  to the constructor as a value (`engine/workflow.py:98-102`, `:148-163`). The constructor is Python,
  so it can branch on an upstream output, and it can add a workflow of its own type again. The
  constructed workflow is cached by the hash of its non-lazy inputs (`engine/workflow.py:107-131`,
  `:207-210`).
- **Typed ports.** Field types are checked when nodes are wired, including unions
  (`utils/typing.py:23-25`).

### 2.2 What is missing

- **No conditional node.** `Workflow.add` takes a task, name, environment, hooks and resources
  (`engine/workflow.py:236-290`); there is no condition. `DiGraph` is fixed once
  `execution_graph` builds it (`engine/workflow.py:309-318`). Routing exists only as Python control
  flow inside a nested workflow's constructor (§2.1). Then:
  - the branch taken is not in the graph before the predicate's node has run, so `plan()` can only
    mark the nested workflow deferred (0001 §3.9);
  - routing is per nested-workflow job. To route each element of a split differently, the nested
    workflow must itself be split, and each element constructs its own workflow;
  - the predicate is whatever the constructor computes, not a node with a recorded result.
    A resumed run takes the same branch only if the constructor's inputs are themselves cached
    outputs.
- **No loop construct.** A loop can be written as a constructor that adds a workflow of its own
  type until a condition on its inputs holds. Nothing in the code prevents or supports it: no bound,
  no per-round naming, and each round is visible only after the previous one has run. No test on
  this branch exercises it.
- **No absent value.** The only "no value" in the engine is `attrs.NOTHING` for workflow outputs
  that the constructor has not yet set (`engine/workflow.py:136-138`, `:165-182`); it never flows
  along an edge. An upstream element either succeeds or fails. A failed element makes every reader
  unrunnable (`submitter.py:1226-1230`). A combine that reads all elements is unrunnable if any one
  failed (`_has_failed` takes the whole-node branch at `:1267-1268`), and a lazy read of an errored
  job raises (`engine/lazy.py:183-187`). A task can return a sentinel object, but then every
  downstream task runs and must test for it, and no type check knows about it.
- **Splitting over an upstream's output waits for the whole upstream node.** `State` can describe a
  nested fan-out, such as segments within each record, through `other_states`. The runner cannot
  release it per element: a node whose splitter reads an upstream output is started only when every
  upstream node is done (`_splits_upstream_output`, `submitter.py:1177-1184`, used at `:1206-1219`).
  This is the barrier that 0001 §3.1 step 2 (E2, the partial-`State` refactor) removes.

### 2.3 Proposal: three additions to the graph and runner, no change to `State`

Gates and absent values change readiness and completion, so they belong where `ready()` and
`complete()` live: in the runner. Loops and route sugar change only what is constructed, so they
can live above the runner, as helpers in `compose` or in the pipeline system. The only `State` work
needed is E2, which 0001 already plans.

**(a) `Absent`, a value.**
```python
@attrs.frozen
class Absent:
    cause: str            # free text; a project defines its own vocabulary
    source: str | None    # the node (and element) that produced it
```
- A task may return `Absent` for any output. It is a successful result, committed like any other.
- An input field declares what it does with one: `absent="propagate"` (default) or `"accept"`.
  The field attribute follows the `hash_eq` precedent (`compose/base/field.py:208-229`)
  and shell's `container_path` (`compose/shell/field.py:77-92`).
- At release (`_release`, `submitter.py:1236-1250`) the runner resolves the element's inputs as
  today. If a field that does not accept `Absent` holds one, the element is completed without being
  dispatched: every output is that `Absent`, with `source` unchanged. Nothing is run and nothing is
  committed; the upstream result that held the `Absent` is cached, so a rerun reaches the same
  outcome at no cost.
- A combine over elements, some absent, gives a list containing `Absent` values. Whether the
  reader accepts that is its field's `absent=`.
- A failure stays a failure: `errored` and `unrunnable` are unchanged, and an `Absent` never
  appears in an error summary.
- Type checking: an `Absent` passes every field's coercion at run time, as a `LazyField` does at
  wiring time. Wiring checks are unchanged.

**(b) A gate on a node: `workflow.add(task, when=lazy_bool)`.**
- `when` is a lazy `bool` read at the element's state index like any other lazy input
  (`_lazy_reads`, `submitter.py:1103-1119`), so a gate on a split node routes each element on its
  own predicate. No `State` change is needed.
- `when` true: the element runs as if ungated. `when` false: the element completes with every
  output `Absent(cause="not-routed", source=node)`, without dispatch. `when` absent: as (a).
- `when` is not part of the job's checksum. A gated-on job therefore has the same checksum and
  cache entry as the same job ungated.
- The predicate is an ordinary node with a stored result. A resumed run reads that result from the
  store and takes the same branch.
- Routing is then gates plus a merge: each branch gated on its predicate, and a small built-in
  `first_present(*values)` task, which accepts `Absent` on every input and returns the first
  present value.
- The graph is static, so `plan()` lists every branch; only the gate's outcome is pending.

**(c) Bounded loops, unrolled at construction: `workflow.loop(body, until=..., max_rounds=n)`.**
- A helper in `compose/workflow.py`, not a runner feature. It adds `n` copies of `body` named
  `<name>.0` … `<name>.<n-1>`, wires round `k`'s inputs from round `k-1`'s outputs, gates round
  `k > 0` on the negation of round `k-1`'s `until` output, and returns the last present round's
  outputs through `first_present` in reverse order.
- Rounds have distinct node names and, because their inputs differ, distinct checksums. Rounds
  after the loop ends complete as absent without dispatch, and the remaining unused rounds cost one
  release each.
- An unbounded loop stays what it is today: a nested workflow that constructs itself again. It is
  not a static graph, and `plan()` marks it deferred.

**Rejected:**
- Routing inside `State`, as a splitter over a predicate. `State` describes how many elements a
  node has and which upstream elements each reads; a gate changes whether an element runs, which is
  readiness. Putting it in `State` would make the element count depend on a result, which brings
  back the barrier E2 is removing.
- A runner-native loop node. It needs a graph that grows during execution. Unrolling gives the same
  semantics for bounded loops with a static graph and distinct ids, and leaves the runner's
  invariants unchanged.

## 3. The interface

### 3.1 `GraphRunner`

```python
class GraphRunner:
    def __init__(self, task: Task, store: ResultStore, *, on_error: OnError,
                 rerun: bool = False, propagate_rerun: bool = True,
                 environment: Environment | None = None, hooks: TaskHooks | None = None): ...

    def ready(self, limit: int | None = None) -> list[Job]:
        """Jobs that may now run, each handed out once.

        Leaf jobs only, across every nesting level. Jobs completed from the store, jobs
        gated off or absent, and jobs whose checksum is already handed out are completed
        here and not returned."""

    def complete(self, job: Job, outcome: Result | BaseException) -> None:
        """Record the outcome of a job handed out by `ready`. Never writes the store."""

    def plan(self) -> Plan:
        """Nodes, split sizes, gates, resources, expected cache hits; deferred parts marked."""

    @property
    def done(self) -> bool: ...
    @property
    def entries(self) -> list[JobErrorEntry]: ...   # as WorkflowError.entries today
    def result(self) -> Result: ...                  # the outer job's, once done
```

**What moves into it, unchanged in behaviour:**
- workflow construction and the execution graph (`submitter.py:602-604`);
- `NodeExecution`: `State` preparation, lazy resolution, per-element release and unrunnable
  marking (`:889-1262`);
- `on_error` (`:376-381`, `:735-750`), and error records (`:1279-1327`);
- deduplication by checksum (`:608-614`, `:700-708`);
- `max_concurrent`, which today truncates the ready list each round (`:865-871`) and becomes
  `ready(limit=...)` with the runner counting jobs in flight.

**Nested workflows** stay inside the runner. A workflow job is expanded when its inputs are ready,
as today (`:710-712`), and its nodes join the runner's ready set. It is never passed to an
executor, so the router still sees only leaf jobs (`:383-400`).

**Completion is reported, not inferred.** `complete()` stores the outcome on the job, so
`update_status` no longer polls `Job.done` for dispatched jobs (`:1061-1096`), and the ten-second
retry loop (`:621-699`) is reduced to a check that the graph is not stuck. An exception passed to
`complete()` is a job that could not be run, recorded as `record_job_error` does now (`:1290-1297`).

**Shared checksums.** When two jobs of a run have one checksum, only the first is handed out. On
its completion the others are completed with the same result, and each is reported through the
audit hook as reused, not executed (the restated J1 of §3.4). An orchestrator that records
provenance per job no longer needs to resend them.

**Resume.** The runner has no durable state of its own; the store is the durable state. A resumed
run is a new `GraphRunner` over the same task and store. `ready()` completes from the store every
job whose result is there and acceptable under the reuse policy, so only missing jobs reach an
executor. Nested workflows are constructed again; a gate reads its predicate's stored result and
takes the same branch. An orchestrator with its own durable store gives the runner a `ResultStore`
whose `lookup` answers from it (`engine/store.py:364-410`).

### 3.2 The `Executor` protocol

```python
class Executor(Protocol):
    batches: ClassVar[bool]      # wants the ready set whole (arrays, batching, pool)

    def submit(self, jobs: Sequence[Job], rerun: bool = False) -> Handle: ...
    def poll(self, handle: Handle) -> tuple[list[tuple[Job, Result | BaseException]], bool]:
        """Outcomes landed since the last poll, and whether all are given; never blocks."""
    def wait(self, handles: Collection[Handle], timeout: float | None = None) -> None:
        """Block until a poll of one of `handles` would give something, or `timeout`."""
    def cancel(self, handle: Handle) -> None: ...
    def close(self) -> None: ...
```

- `submit` takes a sequence. An executor with `batches = False` is given one job per call; one with
  `batches = True` is given every job of a round that the router sent to it, grouped by node.
- `poll` and `cancel` follow the polled protocol of the motivating user. `wait` avoids busy polling
  for in-process executors.
- The executor runs a job by calling `Job.run` (or `load_and_run` in another process,
  `job.py:713`), which takes the lease, checks the store under it and commits (`job.py:392-464`). An
  executor that runs jobs without `Job.run` must commit through the same store, or its results last
  only for the run.
- Not adopted from the motivating user's protocol: `admit`/`release` (a pydra job carries its own
  resolved inputs), `bind_epoch` (a coordinator-fencing concept; see §5) and `fragment_size` (a
  batching executor chunks its own submissions).

**The driver.** `Submitter` becomes a driver over one runner and its executors:

```python
runner = GraphRunner(task, self.store, on_error=self.on_error, ...)
while not runner.done:
    for name, jobs in group_by(self.router, runner.ready(self.max_concurrent)):
        handles.add(self.executors[name].submit(jobs, rerun))
    wait_any(handles)
    for job, outcome in poll_all(handles):
        runner.complete(job, outcome)
```

`Submitter.__call__` keeps its signature and return value (`submitter.py:402-514`), including the
implicit split workflow (`:450-479`). The router keeps its signature (`utils/routing.py:11`).

### 3.3 Today's workers become executors with no user-visible change

- **`WorkerExecutor(worker)`** adapts any `Worker`. It schedules `worker.run(job)` for each job,
  on the loop for an asynchronous worker or on the single helper thread for a synchronous one, as
  `_run_blocking` does now (`submitter.py:819-833`). `batches = False`. Every `Worker` subclass,
  including third-party plugins found through `Worker.plugin` (`workers/base.py:65-95`), keeps
  working through it.
- **`debug`** (`workers/debug.py:14-28`): through the adapter. `on_error` defaults stay as now:
  `fail_fast` for a synchronous worker (`submitter.py:376-381`). The synchronous loop
  `expand_workflow` (`:548-589`) is replaced by the same driver with a synchronous executor, so
  `debug` and `cf` share one code path, which is what the parity tests compare.
- **`cf`** (`workers/cf.py:77-125`): through the adapter.
- **`slurm`** (`workers/slurm.py`): gains `submit_many(jobs)` and `batches = True`. The tick-based
  grouping `_submit_grouped`/`_flush_group` (`:329-367`) is replaced by the ready set it is given;
  `_submit_array` (`:369`) and `max_array_size` (`:74`) are unchanged, as are the array eligibility
  rules (`:311-327`) and polling (`:423-500`).
- **`sge`** (`workers/sge.py`): the same, replacing the `collect_jobs_delay` sleep (`:274-278`).
  `max_job_array_length` keeps its meaning.
- A worker that does not implement `submit_many` keeps per-job submission.

The array composition can change: an array now holds the jobs of one node that became ready in the
same round, where before it held those that reached the worker in the same tick. For a split node
whose elements become ready together the two are the same set.

### 3.4 Caching, identity, the lease and the success record

| Concern | Runner | Executor |
|---|---|---|
| Identity | computes `job.checksum` once, in the submitting process (`job.py:237-246`), through `__checksum__` or `checksum=` when given (K2, `compose/base/task.py:483-508`) or the declared `version` (K1); computes the context once (`job.py:249-262`) | never rehashes: the checksum and context travel with the job (`job.py:192-206`) |
| Cache hit before dispatch | in `ready()`, through the store's success record (#929, `enh/success-record`: `PickleStore.read_succeeded`, `Submitter.read_succeeded`), which reads through `lookup` so a store's reuse policy still decides | — |
| Cache hit under the lease | — | `Job.run` checks the store again after taking the lease (`job.py:407-415`); this closes the race with another submitter |
| Lease | holds only the leases of workflow jobs it expands, as `run_async` does now (`job.py:478`) | taken and renewed by the process that runs the job, inside `Job.run` (#924 `DirectoryLease`, #928 takeover race) |
| Commit and success record | never writes | `store.commit` under the lease (`job.py:450`); with #929 the record is written last and only while the lease is held, and withdrawn under the lease before a rerun |
| Reuse policy | through `lookup(checksum, context, task)` (`store.py:374-389`) | the same, under the lease |
| Provenance (J1, restated) | emits `reused` for store hits and shared checksums | emits `executed` before `commit` |

Two consequences:
- The split does not move any lease or store logic: #924, #928 and #929 sit wholly on one side
  each. The runner reads; the executor writes.
- A `ResultStore` is the only place the two sides meet on disk. An external executor and the
  runner agree on what a job is through `job.checksum` and `job.context` alone.

## 4. Migration

### 4.1 Compatibility

- **Unchanged:** `Submitter(...)` arguments and `__call__`; `Worker` and every plugin; `ResultStore`
  and `PickleStore`; checksums; the cache layout; `WorkflowError`, `errors` and `entries`; the
  `debug` default and `on_error` defaults; error semantics under `debug`.
- **New and opt-in:** `GraphRunner`, `Executor`, `Absent`, `absent=`, `when=`, `workflow.loop`,
  `first_present`.
- **Changed internals:** `Submitter.expand_workflow`, `expand_workflow_async`,
  `get_runnable_tasks`, `fetch_finished` and `NodeExecution` move into `pydra.engine.runner`.
  `compose/workflow.py:388-394` calls the runner instead.

### 4.2 Deprecation

- `Submitter.expand_workflow`, `expand_workflow_async` and `get_runnable_tasks` stay for one alpha
  as wrappers that drive a `GraphRunner`, with a `DeprecationWarning`, then are removed.
- `NodeExecution` keeps its name in `pydra.engine.runner` and is re-exported from
  `pydra.engine.submitter` for one alpha.
- No change to a `Worker` subclass is required at any point. `submit_many` is optional.

### 4.3 Delivery

| PR | Contents | Depends on |
|---|---|---|
| R1 | extract `GraphRunner` (`ready`, `complete`, `done`) from `expand_workflow_async` and `NodeExecution`; `Submitter` drives it through the existing per-job dispatch. No behaviour change; the existing suite and `debug`/`cf` parity are the test | — |
| R2 | one driver for synchronous and asynchronous workers; `expand_workflow` becomes a wrapper | R1 |
| R3 | reported completion: `complete()` sets the job's result; `update_status` stops polling dispatched jobs; the ten-second retry becomes a stuck-graph check | R1 |
| R4 | `Executor` protocol and `WorkerExecutor`; the router dispatches to executors | R2 |
| R5 | `submit_many` for Slurm and SGE; the tick and sleep grouping removed | R4, #929 |
| R6 | shared checksums reported as reuse; `reused`/`executed` audit hook | R1, J1 |
| R7 | `plan()` (subsumes J2) | R1, F2, B2 |
| R8 | `Absent`, `absent=`, `when=`, `first_present` | R1 |
| R9 | `workflow.loop` helper | R8 |
| E2 | partial `State` (0001 §3.1 step 2): per-element release of nested fan-out | R1 |

R1 to R4 change no user-visible behaviour and can merge before any feature. R8 and R9 are the only
new semantics. E2 is independent of the interface but is what nested fan-out needs. Each PR has
tests on `debug` and `cf`, and on Slurm where relevant. R1's test is the existing suite with no
expectations changed.

## 5. Open questions for maintainers

- Is `pydra.engine.runner.GraphRunner` the right public home, and should `ready`/`complete` be
  public API in 1.0 or provisional until R8?
- Polled `Executor` (`submit`/`poll`/`wait`) or an asyncio-native one (`async` iterator of
  outcomes)? The polled form suits external and thread-based executors; the asyncio form suits `cf`
  and the scheduler workers.
- Should a `Worker` gain `submit_many` directly, or should batching executors be a separate base
  class?
- `Absent`: is a free-text `cause` enough, or should pydra fix a small set of causes? Should a
  short-circuited absent result be committed to the store, so that `plan()` counts it as a hit?
- `when=` and `absent=`: acceptable names? Should `when` be excluded from the checksum, as proposed?
- Should `workflow.loop` and `first_present` live in pydra or in a downstream package, given that
  the runner needs only `when=` and `Absent`?
- Should the runner hold a lease on a workflow job it expands, as today, or only on leaf jobs? On a
  shared cache the workflow lease serialises two submitters of the same workflow.
- Does a generic run identity (an epoch or run id passed to executors, so a restarted coordinator
  can fence the old one's jobs) belong in the `Executor` protocol, or in a pool plugin (I)?
- Should the deprecated `Submitter` methods be kept for one alpha, or removed in R2 given the 1.0
  alpha status?
