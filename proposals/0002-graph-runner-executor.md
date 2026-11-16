# Proposal 0002: a graph runner and an executor in place of one Submitter

Status: proposed, for maintainer discussion. Branch `rfc/graph-runner-executor`, off
`enh/execution-extensions` at `3f734f9a` (after #929). This is the separate RFC that proposal 0001
§3.10 names. Line references are to `3f734f9a`.

Proposal 0001 §3.10 said this RFC would follow once 3.1 and 3.3 land. 3.3 (D1, C) and 3.1 step 1
(E1) have landed; 3.1 step 2 (E2) has not. The RFC is written now so that the interface can be
reviewed before downstream systems build against today's `Submitter`. The parts that need E2 are
marked, and §4.3 orders them after it.

Revision 2 answers the review of the first draft (PR #930). Where this revision disagrees with a
finding, the disagreement is stated in place.

## 1. Motivation

`Submitter` does two jobs. It works out which jobs can run: it constructs workflows, prepares
`State`, resolves lazy fields, tracks per-element readiness and applies `on_error`
(`engine/submitter.py:548-787`, `NodeExecution` at `:926-1299`). It also runs them: it picks a
worker, awaits one coroutine per job and collects the futures (`:383-400`, `:720-747`,
`:856-870`). The two are interleaved in `expand_workflow_async` (`:593-759`), with a second loop
for synchronous workers in `expand_workflow` (`:548-591`). Each nested workflow runs its own
`expand_workflow_async` through `Job.run_async` (`:720-722`, `compose/workflow.py:392-394`).

The interleaving has costs that do not depend on any one user:

| Problem | Evidence |
|---|---|
| A worker sees one job at a time, so a worker that wants the ready set rebuilds it | `Worker.run(job)` is awaited per job (`submitter.py:723-726`). The Slurm worker collects the jobs that reach it before the current event-loop tick ends, keyed by `(name, resources, cache_root, rerun)` (`workers/slurm.py:329-367`). The SGE worker sleeps `collect_jobs_delay` for the same purpose (`workers/sge.py:278`). Both groupings depend on asyncio scheduling, not on the graph |
| Completion is inferred from the store, not reported | `NodeExecution.update_status` re-reads each queued or running job's result through `Job.done` (`submitter.py:1098-1133`, `job.py:558-572`). When nothing is ready and nothing is in flight, the loop re-reads for up to eleven one-second rounds (the comment says a minute) and then raises "Something has gone wrong" (`submitter.py:623-701`) |
| Jobs are handed out repeatedly and deduplicated after the fact | a node returns every queued job on every round (`submitter.py:1271`); the loop filters them through `finished`, `futured` and `sharing` (`:610-616`, `:702-719`) |
| No other loop can drive the graph | readiness lives in `NodeExecution` and is only advanced from inside `expand_workflow_async`. An external orchestrator, a notebook stepping a workflow, or a test of readiness without processes, has no entry point |
| No plan | proposal 0001 §3.9 (`plan()`, J2) needs readiness and cache state without execution; today they cannot be separated |
| Batching (0001 §3.6, H) and the pool (§3.7, I) need whole ready sets | the per-job `run` gives them none |

**What pydra users gain from the split:**
- Slurm and SGE arrays formed from the jobs that the runner released in the same round. The round
  still depends on when upstream completions were reported, so arrays are not deterministic across
  runs; they no longer depend on how asyncio interleaves coroutines within a round.
- A batching worker (H) and a pool worker (I) that receive ready sets, not single jobs.
- `plan()` (J2) as a method of the runner, without running anything.
- Readiness and error propagation testable without workers or processes.
- External drivers (a workflow manager, a service, a notebook) that call `ready()` and
  `complete()` and run jobs their own way, while pydra keeps the graph semantics (§3.5).
- One loop instead of two.

**One motivating user.** senselab 2.x (sensein/senselab#617) compiles its pipelines to a pydra
workflow, supplies each node's checksum through `checksum=` (0001 §3.12), and runs the jobs on its
own executors, which take fragments of input references under an admitted plan and return result
references (senselab `2.x` at `46608622`,
`packages/senselab-core/src/senselab_core/runtime/execution/executor.py:355-389`). Without a seam
it subclasses `Worker`, rebuilds the ready set by collecting jobs until `loop.call_soon` fires
(`runtime/execution/runner.py:548-574`), recovers which upstream job fed each input from the
resolved values (`:734`), resends jobs that pydra folded into another of the same checksum
(`:404`), and keeps an in-memory store to answer pydra's `lookup` (`:118`). Its design asks the
graph layer to release a job when its inputs exist, to mark descendants complete, typed-absent or
failed, to store route choices and loop rounds as results, and to treat infrastructure faults and
held queues as neither successes nor failures (its `execution.md` §3.1, §3.3; ADR-009). Its
"Route B" plan (`route-b-analysis.md` §3.1) has two parts: driving the graph through `ready()` and
`complete()`, and moving its executors behind a pydra `Executor`. This RFC supports the first part
as a first-class external driver (§3.5). It does not adopt the reference-passing executor protocol
as pydra's `Executor` (§3.2 says why), so the second part would need those executors to take pydra
jobs. The rest of this RFC does not refer to senselab.

## 2. The first question: can `State` and the graph express routing, conditional nodes, loops and typed absent values?

A pipeline system layered on pydra wants, per node: typed ports, nesting, fan-out and gather,
routing on a recorded predicate, bounded loops with distinct ids per round, and a typed absent value
that is not a failure. This section answers from the code which of these pydra has.

### 2.1 What pydra has

- **Fan-out and gather.** `State` holds a scalar or outer splitter, a combiner, the container depth
  per input, and links to upstream states (`engine/state.py:82-117`, `other_states` at `:424-445`,
  `_connect_splitters` at `:478-526`). `prepare_states` builds indices and values once the inputs are
  known (`:849-877`). Split and combine over records map onto this directly.
- **Nesting.** A workflow task is a node. Its job is never routed to a worker
  (`submitter.py:389-390`); `Job.run_async` leases it, checks the store, and if there is no result
  runs a nested `expand_workflow_async` with its own `futured`, `sharing` and `max_concurrent`
  (`job.py:475-538`, `submitter.py:593-759`, `:902-908`).
- **Per-element readiness (E1, landed).** An element is released when the upstream elements it
  reads have succeeded and is marked unrunnable when one has failed (`submitter.py:1260-1271`,
  `_has_failed`/`_has_succeeded` at `:1301-1314`).
- **Data-dependent construction.** A nested workflow is constructed when its job runs
  (`submitter.py:604`). Its inputs are resolved by then, and every input that is not lazy is passed
  to the constructor as a value (`engine/workflow.py:98-102`, `:148-163`). The constructor is Python,
  so it can branch on an upstream output, and it can add a workflow of its own type again. The
  constructed workflow is cached by the hash of its non-lazy inputs (`engine/workflow.py:107-131`,
  `:211-214`).
- **Routing by an empty split.** A predicate task that returns `[x]` or `[]`, a node split over it
  and a combine give a static graph in which the branch not taken has zero elements and its reader
  receives an empty list. A node with no elements yields an empty `StateArray`
  (`engine/lazy.py:222-223`). The review of the first draft ran this on `debug`: `go=True` gives
  `out=[6], n=1`; `go=False` gives `out=[], n=0`. Because the splitter reads an upstream output, the
  split node waits for the whole predicate node (`submitter.py:1214-1221`, `:1244-1256`).
- **Typed ports.** Field types are checked when nodes are wired and coerced at run time by
  `TypeParser` (`utils/typing.py:87`, `check_type` at `:466`).

### 2.2 What is missing

- **No conditional node.** `Workflow.add` takes a task, name, environment, hooks and resources
  (`engine/workflow.py:236-290`); there is no condition. `DiGraph` is fixed once `execution_graph`
  builds it (`engine/workflow.py:309-318`). Routing exists as constructor control flow (§2.1) or as
  an empty split (§2.1). Each has limits:
  - constructor routing is not in the graph before the predicate's node has run, so `plan()` can
    only mark the nested workflow deferred (0001 §3.9); its predicate is whatever the constructor
    computes, not a node with a recorded result;
  - empty-split routing has no cause: "not routed" and "routed, no items" are both `[]`. Every
    branch output is list-wrapped, so a reader that wants a value must unwrap it, and a merge of
    branches is a concatenation. `plan()` cannot tell a route from an empty fan-out. Routing each
    element of an already split node needs a nested split, which waits on the E2 barrier below.
- **No loop construct.** A loop can be written as a constructor that adds a workflow of its own
  type until a condition on its inputs holds. Nothing in the code prevents or supports it: no bound,
  no per-round naming, and each round is visible only after the previous one has run. A search of
  the test names on this branch found no recursive-workflow test; the tests were not read one by one.
- **No absent value.** The only "no value" in the engine is `attrs.NOTHING` for workflow outputs
  that the constructor has not yet set (`engine/workflow.py:136-138`, `:165-182`); it never flows
  along an edge. An upstream element either succeeds or fails. A failed element makes every reader
  unrunnable (`submitter.py:1263-1267`). A combine that reads all elements is unrunnable if any one
  failed (`_has_failed` takes the whole-node branch at `:1304-1305`), and a lazy read of an errored
  job raises (`engine/lazy.py:184-188`). A task can return a sentinel object, but then every
  downstream task runs and must test for it, and no type check knows about it.
- **Splitting over an upstream's output waits for the whole upstream node.** `State` can describe a
  nested fan-out, such as segments within each record, through `other_states`. The runner cannot
  release it per element: a node whose splitter reads an upstream output is started only when every
  upstream node is done (`_splits_upstream_output`, `submitter.py:1214-1221`, used at
  `:1244-1256`). This is the barrier that 0001 §3.1 step 2 (E2, the partial-`State` refactor)
  removes.

### 2.3 Proposal

Gates and absent values change readiness and completion, so they belong where `ready()` and
`complete()` live: in the runner. Loops and merges change only what is constructed, so they are
helpers in `compose` over the gate. The `State` class itself does not change. The node's
upstream-state collection, its lazy reads and the graph's edges do (§2.3(b)). E2 is a `State`
change and stays as 0001 plans it.

**Why gates and `Absent`, given the empty split.** After E2 the empty split routes per element
with no further change to pydra, and it should stay supported. It cannot give a reader a typed
cause, a merge that returns a value rather than a list, a branch whose readers are not dispatched
at all, or a `plan()` that shows a route as a route. If maintainers judge those not worth the
additions, the smallest acceptable outcome is `Absent` alone (R10a in §4.3) and documenting the
empty split as the routing idiom. §5 asks this.

**(a) `Absent`, a value.**
```python
@attrs.frozen
class Absent:
    cause: str            # free text; a project defines its own vocabulary
    source: str | None    # the node and element that produced it; provenance only
```
- **Producing one.** A task may return `Absent` for any output. That is a successful result,
  committed like any other under the task's checksum, because it is a function of the task's
  inputs.
- **Hashing.** `Absent.__bytes_repr__` hashes `cause` only. `source` is provenance, recorded and
  not keyed, like the execution context (0001 §3.11): renaming an upstream node does not change a
  reader's checksum. Two causes give two checksums, so a reader that behaves differently per cause
  is never served the other's result. A `checksum=` or `__checksum__` hook (K2) receives the
  `Absent` object itself among the resolved inputs and must digest it; the hook's documentation
  says so.
- **What a reader does with one.** An input field declares `absent="propagate" | "accept" |
  "drop"`. It is a field attribute like `hash_eq` (`compose/base/field.py:208-229`). The default
  depends on what the field reads:
  - a single value (no combine): `"propagate"`;
  - a combined value (the field reads a combine): `"accept"`. Propagating here would make one
    absent element erase the whole gather, which is the "combine fails whole" problem of §2.2.
    Dropping would misalign the list with sibling combined inputs of the same node.
- **The check is one level deep.** For a single value, the field holds an `Absent` if its resolved
  value is an `Absent` instance. For a combined value, each element is checked. Lists, dicts and
  `StateArray`s inside a value are not scanned.
- **Propagation.** At release (`_release`, `submitter.py:1273-1287`) the runner resolves the
  element's inputs as today. If a `"propagate"` field holds an `Absent`, or any element of a
  combined `"propagate"` field does, the element completes without being looked up, shared,
  dispatched or committed (the precedence rule of §2.3(b)). Every output is the `Absent` of the first such field
  in declaration order. Its `cause` is unchanged, and `source` keeps the node that produced it. A
  `"drop"` field on a combined value receives the list without its absent elements. `"accept"` passes
  the value as resolved.
- **Fields excluded from the checksum.** A field whose value is not hashed (the `container_path`
  exclusion at `compose/base/task.py:561`, or a K1/K2 task whose hook ignores the field) could
  change from a value to an `Absent` with no checksum change. Such a field may not declare
  `"accept"`; defining it raises at definition time.
- **Splitter inputs.** A splitter input that resolves to an `Absent` gives the node no elements.
  The node is absent as a whole, and a reader of it receives that `Absent` rather than an empty
  `StateArray`, so absence is distinguishable from an empty list. Today a splitter over an upstream
  output is resolved once the whole upstream is done (`start`, `submitter.py:1039-1075`). Per
  element, after E2, each upstream element's `Absent` gives that element's sub-state no elements.
  This case is specified with E2 and implemented after it (§4.3).
- **Coercion.** An `Absent` passes `TypeParser` coercion for a field that accepts it, and for the
  elements of a combined value under `"accept"`. Wiring checks are unchanged.
- **Workflow outputs.** `WorkflowOutputs._from_job` (`compose/workflow.py:319`) resolves output
  lazy fields; an `Absent` there passes coercion and is part of the workflow's result. The result is
  committed. That is safe because every gate inside the workflow reads inner results that are
  themselves keyed by the workflow's inputs.
- **A failure stays a failure.** `errored` and `unrunnable` are unchanged, and an `Absent` never
  appears in an error summary.

**(b) A gate on a node: `workflow.add(task, when=lazy_bool)` or `unless=lazy_bool`.**
- **Precedence (the safety rule).** For each released element the runner evaluates, in order:
  1. the gate;
  2. propagation of `Absent` inputs (§2.3(a));
  3. the store, for a job whose result is there;
  4. sharing with a job of the same checksum already handed out;
  5. dispatch.

  An element that stops at step 1 or 2 completes absent and is never looked up, shared or
  committed under its checksum. It does not enter the sharing table. Consequences:
  - A gated-off job is never served a result committed by a gated-on run of the same task and
    inputs. A gated-on twin is never completed absent.
  - Of two same-checksum jobs in one run, the gated-off one completes absent and the gated-on one
    runs.
  - An absent result made by a gate is never committed, so it cannot be served to a later gated-on
    run or to an ungated twin.
  - `plan()` counts gated elements from the predicate's stored result, not from the gated job's
    slot.
- **Why the gate stays out of the checksum** rather than entering identity. A gated-on job
  computes exactly what the ungated job computes, so keying it on the gate would split one result
  across two cache slots and two identities. A gated-off job computes nothing and needs no identity.
  The precedence rule is what makes this safe. An accepting reader sees the gate's effect only
  through an `Absent` input, which is hashed by `cause` (§2.3(a)), so a gate flip changes the
  reader's checksum.
- **The gate is an input of the node.** The `State` class is unchanged, but three places that list
  a node's inputs must include `when`/`unless`:
  - `Node._get_upstream_states` iterates `self.input_values` (`engine/node.py:209-230`). It must
    include the gate, so a gated node takes the predicate's state as it would a data input's.
  - `NodeExecution._lazy_reads` iterates the task's fields (`submitter.py:1140-1157`). It must yield
    the gate, so the gate is read at the element's state index and the element waits for it.
  - The graph's edges (`engine/workflow.py:309-318`, `_create_graph`) must include an edge from the
    predicate's node.

  `Download(url="x")` gated on a `check` split over three records therefore has three elements, as
  it would with a data input from `check`. All three have one checksum. By the precedence rule, the
  gated-off ones complete absent, and the gated-on ones run once and share the result.
- **Gate and data from different splits** combine as any two inputs from different upstream states
  do: the upstream states are combined outer unless the node's own splitter says otherwise
  (`state.py:478-526`).
- **Values.** `when` true or `unless` false: the element proceeds to step 2. Otherwise it completes
  with every output `Absent(cause="not-routed", source=node)`. A gate that is itself `Absent`
  propagates. A gate is typed `bool`; anything else is a wiring error.

**(c) Merging routes: `Select`, keyed on the gates.**
- A route is gated branches plus a merge. The merge selects on the gates, not on presence: a
  routed branch may itself return a content `Absent`, which must reach the reader with its cause.
- `pydra.compose.select(n)` builds a task class with inputs `when_0 … when_{n-1}` (bool, accept)
  and `value_0 … value_{n-1}` (accept). It returns `value_k` for the first `k` whose `when_k` is
  true, whatever that value is, and `Absent("not-routed")` if none is true.
- It is generated per arity because only top-level lazy fields are resolved
  (`submitter.py:1153-1157`); a variadic or list-of-lazy input is not. Nested-lazy resolution would
  remove the need and is not proposed here.

**(d) Bounded loops, unrolled at construction: `workflow.loop(body, until="done", max_rounds=n, index=None)`.**
- A helper in `compose/workflow.py`, not a runner feature. `until` names a `bool` output of `body`.
  The helper adds `n` copies of `body` named `<name>.0` … `<name>.<n-1>` and wires round `k`'s
  inputs from round `k-1`'s outputs. It gates round `k > 0` with `unless=<round k-1>.done`, so no
  negation node is needed. A round whose predecessor was gated off reads `Absent` inputs and
  propagates.
- **Identity per round.** Node names differ per round; checksums need not. A round whose carried
  outputs equal its inputs has the checksum of the round before. The helper offers two contracts:
  - `index="<field>"`: the round number is passed to that input of `body`, so every round has its
    own checksum. A retry or poll body, which carries nothing forward, must use this.
  - `index=None`: the loop is a fixed-point iteration of a pure body. Equal rounds are equal
    computations, so folding them onto one checksum is correct, and the loop's documentation says
    so.
- **Outputs.** The helper returns the body's outputs from the last round that ran, chosen with
  `Select` on the gates (not on presence, so a last round that returns a content `Absent` is
  returned as such). It adds `rounds` (int) and `converged` (bool: the last round that ran had
  `done` true), so exhausting `max_rounds` is distinguishable from convergence.
- Unused rounds complete absent at step 1 and cost one release each.
- An unbounded loop stays what it is today: a nested workflow that constructs itself again. It is
  not a static graph, and `plan()` marks it deferred.

**Rejected:**
- Routing as a splitter over a predicate inside `State`. The empty split already does this
  (§2.1), behind the same E2 barrier as any split over an upstream output; making it a `State`
  feature adds nothing the idiom lacks except what gates add without a `State` change.
- A runner-native loop node. It needs a graph that grows during execution. Unrolling gives the same
  semantics for bounded loops with a static graph, and leaves the runner's invariants unchanged.

## 3. The interface

### 3.1 `GraphRunner`

```python
class GraphRunner:
    def __init__(self, task: Task, store: ResultStore, *, on_error: OnError,
                 rerun: bool = False, propagate_rerun: bool = True,
                 environment: Environment | None = None, hooks: TaskHooks | None = None,
                 lookup_before_dispatch: bool = False): ...

    def ready(self, limit: int | None = None) -> Round: ...
    def complete(self, job: Job, outcome: Outcome) -> None: ...
    def upstream(self, job: Job) -> dict[str, list[tuple[Job, int | None]]]: ...
    def plan(self) -> Plan: ...
    def stuck(self) -> str | None: ...        # a diagnosis when nothing can progress
    @property
    def done(self) -> bool: ...
    @property
    def entries(self) -> list[JobErrorEntry]: ...
    @property
    def not_run(self) -> list[tuple[Job, NotRun]]: ...
    @property
    def failed(self) -> bool: ...             # any Failed or errored Result so far
    @property
    def first_error(self) -> BaseException | None: ...  # as raised, for fail_fast
    def retry(self, job: Job) -> None: ...    # hand a NotRun job out again
    def result(self) -> Result: ...           # the outer job's, read from the store

@attrs.frozen
class Round:
    dispatch: list[Job]           # each handed out once; job.rerun already set
    settled: list[Settled]        # completed inside this call, never dispatched

@attrs.frozen
class Settled:
    job: Job
    how: Literal["stored", "shared", "gated", "absent", "workflow-stored"]
    result: Result

Outcome = Result | Failed | NotRun
@attrs.frozen
class Failed:  error: BaseException
@attrs.frozen
class NotRun:  reason: str; held: bool    # cancelled, never started, or held by infrastructure
```

**`ready()`** applies the precedence of §2.3(b) to every released element. It returns the jobs to
dispatch and, separately, every job it completed itself, with how. Each job is handed out once.
`limit` caps the jobs handed out by this call (`max_concurrent`, below). `job.rerun` is set from
`rerun` and `propagate_rerun`: the outer job gets `rerun`, inner jobs get
`rerun and propagate_rerun` (today's `:575`, `:702`).

**`complete(job, outcome)`:**
- A `Result` (errored or not) is the job's result. It is held on the job, so lazy reads by readers
  use it without a store read.
- `Failed` is a job that could not be run, recorded as `record_job_error` does now
  (`submitter.py:1327-1333`). Readers become unrunnable.
- `NotRun` is neither. The job stays pending: its readers stay blocked, no error is recorded, and
  `held=True` says the cause was infrastructure. A later `ready()` hands the job out again only after
  `runner.retry(job)`. A run that ends with `NotRun` jobs is incomplete, not failed: `done` is true
  once nothing else can progress, `not_run` lists them, and `Submitter` raises `RunIncomplete`, a
  `WorkflowError` subclass, so callers that catch `WorkflowError` keep working.

**`upstream(job)`** returns, for each lazy input, the upstream jobs and state indices it was
resolved from (`NodeExecution._upstream`, `submitter.py:1179-1196`), so a driver need not recover
them from values.

**`stuck()`.** When `ready()` returns nothing, nothing is in flight and `done` is false, the driver
calls `stuck()`. It returns today's diagnosis (`submitter.py:634-701`) for the driver to raise. The
store-visibility retry that today precedes that diagnosis moves into the executor adapter
(§3.3), because only an adapter that reads results from the store needs it.

**What moves into it, unchanged in behaviour:**
- workflow construction and the execution graph (`submitter.py:604-607`);
- `NodeExecution`: `State` preparation, lazy resolution, per-element release and unrunnable
  marking (`:926-1299`);
- `on_error` (`:376-381`, `:741-750`), and error records (`:1316-1363`);
- deduplication by checksum (`:610-616`, `:702-719`), now after the gate (§2.3(b));
- the success-record read before dispatch (#929, `submitter.py:761-787`).

**Nested workflows and flattening.** The end state is one runner per `Submitter` call that expands
nested workflows itself, so `ready()` returns leaf jobs across every level. This flattening changes
two behaviours, which §4.3 delivers in their own PR (R6):
- `max_concurrent` becomes one cap over the whole run. Today each nested workflow applies its own
  cut (`submitter.py:902-908`), so with `max_concurrent=4` and eight nested workflows of four
  ready leaves each, up to 4 run per workflow. After flattening, 4 run in all.
- Equal-checksum jobs in two nested workflows are folded into one. Today each nested loop has its
  own `sharing` table (`:610-616`), so both run, and the second waits on the first's lease.

Until R6, the runner of R1 keeps today's structure: one runner per workflow job, driven by
`Job.run_async` as now.

**Workflow jobs under the split (after R6).** Leaf jobs are written by executors; workflow jobs
are written by the runner. For each workflow job the runner does what `Job.run_async` does now
(`job.py:475-538`):
- **Lease.** It takes the workflow job's lease and keeps its heartbeat (one thread per lease,
  `engine/lease.py:251`) until the commit. An external driver's `ready()` must not block, so the
  runner takes the lease with one non-blocking attempt (`DirectoryLease._step`, `lease.py:715-731`,
  made public as `Lease.try_take()`). A workflow job whose lease another process holds is tried
  again on later calls. It is neither dispatched nor settled, so it appears as `waiting` in
  `plan()` and in `stuck()`'s diagnosis, and the driver keeps waiting rather than raising.
- **Store check.** Under the lease it checks the store first. A stored, non-errored, acceptable
  result completes the workflow job as `workflow-stored`, and its nodes are not expanded. This is
  today's behaviour (`job.py:489-497`), not a resume regression.
- **Expansion.** Otherwise the runner withdraws the success record (`job.py:499`), runs
  `pre_run_task` hooks and starts the audit, then expands the workflow.
- **Commit.** When the workflow's nodes are done, it builds the outputs with
  `WorkflowOutputs._from_job`, runs `post_run_task` and finalises the audit. It then commits through
  `store.commit`, which writes the result and the success record (`store.py:587-620`), and releases
  the lease. An error in a node commits an errored workflow result and records the error, as
  `job.py:516-528` does.
- **`LeaseLost`.** If the lease was taken over, `commit` raises `LeaseLost` (`store.py:591`). The
  runner records it as an error entry of the workflow job with `phase="commit"`, visible in
  `entries` and in the raised `WorkflowError`. `ready()` and `complete()` do not raise it.

The outer job of a `Submitter` call is a workflow job whenever the task is a workflow or is split
(`submitter.py:443-465`), so its result is committed by the runner, and `runner.result()` and
`Submitter.__call__`'s check (`:503-513`) read it from the store as today. An outer leaf job is
handed out by `ready()` and committed by its executor.

**Resume.** The runner has no durable state of its own; the store is the durable state. A resumed
run is a new `GraphRunner` over the same task and store. What it skips, precisely:
- A workflow job whose result is stored and acceptable is not expanded.
- A leaf job with a success record is completed in `ready()` through `read_succeeded`, which reads
  through `lookup` (`store.py:547-568`). `read_succeeded` is not part of the `ResultStore`
  protocol (`store.py:379-425`); `Submitter` finds it with `getattr` (`submitter.py:774`). This RFC
  adds it to the protocol as an optional method.
- A leaf job without a record is dispatched. This covers a read-only cache, a cache written before
  #929, and a slot whose record write failed. So does a job whose stored result is errored, or is
  from another context: `read_succeeded` returns `None` for errored and cross-context results.
  `Job.run` then checks the store under the lease (`job.py:415-424`), returns an acceptable result
  without running, and reruns an errored one.
- An executor that does not call `Job.run` gets no lease-time check. A runner built with
  `lookup_before_dispatch=True` calls `store.lookup` for every job without a record before handing
  it out, so such an executor is given only jobs with no acceptable result. `plan()` counts hits
  from records, and from lookups when that option is set.
- A gate reads its predicate's stored result and takes the same branch.

### 3.2 The `Executor` protocol

```python
class Executor(Protocol):
    batches: ClassVar[bool]       # wants the jobs of a round together (arrays, batching, pool)
    checks_store: ClassVar[bool]  # runs jobs through Job.run, which checks the store under the lease
    inline: ClassVar[bool]        # runs each job to completion inside submit, in the caller's thread

    def bind(self, report: Callable[[Job, Outcome], None]) -> None:
        """Called once by the driver; `report` may be called from any thread."""
    def submit(self, jobs: Sequence[Job]) -> Handle: ...
    def cancel(self, handle: Handle) -> None:
        """Stop the jobs of `handle` not yet finished; each is reported `NotRun`."""
    def close(self) -> None: ...
```

- **Outcomes are pushed.** Every job submitted is reported exactly once, through `report`, as a
  `Result`, `Failed` or `NotRun`. The driver turns each report into `runner.complete()` on its own
  loop (`loop.call_soon_threadsafe`). One `asyncio.Event` therefore waits across every executor of
  a run. That closes the first draft's gap: it had a per-executor `wait` and a driver waiting across
  several.
- **`rerun`** is read from each job (`job.rerun`, §3.1), not passed per call, so `propagate_rerun`
  is kept.
- **`submit`** takes a sequence. An executor with `batches = False` is given one job per call. One
  with `batches = True` is given every job of a round that the router sent to it, in node order.
- **`cancel`** is called by the driver on every open handle when `fail_fast` stops a run, and when
  the driver itself is interrupted, as the `finally` at `submitter.py:751-756` does now.
- **A handle** is removed by the driver once all its jobs are reported.
- **Running a job.** The executor runs a job by calling `Job.run` (or `load_and_run` in another
  process, `job.py:723`), which takes the lease, checks the store under it and commits
  (`job.py:400-473`). An executor that runs jobs another way sets `checks_store = False`. It must
  then commit through the same store, or its results last only for the run. The runner should be
  built with `lookup_before_dispatch=True` (§3.1).
- **Not adopted:** `admit`/`release` of a plan's input values, `bind_epoch`, `fragment_size`, and
  reference-passing fragments. A pydra job carries its resolved inputs. A coordinator epoch is a
  fencing concept of a particular queue system (§5). A batching executor chunks its own
  submissions. A system whose executors pass references drives the runner from outside instead
  (§3.5).

**The driver.** `Submitter` becomes a driver over one runner and its executors. It owns the event
loop, as now (`get_open_loop`, `submitter.py:1400`, and `_own_loop`). Inside a running loop, such
as Jupyter, it behaves as today.

```python
async def drive(runner, executors, router):
    progressed = asyncio.Event()
    for ex in executors.values():
        ex.bind(lambda job, out: loop.call_soon_threadsafe(on_outcome, job, out))
    try:
        while not runner.done:
            rnd = runner.ready(limit=max_concurrent - in_flight)
            audit(rnd.settled)
            for name, jobs in group_by(router, rnd.dispatch):
                ex = executors[name]
                if ex.inline:                    # debug: one job, completed before the next
                    for job in jobs:
                        ex.submit([job])         # reports before returning
                        if runner.failed and on_error == "fail_fast":
                            raise runner.first_error   # the task's own exception object
                else:
                    open_handles.add(ex.submit(jobs))
            if not rnd.dispatch and not open_handles and not runner.done:
                if diagnosis := runner.stuck():
                    raise RuntimeError(diagnosis)
            await progressed.wait(); progressed.clear()
            if runner.failed and on_error == "fail_fast":
                break
    finally:
        for handle in open_handles:
            handle.executor.cancel(handle)
```

`Submitter.__call__` keeps its signature and return value (`submitter.py:402-514`), including the
implicit split workflow (`:443-465`). The router keeps its signature (`utils/routing.py:11`).

### 3.3 Today's workers become executors with no user-visible change

- **`InlineExecutor(worker)`, for `debug`** (`workers/debug.py:14-28`). `inline = True`. `submit`
  calls `worker.run(job)` in the caller's thread, as `Submitter.submit` does now
  (`submitter.py:531`), and reports before returning. Under `fail_fast` the driver re-raises the
  task's own exception object before submitting the next job, as `:579-580` does now. Signal
  handlers, `ContextVar`s set by the caller, `pdb` and main-thread-only libraries keep working, and
  no job runs after the first failure. The helper thread of `_run_blocking` (`:856-870`) is used
  only for a synchronous worker routed beside asynchronous ones, as now.
- **`WorkerExecutor(worker)`, for every asynchronous worker,** including third-party plugins found
  through `Worker.plugin` (`workers/base.py:65-95`). It schedules `worker.run(job)` per job on the
  driver's loop and reports when the coroutine ends:
  - **`cf`** (`workers/cf.py:77-126`) returns the `Result`, which is reported as is.
  - **`slurm` and `sge`** return `True`, not a `Result` (`workers/slurm.py:263`,
    `workers/sge.py:387`, `:419`). The `Result` exists only in the store. The adapter reads it with
    `job.result()`. The file may not yet be visible on the submitting host through NFS attribute
    caching. The adapter then retries with backoff up to `visibility_timeout`, by default today's
    eleven seconds (`submitter.py:628-634`). Only after that does it report `Failed` with today's
    diagnosis text. The tolerance moves here because only a store-reading adapter needs it.
  - `batches = False`, except as below.
- **`slurm`** gains `submit_many(jobs)`; `WorkerExecutor` sets `batches = True` for a worker that
  has it. The tick-based grouping (`workers/slurm.py:329-367`) is replaced by the round it is given.
  The array key `(name, resources, cache_root, rerun)` (`:339`) is kept. Before flattening (R6) a
  round comes from one workflow, so names do not repeat within it. After R6 the key adds the
  workflow path, because node names repeat across nested workflows. `_submit_array` (`:369`),
  `max_array_size` (`:74`), the eligibility rules (`:311-327`) and polling (`:423-536`) are
  unchanged.
- **`sge`**: the same, replacing the `collect_jobs_delay` sleep (`workers/sge.py:278`).
  `max_job_array_length` keeps its meaning.
- A worker without `submit_many` keeps per-job submission.

An array can be composed differently: it holds the jobs of one node released in the same round,
where before it held those that reached the worker in the same tick. For a split node whose elements
are released together the two are the same set.

### 3.4 Caching, identity, the lease and the success record

| Concern | Runner | Executor |
|---|---|---|
| Identity | computes `job.checksum` once in the submitting process (`job.py:237-246`), through `__checksum__` or `checksum=` (K2, `compose/base/task.py:483-508`) or the declared `version` (K1), and the context once (`job.py:249-262`) | never rehashes: checksum and context travel with the job (`job.py:192-206`) |
| Gate and absence | evaluated first, never keyed, never committed (§2.3(b)) | never sees a gated-off or propagated-absent job |
| Hit before dispatch | `ready()`, through `read_succeeded` (`store.py:547-568`), and through `lookup` when `lookup_before_dispatch` | — |
| Hit under the lease | — | `Job.run` checks the store after taking the lease (`job.py:415-424`) |
| Lease | workflow jobs (§3.1) | leaf jobs, inside `Job.run` (#924 `DirectoryLease`, #928) |
| Commit and success record | workflow jobs: `store.commit`, record written last and only while the lease is held (`store.py:587-620`); withdrawn before re-expansion (`job.py:499`) | leaf jobs: the same calls inside `Job.run` (`job.py:427`, `:459`) |
| Other store writes | — | Slurm requeue removes the job's lease file (`workers/slurm.py:252`) |
| Reuse policy | through `lookup(checksum, context, task)` (`store.py:389-404`) | the same, under the lease |
| Provenance (J1, restated) | `Round.settled` and the audit hook: `reused` for stored and shared jobs, `absent` for gated and propagated ones | `executed` before `commit` |

- So the rule is: **leaf jobs, the executor writes; workflow jobs, the runner writes.** The lease
  (#924, #928) and success-record (#929) code is used unchanged on both sides.
- A `ResultStore` is the only place the two sides meet on disk.
- **A hit served in `ready()` passes only the store's own rules.** No executor sees it. A project
  that must refuse some hits at serve time, for example on licence grounds, must apply the refusal
  in its store's `lookup` or `ReusePolicy.accept`. `read_succeeded` reads through `lookup`, so the
  refusal then applies there too.

### 3.5 Driving the runner from outside

A system that runs jobs its own way, for example by passing references rather than values, uses
`GraphRunner` directly, without `Submitter` or `Executor`:

```python
runner = GraphRunner(task, store, on_error="raise_at_end", lookup_before_dispatch=True)
while not runner.done:
    rnd = runner.ready()
    record(rnd.settled)                          # stored, shared, gated, absent: with causes
    for job in rnd.dispatch:
        send(job, runner.upstream(job))          # which upstream jobs fed each input
    for job, outcome in receive():               # Result, Failed, or NotRun(held=True)
        runner.complete(job, outcome)
```

What such a driver must and need not supply:
- **Outputs may be references,** except where the runner reads values. A gate's predicate must
  resolve to a `bool` or an `Absent`. Absence must be reported as pydra's `Absent`, not as a
  reference to one. A value that a downstream splitter splits over must be a real sequence.
- **Every settled job is visible** in `Round.settled`, with how and, for absences, the cause. A
  driver can therefore log every node as run, reused or typed-absent without the audit hook.
- **Infrastructure faults and held queues** are `NotRun(held=True)`. They are not errors, readers
  stay blocked, and the run ends incomplete and resumable. Content failures are `Failed`, or an
  errored `Result`.
- **Its store** is a `ResultStore` adapter over the driver's own store. The runner commits workflow
  jobs to it (§3.1). If the adapter is not durable, workflow results last only for the run, and a
  resumed run re-expands them.

## 4. Migration

### 4.1 Compatibility

- **Unchanged:** `Submitter(...)` arguments and `__call__`; `Worker` and every plugin;
  `PickleStore`; checksums; the cache layout; `WorkflowError`, `errors` and `entries`; the `debug`
  default; `on_error` defaults; error semantics under `debug`, including the caller's thread and no
  job after the first failure (§3.3).
- **Named behaviour changes, in R6 only:** `max_concurrent` caps the whole run rather than each
  nested workflow; equal-checksum jobs in different nested workflows run once.
- **New and opt-in:** `GraphRunner`, `Round`, `Outcome`, `Executor`, `Lease.try_take`,
  `ResultStore.read_succeeded` (optional), `RunIncomplete`, `Absent`, `absent=`, `when=`/`unless=`,
  `select(n)`, `workflow.loop`.
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

The hard steps are R6 (flattening, with workflow-job commits moving into the runner), E2 (partial
`State`) and R10b (the gate's place in node inputs, state and edges). The others are refactors or
additions behind them.

| PR | Contents | Depends on | Behaviour |
|---|---|---|---|
| R1 | extract `GraphRunner` (`ready`, `complete`, `done`, `stuck`) from `expand_workflow_async` and `NodeExecution`, one runner per workflow job, driven by `Job.run_async` as now | — | none; existing suite unchanged |
| R2 | one driver for synchronous and asynchronous workers; `InlineExecutor` for `debug`; `expand_workflow` becomes a wrapper | R1 | none; `debug` keeps the caller's thread and `fail_fast` order |
| R3 | `Executor` protocol (`bind`, `submit`, `cancel`, `close`), `WorkerExecutor` with the store read and visibility retry for workers that return `True`; the router dispatches to executors | R2 | none |
| R4 | reported completion: `complete()` holds the result on the job; `update_status` stops polling dispatched jobs; the visibility retry leaves the loop for the adapter | R3 | none |
| R5 | `submit_many` for Slurm and SGE; tick and sleep grouping removed | R3 | array composition per round (§3.3) |
| R6 | flattening: one runner per call; the runner leases, checks, expands and commits workflow jobs; `Lease.try_take`; array key gains the workflow path | R4, R5 | named changes (§4.1); tests updated for them |
| R7 | driver model: `Round.settled`, `NotRun`/`RunIncomplete`, `upstream()`, `lookup_before_dispatch`, `read_succeeded` in the protocol | R4 | opt-in |
| R8 | `plan()` (subsumes J2) | R6, F2, B2 | opt-in |
| E2 | partial `State` (0001 §3.1 step 2): per-element release of nested fan-out, including an `Absent` splitter input per element | R1 | 0001's |
| R10a | `Absent`: hashing, `absent=` with its defaults, propagation at release, coercion, `WorkflowOutputs`, the definition-time check on unhashed fields; whole-node absence for a splitter input | R6, E2 | opt-in |
| R10b | `when=`/`unless=`: node inputs, upstream states, lazy reads and edges; the precedence rule; `select(n)` | R10a | opt-in |
| R11 | `workflow.loop` with `index=`, `rounds`, `converged` | R10b | opt-in |

R5 no longer depends on #929, which has merged; the first draft listed it because arrays and the
success-record read changed the same dispatch code. Each PR has tests on `debug` and `cf`, and on
Slurm where relevant. Only R5 and R6 change any existing test expectation, and only for the
behaviours named.

## 5. Open questions for maintainers

- Is `pydra.engine.runner.GraphRunner` the right public home? Should `ready`/`complete` be public
  API in 1.0, or provisional until R10b?
- Should a `Worker` gain `submit_many` directly, or should batching executors have a separate base
  class?
- Is the empty-split idiom enough for routing after E2? If so, ship R10a (`Absent`) and drop R10b
  and R11.
- **Gate precedence.** Is "gate, then absence, then store, then sharing" acceptable, with gated and
  propagated absences never committed? The alternative is to key the gate into identity, at the
  cost of one result in two slots.
- **Who commits a workflow job.** Should the runner take the workflow lease at all after
  flattening, or should workflow jobs be committed without a lease? Their result is a function of
  results that are themselves leased. Today the workflow lease serialises two submitters of the same
  workflow on a shared cache.
- **`Absent` in checksums.** Is hashing `cause` and not `source` right? Is a free-text `cause`
  enough, or should pydra fix a small vocabulary?
- **`absent=` defaults.** `"propagate"` for single values and `"accept"` for combined values: one
  default or two?
- **Where a synchronous executor runs.** Inline in the caller's thread for `debug` (proposed), and on
  the helper thread for a synchronous worker beside asynchronous ones: is that split right?
- **Store visibility.** Is moving the eleven-second tolerance into the adapter, as a
  `visibility_timeout`, acceptable, and what should its default be on NFS?
- **What `ready()` reports.** Is `Round.settled` enough for external drivers, or should settled jobs
  also go through a callback?
- **A not-run outcome.** Is `NotRun(held=...)` with `RunIncomplete` the right shape for
  cancellation and infrastructure holds, and should `retry(job)` be public?
- Does a run identity (an epoch passed to executors, so that a restarted coordinator can fence the
  old one's jobs) belong in the `Executor` protocol, or in the pool plugin (I)?
- Should `workflow.loop` and `select` live in pydra or in a downstream package?
- Should the deprecated `Submitter` methods be kept for one alpha, or removed in R2, given the 1.0
  alpha status?
