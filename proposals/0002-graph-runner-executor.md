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

RFC 0003 (PR #931, `PortableStore`) needs six additions here, which its §13 lists. Each is stated
where it belongs and marked "Under a `PortableStore`": `job.attempt` (§3.1), `Result.committed`
(§3.1), what travels with a job (§3.4), `find(job)` in `ready()` (§3.1), result reads through
`_committed.json` (§3.3) and the workflow-job steps (§3.1). With a `PickleStore` none of them
changes anything.

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
- One readiness implementation instead of two expansion loops. It is driven by two small drivers,
  a synchronous one for `debug` and an asynchronous one for every other worker (§3.2).

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
routing on a recorded predicate, loops with distinct ids per round, and a typed absent value
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
  to the constructor as a value (`engine/workflow.py:99-103`, called at `:164`). The constructor is
  Python, so it can branch on an upstream output, and it can add a workflow of its own type again.
  The constructed workflow is cached by the hash of its non-lazy inputs
  (`engine/workflow.py:107-131`, `:211-214`).
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
  Written that way, round `k` holds `k` open nested workflows, `k` leases and a work-directory path
  `k` levels deep, and nothing is released until the innermost round returns. §2.3(d) adds a loop
  whose rounds the runner instantiates one at a time inside one workflow job.
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
  `:1244-1256`). 0001 §3.1 step 2 (E2, the partial-`State` refactor) removes this barrier for
  every shape except six, which keep it (below and the E2 row of §4.3).
- **E2 as implemented (pydra #949).** A node whose splitter reads an upstream output is partial:
  each upstream element's sub-state expands as a block when that element succeeds, and its elements
  are released then. Each element carries `state_key`, a flat tuple of field coordinates such as
  `(k, a, j)`, assigned when its block expands and never changed. `state_index` is the flat rank,
  filled only when the node seals; a node with a failed block never seals and its elements keep
  `state_index=None`. Every report on a partial node (error entries, error strings such as
  `Job 'B[1,0]'`, `Unrunnable`, `stuck()`) names the key, so names do not depend on timing. Element
  order and checksums equal today's over every rectangular split. Six shapes keep the barrier:
  (1) `.` between an inner axis and a static axis or an axis from another source; (2) `.` over two
  partial upstream states; (3) a user-supplied `container_ndim > 1` on an inner input; (4) a
  combiner that removes a block coordinate but keeps that block's inner coordinate; (5) an inner
  split beside a non-split input from the same source, or from a stateful upstream not reached
  through an upstream-state term; (6) a combiner name that matches no path, or more than one. A
  barrier node that would mispair ragged or combined inner inputs records a node-level resolution
  error instead of a wrong result.
- **Wrong today, corrected by E2 (pydra #945).** Five cases give wrong results or raise now, and are
  the only ones whose outputs and checksums change: a ragged inner split (`n=[1,3]` gives
  `[11, 31]`, not `[11, 31, 32, 33]`); a per-element empty split, which raises `IndexError`; a split
  over a combined upstream, which mispairs elements; a `typing.Any` upstream that mixes scalars and
  lists; and a reader that combines a static field of a partial upstream, which gets six extra
  empty groups.

### 2.3 Proposal

Gates and absent values change readiness and completion, so they belong where `ready()` and
`complete()` live: in the runner. Merges change only what is constructed, so `select(n)` is a
helper in `compose` over the gate. A loop is a workflow job whose graph the runner grows one round
at a time, inside that job's frame only (§2.3(d)). The `State` class itself does not change. The node's
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
  `Absent` object itself among the resolved inputs, but whether it digests it is not known: the
  hook is an opaque callable (`compose/base/task.py:489-507`), and a hook may key an input on its
  upstream's key rather than on its value. The runner therefore does not rely on it ("Absence the
  checksum may not see", below).
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
- **A propagated job has no task checksum.** Its checksum is the absence record, so no `checksum=`
  hook ever receives an `Absent`: the task type, `-absent-` and the digest of the reader's task
  definition (its executor and `Outputs` class, hashed as the checksum hashes them, without inputs),
  the unsuffixed checksum of the job the `Absent` was read from, and the `(field, element index,
  cause)` triple. Two tasks reading one `Absent` differ; one task reading it at several elements
  has one identity, as its results are equal. Pickling such a job reads no context and no store.
- **Absence the checksum may not see.** A field whose value pydra does not hash (the
  `container_path` exclusion at `compose/base/task.py:561`), and every field of a task with a
  supplied checksum (`__checksum__` or `checksum=`, `:489-507`), could change from a value to an
  `Absent`, or lose an element to `"drop"`, with no checksum change. Neither case can be detected
  at definition time. For such a task, when an `"accept"` field holds an `Absent`, or a combined
  `"accept"` or `"drop"` field has absent elements (kept or dropped), the runner appends a suffix
  to `job.checksum` (`job.py:237-246`) before the store, sharing and commit steps: `-absent-` and
  the hex digest of the sorted `(field, element index, cause)` of every such `Absent`, where the
  index is the element's position before dropping. It stays one path component, as `:496-506`
  requires. A task with no accepted or dropped `Absent` keeps its checksum unchanged, so no
  existing cache slot moves. Example: a gather over records whose key is built from its upstreams'
  keys gets a different checksum in a run where record 7 was routed off than in a run where
  record 7 ran, although the upstream keys are the same in both. An `Absent` a field holds as a
  constant (a node of a nested workflow given one through the workflow's input, or a root job's
  input) counts like a resolved one: a whole value at index -1, a list element at its position.
  `"drop"` is not applied to a constant list: its `Absent` elements reach the task, which holds
  them, and they enter the suffix.
- **An absent-aware supplied checksum gets no suffix.** A task with a supplied checksum may declare
  that its hook sees absence (`define(checksum=..., absent_aware=True)`, or
  `__checksum_absent_aware__ = True` on a class that defines `__checksum__`). The runner then
  passes the sorted `(field, element index, cause)` triples to the hook as the `absent` keyword,
  beside the resolved inputs, and appends no suffix; the hook is responsible for folding them into
  its value. The declaration is optional. A task without it gets the suffix as above.
- **The checksum-change check compares the unsuffixed checksum.** After a job with
  `__checksum__` runs, `Job._check_for_hash_changes` recomputes `task._checksum` and raises if it
  differs from `job.checksum` (`job.py:642-647`, called at `:468` and `:537`). The suffix is the
  runner's, not the task's, so the job keeps its unsuffixed checksum beside the suffixed one, and
  the check compares `task._checksum` with the unsuffixed value. Without that, every
  supplied-checksum job that runs with an absence would fail the check.
- **The suffix is one level deep in pydra; a driver carries it further.** A reader of a suffixed
  job holds no `Absent` and gets no suffix. A reader whose checksum pydra computes from values
  needs none: the suffixed job's output differs between the two runs, and so does the reader's
  hash. Only a supplied checksum built from upstream keys can miss it, and such keys come from the
  system that supplies them. §3.5 therefore requires that a driver's reference for a job carry the
  whole `job.checksum` and that downstream supplied keys be built from it. Carrying the suffix
  forward inside the runner was rejected: pydra cannot see how a hook builds its key, and a
  driver that builds keys from upstream keys can fold absence in itself, through an absent-aware
  hook or by building from the suffixed checksum (§3.5).
- **Splitter inputs.** A splitter input that resolves to an `Absent` gives the node no elements.
  The node is absent as a whole, and a reader of it receives that `Absent` rather than an empty
  `StateArray`, so absence is distinguishable from an empty list. A node whose state inherits that
  node's is absent as a whole too, with the same `Absent`. A combining reader receives the
  `Absent` and applies its mode: `"accept"` passes it, `"drop"` gives an empty list, `"propagate"`
  completes the reader absent. Such nodes have no elements and so no `Settled`; they are reported
  in `Round.absent_nodes` (§3.1). Today a splitter over an upstream
  output is resolved once the whole upstream is done (`start`, `submitter.py:1039-1075`). Per
  element, after E2, each upstream element's `Absent` gives that element's sub-state no elements.
  The node is not absent as a whole, and no element of the sub-state is settled. The details:
  - A reader of an absent sub-state receives the `Absent`, not an empty list. A sub-state is read
    as a group of a combining node, and a group is absent when every upstream element it covers
    gave an `Absent`. A combining reader applies its mode as above. A reader of a group with no
    elements for another reason, such as an empty list, receives an empty list. The group is
    formed wherever the combine sits: a partial node records its absent sub-states, own and
    inherited, by the coordinates they fix, so a partial node downstream that combines the
    inner axis forms the absent group too, through any chain of partial nodes. When the group
    also keeps an axis of the combining node's own static split, one absent group is formed per
    coordinate of that axis. A group that keeps an axis inside the absent sub-state has no
    absent group, as that sub-state has no coordinates there. A group that keeps an axis of
    another upstream output cannot be enumerated, so it raises `AbsentGroupError`, and the node
    never seals.
  - A read of every element holds only the elements that exist: a node split over `[[0, 1],
    Absent]` reads as `[1, 2]`, as if the absent element were not there. A read of every group
    holds the `Absent` in place of each absent group (`[[1, 2], Absent]`). Neither read is ever
    the `Absent` itself, even when every sub-state is absent: only a whole-value splitter input
    makes a node absent as a whole.
  - A propagated reader of an absent group takes, as the producer in its identity, the unsuffixed
    checksum of the upstream job whose `Absent` emptied the group.
  - A gated element is still an element: it completes absent with `Absent("not-routed")` and is
    settled. An element of an absent sub-state never exists, so it is neither gated nor settled.
  - A barrier node (§4.3, E2), and a barrier reader of a partial node, read an upstream element
    that is an `Absent` as an empty sub-state, which today's `State` indexes when every sibling
    is empty too: a reader combining the block axis of an all-absent upstream gets `[]`. Beside a
    present element the empty sub-state is ragged and is refused as other ragged shapes are.
    Today's `State` cannot form an absent group, so a combine that would keep one where a barrier
    node, or any node that is not partial, lies between the absent element and the combine
    raises `AbsentGroupError` (a `PydraStateError`) naming the axis and the upstream node. This
    covers a combining barrier node itself and a combining reader downstream of one. A combine
    that keeps no absent group (one over the block axis only) is not refused.
  - A key written while an absent element opened one propagating element (the behaviour before
    this rule was implemented) carries the `Absent`, hashed by `cause` or in the absence suffix.
    Under this rule no reader holds that `Absent`, so such a key is never looked up: it is
    orphaned, not served. The reader's key is the key of a run in which the element is not there.
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
  through an `Absent` input, which is hashed by `cause`, or enters the absence suffix when the
  reader's checksum is supplied or the field is not hashed (§2.3(a)). Either way a gate flip
  changes the reader's checksum.
- **The gate is an input of the node.** The `State` class is unchanged, but three places that list
  a node's inputs must include `when`/`unless`:
  - `Node._get_upstream_states` iterates `self.input_values` (`engine/node.py:210-230`). It must
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
  with every output `Absent(cause="not-routed", source=node)` and is reported as
  `Settled(how="gated")`. A gate that is itself `Absent` propagates (`how="absent"`). The gate is
  read before the element's data inputs are resolved: an element that stops at step 1 waits for its
  data upstreams, as for any input, but its data are never read or coerced. A gate is typed `bool`;
  anything else is a wiring error. A plain `bool` (a workflow input, concrete at construction) is a
  constant gate, and a lazy field typed `Any` is checked when read; a value that is neither a `bool`
  nor an `Absent`, such as a driver's reference, is refused by `ready()` with a `TypeError` naming
  the node and the gate.
- **A gated-off job's identity** is an absence record, as for a propagated job (§2.3(a)): the task
  type, `-absent-` and the digest of the reader's task definition, the unsuffixed checksum of the
  job the gate was read from, and the triple `(_when or _unless, -1, "not-routed")`. Two nodes with
  one task definition gated by one predicate element share it; their results differ only in
  `source`, which is provenance. It is never looked up, shared or committed.

**(c) Merging routes: `Select`, keyed on the gates.**
- A route is gated branches plus a merge. The merge selects on the gates, not on presence: a
  routed branch may itself return a content `Absent`, which must reach the reader with its cause.
- `pydra.compose.select(n)` builds a task class with inputs `when_0 … when_{n-1}` (bool, accept)
  and `value_0 … value_{n-1}` (accept). It returns `value_k` for the first `k` whose `when_k` is
  true, whatever that value is, and `Absent("not-routed")` if none is true.
- It is generated per arity because only top-level lazy fields are resolved
  (`submitter.py:1153-1157`); a variadic or list-of-lazy input is not. Nested-lazy resolution would
  remove the need and is not proposed here.

**(d) Loops, instantiated at run time: `workflow.loop(body, until="done", carry=None, index=None, max_rounds=None)`.**
- `loop` returns a task class. Its inputs are `body`'s (without `index`); its outputs are `body`'s, with
  their types, plus `rounds` (int), `converged` (bool) and `stopped` (str). Added to a workflow it is one
  node: it can be split, gated and read like any other, and under a split there is one loop per element.
- **Rounds.** The loop is a workflow job. The runner expands it with one node, `r0`, a copy of `body`
  given the loop's inputs. When round `k` is done and its `until` output is `False`, the runner adds
  `r<k+1>` to that job's graph: each `carry` input reads the named output of round `k`, `index` (if
  named) is given `k+1`, and every other input is the loop's. Nothing is constructed for a round before
  the round before it has returned `False`, so the cost of a loop is the rounds it runs. `until` is read
  by the runner and must resolve to a `bool` or an `Absent`, as a gate must (§2.3(b)); anything else
  fails the loop job with a `GateTypeError` naming the loop, `until` and the round; on a root the
  runner does not own, it becomes an entry of the run, as it would for an owned root. The loop's
  `construct()` builds a new workflow on every call and is never cached, because the runner grows it.
- **`carry`** maps a body input to the body output that feeds it in the next round. Left unset it is
  the inputs and outputs that share a name. A pair whose types do not match, a name that matches
  nothing, and an empty carry without `index` are refused at construction.
- **Identity.** A round is an ordinary job: its checksum is `body`'s over that round's resolved inputs,
  and it passes the precedence rule of §2.3(b) like any element. Two contracts:
  - `index="<field>"`: the round number is an input, so every round has its own checksum. A retry or
    poll body, which carries nothing forward, must use this.
  - `index=None`: the loop iterates a pure body. A round whose inputs equal an earlier round's has its
    checksum and is served its result.

  The loop job has its own checksum, over its inputs, `body`'s definition and the loop's parameters
  (`until`, `carry`, `index`, `max_rounds`, environment). A rerun of a finished loop is served whole
  from that slot; where the slot is missing, each round is served from its own and the next is
  instantiated from it. The runner keeps no loop state between runs.
- **Stopping.** The loop ends after the first round for which one of these holds, tested in order:
  `until` is an `Absent` (`stopped="absent"`); `until` is `True` (`"until"`); under `index=None`, the
  round's checksum equals an earlier round's of this loop (`"cycle"`: the loop's state repeats from
  here, so `until` can never become true; a fixed point is the case of length one, and a poll that
  forgot its `index` stops after two rounds); `max_rounds` is set and reached (`"max-rounds"`).
  `max_rounds` is optional and is only a stop: no graph, plan or store structure is sized by it. A
  round that fails fails the loop job, whose error is the round's. A round reported `NotRun` leaves the
  loop open until `retry`.
- **Outputs.** `body`'s outputs are those of the last round instantiated. `converged` is that round's
  `until` as returned: `True`, `False`, or the `Absent` with its cause. `rounds` is the number of
  rounds instantiated, less a last round that completed absent by propagation. A carried `Absent`
  reaches the next round through an ordinary lazy read, so the reading field's `absent` mode decides
  whether that round runs. A gated-off loop is never entered: every output, `rounds`, `converged` and
  `stopped` included, is `Absent("not-routed")`.
- **What a driver sees.** Each round is in `Round.dispatch` or `Round.settled` under the loop's path as
  `r<k>`. A loop that ends is reported once in `Round.loops` with its rounds and why it stopped
  (`"error"` included), and `done` waits for that report. `plan()` lists a loop not yet entered as
  `deferred` and an entered one as `looping`, with the rounds done, its `max_rounds` if any and no
  element total, followed by its live rounds; `plan()` never adds a round. One `ready()` adds at most
  `MAX_ADVANCES` (`pydra.engine.loop`, 64) rounds to each loop, so a loop replayed from the store
  returns to its driver between them. The cap is per loop: a split of n replayed loops may settle n
  times as many rounds in one call, as a wide stored node settles all its elements in one call.
- **Memory.** Between calls of `ready()` the runner holds the two newest rounds of a loop; older
  rounds are released at the end of the call that superseded them, after its records (a workflow
  round's absent nodes and unrunnable elements included) are collected, with their jobs, their entries
  in its tables and, for a round that is a workflow, its frame. Their results stay in the store. Under a
  store that commits nothing (a test harness), a round repeated by a cycle of length two or more whose
  first occurrence was released in an earlier call is computed again before the cycle stop. `runner.upstream(job)` is therefore valid
  from the job's hand-out until its outcome is reported.
- **Limits.** `body` must be a task class, so it carries no splitter of its own; a body that splits is
  a workflow that combines before `until`. Rounds of different elements' loops are separate dispatch
  groups, so they are not batched together; a loop whose rounds should batch across elements (one GPU
  call over many recordings per round) puts the split inside a workflow `body` with an aggregate
  `until`, and per-element early exit inside it is a gate on the element. One loop holds one workflow
  lease for its whole life; under `PickleStore` a split of n loops holds n heartbeat threads, as n
  nested workflows do.

**Rejected:**
- Routing as a splitter over a predicate inside `State`. The empty split already does this
  (§2.1), behind the same E2 barrier as any split over an upstream output; making it a `State`
  feature adds nothing the idiom lacks except what gates add without a `State` change.
- Unrolling `max_rounds` copies at construction (the first R11). Memory and `plan()` cost grow with
  the bound rather than with the rounds run, a loop over a split needs a merge across n rounds of one
  state, and a while condition without a meaningful bound cannot be written. Since R6 and E2 the
  runner already adds frames and blocks during a run; a loop adds rounds only inside the frame of its
  own job, so no graph that a reader is wired into changes.
- A loop as a nested workflow that contains a gated copy of itself (tail recursion). Round `k` holds
  `k` open frames and leases, the path grows by one level per round, and nothing is released until
  the innermost round returns.

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
    def plan(self, executors: Mapping[str, Executor] | None = None) -> Plan: ...
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

`plan(executors=None)` mirrors `drive(runner, executors, ...)`: the runner holds no executors, so an
external driver passes them by the names its router returns, and the default is the submitter's
workers. It changes nothing a run depends on: no job is released, handed out or settled, no lease is
taken, no job table of a live node (`_tasks`, `blocked`, `queued`, `successful`, `errored`) and no
static node's `State` is written, no in-flight result is read, and a job is formed only from upstream
elements the runner has accepted (never from one in flight, whose store slot may hold an earlier
attempt). On a partial node (E2) `plan()` may advance derived, monotone bookkeeping that the next
`ready()` would perform the same way: registering blocks for source keys that exist, closing groups,
sealing (which assigns `state_index`), and recording an inherited failed block as unrunnable. A partial
node is probed on a copy that expands the blocks whose upstream elements have been accepted, and
`PlanNode.known_elements` counts their elements while `elements` stays None until every block is
expanded. Where an
element's job cannot be formed yet, the router and `place(jobs)` are shown a stand-in (the node's name,
task with lazy inputs, resources, no state index) and `PlanNode.provisional` is true. A router error
on a real job raises, as the run would; on a stand-in the worker is None. `PlanNode.expected_hits` is 0
under a rerun, as `ready()` skips the store then. The router is called once per node, so a stateful
router sees these calls. Cost: one checksum per formable element, and one `expected_hits(jobs)` call
(`PickleStore`: a single listing of its record directory) or one existence check per element.
Elements that would complete absent (§2.3(a)) are counted in `PlanNode.absent`, and elements a gate
switches off in `PlanNode.gated`, read from the predicate's result (§2.3(b)); neither is routed,
placed or looked up. A node that would be absent as a whole is listed with `state="absent"` and no
elements. An entered loop (§2.3(d)) is listed as one entry with `state="looping"`,
`PlanNode.rounds` (the rounds done so far), `PlanNode.max_rounds` and `elements=None`, followed by
its live rounds; no round is instantiated by `plan()`.

@attrs.frozen
class Round:
    dispatch: list[Job]           # each handed out once; job.rerun and job.attempt already set
    settled: list[Settled]        # completed inside this call, never dispatched
    unrunnable: list[Unrunnable]  # marked unrunnable inside this call
    absent_nodes: list[AbsentNode]  # found absent as a whole inside this call, each once
    loops: list[LoopClosed]       # loops that ended inside this call, each once (§2.3(d))
    retry_after: float | None     # seconds until a waiting workflow lease is tried again

@attrs.frozen
class LoopClosed:
    loop: str                     # the loop job's path, element index or key included
    rounds: int                   # as the loop's `rounds` output
    stopped: Literal["until", "max-rounds", "cycle", "absent", "error"]

@attrs.frozen
class AbsentNode:
    node: str                     # the node's workflow path and name
    absent: Absent                # what every read of it receives
    inherits: str | None          # the node whose absence its state inherits; None at the origin

@attrs.frozen
class Settled:
    job: Job
    how: Literal["stored", "shared", "gated", "absent", "workflow-stored"]
    result: Result

@attrs.frozen
class Unrunnable:
    node: str                     # the node's name; its workflow path after R6
    index: int | None             # flat index; None on a partial node (key set) or whole node
    key: tuple | None             # the element's state_key on a partial node (§2.2, E2)
    block: tuple | None           # a block that failed before expanding (index and key None)
    upstream: list[str]           # the upstream nodes whose failed or unrunnable elements it reads
    # the whole node <=> index, key and block are all None

Outcome = Result | Failed | NotRun           # Result.committed: Committed | None (RFC 0003 §7.2)
@attrs.frozen
class Failed:  error: BaseException
@attrs.frozen
class NotRun:  reason: str; held: bool    # cancelled, never started, or held by infrastructure
```

**One thread.** `GraphRunner` is not thread-safe. Every call is made from one thread, the driver's;
an executor that reports from another thread has its outcomes marshalled to that thread
(`call_soon_threadsafe` in the asynchronous driver of §3.2, a queue in an external driver of
§3.5).

**`ready()`** applies the precedence of §2.3(b) to every released element. It returns the jobs to
dispatch and, separately, every job it completed itself, with how. Each job is handed out once.
`dispatch` is grouped by node, in node order (workflow path, then name).
- **`limit`** caps the jobs handed out by this call (`max_concurrent`, below) by whole node groups:
  groups are taken in order while they fit, and only the group that crosses the cap is sliced. With
  `limit=4` and two nodes of three released jobs, the first node is handed out whole and one job of
  the second. A driver of an executor with `batches = True` passes no cap (`max_concurrent` is
  infinite) and lets the executor cap inside itself, so a node's elements stay in one array or
  batch: it gives its worker at most `max_concurrent` jobs at once and holds the rest of a round
  until jobs end. Each batching executor caps its own jobs, and the driver's cap counts only the
  jobs of the other executors, so a run with several batching executors, or one beside other
  executors, can have more than `max_concurrent` jobs in flight in all. An external driver
  (§3.5) passes no `limit`.
- **Sharing is the runner's.** The same-checksum table (today's `futured`, `sharing` and
  `finished`, `submitter.py:610-616`, `:702-719`) is runner state applied in `ready()` after the
  gate and absence steps. A twin of a job in flight is held and settled as `Settled(how="shared")` when
  the first completes; a twin of a finished job settles at once. An external driver therefore sees
  each shared job in `Round.settled` and never receives a twin to dispatch. The drivers of §3.2
  keep no checksum table; they only dispatch.

`job.rerun` is set from
`rerun` and `propagate_rerun`: the outer job gets `rerun`, inner jobs get
`rerun and propagate_rerun` (today's `:575`, `:702`).

- **Under a `PortableStore`: `job.attempt`.** `ready()` gives each job in `Round.dispatch` an
  `attempt` id, a fresh UUID, each time it hands the job out (a job handed out again after
  `retry` gets a new one). A task with `can_resume` gets a UUID too: its work directory does not
  depend on the attempt, it is the stable `work/resume-<hash>` of its address, guarded by the
  resumable lease of the same hash, and the outcome files there (`_committed.json`, `_error.json`)
  carry the attempt, so an earlier attempt's outcome is never read as this one's. A job
  that is settled, propagated or marked unrunnable is never handed out and has none. A workflow
  job is never in `Round.dispatch`, so the runner also gives an attempt to each workflow job when it
  starts to expand it (after the store check, §3.1 "Workflow jobs under the split"), and to the
  outer job when it is created (`submitter.py:476-481`). A workflow job completed as
  `workflow-stored` is not expanded and has none. The store
  derives the job's work directory from `job.checksum`, the lookup context's fingerprint and
  `job.attempt` (RFC 0003 §5.4), so the checksum there is the suffixed one of §2.3(a). A
  `PickleStore` ignores `attempt`.
- **Under a `PortableStore`: `find(job)` in `ready()`.** Where `ready()` reads a store, through
  `read_succeeded` or for every job under `lookup_before_dispatch`, a runner over a store that has
  `find` (found with `getattr`, as `read_succeeded` is at `submitter.py:774`) calls `find(job)`
  instead. A `Hit` settles the job as `stored`; a `Miss` dispatches it; a `Verify` dispatches it
  with `job.verify` set to the candidate, so the executing process compares its fresh outputs with the
  stored entry before publishing (RFC 0003 §5.2, §7.1). `read_succeeded` returns `Result | None`
  and cannot carry the verify mark, which is why `find` replaces it there. The lookup comes after
  the gate and absence steps of §2.3(b), as every store read does.

**`complete(job, outcome)`:**
- A `Result` (errored or not) is the job's result. It is held on the job, so lazy reads by readers
  use it without a store read. Under a `PortableStore` the `Result` carries `committed`, the
  `Committed` that `publish` returned (RFC 0003 §7.2), and `complete()` keeps it with the `Result`.
  `Outcome` is unchanged. A `Result` with `stored=False` (outputs computed but not committed, RFC
  0003 §7.3) is a success: its readers run, and it is not reported in `Round.unrunnable`.
- `Failed` is a job that could not be run, recorded as `record_job_error` does now
  (`submitter.py:1327-1333`). Readers become unrunnable.
- **Unrunnable elements are reported.** An element that reads a failed or unrunnable upstream
  element has no `Job`, no `Settled` and no error entry: entries have `phase="run"` or
  `"resolution"` only (`submitter.py:1286`, `:1336-1348`). After E2 there is a third phase,
  `"expansion"`, for a block that could not be expanded: its entry has `state_index` and
  `state_key` `None` and `block` set, and its label is `B[block 1]`. `JobErrorEntry` gains
  `state_key` and `block`, both defaulting to `None`. A fourth phase, `"stopped"`, is recorded
  for an element that was handed out and had no outcome when the driver closed its executors
  (`fail_fast`, interruption, `close()`). A job reported `NotRun` already has its outcome and gets
  no `"stopped"` entry. `NodeExecution` already records, per
  index, the upstream nodes that made it unrunnable (`unrunnable` at `:942`, set at `:1251` for a
  whole node and `:1267` per element). `ready()` reports each newly marked entry once, in
  `Round.unrunnable`, so a driver can record a descendant of a failure as blocked by it.
- `NotRun` is neither. The job stays pending: its readers stay blocked, no error is recorded, and
  `held=True` says the cause was infrastructure. A later `ready()` hands the job out again only after
  `runner.retry(job)`. A run that ends with `NotRun` jobs is incomplete, not failed: `done` is true
  once nothing else can progress, `not_run` lists them, and `Submitter` raises `RunIncomplete`, a
  `WorkflowError` subclass, so callers that catch `WorkflowError` keep working.

**`upstream(job)`** returns, for each lazy input, the upstream jobs and state indices it was
resolved from (`NodeExecution._upstream`, `submitter.py:1179-1196`), so a driver need not recover
them from values. It lists data inputs only: a gate (§2.3(b)) is not an input of the task a driver
sends, so it is not listed. It stays a 2-tuple after E2: the index is `None` when the upstream is unsealed
or read whole, and the element's key is `job.state_key`. `Round.settled` is unchanged too;
`Settled.job` carries `state_key`. It is valid from the job's hand-out until its outcome is reported:
a loop releases its older rounds (§2.3(d)), and `upstream()` of a job the runner no longer holds
raises a `LookupError` naming it.

**`stuck()`.** When `ready()` returns nothing, nothing is in flight, `retry_after` is `None` and
`done` is false, the driver calls `stuck()`. It returns today's diagnosis (`submitter.py:634-701`)
for the driver to raise. A workflow job waiting for its lease is not stuck: while one waits,
`retry_after` is set and `stuck()` returns `None`. The store-visibility retry that today precedes
that diagnosis moves into the executor adapter (§3.3), because only an adapter that reads results
from the store needs it.

**What moves into it, unchanged in behaviour:**
- workflow construction and the execution graph (`submitter.py:604-607`);
- `NodeExecution`: `State` preparation, lazy resolution, per-element release and unrunnable
  marking (`:926-1299`);
- `on_error` (`:376-381`, `:745-750`), and error records (`:1316-1363`);
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
  runner takes the lease one attempt at a time. `Lease.try_take() -> float | None` makes public
  one pass of the loop in `DirectoryLease.__aenter__` (`lease.py:773-781`). The first call runs
  `_prepare()` (`:392-400`), which sets the wait's bounds; every call runs `_step()`
  (`:715-728`), which keeps `_deadline` and `_newest_seen` between attempts (`:669-676`), takes
  over a stale lease (`_judge`, `:521`), and returns `None` once the lease is held or the delay
  before the next attempt. A workflow job whose lease another process holds is neither dispatched
  nor settled. The runner keeps its `Lease` object and tries again on the first `ready()` after the
  delay, and `Round.retry_after` is the smallest delay over every waiting lease. A driver waits for
  the next outcome or for `retry_after`, whichever comes first, then calls `ready()` again (§3.2,
  §3.5). The job appears as `waiting` in `plan()`, and `stuck()` returns `None` while it waits.
  Two cases this covers: a run resubmitted after its coordinator was killed while expanding finds
  the old lease, waits, and takes it over once it is `STALE_AFTER` old (`:27`), as today; a second
  submitter of the same workflow on a shared cache waits, takes the lease when the first releases
  it, and finds `workflow-stored` under it.
- **`LeaseTimeout`.** If a bound passes (`LeaseSettings.timeout` without a renewal seen, or
  `max_wait`), `_step` raises `LeaseTimeout` (`:721-724`). The runner records it as an error
  entry of the workflow job with `phase="lease"`, the job's readers become unrunnable, and
  `ready()` does not raise it.
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
- **Under a `PortableStore`** the four steps take this form (RFC 0003 §13 item 6):
  - *Lease:* `try_take` on what `store.acquire` returns (`store.py:579-585` for `PickleStore`).
    That lease is always held, so `try_take` returns `None` at once, unless `dedupe` is on, the task
    is resumable or the job is a rerun (RFC 0003 §5.3, §10).
  - *Store check:* `find(job)`, before or under the lease. A `Hit` completes the job as
    `workflow-stored` and its nodes are not expanded; a `Verify` sets `job.verify` and the workflow
    is expanded.
  - *Expansion:* no success record is withdrawn, because there is none (RFC 0003 §5.2).
  - *Commit:* `publish` of the `WorkflowOutputs` under the workflow job's `job.checksum`, suffix
    included, and its execute context, returning `Committed`. `publish` never raises `LeaseLost`, so
    the `phase="commit"` entry does not arise; a `LeaseTimeout` from a `dedupe` lease is still
    recorded with `phase="lease"`. An error in a node publishes nothing, writes `_error.json` to the
    workflow job's work directory and records the error entry as today (RFC 0003 §11).

The outer job of a `Submitter` call is a workflow job whenever the task is a workflow or is split
(`submitter.py:443-465`), so its result is committed by the runner, and `runner.result()` and
`Submitter.__call__`'s check (`:503-513`) read it from the store as today. An outer leaf job is
handed out by `ready()` and committed by its executor. Both reads go through `Job.result()`, so
under a `PortableStore` they read as §3.3 describes for the Slurm adapter.

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
- Under a `PortableStore`, `read_succeeded` and `lookup` give way to `find(job)` (above), and a
  `Verify` is dispatched, not skipped.

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
        """Stop the jobs of `handle` not yet finished, as the run ends; they are not reported."""
    def close(self) -> None: ...
```

- **Outcomes are pushed.** Every job submitted is reported exactly once, through `report`, as a
  `Result`, `Failed` or `NotRun` (a job the worker itself cancelled is `NotRun`), unless its handle
  is cancelled as the run ends (below). The asynchronous driver turns each report into
  `runner.complete()` on its own loop (`loop.call_soon_threadsafe`). One `asyncio.Event` therefore
  waits across every executor of a run. The synchronous driver binds `runner.complete` itself, so
  an inline executor's report is applied before `submit` returns. The asynchronous driver closes
  the first draft's gap: it had a per-executor `wait` and a driver waiting across several.
- **`rerun`** is read from each job (`job.rerun`, §3.1), not passed per call, so `propagate_rerun`
  is kept.
- **`submit`** takes a sequence. An executor with `batches = False` is given one job per call. One
  with `batches = True` is given every job of a round that the router sent to it, in node order.
- **`cancel`** is called by the driver on every open handle when `fail_fast` stops a run, and when
  the driver itself is interrupted, as the `finally` at `submitter.py:751-756` does now. It is
  called only as the run ends, after `runner.close()`: the jobs of a cancelled handle are not
  reported, and each is recorded `"stopped"` where its workflow failed (§3.1). Cancelling a handle
  in the middle of a run is not supported.
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

**The drivers.** `Submitter` becomes a driver over one runner and its executors. There are two
drivers over the same `ready()`/`complete()`, because one `async` driver cannot keep what `debug`
does now.

A run whose executors are all inline (`debug`, the default) is driven by a synchronous loop, as
`expand_workflow` is now (`submitter.py:548-591`). No event loop runs while a task runs:
`Submitter.submit` calls `worker.run` directly (`:531`). A task may therefore call `asyncio.run`
or run a nested `Submitter` with an asynchronous worker (`run_until_complete`, `:529`), and a
`debug` run inside a notebook's running loop needs no `nest_asyncio`, as today. Each report is
applied before `submit` returns, so `runner.failed` is current when the next job is considered.

```python
def drive_inline(runner, executor):
    executor.bind(runner.complete)           # reports straight into the runner, same thread
    while not runner.done:
        rnd = runner.ready()
        audit(rnd.settled, rnd.unrunnable)
        for job in rnd.dispatch:
            executor.submit([job])           # runs the job; complete() has run on return
            if runner.failed and on_error == "fail_fast":
                raise runner.first_error     # the task's own exception object, as :579-580
        if not rnd.dispatch and not rnd.settled and not rnd.unrunnable:
            if rnd.retry_after is not None:
                time.sleep(rnd.retry_after)  # a workflow lease held by another process
            elif diagnosis := runner.stuck():
                raise RuntimeError(diagnosis)
    raise_entries(runner)                    # WorkflowError from runner.entries, as :757-759
```

Every other run is driven by an asynchronous loop. It owns the event loop as now (`get_open_loop`,
`:1400`, and `_own_loop`, `:323`); inside a running loop it behaves as today's asynchronous path.
An inline executor routed beside asynchronous ones runs on the helper thread of `_run_blocking`
(`:856-870`) and reports through `call_soon_threadsafe` like the others, as a synchronous worker
does beside asynchronous ones now.

```python
async def drive(runner, executors, router):
    loop, progressed = asyncio.get_running_loop(), asyncio.Event()
    def on_outcome(job, out):
        runner.complete(job, out); progressed.set()
    for ex in executors.values():
        ex.bind(lambda job, out: loop.call_soon_threadsafe(on_outcome, job, out))
    try:
        while not runner.done:
            rnd = runner.ready(limit=max_concurrent - in_flight)
            audit(rnd.settled, rnd.unrunnable)
            for name, jobs in group_by(router, rnd.dispatch):
                open_handles.add(executors[name].submit(jobs))
            if rnd.settled or rnd.unrunnable:
                continue                     # their readers may be ready now
            if not rnd.dispatch and not open_handles and rnd.retry_after is None:
                if diagnosis := runner.stuck():
                    raise RuntimeError(diagnosis)
            try:                             # the next outcome, or the next lease attempt
                await asyncio.wait_for(progressed.wait(), rnd.retry_after)
            except TimeoutError:
                pass
            progressed.clear()
            if runner.failed and on_error == "fail_fast":
                break
    finally:
        for handle in open_handles:
            handle.executor.cancel(handle)
    raise_entries(runner)                    # including the first error after a fail_fast break
```

`Submitter.__call__` keeps its signature and return value (`submitter.py:402-514`), including the
implicit split workflow (`:443-465`). The router keeps its signature (`utils/routing.py:11`).

### 3.3 Today's workers become executors with no user-visible change

- **`InlineExecutor(worker)`, for `debug`** (`workers/debug.py:14-28`). `inline = True`. `submit`
  calls `worker.run(job)` in the caller's thread, as `Submitter.submit` does now
  (`submitter.py:531`), and reports before returning, straight into `runner.complete` under the
  synchronous driver (§3.2). Under `fail_fast` that driver re-raises the task's own exception
  object before submitting the next job, as `:579-580` does now. No event loop is running, so
  signal handlers, `ContextVar`s set by the caller, `pdb`, main-thread-only libraries, a nested
  `asyncio.run` and a notebook keep working, and no job runs after the first failure. The helper
  thread of `_run_blocking` (`:856-870`) is used only for a synchronous worker routed beside
  asynchronous ones, as now.
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
  - **Under a `PortableStore`** `job.result()` (`job.py:617-624` reads through `lookup` today)
    reads by `job.committed` when set, otherwise by `<work dir>/_committed.json`, and when that is
    absent after the visibility retry, by `store.read` of the entry at the job's own address. Only
    when that also finds nothing does the adapter report `Failed`, naming the work directory (RFC
    0003 §5.4, §7.2). A job whose outputs were not stored (`stored=False`) is reported as its
    `Result` and its readers run. The same `Job.result()` serves `runner.result()` (§3.1) and the
    final read of `Submitter.__call__` (`submitter.py:503-513`).
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
| Identity | computes `job.checksum` once in the submitting process (`job.py:237-246`), through `__checksum__` or `checksum=` (K2, `compose/base/task.py:483-508`) or the declared `version` (K1), and the context once (`job.py:249-262`) | never rehashes: checksum and context travel with the job (`job.py:192-206`); under a `PortableStore` so do `attempt` and `verify`, and the context is the identity's lookup context (RFC 0003 §4, §7.1) |
| Gate and absence | evaluated first, never keyed, never committed (§2.3(b)) | never sees a gated-off or propagated-absent job |
| Hit before dispatch | `ready()`, through `read_succeeded` (`store.py:547-568`), and through `lookup` when `lookup_before_dispatch`; `find(job)` in their place when the store has it (§3.1) | — |
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
    record(rnd.settled, rnd.unrunnable, rnd.absent_nodes, rnd.loops)  # ...; loops ended
    for job in rnd.dispatch:
        send(job, runner.upstream(job))          # which upstream jobs fed each input
    if rnd.settled or rnd.unrunnable or rnd.absent_nodes or rnd.loops:
        continue                                 # their readers may be ready now
    for job, outcome in receive(timeout=rnd.retry_after):  # returns empty on timeout
        runner.complete(job, outcome)
```

What such a driver must and need not supply:
- **Outputs may be references,** except where the runner reads values. A gate's predicate must
  resolve to a `bool` or an `Absent`. Absence must be reported as pydra's `Absent`, not as a
  reference to one. A value that a downstream splitter splits over must be a real sequence; a
  reference there is refused with a `TypeError` naming the node and the field. The runner also
  resolves each reader's inputs into its task, which coerces them to the declared types, so for now
  every port that carries a reference, output or input, must be typed `Any`; a reference reaching a
  typed port is refused by the type check with an error naming the field. A reference marker that
  type coercion accepts for any declared type is a follow-up.
- **Every settled job is visible** in `Round.settled`, with how and, for absences, the cause, and
  every element blocked by a failure in `Round.unrunnable`, with the upstream nodes that blocked
  it. A node absent as a whole has no job; it is reported once in `Round.absent_nodes`, with its
  `Absent` and the node it inherits from, so following `inherits` from each entry rebuilds every
  absent subgraph from the rounds alone. A driver can therefore log every node as run, reused,
  typed-absent, gated or blocked without the audit hook. `runner.done` stays false while any such
  record is unreported (a twin shared by the last `complete()` is settled after every node is
  done), so the loop above sees every one. A round with only absent nodes, or only loops that
  ended, is progress, as one with only settled jobs is: the driver calls `ready()` again. Each round
  of a loop is an ordinary job, and a loop that ends is reported once in `Round.loops`.
- **It must not block without a bound.** `receive` waits at most `retry_after` when that is set,
  because a workflow job waiting for its lease is advanced only by a later `ready()` (§3.1). A
  driver with nothing in flight and `retry_after` set is waiting, not stuck; it calls `stuck()`
  only when `retry_after` is `None`.
- **Absence reaches supplied keys, transitively.** A driver that supplies checksums has two
  choices, and needs one. Its tasks may declare themselves absent-aware (§2.3(a)): the hook
  receives the absence triples, folds them into its own key, and gets no suffix, so the key keeps
  whatever form the driver gives its keys. This is optional. Otherwise the driver uses the suffixed
  `job.checksum` as it is: the reference it returns for a job carries the whole value, and a
  downstream supplied key that names an upstream output is built from it, not from the upstream's
  key before the suffix. pydra puts no constraint on the form of a supplied checksum beyond one
  path component, and a consumer may re-key the suffixed value into its own form. A key built from
  unsuffixed upstream keys of a task that is not absent-aware does not see an accepted or dropped
  `Absent` two levels down: a report read from a gather would be served from the run in which
  record 7 was present.
- **Infrastructure faults and held queues** are `NotRun(held=True)`. They are not errors, readers
  stay blocked, and the run ends incomplete and resumable. Content failures are `Failed`, or an
  errored `Result`. The mapping, per job:

  | What happened to the job | Outcome |
  |---|---|
  | its content failed and the driver's retries are exhausted | `Failed` (or an errored `Result`) |
  | held by its environment or queue | `NotRun(held=True)`, for each job held |
  | refused by a policy, such as a licence | `Failed`, for each job refused; never an exception raised out of the driver loop |
  | being re-placed after a resource or infrastructure fault | no outcome yet; the job is still in flight |

  A `Failed` makes that job's readers unrunnable and leaves its siblings' readers alone. A run
  whose only remaining jobs are `NotRun` is `done`, lists them in `not_run`, and under `Submitter`
  raises `RunIncomplete`.
- **Its store** is a `ResultStore` adapter over the driver's own store. The runner commits workflow
  jobs to it (§3.1). If the adapter is not durable, workflow results last only for the run, and a
  resumed run re-expands them.
- **What it forwards with a job.** If its store is a `PortableStore`, `send` carries `job.attempt`
  and `job.verify` with the checksum and context, and the `Result` it reports carries `committed`
  (RFC 0003 §5.4, §7.1, §7.2). A job settled by `find` in `ready()` is never sent.

## 4. Migration

### 4.1 Compatibility

- **Unchanged:** `Submitter(...)` arguments and `__call__`; `Worker` and every plugin;
  `PickleStore`; checksums; the cache layout; `WorkflowError`, `errors` and `entries`; the `debug`
  default; `on_error` defaults; error semantics under `debug`, including the caller's thread and no
  job after the first failure (§3.3).
- **Named behaviour changes, in R6 only:** `max_concurrent` caps the whole run rather than each
  nested workflow; equal-checksum jobs in different nested workflows run once.
- **New and opt-in:** `GraphRunner`, `Round`, `Unrunnable`, `Outcome`, `Executor`, `Lease.try_take`,
  `ResultStore.read_succeeded` (optional), `RunIncomplete`, `Absent`, `absent=`, `when=`/`unless=`,
  `select(n)`, `workflow.loop`, `Round.loops`/`LoopClosed`, `PlanNode.rounds`/`max_rounds` and
  the `"looping"` plan state.
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
| R2 | the two drivers over one runner: synchronous for all-inline runs, asynchronous otherwise; `InlineExecutor` for `debug`; `expand_workflow` becomes a wrapper | R1 | none; `debug` keeps the caller's thread, no running loop, and `fail_fast` order |
| R3 | `Executor` protocol (`bind`, `submit`, `cancel`, `close`), `WorkerExecutor` with the store read and visibility retry for workers that return `True`; the router dispatches to executors | R2 | none |
| R4 | reported completion: `complete()` holds the result on the job, with `Result.committed` when present; `update_status` stops polling dispatched jobs; the visibility retry leaves the loop for the adapter | R3 | none |
| R5 | `submit_many` for Slurm and SGE; tick and sleep grouping removed | R3 | array composition per round (§3.3) |
| R6 | flattening: one runner per call; the runner leases, checks, expands and commits workflow jobs; `Lease.try_take` and `Round.retry_after`; array key gains the workflow path; the workflow-job steps take their `PortableStore` form when RFC 0003 P4 and P5 are in | R4, R5 | named changes (§4.1); tests updated for them |
| R7 | driver model: `Round.settled`, `Round.unrunnable`, `NotRun`/`RunIncomplete`, `upstream()`, `lookup_before_dispatch`, `read_succeeded` in the protocol; `job.attempt` on handed-out jobs, and `find(job)` with `job.verify` when the store has `find` | R4 | opt-in |
| R8 | `plan()` (subsumes J2) | R6, F2, B2 | opt-in |
| E2 | partial `State` (0001 §3.1 step 2): per-element release of nested fan-out, `state_key`, the `"expansion"` phase and `Unrunnable.key`/`block`, including an `Absent` splitter input per element | R1 | 0001's; barrier kept for the six shapes of §2.2; corrects the ragged, empty, combined-source, mixed-scalar and static-combine cases (#945), whose checksums change; the `test_readiness` resplit and bad-input tests change expectations to keys |
| R10a | `Absent`: hashing, `absent=` with its defaults, propagation at release, coercion, `WorkflowOutputs`, the absence suffix for supplied checksums and unhashed fields; whole-node absence for a splitter input; after E2, per-element absence of a splitter input (no elements in that element's sub-state, an absent group read as the `Absent`, downstream partial combines form the absent group, a barrier path reads an empty sub-state and raises `AbsentGroupError` where a combine would keep an absent group). A single value that is an `Absent` propagates under `"drop"` (it has no elements to drop); a combined read of a wholly absent node is that `Absent`, so `"drop"` gives an empty list. A whole-value `Absent` has element index -1 in its triple | R6, E2 | opt-in |
| R10b | `when=`/`unless=`: node inputs, upstream states, lazy reads and edges; the precedence rule, with the gate read before data inputs are resolved; `Settled(how="gated")`, the gated-job identity and `PlanNode.gated`; a `bool` or an `Any`-typed lazy gate checked when read; `select(n)` (`pydra.compose.select`, a callable module, as `pydra.compose` is a namespace package; `from pydra.compose.select import select` also works) | R10a | opt-in |
| R11 | `workflow.loop` as a runtime while-loop: a `LoopTask` whose frame the runner grows one round at a time; `carry=`, `index=`, optional `max_rounds`; outputs `rounds`, `converged`, `stopped`; the cycle stop; a two-round memory window; `Round.loops`, `PlanNode.rounds`/`max_rounds` and state `"looping"`; `GraphRunner.done` waits for an open loop on an unowned root | R6, R10b, E2 | opt-in |

RFC 0003's P8 depends on R4, which passes `Result.committed` through. Its P4 changes how
`Job.result()` reads (§3.3), which the R3 adapter uses unchanged. The `find` and `attempt` parts of
R7 and the `PortableStore` form in R6 can land with or after RFC 0003 P5 and P4; with a
`PickleStore` they are inert. RFC 0003's P4 defines `work_dir` and `attempt`; R7 assigns `attempt`
in `ready()`. Whichever lands first uses the rule of §3.1 ("Under a `PortableStore`:
`job.attempt`"): before R7 the submitter's expansion loop gives the attempt at hand-out, and to
workflow and outer jobs as stated there.

R5 no longer depends on #929, which has merged; the first draft listed it because arrays and the
success-record read changed the same dispatch code. Each PR has tests on `debug` and `cf`, and on
Slurm where relevant. Only R5, R6 and E2 change any existing test expectation, and only for the
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
- Should `select` live in pydra or in a downstream package? (`workflow.loop` needs the runner, so it lives in pydra.)
- Should the deprecated `Submitter` methods be kept for one alpha, or removed in R2, given the 1.0
  alpha status?


## Amendment (2026-11-24): absent modes and the loop's per-call bound

**Section 2.3, absent modes (owner decision D2).** An input field's ``absent=`` is one of:

- ``"propagate"`` (the default, for a single and a combined read alike): the element completes
  absent; the function is not called.
- ``"accept"`` (opt-in): the value is passed as resolved, ``Absent`` included.
- ``"drop"``: a combined value is passed without its absent elements.
- ``"fill"`` with ``fill=<value>``: each ``Absent`` is replaced with the value, and its
  ``(field, element, cause)`` triple always enters the job's absence suffix, so a filled run is
  keyed apart from a run that read the same value for real.

An ``Absent`` left inside the value of a field that does not declare ``accept`` (for example a
single read of a list that holds one) is a resolution error of that element,
``AbsentNotAccepted``; it never reaches the function. This replaces the earlier default under
which a combined read accepted ``Absent``.

**Section 2.3(d), ``max_advances``.** ``GraphRunner(max_advances=)`` (default
``pydra.engine.loop.MAX_ADVANCES`` = 64) bounds the rounds each loop adds during one ``ready``
call. 64 keeps a store replay of a loop to a few driver round trips while every call still hands
control back to the driver within a bounded amount of work; a site with very cheap rounds may raise
it, one whose driver must stay responsive may lower it.


## Amendment (2026-11-25): bounded work per call, recorded errors, declared capabilities

**Work per `ready` call.** A pass walks each graph from the first node not done (a cursor
reset when the node order changes or a job is retried) up to the first node whose
predecessors include one not started; the progress test reads the same window. A pass walks
only frames not finished, and a released lease clears only the frames waiting on one. A chain
of N stored jobs therefore costs O(N) node looks, not O(N^2).

**Recorded errors.** A worker records the qualified type of a failure, and the exception
itself when it pickles, beside the traceback. The runner reads that exception back; otherwise
it reports a `RecordedError` carrying the type's name. The traceback text is never parsed.

**Executor capabilities.** A worker declares `batches` and `reports_done`; the executor reads
them instead of inspecting signatures. A round refused for one group (an SGE job asking for
more threads than `max_threads`) fails only that group's jobs.

**Absent values.** A `TypeParser` refuses an `Absent` unless built with `accepts_absent=True`
(a field that accepts one, and output reads).

### Moved from docstrings

### plan() (from PlanNode, C2)
    """One node of the graph, as the runner would run it now

    Parameters
    ----------
    path : str
        The node's name, with the nested workflows that hold it (``"outer/inner/node"``);
        a workflow job whose lease is not yet held is named by its own path
    elements : int, optional
        How many elements the node releases, whatever a concurrency cap would hand out at
        once; None if that depends on an upstream output that has not been read
    resources : ComputeResources, optional
        What the node asks of its worker
    worker : str, optional
        The name of the worker the submitter's router sends the node's first element to.
        An element whose job cannot be formed yet is routed as a stand-in (name, task with
        lazy inputs, resources), and the answer is ``provisional``; a stand-in the router
        fails on gives None. A real job the router fails on raises, as the run would
    expected_hits : int
        How many of the node's unfinished elements have a stored result to reuse. The store
        may answer with an ``expected_hits(jobs)`` method, which then counts everything
        ``ready`` would settle; otherwise the records are counted, and the results
        ``lookup`` returns when the runner has ``lookup_before_dispatch``. A store with a
        ``recorded(jobs)`` method (``PickleStore``) is asked for its recorded jobs and the
        others are looked up under that option. A store with neither (an adapter) reports 0. Only
        elements whose inputs are resolvable now have a checksum and are counted.
    state : "pending" | "waiting" | "deferred"
        ``pending``: its elements are released or can be. ``waiting``: a workflow job
        whose lease another process holds. ``deferred``: what it holds is not known yet (a
        workflow job not yet expanded, a split over an upstream output)
    gated : int
        Elements a gate switches off; always 0 until gates exist
    placement : object, optional
        What the worker's executor's ``place(jobs)`` returned for the node's elements,
        when it has one (RFC 0005 section 7.4). Elements whose jobs cannot be formed yet are
        shown to ``place`` as the same stand-ins the router sees, so the answer is then
        ``provisional``
    provisional : bool
        Whether ``worker`` or ``placement`` was derived from a stand-in rather than from
        the jobs the run will dispatch, so a later plan may differ


### 2.3(d) workflow.loop (from compose.workflow.loop, C6)
    """The task class of a loop over ``body``, whose rounds are instantiated at run time
    (RFC 0002 section 2.3(d))

    Added to a workflow, the loop is one node that can be split, gated and read like any
    other; under a split there is one loop per element. The runner expands it with one
    round, ``r0``, given the loop's inputs. When round ``k`` is done and its ``until``
    output is false, it adds ``r<k+1>``, whose ``carry`` inputs read round ``k``'s
    outputs, whose ``index`` (if named) is ``k+1`` and whose other inputs are the loop's.
    Nothing is built for a round before the one before it has returned false.

    The loop ends after the first round for which one of these holds, in order: its
    ``until`` is an ``Absent`` (``stopped="absent"``); it is true (``"until"``); with
    ``index=None``, the round's checksum equals an earlier round's (``"cycle"``: its
    state repeats, so ``until`` can never become true); ``max_rounds`` is reached
    (``"max-rounds"``). A round that fails fails the loop.

    The loop's inputs are ``body``'s without ``index``. Its outputs are ``body``'s, from
    the last round instantiated, plus ``rounds`` (the rounds instantiated, less a last
    round that completed absent by propagation), ``converged`` (that round's ``until``
    as returned: ``True``, ``False`` or the ``Absent``) and ``stopped``.

    Rounds of different elements' loops are dispatched separately, so a loop whose
    rounds should be batched across elements puts the split inside a workflow ``body``
    that combines before ``until``.


index: with index=None the loop iterates a pure body, and a round whose inputs equal an earlier round's is served that round's result. max_rounds is only a stop, which sizes nothing.


## Amendment (2026-11-25, round 2): progress as a count, the R7 split

**Progress.** `GraphRunner.ready` repeats its pass until a pass hands out a job or changes nothing.
It now compares a count of changes (`pydra.engine.progress`), not a snapshot of the graph: the node
tables (`successful`, `errored`, `unrunnable`, `queued`, `blocked`), a node's mode and partial state,
a partial node's version and seal, a frame's state, a loop's rounds and stop, and the runner's
handed, checked and reading sets count each change in the thread that makes it. Reading the count
walks no node, so a pass costs what `_runnable` costs. A runner is driven from one thread; a change
made by planning (`plan`'s scratch views) only adds a pass.

**R7 in three PRs.** The driver model, `max_concurrent` inside a batching `WorkerExecutor` (with the
driver's backlog), and releasing workflow leases off the event loop are separate branches; the
second and third each add their tests.

### Moved from docstrings (round 2)

The text the cut removed, by module and name, as it stood at the stack top.

- `pydra/engine/absent.py` `<module>`:

  A task may return :class:`Absent` for any output. A reader's input field declares what it does with
  one through its ``absent`` attribute: ``"propagate"`` (the default: the element completes absent
  without being looked up, shared, dispatched or committed), ``"accept"`` (the value is passed as
  resolved), ``"drop"`` (a combined value is passed without its absent elements) or ``"fill"`` (each
  is replaced with the field's ``fill``). A single value that is an ``Absent`` propagates under
  ``"drop"``, as there are no elements to drop. An ``Absent`` left inside the value of a field that
  does not accept it raises :class:`AbsentNotAccepted`. The triple of an ``Absent`` that is a field's
  whole value has the element index -1 (``WHOLE``).
  Outside the runner's resolution of an element, an ``Absent`` passes the type coercion of an output
  field and of an input field whose ``absent`` is ``"accept"`` or ``"drop"``.
  Every read of a node that is absent as a whole (a splitter input resolved to an ``Absent``, or its
  state inherits such a node's) is that ``Absent``. A combining reader then applies its mode:
  ``"accept"`` passes the ``Absent``, ``"drop"`` an empty list, ``"fill"`` the fill, and
  ``"propagate"`` completes the reader absent.
  Per element (a partial node), an upstream element whose output is an ``Absent`` gives that element's
  sub-state no elements. A read of a group all of whose sub-states are absent is the ``Absent``, and
  the reader applies its mode as above; a read of every element holds only the elements that exist. A
  partial node downstream forms the group too, from the absent sub-states its sources record. A
  barrier node, or a barrier reader of a partial node, reads such an element as an empty sub-state. A
  combine that would keep its group where no chain of partial nodes forms it raises
  ``AbsentGroupError``.
  A task whose checksum may not see an accepted or dropped ``Absent`` (a supplied checksum, or a field
  the checksum skips), or that filled one, has :func:`absence_suffix` of :func:`absence_triples`
  appended to its job checksum.

- `pydra/engine/absent.py` `Absent`:

  cause: It enters the hash

- `pydra/engine/absent.py` `absence_identity`:

  Different tasks reading the same ``Absent`` differ; the same task reading it at several elements has
  one identity, as its results are equal.

- `pydra/engine/absent.py` `absence_suffix`:

  The digest is taken over the RFC 8785 canonical JSON of ``[[field, {"$int": "<index>"}, cause],
  ...]``, the encoding RFC 0004's key record carries the triples in, so a store can recompute the
  suffix from the record. The suffix holds no path separator, so the checksum stays one path
  component.

- `pydra/engine/absent.py` `absence_triples`:

  These are the absent values a field filled, and those it accepted or dropped (a field's whole value
  has index -1, an element of a combined value its position before dropping) restricted to the fields
  the checksum does not hash: every field of a task with a supplied checksum (``__checksum__`` or
  ``checksum=``), else the fields with ``container_path``. A task whose checksum pydra computes from
  hashed values sees an ``Absent`` through its ``cause``, so it has none. Besides those the runner
  recorded, an ``Absent`` a field holds as a constant (a node of a nested workflow given one through
  the workflow's input) counts, at index -1 or its position in a list.

- `pydra/engine/drivers.py` `drive`:

  The jobs of each round are grouped by the executor the router names, and handed to it in node order,
  one call per job unless the executor batches. Outcomes are reported onto this loop. The driver keeps
  no checksum table: the runner settles twins itself. With ``on_error="fail_fast"`` the first failure
  stops the run. On the way out, whether complete, failed or interrupted, every open handle is
  cancelled and the runner's leases are released (off the loop). The executors are never closed:
  whoever made them closes them, so an executor that holds warm workers (a ``WarmPool``) keeps them
  from one run to the next. Failures are left on ``runner.errors``.
  ``max_concurrent`` caps the jobs in flight through ``runner.ready(limit)``, unless an executor
  batches (``batches = True``): the runner is then asked for every ready job, so a node's elements
  stay in one array or batch, and the batching executor caps inside itself, each to its own jobs. Jobs
  routed to the other executors then wait in the driver for the cap, which counts only theirs, so the
  run as a whole can have more than ``max_concurrent`` jobs in flight. Under ``fail_fast`` an executor
  whose ``serial`` attribute is true (a synchronous worker) has one job in flight at a time, so none
  starts after the first failure.
  router: An exception it raises is the job's failure

- `pydra/engine/drivers.py` `drive_inline`:

  With ``on_error="fail_fast"`` the first failure stops the run and its exception is raised as the
  task raised it; no job is started after it. Otherwise jobs that do not read a failed job still run,
  and the failures are left on ``runner.errors``. The driver keeps no checksum table: the runner
  settles twins itself.

- `pydra/engine/executor.py` `Executor`:

  Every job submitted is reported exactly once through the callable given to :meth:`bind`, as a
  ``Result``, a :class:`~pydra.engine.runner.Failed` or a :class:`~pydra.engine.runner.NotRun` (a job
  the worker cancelled). The callable may be invoked from any thread. :meth:`cancel` is called only as
  the run ends, after ``runner.close()``: the jobs of a cancelled handle are not reported, and are
  recorded ``"stopped"`` where their workflow failed. Cancelling in the middle of a run is not
  supported.

- `pydra/engine/executor.py` `InlineExecutor`:

  worker_for: Without it every job runs on ``worker``

- `pydra/engine/executor.py` `WorkerExecutor`:

  ``worker.run(job)`` is scheduled for each job, in one go for the jobs of a ``submit`` call, and the
  job is reported when it ends. A worker that returns ``True`` instead of a result (slurm, sge) is
  read back with ``job.result()``, retried with backoff for up to ``visibility_timeout`` seconds, then
  reported :class:`~pydra.engine.runner.Failed`. A synchronous worker runs on a helper thread. A
  worker declaring ``batches`` (slurm, sge) is given the jobs of a ``submit`` call together through
  ``submit_many(jobs)``, so it can place them in arrays. One declaring ``reports_done`` is also passed
  ``done(job, outcome)``, so a job is reported when it ends rather than when the round does.
  A batching executor is driven with no limit (RFC 0005 section 7.1) and caps inside itself: it gives
  the worker at most ``max_concurrent`` jobs at a time, in node order, and holds the rest of a round
  until jobs end (as each is reported through ``done``, or when its ``submit_many`` call returns). The
  cap is this executor's own: a run with several batching executors, or one beside executors capped by
  the driver, can have more than ``max_concurrent`` jobs in flight in all.

- `pydra/engine/job.py` `Job._finish_workflow`:

  Returns the error of the workflow (``error``, or the one raised building its outputs), None if it
  succeeded. Raises what the commit raises (`LeaseLost`), after which nothing is stored.

- `pydra/engine/job.py` `Job._pickle_to_commit`:

  A store with work directories is always given the job, which names the directory; it never writes
  the job to an entry.

- `pydra/engine/job.py` `Job._record_pool_facts`:

  They enter the execute context only through the store (a `PortableStore` whose identity lists
  ``pool`` in ``recorded_only``).

- `pydra/engine/job.py` `Job._reusable`:

  A store with ``find`` answers through its reuse hook (RFC 0003 section 7.1): a `Verify` marks the
  job to verify against that entry (and the job runs), anything else clears the mark. A job
  ``looked_up`` before dispatch is not looked up again under a lease that excludes nobody
  (``exclusive`` False): it keeps the mark that lookup set. An exclusive lease may have been held by a
  process that published the job since, so it looks again (RFC 0003 section 5.3). A work directory's
  ``_committed.json`` is the outcome of an earlier run and is never served to a new one. A deferred
  workflow job (`deferred`) is never served whole.

- `pydra/engine/job.py` `Job._run_task`:

  The state of each provided field is bound for the duration of the run and handed to the task through
  ``provided_values``; a task with no provided field costs nothing here.

- `pydra/engine/job.py` `Job._start_workflow`:

  Returns the result the workflow's outputs are later put in (see `_finish_workflow`).

- `pydra/engine/job.py` `Job.context`:

  It selects the job's slot in the store, is recorded on its result and is what a later job's context
  is compared with, so every process uses this one object. Building it reports a declared version
  whose stored results were computed from other code (see `pydra.engine.store.check_declared_code`). A
  store with an ``identity`` supplies the context (``identity.context(job, "lookup")``).

- `pydra/engine/job.py` `Job.record`:

  ``Job.record``, ``Job.checksum`` and ``PydraIdentityV2.key`` use the carried record and never
  rebuild it.

- `pydra/engine/job.py` `Job.resolved`:

  It is resolved again when the job's environment is replaced.

- `pydra/engine/job.py` `Job.resolves_reference`:

  A python task has nothing to resolve, whatever environment it was given.

- `pydra/engine/job.py` `Job.result`:

  settled: Otherwise nothing found is None, a job not yet run.

- `pydra/engine/loop.py` `<module>`:

  A loop is a workflow job whose graph the runner grows one round at a time (see
  :mod:`pydra.engine.runner`): round ``r<k+1>`` is added only once round ``r<k>`` is done and its
  ``until`` output is false.

- `pydra/engine/loop.py` `LoopConstructor`:

  It is hashed as the loop's :class:`LoopSpec`, so the loop job's checksum covers the body's
  definition and the loop's parameters.

- `pydra/engine/loop.py` `decide`:

  The tests are taken in order: an ``Absent`` until, a true until, a round whose checksum an earlier
  round of the loop had (under ``index=None`` the loop's state repeats from there, so until can never
  become true; a fixed point is the cycle of length one), and ``max_rounds``

- `pydra/engine/node.py` `Node.gates`:

  A lazy gate is an input of the node, so the node takes its upstream's state as it would a data
  input's

- `pydra/engine/node.py` `check_gate`:

  A ``bool`` is a workflow input whose value is known when the workflow is constructed. A lazy field
  typed ``Any`` passes, and its value is checked when it is read: an external driver's predicate port
  may be typed ``Any`` (RFC 0002 section 3.5).

- `pydra/engine/partial.py` `<module>`:

  A :class:`~pydra.engine.runner.NodeExecution` classifies itself once its upstream nodes have started
  (:func:`classify`) and, when it is partial, delegates its blocks, element keys, groups and seal to a
  :class:`PartialExpansion`. The index algebra itself (paths, entries, the RPN evaluation) is in
  :mod:`pydra.engine.state`.
  Terms ----- path A tuple naming a splitter field through the prev-state terms that lead to it:
  ``("_A", "A.x")`` is field ``A.x`` reached through the prev-state term ``_A``. element key
  (``state_key``) The coordinates of an element, one per path of ``elem_paths``. It never changes; its
  flat ``state_index`` is assigned when the node is sealed. block One combination of the final keys of
  the node's sources (its inner sources, and the partial nodes it reads through prev-state terms).
  settled / sealed Settled: every element the node will have is known. Sealed: settled, with every
  block expanded and every source sealed.

- `pydra/engine/partial.py` `PartialExpansion._absent_groups`:

  A group that keeps an axis inside the absent sub-state has no such group. A group that keeps an axis
  of another source cannot be enumerated, so it is refused as a block error (``AbsentGroupError``) and
  the node never seals

- `pydra/engine/partial.py` `PartialExpansion.expand_ready`:

  Only the blocks that may have changed since the last call are visited (``_candidates``)

- `pydra/engine/partial.py` `PartialExpansion.legacy_prepare`:

  Returns True, or the reason it cannot be (a ragged or combined source, or an order today's ``State``
  gets wrong)

- `pydra/engine/plan.py` `PlanNode`:

  state: a workflow job whose lease another process holds; not known yet (a workflow job not expanded,
  a split over an upstream output); or absent as a whole (RFC 0002 section 2.3(a)), with no elements
  ``looping``: an entered loop (RFC 0002 section 2.3(d)) and its live rounds, whose total is never
  predicted (``elements`` is None)

- `pydra/engine/plan.py` `_elements`:

  A job is formed only when every upstream element it reads is in the upstream node's ``successful``:
  a job handed out and not yet reported is never read, and nothing is kept on a live node. ``live`` is
  the node itself, whose tables say which elements are finished and which jobs exist when ``view`` is
  a copy of a started partial node.

- `pydra/engine/plan.py` `_hits`:

  A store's own ``expected_hits(jobs)`` answers for all of them, unless it has the finer ``recorded``
  (``PickleStore``), which names the recorded jobs so the rest can be looked up.

- `pydra/engine/plan.py` `_view`:

  The copy has its own ``State`` and, for a partial node, its own blocks. A copy of a partial node
  expands every block whose upstream elements are in the upstream nodes' ``successful``, as the next
  ``ready`` would, including the blocks of a live partial node that ``ready`` has not expanded yet;
  its element keys are the live node's.
  No job table of a live node is written (``_tasks``, ``blocked``, ``queued``, ``successful``,
  ``errored``) and no in-flight result is read. Reading a live partial node (its ``done``, ``settled``
  or a whole read's seal check) may advance its derived bookkeeping: registering blocks for source
  keys that exist, closing groups, sealing (which assigns ``state_index`` to its jobs), and recording
  an inherited failed block in ``unrunnable``. Each step is monotone and a function of tables
  ``plan()`` does not write, so the next ``ready()`` would make the same change. A barrier copy may
  also prepare the ``State`` of a sealed partial upstream, as its seal-time check does.

- `pydra/engine/runner.py` `<module>`:

  Drivers (see :class:`~pydra.engine.submitter.Submitter`) loop over a :class:`GraphRunner`, taking
  the jobs it hands out, running them and reporting each outcome.

- `pydra/engine/runner.py` `AbsentNode`:

  It has no elements, so no job of it is dispatched or settled; this is its record.

- `pydra/engine/runner.py` `GateTypeError`:

  It is raised from :meth:`GraphRunner.ready`, not recorded as the element's resolution error

- `pydra/engine/runner.py` `GraphRunner`:

  A driver calls :meth:`ready` for the jobs to run, runs them however it chooses, reports each with
  :meth:`complete`, and repeats until :attr:`done`. The runner reads the store, but never runs a leaf
  job.
  The runner expands nested workflows itself, so :meth:`ready` returns the leaf jobs of every level.
  For each workflow job it takes the lease one attempt at a time (a job whose lease another process
  holds is waited for: :attr:`Round.retry_after`), checks the store, expands the workflow and, when
  its nodes are done, commits its result. A driver ends with :meth:`close`, which releases the leases
  of unfinished workflow jobs.
  Every job the runner completes itself is reported in :attr:`Round.settled`: read from the store,
  shared with a job of the same checksum (the same-checksum table is the runner's, so a driver never
  receives a twin to dispatch), or a stored workflow job. Every element marked unrunnable is reported
  once in :attr:`Round.unrunnable`. An outcome is a ``Result``, a :class:`Failed` or a
  :class:`NotRun`; a run whose only remaining jobs are ``NotRun`` is :attr:`done`, lists them in
  :attr:`not_run` and is incomplete, not failed.
  The runner is not thread-safe: every call is made from one thread, the driver's, and an executor
  that reports from another thread has its outcomes marshalled to it.
  submitter: Without it a ``debug`` submitter over ``store`` is created

- `pydra/engine/runner.py` `GraphRunner._name_stopped`:

  An element that completed has its own outcome, and one never handed out (blocked behind an
  unfinished element, or released but held back by the ``ready`` limit) is not an entry: the driver
  reports it as unrunnable.

- `pydra/engine/runner.py` `GraphRunner._progress`:

  ``ready`` repeats its pass until a pass hands out a job or leaves the count unchanged.

- `pydra/engine/runner.py` `GraphRunner._read_store`:

  A store that `finds` (`PortableStore`) is asked that instead (RFC 0002 section 3.1): a `Hit` is
  settled, a `Verify` is dispatched with ``job.verify`` set, a `Miss` is dispatched. Without
  ``lookup_before_dispatch`` a job whose own address holds no entry (`present`) costs one existence
  check and is dispatched. A job read once without a result is not read again while the ``ready``
  limit holds it back.

- `pydra/engine/runner.py` `GraphRunner._read_succeeded_jobs`:

  Each job costs one existence check of its own record (`PickleStore.read_succeeded`). A job read this
  way is complete: its result is cached on it and it is not dispatched, so it takes no lease. A job
  without a record (its slot has errored, is running, is in a read-only cache or was written before
  records existed), or whose result the store's ``lookup`` does not return, returns errored or returns
  from another context, is dispatched and runs under its lease as before. A store without records
  dispatches every job.

- `pydra/engine/runner.py` `GraphRunner._release_node`:

  Its predecessors must have been released before it.

- `pydra/engine/runner.py` `GraphRunner.close`:

  Idempotent; a driver calls it however the run ends

- `pydra/engine/runner.py` `GraphRunner.complete`:

  outcome: A :class:`Failed` is recorded as the job's error and the elements that read the job become
  unrunnable. A ``Result`` is held on the job as given (whatever attributes it carries stay on it), so
  readers use it without a store read; an errored one is a failure of the run. A :class:`NotRun`
  leaves the job pending, its readers blocked and no error recorded, until :meth:`retry`. The twins
  held behind the job take a ``Result`` or ``Failed`` as theirs.

- `pydra/engine/runner.py` `GraphRunner.done`:

  It stays False while a settled job, an unrunnable element or an absent node has not yet been
  reported by :meth:`ready`, such as a twin shared by the last :meth:`complete`

- `pydra/engine/runner.py` `GraphRunner.plan`:

  No job is released, handed out or settled, no lease is taken and no live node or its ``State`` is
  written, so a planned run ends as an unplanned one (up to the classification of finished jobs that
  :attr:`NodeExecution.done` performs). A workflow job waiting for its lease is ``waiting``, one not
  yet tried is ``deferred``. Nodes that are done are not listed.
  A job is formed only from upstream elements the runner has already accepted, never from one in
  flight, so a node behind a running one is counted from its element count and not from stored
  results. Cost: one checksum per formable element, and for the store's records one
  ``expected_hits(jobs)`` call (``PickleStore``: a single listing of its record directory) or one
  existence check per element. The submitter's router is called once per node, with a real job when
  one can be formed, so a stateful router sees these calls.
  executors: one with a ``place(jobs)`` method is asked where the node's unfinished elements would
  run, and its answer is carried as :attr:`PlanNode.placement`

- `pydra/engine/runner.py` `GraphRunner.read_off_thread`:

  A job whose own address holds an entry is not handed out until its read ends; ``notify`` is called
  (from the helper thread) as each ends, and the next :meth:`ready` settles it or hands it out.
  :attr:`reading` counts the reads not yet taken up. ``readers`` is how many reads run at once
  (default `STORE_READERS`).

- `pydra/engine/runner.py` `GraphRunner.ready`:

  Each released element is taken through the store (a success record, then ``lookup`` when
  ``lookup_before_dispatch``) and the same-checksum table: a twin of a job whose outcome is known is
  settled at once, a twin of a job handed out and not yet reported is held and settled when that job's
  outcome is reported, and the rest are handed out, each once, with a fresh ``job.attempt``.
  limit: A driver of an executor with ``batches = True`` passes none, and the executor caps inside
  itself; so does an external driver.
  Round: Each loop adds at most ``max_advances`` rounds per call, so a loop replayed from the store
  returns to its driver between them; the rounds a loop no longer needs are released at the end of the
  call

- `pydra/engine/runner.py` `GraphRunner.release_off_thread`:

  The driver then awaits :meth:`released` once it has called :meth:`close`.

- `pydra/engine/runner.py` `GraphRunner.upstream`:

  The index is None for an upstream node without a state and for an input that reads the upstream
  whole (combined, or the sequence a splitter splits over).
  It is valid from the job's hand-out until its outcome is reported: a loop releases its older rounds
  (RFC 0002 section 2.3(d)).

- `pydra/engine/runner.py` `NodeExecution`:

  A node is classified once its upstream nodes have started (``mode``):
  * ``"static"``: its elements are known when it starts, indexed by int as before; * ``"partial"``: it
  splits over an upstream output (or reads a partial node through its state), so it is expanded one
  *block* per upstream element as each succeeds. Its elements are keyed by ``state_key``, a tuple of
  coordinates that never changes; ``state_index`` is filled in when the node is *sealed* (every block
  expanded); * ``"barrier"``: a shape that cannot be expanded per block, which waits for every
  upstream node to be done as before; * ``"error"``: a combiner that cannot apply, reported as a
  resolution error.

- `pydra/engine/runner.py` `NodeExecution._absent_axes`:

  Each maps to whether a reader combining it forms the absent group (a chain of partial nodes does)
  and the upstream node whose element was absent

- `pydra/engine/runner.py` `NodeExecution._become_absent`:

  Only called before the node has an element (a partial node that has expanded no block)

- `pydra/engine/runner.py` `NodeExecution._release`:

  An element whose propagating input holds an ``Absent`` completes absent at once: it is never looked
  up, shared, dispatched or committed.

- `pydra/engine/runner.py` `NodeExecution._resolve`:

  A whole-value ``Absent`` propagates under ``"drop"`` too, as it has no elements to drop.
  propagated: None if it runs

- `pydra/engine/runner.py` `NodeExecution.settled`:

  A node that is not partial is settled once started

- `pydra/engine/runner.py` `NotRun`:

  The job stays pending and its readers stay blocked until :meth:`GraphRunner.retry` hands it out
  again; no error is recorded.

- `pydra/engine/runner.py` `Settled`:

  how: A gated or absent job was never looked up, shared or committed

- `pydra/engine/runner.py` `_Frame`:

  The root frame is the runner's own task, with a job only when the runner is given one.

- `pydra/engine/runner.py` `accept_found`:

  ``complete`` says ``found`` is the answer of ``find(job)``: a job it does not serve is marked
  ``looked_up``.

- `pydra/engine/runner.py` `find_before_dispatch`:

  ``(None, False)`` when there is nothing at the address or the store cannot be read (the job is then
  looked up under its lease).

- `pydra/engine/runner.py` `result_error`:

  The recorded traceback is kept as ``recorded_traceback``.

- `pydra/engine/state.py` `expand_rpn`:

  ``*`` is an outer product with the left operand outer, ``.`` pairs the entries of its operands,
  which must be equally many. The result is in the order today's ``State.splits`` enumerates the same
  splitter.

- `pydra/engine/submitter.py` `RunIncomplete`:

  The run is incomplete, not failed, and a later run resumes it from the store.

- `pydra/engine/submitter.py` `Submitter`:

  cache_root: Under the default ``store="portable"`` it is the root of a
  :class:`~pydra.engine.portable.store.PortableStore` opened as ``private`` (owned by the current
  user, writable by no one else) that may hold pickles: it must be empty or such a root, and is by
  default ``store`` in the user cache directory (see :func:`~pydra.utils.general.default_store_root`).
  Under ``store="pickle"`` it holds :class:`~pydra.engine.store.PickleStore` slots and is by default
  ``run-cache`` there; loading from such a cache unpickles its files, so pointing Pydra at it trusts
  everyone who can write to it with code execution as the reading user: only use caches whose writers
  you trust. Given a store object (or an executor's store) and no ``cache_root``, it is that store's
  root when the store has one a job may write to (a ``PickleStore``, or a private or trusted
  ``PortableStore``), so a job run on another host keeps its run files where the store is; otherwise
  ``run-cache`` in the user cache directory
  worker: It runs every job the router does not send elsewhere, and is available to the router under
  the name ``"default"``. An executor object (:class:`~pydra.engine.executor.Executor`, such as a
  :class:`~pydra.workers.pool.pool.WarmPool`) is handed each run's jobs itself; it belongs to its
  caller, who closes it, so its workers stay resident from one call to the next, and its store, if it
  has one, is the submitter's. ``"pool"`` makes a ``WarmPool`` over the submitter's store from
  ``**kwargs`` (``root``, ``classes`` and its other parameters; ``substrate`` defaults to a
  ``LocalSubstrate``), which the submitter closes when it is closed. Several submitters may run on one
  ``WarmPool`` at once: each run receives its own jobs' outcomes
  workers: The name ``"default"`` is reserved for ``worker``
  router: By default every job goes to ``worker``. See :func:`pydra.utils.routing.tag_router`
  readonly_caches: Under ``store="portable"`` each is a portable root read as ``shared`` (its pickle
  members are never opened) unless given as a :class:`~pydra.engine.portable.trust.Root` with another
  trust. Under ``store="pickle"`` they are unpickled like ``cache_root``, so they trust everyone who
  can write to them with code execution as the reading user: only list caches whose writers you trust
  max_concurrent: A batching worker (one with ``submit_many``, such as slurm or sge) applies it inside
  its executor, to its own jobs only; the other workers share the cap the driver applies. A run that
  routes jobs to several batching workers, or to a batching worker beside other workers, can therefore
  have more than ``max_concurrent`` jobs in flight in all
  messenger: Saved in the `audit` attribute. See available flags at
  :class:`~pydra.utils.messenger.Messenger`.
  messengers_args: Saved in the `audit` attribu
  clean_stale_locks: A lease whose holder stopped renewing it is taken over once it is older than
  ``stale_after`` (see :class:`~pydra.engine.lease.LeaseSettings`), whichever host left it
  store: ``"pickle"`` a :class:`~pydra.engine.store.PickleStore` over ``cache_root`` and
  ``readonly_caches``; or a store object, which holds its own roots (a portable root on shared storage
  is passed this way, with ``trust="shared"`` and a ``work_dir``)
  reuse: A result whose context holds an unknown version is reused by no other process under
  ``"consistent"`` or ``"exact"``. Set the policy on the store itself when passing ``store``
  on_error: By default (None), the mode of the loop that expands the workflow: "fail_fast" for a
  synchronous ``worker`` such as *debug*, otherwise "raise_at_end" (workflows are always expanded by
  ``worker``, never routed, so the ``on_error`` attribute holds the mode used). Under "fail_fast" jobs
  that are already running on an asynchronous worker are not stopped (cancelling their futures does
  not stop the executor's processes), so they finish after the call returns, and their lock files
  remain until they do or until the pool is torn down with ``shutdown(cancel_futures=True)``
  telemetry: The ``NO_ET`` environment variable also turns the check off
  chdir: With ``chdir=False`` Python and workflow tasks run in the caller's working directory and the
  process directory never changes, so several jobs can share one process. Shell tasks run their
  command with the cache directory as its working directory, and relative outputs of shell tasks are
  resolved under it. A relative path a Python function writes lands in the caller's directory, not the
  job's cache directory, and is not moved: another job can overwrite it, so a cache hit can return a
  different file's content (a warning is emitted when a ``FileSet`` output lies outside the cache
  directory). The ``pre_run_task``/``post_run_task`` hooks see the caller's directory, and audit
  messages without an explicit ``message_dir`` go to ``<cache_dir>/messages``
  providers: Under ``debug`` the submitter holds one ``ResidentSet`` for its lifetime, so the state a
  provider loads is reused by later calls; under ``cf`` each process has its own

- `pydra/engine/submitter.py` `Submitter._failed_here`:

  Under a store with work directories that is a failure recorded for this attempt, never what the
  store holds at the job's address, which an earlier run may have stored; the job is then marked
  errored so that its result is the errored one.

- `pydra/engine/submitter.py` `Submitter._served`:

  A `Verify` marks the job, which then runs; a job given the answer of ``find(job)`` is marked
  ``looked_up`` (see `Job.looked_up`).

- `pydra/engine/submitter.py` `Submitter.expand_workflow`:

  Use a :class:`~pydra.engine.runner.GraphRunner` with :func:`~pydra.engine.drivers.drive_inline`

- `pydra/engine/submitter.py` `Submitter.route_name`:

  Only leaf jobs are routed: a workflow job is expanded by the submitter's own ``worker``, whatever
  its resources, and its nodes are placed individually.

- `pydra/engine/submitter.py` `_make_private_dir`:

  An existing directory is left as it is.

- `pydra/engine/submitter.py` `_portable_store`:

  On Windows, where a shared root is unsupported (RFC 0003 section 9), it is read ``private``, which
  is accepted only under ``%LOCALAPPDATA%``; elsewhere it is skipped with a warning.

- `pydra/engine/submitter.py` `check_identity_marker`:

  When it names an older scheme, or is missing from a root that holds slots, log one line (their
  results are recomputed once, as every key changed) and write it. A marker naming a newer or an
  unknown scheme is left as it is, with one line, so installs sharing a cache never flip it; an empty
  or garbled one (not ``<name>@<N>``) counts as missing. One read when the marker is current; a root
  that cannot be written is left as it is. Each root is logged at most once per process.

- `pydra/engine/workflow.py` `Workflow.add`:

  when: A gate that is itself ``Absent`` propagates

- `pydra/utils/reach.py` `<module>`:

  A development tool: it runs a Python task once under `sys.setprofile` and lists the user functions
  that ran but that the code walk of `pydra.utils.hash` did not reach, which are the places where
  ``getattr``, ``importlib`` or a registry hides a dependency. It is never used while a workflow runs.

- `pydra/utils/reach.py` `reach_report`:

  Everything runs in this process; code run in a thread is profiled, code run in another process is
  not.
  Raises: TypeError: ``task`` is not a Python task class Exception: whatever building the task or the
  function raises

- `pydra/utils/resources.py` `ComputeResources`:

  The resources are scheduler metadata: they never enter a task's checksum.
