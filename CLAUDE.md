# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pydra is a dataflow engine for constructing and executing directed acyclic graphs (DAGs) of tasks. It is the core for Nipype 2.0. Requires Python 3.11+. Uses `attrs` extensively for dataclass-like definitions and `hatchling` + `hatch-vcs` as the build system (version derived from git tags).

## Commands

### Install

```bash
pip install -e ".[dev]"   # full dev install (includes test + lint deps)
pip install -e ".[test]"  # test deps only
pip install -e ".[doc]"   # doc deps only
```

### Testing

```bash
pytest pydra                                          # full test suite (parallel, with coverage)
pytest pydra/engine/tests/test_job.py                 # single test file
pytest pydra/engine/tests/test_job.py::test_my_func   # single test
pytest pydra --only-worker=debug                      # single-process worker (good for debugging)
pytest pydra --only-worker=cf                         # ConcurrentFutures worker
pytest pydra --only-worker=slurm                      # SLURM (requires sbatch)
pytest pydra --with-dask                              # Dask worker
```

Set `_PYTEST_RAISE=1` for IDE breakpoint-friendly exception propagation.

Tests include doctests (`--doctest-modules` is on). `xfail_strict = true` means unexpected passes fail.

### Linting / Formatting

```bash
tox -e style        # check style (ruff)
tox -e style-fix    # auto-fix style
tox -e spellcheck   # codespell check
pre-commit run --all-files   # black + flake8 + codespell + nbstripout
black pydra
flake8              # max-line-length=105, ignores E203/W503/F541
```

### Tox Environments

```bash
tox -e py311-latest   # Python 3.11, latest deps
tox -e py313-pre      # Python 3.13, pre-release deps
tox -e py311-min      # Python 3.11, minimum pinned deps
```

### Docs

```bash
make -C docs html
```

### CLI

```bash
pydracli crash <crashfile>           # display crash file
pydracli crash <crashfile> --rerun   # rerun crashed job
```

## Architecture

### Layer Overview

```
compose/    Task definition (decorators + task types)
engine/     Execution engine (graph, jobs, state, submitter)
workers/    Execution backends (cf, debug, slurm, sge, + plugins)
environments/ Execution environments (native, docker, singularity, lmod)
utils/      Hashing, typing helpers, plugin discovery, profiling
tasks/      Built-in reusable tasks
scripts/    CLI entry points
```

### Task Definition (`compose/`)

Three task flavors, each defined by a decorator:

**Python tasks** — wrap a Python function:
```python
@python.define
def Add(a: int, b: int) -> int:
    return a + b
```

**Shell tasks** — wrap a CLI command:
```python
@shell.define
class BET(shell.Task["BET.Outputs"]):
    executable = "bet"
    input_image: File = shell.arg(argstr="{input_image}", position=1)
```

**Workflow tasks** — compose other tasks into a DAG:
```python
@workflow.define
def MyWorkflow(x: int) -> int:
    node_a = workflow.add(Add(a=x, b=1))
    return node_a.out
```

The decorator machinery is in `compose/base/builder.py` (`build_task_class()`). It converts `Arg`/`Out` field specs into `attrs` fields and dynamically creates a `Task` class + a paired `Outputs` class.

Key base types: `compose.base.Field`, `Arg`, `Out`, `Task[OutputType]`, `Outputs`.

### Execution Engine (`engine/`)

**Workflow construction** (`engine/workflow.py`): `Workflow.construct(task)` runs the workflow definition function to discover nodes and wire the `DiGraph`. Constructed workflows are cached by content hash.

**Node** (`engine/node.py`): Wraps a `Task` inside a workflow. Holds the task, its `State`, optional `Environment`, and `TaskHooks`. Exposes `lzout` — a lazy output proxy for wiring downstream nodes.

**LazyField** (`engine/lazy.py`): Promises between nodes. `node.lzout.x` returns a `LazyOutField`. Assigning it to another node's input creates a dataflow edge.

**State / Splitter / Combiner** (`engine/state.py`): Implements map-reduce semantics. A node can be split over an iterable input (producing parallel jobs) and combined (reducing results). Splitters can be scalar (zip) or outer (cartesian product), expressed in RPN. Each concrete state index corresponds to one `Job`.

**Job** (`engine/job.py`): The concrete unit of work submitted to a worker. Holds the fully-resolved `Task`, a `cache_dir` (from content hash of task inputs + definition), and uses `filelock.SoftFileLock` for safe parallel execution.

**Submitter** (`engine/submitter.py`): The async dispatch loop. Constructs the `DiGraph` of `NodeExecution` objects, drives an asyncio event loop to submit ready jobs to the configured `Worker`, handles caching (skips jobs with a valid cached result), and manages concurrency.

```python
with Submitter(worker="cf", cache_root="/tmp/cache") as sub:
    result = sub(my_task)
```

### Workers (`workers/`)

| Class | Module | Description |
|---|---|---|
| `ConcurrentFuturesWorker` | `cf.py` | `ProcessPoolExecutor`-based (default) |
| `DebugWorker` | `debug.py` | Single-process, synchronous |
| `SlurmWorker` | `slurm.py` | Submits via `sbatch`, polls with `sacct` |
| `SGEWorker` | `sge.py` | SGE qsub |

Workers are discovered via a plugin system (`get_plugin_classes` in `utils/general.py`). External workers (`pydra-workers-psij`, `pydra-workers-dask`) are installable as separate packages.

### Environments (`environments/`)

Control *how* shell tasks execute: `native.py` (bare OS), `docker.py`, `singularity.py`, `lmod.py` (load environment modules before executing).

### Caching (`utils/hash.py`, `engine/job.py`)

Cache keys are content hashes of the `Task` (all inputs + task definition). Results are stored as cloudpickled files under `~/.cache/pydra/<version>/run-cache/` (via `platformdirs`). Lock files prevent race conditions.

### Provenance Tracking (`engine/audit.py`)

Optional JSON-LD provenance tracking controlled via `AuditFlag` bits. Messengers: `PrintMessenger`, `FileMessenger`, `RemoteRESTMessenger`. Schema at `schema/context.jsonld`.

### Data Flow Summary

```
User code
  │
  ├─ @python.define / @shell.define / @workflow.define
  │     └─> compose/base/builder.py: build_task_class()
  │             creates Task(attrs) + Outputs(attrs)
  │
  ├─ Submitter(worker="cf", cache_root=...)
  │     ├─ Workflow.construct(task)  → DiGraph of Nodes  (engine/graph.py)
  │     ├─ State resolution          → list of state indices (engine/state.py)
  │     ├─ per state-index: Job(task, cache_dir)         (engine/job.py)
  │     │     └─ if not cached → Worker.run(job)         (workers/)
  │     │           └─ Environment.execute(job)          (environments/)
  │     └─ Result(outputs, runtime, cache_dir)           (engine/result.py)
  │
  └─ LazyField wiring between Nodes                      (engine/lazy.py)
```

## Key Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Build, deps, pytest config, coverage config |
| `tox.ini` | tox envs: test, style, style-fix, spellcheck, build, publish |
| `.flake8` | Flake8: max-line-length=105 |
| `pydra/conftest.py` | `worker`/`any_worker` fixtures; `--only-worker`, `--with-dask` flags |
| `pydra/compose/base/builder.py` | Decorator machinery (`build_task_class`) |
| `pydra/compose/base/field.py` | `Field`, `Arg`, `Out`, `NO_DEFAULT`, `Requirement` |
| `pydra/compose/base/task.py` | `Task` and `Outputs` base classes |
| `pydra/compose/python.py` | `@python.define` |
| `pydra/compose/shell/task.py` | Shell `Task`, CLI construction |
| `pydra/compose/workflow.py` | `@workflow.define`, `workflow.add`, `workflow.this` |
| `pydra/engine/submitter.py` | Async dispatch loop |
| `pydra/engine/job.py` | Single unit of work, caching, locking |
| `pydra/engine/state.py` | Splitter/combiner map-reduce |
| `pydra/engine/lazy.py` | `LazyField` — dataflow wiring |
| `pydra/engine/workflow.py` | DAG construction and caching |
| `pydra/engine/node.py` | `Node` — task wrapper in workflow graph |
| `pydra/utils/hash.py` | Content hashing for cache keys |
| `pydra/utils/general.py` | Plugin discovery, cache root, platform utils |
| `pydra/utils/typing.py` | `StateArray`, `TypeParser`, type helpers |
