# Proposal 0004: portable automatic task identity

Status: proposed, for maintainer discussion. Branch `rfc/portable-task-identity`, off
`enh/execution-extensions` at `3f734f9a`. Line references are to `3f734f9a` unless a branch is named.
Proposal 0001 names are used throughout: **K1** is a declared `version=` (0001 §3.11), **K2** a
supplied `checksum=` with the reuse hook (0001 §3.12), **G1** the environment checksum recorded and
never keyed (0001 §3.5). RFC 0003 is `rfc/portable-store` at `fe8943a9`. The probes cited in §8 were
run on one macOS arm64 machine under load. Timings are upper bounds.

## 1. Motivation

A job checksum should change exactly when the computation changes. It should not change because
the run moved to another machine or another Python version. It should also not ask anything of a
user who wires functions from existing libraries, or command-line tools in different
environments, into a workflow. K1 and K2 let an author take identity over, but most pydra users
will never declare a version, so the automatic default has to be right on its own.

At `3f734f9a` the code part of the default is in better shape than its reputation. FH1 to FH4
replaced bytecode with a version-neutral AST encoding, `ast@1` (`pydra/utils/hash.py:763-875`), and
replaced the function-local hash with a walk that follows closures and globals through user
modules (`hash.py:1246-1583`). Issue #777 ("doesn't pick up changes in functions defined in the
either in the current or other modules") is still open, but the walk now handles the cases it
describes (§8.3). The remaining gaps are in what the walk stops at and in how values are encoded:

| Problem | Evidence |
|---|---|
| A library function is identified by its name only | `lib:{module.qualname}` (`hash.py:1385-1393`); a module global as `module:{name}` (`hash.py:1366-1369`). Probe §8.4: one task under numpy 2.4.6 and 2.5.4 has the same checksum. Library versions are recorded in the context (`engine/context.py:468-486`), so `ConsistentReuse` refuses the entry (`engine/store.py:246-260`), but `AnyReuse` serves it and a portable store files both under one key |
| A shell task's command template is not in its checksum | `_compute_hashes` hashes the input *values* and the `Outputs` class (`compose/base/task.py:568-585`), not the input field definitions. Probe §8.2: `ls -l <dir>` and `ls -a <dir>` give one checksum, and so do `argstr="--a"` and `argstr="--b"` |
| A native executable is its name | `executable` is a `str` field (`compose/shell/task.py:243`); `Native` has no fields, so its environment identity is `{}` (`environments/native.py:12-30`, `environments/base.py:53-69`). An upgraded tool changes neither key nor context |
| A container tag is not resolved | `Container._identity` returns `digest: None` when `_resolve_digest` cannot resolve one (`environments/base.py:142-156`); `Singularity` does not override it; `Docker` returns the local image id, or None (`environments/docker.py:15-33`). Probe §8.6: a local `python:3.13-slim` pulled earlier has digest `09a17cad…`, while the registry tag now points to `e48f59cd…` |
| Arrays are hashed without dtype or shape | class name, `size` and `tobytes()` (`hash.py:1790-1797`). Probe §8.2: int64 and float64 zeros of one size hash equal, as do shapes (2,3), (3,2) and (6,) of one buffer |
| A path is hashed as its string | `bytes_repr_pathlike` (`hash.py:498-501`). A `Path` input gets a hit after its file changes and a miss after it moves. Only `FileSet` values are hashed by content (`hash.py:692-708`) |
| An instance of a user class is hashed without its class's code | the generic serialiser hashes the class name and the instance's fields (`hash.py:417-437`). Probe §8.3: editing a method of a class whose instance is passed as an input leaves the checksum unchanged |
| The checksum cannot be explained | it is a 128-bit blake2b over nested byte streams (`hash.py:358`), and only the hashes per field are kept (`task.py:577-585`). A store cannot re-derive a checksum from what it stores, and a user cannot see why two runs missed |

**One motivating user.** senselab 2.x (sensein/senselab#617) drives pydra only through K2. Its
ADR-002 rejects code hashing because it "is not portable across platforms, and it misses called
functions (pydra #777)". The probes in §8 show that both statements describe pydra before FH1–FH4
for user code, and that they still hold for library code, shell tasks and values. A senselab-style
declaration cannot be the default, because people calling library functions and command-line tools
will not write one. The rest of this proposal does not refer to senselab.

## 2. Goals, non-goals, compatibility

**Goals:**
- the automatic checksum of a task is equal on every platform and every supported Python minor,
  for the same code, the same library versions, the same tools and the same input content;
- it changes when user code, a reached library's version, a shell command template, a tool's
  version or an image digest changes;
- inputs are identified by content, with a type tag, using the encoding RFC 0003 uses for entries;
- what is keyed and what is only recorded follows one rule (§3.2);
- a checksum can be re-derived from a small, stored record;
- one recompute per cache when this lands, announced, and a versioned scheme after that.

**Non-goals:**
- Proving that a function is pure. Values read from mutable globals are hashed as they are at
  submission (`hash.py:1459-1461`), as now.
- Tracing dynamic dispatch (§4.3).
- Measuring how close two contexts are. That is the reuse policy's job (RFC 0003 §6).

**Defaults that do not change:** `code_hash="user"` (`hash.py:944-947`); `ast@1`; K1 and K2; the
context fields and the three reuse policies; `PersistentCache` and its location. **Checksums do
change**, once (§6). This reverses 0001 §2 ("checksums of tasks that use no environment, or
`Native`/`Lmod`" do not change), for the reasons in §1.

## 3. The scheme

### 3.1 The identity record

A job's checksum is the digest of a small JSON document, its identity record:

```json
{
  "scheme": "pydra-identity@1",
  "kind": "python",
  "task": {"code": "code@1:9f2c…", "refs": ["lib:numpy==2.4.6:numpy.median", "…"]},
  "outputs": "type@1:51ab…",
  "inputs": {"x": "float@1:3ff0000000000000", "img": "file@1:sha256:7d1e…"}
}
```

- `checksum = f"{kind}-{sha256(JCS(record))}"`, where JCS is RFC 8785 canonical JSON, the encoding
  RFC 0003 already uses for `pydra-context@1` (0003 §3.3).
- Every input is one typed digest string (§3.4). A value with no portable encoding falls back to
  today's `bytes_repr`, tagged `py@1:`, so nothing that hashes today stops hashing.
- `task` is the code part: §4 for Python tasks, §5 for shell tasks. A supplied `checksum=` (K2)
  replaces the whole record, as today (`task.py:482-508`).
- The record is kept with the result. RFC 0003's reader can recompute the checksum from it and
  check that it equals the entry's path (0003 §5.2), and `pydracli explain <job>` can show the
  fields where two records differ.

### 3.2 What is keyed and what is recorded

One rule decides: **a fact enters the key if it names what is computed and is the same on every
platform that can run it. A fact that depends on the machine is recorded in the context and judged
by the reuse policy** (G1, RFC 0003 §6).

| Keyed (portable) | Recorded (`job_context`, `engine/context.py:489-522`) |
|---|---|
| user code, `ast@1` | Python full version, implementation, platform, C library |
| a reached library's distribution, public version and VCS commit | the local version label (`+cu121`), the transitive requirements of reached distributions, the archive hash of an installed wheel |
| the shell command template | the resolved executable path and the sha256 of its file |
| a declared tool version, or the output of a declared version command | the owning OS or conda package of the executable |
| a container image's index digest | the local image id and the per-platform manifest digest |
| an Lmod module's resolved name and version | the Lmod tree, `MODULEPATH` |
| input content, with dtype and shape | hardware, CUDA, environment variables |

## 4. Python tasks

### 4.1 Functions from installed libraries

Where the walk reaches library code today it writes `lib:{module.qualname}` (`hash.py:1385-1393`),
`module:{name}` (`hash.py:1366-1369`), or the reference of a builtin or descriptor
(`hash.py:1407-1431`). Under this proposal each such reference also carries its owner:

```
lib:{dist}=={public version}[@{commit}]:{module.qualname}
```

- **The owner** is the distribution whose `RECORD` lists the module's file, which is
  `context._owners` (`engine/context.py:426-444`) moved to `pydra/utils/hash.py` so the walk and
  the context use one lookup. It is memoised per module.
- **The version** is the PEP 440 public version. The local segment (`2.5.1+cu121`) is recorded, not
  keyed: it names a build for a platform. A VCS install appends its `commit_id`
  (`engine/context.py:261-262`), which is portable. An archive hash is not keyed, because each
  platform installs a different wheel.
- **The standard library and builtins** keep `lib:{module.qualname}` with no version. The Python
  minor is already compared by `ConsistentReuse` (`engine/store.py:246-260`). Keying it would bring
  back the false misses across minors that `ast@1` removed.
- **C extensions** (`numpy.add`, a ufunc whose `__module__` is `numpy`) are owned like any other
  module. The owner is found from the module's file (§8.5). No source is read.
- **Editable installs** live outside `site-packages`, so `_path_is_library` already classifies them
  as user code (`hash.py:961-973`). Their source is walked under `ast@1`, so the version string,
  which an editable install does not change, is never used. An editable install inside
  `site-packages` is found through `direct_url.json` (`engine/context.py:195-205`) and walked the
  same way.
- **Library modules no distribution owns** (a vendored module, a `PYTHONPATH` entry inside a
  library root, a conda package without `.dist-info`): a pure-Python module is walked as user code.
  An extension module gets `lib:?:{module.qualname}`, and the context gets `unknown`, so the result
  is not reused by another process (`engine/context.py:520-521`). That is today's behaviour for an
  unknown package.
- **Only directly reached distributions are keyed.** A task that calls `scipy.signal.resample`
  keys scipy. numpy, which scipy requires, stays in the context's `packages` closure
  (`engine/context.py:448-486`) and is compared by `ConsistentReuse`. Keying the closure would make
  one unrelated upgrade in a large environment invalidate every task.

**Classification has to agree between sites.** A package installed as a wheel on a cluster is
library code, keyed by name and version. The same package in an editable checkout on a laptop is
user code, keyed by its AST. The two checksums differ although the code is the same. A project
that wants one checksum for both sets `PYDRA_USER_PACKAGES=name` (`hash.py:976-997`) on both,
which walks the package as user code everywhere. The documentation has to say this prominently.

### 4.2 User-defined functions

This part exists. The proposal keeps it and fixes two gaps.

- **What is traced.** The walk starts at the task function. It reads the globals named by the
  function's bytecode, with the attribute chains read from them (`hash.py:1100-1131`). It follows a
  chain through user modules and stops at the first library module (`hash.py:1322-1336`). It reads
  closure cells and the defaults of called functions (`hash.py:1311-1341`), and the bases and
  members of user classes (`hash.py:1343-1347`, `:1157-1236`). Each definition it reaches is
  encoded once under `ast@1` (`hash.py:1134-1154`), so cycles terminate. It is memoised per process
  and checked against source mtimes and weak references (`hash.py:1474-1642`).
- **Why it is portable.** `ast@1` drops positions, `ctx`, docstrings, the root's name, annotations
  and decorators, and fields at their defaults, and it merges f-string parts
  (`hash.py:790-875`). Probe §8.1: 14,986 functions encode identically under Python 3.11, 3.12,
  3.13, 3.14 and 3.15. Probe §8.3: one task has one checksum under 3.11, 3.12 and 3.13, and from two
  directories. The walk uses live objects, not a static import graph, so `from x import y`, aliases
  and re-exports resolve as Python resolves them.
- **Gap 1: inputs that are user-class instances.** Hash the class of a user-class instance with
  its code, by appending `hash_single(type(obj))` in the generic serialiser when the type is not
  library code (`hash.py:417-437`). `bytes_repr_type` already hashes a class's `__dict__`,
  functions included (`hash.py:612-636`). Probe §8.3: with this change, editing the input's method
  changes the checksum, and editing an unrelated class does not.
- **Gap 2: library references carry versions** (§4.1).

**What falls back, and how the user overrides it:**

| Case | Behaviour | Override |
|---|---|---|
| `getattr(module, name)`, `importlib.import_module(...)`, dispatch through a registry or string | the target is not seen (probe §8.3) | pass the callable as an input; or `version=`; or `checksum=` |
| a method called on an input object | its class's code, after gap 1 | — |
| lambdas | walked as functions; lambdas that share a line are told apart by instruction positions (`hash.py:902-931`) | — |
| closures, factory-made functions | the closure cells are walked (probe §8.3) | — |
| decorators | a user decorator's wrapper is walked through its closure; a library wrapper with `__wrapped__` is walked through it (`hash.py:1385-1393`, `:1453-1458`); decorators of the task function itself are not encoded (`hash.py:803-810`) | `version=` |
| module constants, mutable globals | hashed by their value at submission (`hash.py:1459-1464`) | pass them as inputs |
| notebooks | cells registered in `linecache` (IPython does) are walked under `ast@1` (`hash.py:1186-1191`) | — |
| no source (REPL, `exec` without `linecache`) | bytecode tagged with the Python minor (`hash.py:1147-1150`): correct, but a miss across minors | define it in a file; or `version=` |
| `code_hash="function"` or `"module"` | the walk stops early, by the user's choice (`hash.py:1300-1303`) | — |

Dynamic dispatch cannot be traced statically, and tracing it at run time costs too much for every
job. The proposal adds an opt-in check, `pydracli check-reach <module:task> [inputs]`. It runs the
task once under `sys.setprofile` and lists the user functions that ran but that the walk did not
reach, which are the places a `getattr` hides a dependency. It is a development tool and never
runs during a workflow.

**#777.** Close it, citing FH1–FH4 and the probe §8.3 cases as regression tests, with gap 1 in the
same PR.

### 4.3 Workflows

A workflow's constructor is walked like a function today (`compose/workflow.py:141-153`). Its nodes
are keyed by their own records. Nothing changes except that `version=` becomes available (§6.1).

## 5. Shell tasks

### 5.1 The command template

The record of a shell task gets `task.template`. It holds the executable, then the definition of
every input field that shapes the command line: `argstr`, `position`, `sep`, `formatter`
(as a function reference, walked like any user function), `path_template`, `keep_extension`,
`copy_mode` and the field's type. Each is encoded through the existing `bytes_repr` of attrs
fields, as `_versioned_fields` already does for the `Outputs` class (`hash.py:602-609`,
`:1676-1690`), so input and output fields are treated the same way. `help` and other documentation
fields are left out. That needs a `hash_eq`-style metadata flag on them, because attrs `eq` cannot
be reused for this.

### 5.2 The executable

- **Keyed:** the executable as written (`"bet"`, `["fslmaths", "-dt", "float"]`), and the tool's
  version if it is known portably:
  - `shell.define(..., version="6.0.7")`, K1 extended to shell tasks;
  - or `shell.define(..., version_command=["bet", "-V"])`. Its stdout, stripped, is keyed. It runs
    once per process per resolved executable, inside the task's environment, and is cached in
    `PersistentCache` under the stat key of the executable's file (`hash.py:644-689`). Probe §8.5:
    `--version` costs 3 to 70 ms.
- **Recorded, not keyed** (a `tools` field in the context): the resolved path (`shutil.which`, then
  `realpath`), the sha256 of that file, and its owning package if one can be read cheaply (a
  Homebrew `Cellar/<name>/<version>` path, a conda `conda-meta` record, `dpkg -S` / `rpm -qf`
  behind a time limit). `ConsistentReuse` adds `tools` to the fields it compares
  (`engine/store.py:252`).
- **Why the binary digest is not keyed:** it differs between architectures for one release, so
  keying it would split every cross-platform cache. It also misses the shared libraries that do
  the work: the `ffmpeg` binary is 0.4 MB and its code is in `libav*` (probe §8.5). As context it
  is still exact on one platform.
- **Cost:** `which` plus `stat` take under 1 ms. A sha256 takes 0.3 ms for a 0.1 MB binary and
  34 ms for 40 MB. It is computed once per process per path, then served from `PersistentCache` in
  microseconds.
- **Not found on the submitting host** (it exists only in the container or on the compute node):
  the executable is keyed as written. `tools` is filled where the job runs, by the environment
  (§5.3), as `environment` is today.

### 5.3 Environments

The environment a shell task runs in is part of the tool, so for shell tasks its portable identity
is keyed. Python tasks keep G1 as it is: their code is keyed by §4, and the environment is context.

- **Container index digest.** The keyed value is the digest the tag resolves to in its registry:
  for a multi-platform image this is the image index digest, which is the same on every
  architecture. Resolution order:
  1. `image@sha256:…` given, or `identity=` given (`environments/base.py:128-137`): use it, with no
     lookup.
  2. Local: `docker image inspect --format '{{json .RepoDigests}}'` gives the digest the image was
     pulled by. Probe §8.6: about 30 ms, no network, and for `busybox:latest` it equals the
     registry index digest. The local `Id` used today is the per-platform config digest. It is
     recorded and not keyed.
  3. Registry, once per tag per `Submitter` run: an anonymous token request, then `HEAD
     /v2/<repo>/manifests/<tag>` with the index media types in `Accept`, reading
     `Docker-Content-Digest`. Probe §8.6: 60–90 ms for the token and 100–120 ms for the `HEAD` on
     Docker Hub. Never a pull.
  4. Singularity/Apptainer: a `.sif` path is hashed by content through `PersistentCache` (a full
     read the first time, at about 0.5 ms/MB, then a stat). A `docker://` URI goes through step 3.
     A `library://` or `shub://` URI is unresolved.
- **When the local digest and the registry disagree** (the tag moved after the pull; probe §8.6
  found one), the key uses the local digest, because that image is the one that will run. A
  warning names both digests.
- **Offline, or a tag that cannot be resolved:** the key carries `image:tag` and
  `unresolved`, and the context carries `unknown` (`engine/context.py:520-521`). The result is used
  within the process and never served to another. `Submitter(resolve_images="require")` raises
  instead, for runs that must be reproducible.
- **Lmod.** The key carries the resolved full module names (`fsl/6.0.7.9`, not `fsl`). They are
  read from `LOADEDMODULES` in the output of the `lmod python load` that `execute` already runs
  (`environments/lmod.py:32-41`), once per process per module list. Today the identity is the list
  as written (`environments/lmod.py:23`). `MODULEPATH` is recorded. Sites that name modules
  differently get different keys. `version=` on the task overrides this when an author knows the
  sites are equivalent.
- **Native** contributes nothing to the key beyond §5.2.
- **Venv (G2, not yet landed).** A Python task in a `Venv` must resolve the distribution versions of
  §4.1 inside the venv, not in the submitting interpreter. The lock digest is recorded. G2 has to
  carry this; it is noted here so the two proposals agree.

## 6. Inputs, overrides and context

### 6.1 Overrides

- `version=` (K1) replaces the `ast@1` code of user definitions with `qualname==version`, as today
  (`hash.py:1495-1501`, `compose/python.py:223-234`). Library references keep their distribution
  versions, so a declared task still reruns when numpy changes. The recorded code digest and
  `DeclaredVersionWarning` stay as they are (`engine/store.py:277-337`). `version=` is added to
  `shell.define` and `workflow.define`.
- `checksum=` (K2) replaces the whole record (`compose/base/builder.py:152-153`, `task.py:482-508`).
  pydra validates only its form. This is how a project with its own identity discipline drives
  pydra.
- `PYDRA_USER_PACKAGES` (`hash.py:96-99`, `:976-997`) moves a package between library and user
  code.

### 6.2 Canonical typed encoding of inputs

Each input is one digest string `<tag>@<n>:<hex>`, from a registry that extends `bytes_repr`
(`register_digest(type, fn)`):

| Value | Digest |
|---|---|
| `None`, `bool`, `int`, `str`, finite `float` | `json@1:` sha256 of its JCS; a non-finite float as RFC 0003's `{"$float": …}` (0003 §5.3) |
| `list`, `tuple`, `dict` with `str` keys of the above | `json@1:` over the JCS of the digests, with the container type tagged |
| `numpy.ndarray` (not object dtype), numpy scalars | `npy@1:` sha256 over `dtype.str`, `shape` and the C-ordered bytes, read through a `memoryview` with no copy. Object arrays are digested element by element, as now (`hash.py:1794-1795`) |
| `FileSet` | `file@1:` sha256 over the sorted (relative name, content sha256) pairs. Probe §8.2: content-only hashing is already independent of directory and file name |
| `os.PathLike`, and `str` fields typed `Path` | an existing path: its content, as `file@1:`. A missing path, such as an output location: `path@1:` of the string |
| a user-class instance | `py@1:` from `bytes_repr`, with its class's code (§4.2) |
| anything else | `py@1:` from today's `bytes_repr` |

- **Shared with RFC 0003.** `npy@1:<sha256>` is the name of the entry's array member
  (`<sha256>.npy`) and `file@1:<sha256>` is the `files/<sha256>/` member (0003 §5.1). An upstream
  output that a store has already digested is then a downstream input whose digest is read from
  the entry, not recomputed. Both proposals use one implementation of the digests in
  `pydra/utils/digest.py`. RFC 0003 P3 depends on it.
- No value is normalised. NaN payloads and `-0.0` give misses, which are safe; senselab-style
  normalisation is left to a registered digest.
- The persistent stat-to-digest cache (`hash.py:148-255`) is used for `file@1` and `path@1` alike,
  keyed by `(path, mtime, size, inode, ctime)` with the 2 s settle window (`hash.py:50`,
  `:658-689`). A hit should also populate the in-memory dict, which it does not today
  (`hash.py:222-224`).

### 6.3 Context and reuse

Nothing in §3.2's right column enters the key. `job_context` gains `tools` (§5.2) and the
per-platform image digests (§5.3), and loses nothing. Whether an entry recorded elsewhere is served
remains the reuse policy's decision: `ReusePolicy` today, and RFC 0003's `Reuse` verdict later.
Because library versions are now keyed, `AnyReuse` no longer serves a result across a library
upgrade. Across platforms and Python minors it serves exactly the entries with the same code,
versions, tools and input content.

## 7. Cost

Per process, once:
- `importlib.metadata.packages_distributions()`: 78–213 ms (§8.5, §8.7). `job_context` already pays
  it (`engine/context.py:317-322`), so the walk adds one memoised dict lookup per reached module.
- An owner lookup from a module's file through the cached `RECORD` files: 0.25 ms uncached, then
  memoised.
- Images: 30 ms local, about 200 ms for a registry round trip, per tag per run. Lmod: one `lmod`
  call per module list, which `execute` already makes.

Per task definition, memoised per process: the walk (`hash.py:1601-1642`), unchanged. A checksum of
a trivial Python task on a fresh instance takes 0.21 ms (§8.7). The record adds one JCS
serialisation and one sha256 of a few hundred bytes.

Per item:
- arrays: 0.42 ms/MB with dtype and shape and no copy, against 1.1 ms/MB today, which copies
  through `tobytes` (§8.7);
- files and paths: 0.5 ms/MB on the first read, 2 µs on a `PersistentCache` hit;
- scalars: unchanged in order.

Nothing grows faster than linearly in the number of items. The extra work is per process (owners,
tags), per tool (one digest) or per byte of input. Bytes are already read once for `FileSet`
today, and are now read once for `Path` inputs too.

## 8. Feasibility

The probes are kept outside the repository, in `probes-identity/` of the job scratch directory, and
are not committed. Interpreters: CPython 3.11.15, 3.12.11, 3.13.11, 3.14.3 and 3.15.2 on macOS
arm64. pydra at `3f734f9a`.

1. **`ast@1` across Python minors.** `encode_ast` (`hash.py:763-875`), copied unchanged, was run
   over every function and lambda of `pydra/` (3,185) under all five interpreters. All five
   agreed on every digest. Over numpy 2.5.4's source (11,801 functions under 3.11), 3.13 and 3.15
   agreed with 3.11 on all of them. The 8 files that differed fail to parse under 3.11 (PEP 695
   `type` statements), so they are not encoding differences.
2. **Present false hits.** int64 against float64 zeros, equal hashes. (2,3), (3,2) and (6,) over one
   buffer, equal hashes. A `Path` input whose content changed, same checksum. `ls -l <dir>` against
   `ls -a <dir>`, same checksum. `argstr="--a"` against `"--b"`, same checksum. `File` inputs with
   one content in different directories or under different names, equal hashes (content only, as
   intended).
3. **Call-graph tracing** (11 single-edit variants of a user package, each run in a fresh
   process). Seen: a direct helper, a transitive helper, a module constant, an `lru_cache`d
   helper, a helper with a user decorator, a lambda, a factory closure, and a method of a class
   built in the body. Not seen: a method of an input instance, `getattr(module, name)`,
   `importlib.import_module`. With the gap 1 prototype (§4.2), the input-instance edit is seen and
   an edit to an unrelated class is not. The base checksums were equal under 3.11, 3.12 and 3.13,
   and with the package copied to another directory.
4. **Library versions in the key.** One task calls `np.median` and `numpy.mean`. Unpatched, its
   checksum is the same under numpy 2.4.6 (3.11 and 3.13) and 2.5.4 (3.12). With `_CodeWalk.library`
   patched to append `@dist==version`, the checksum is equal at 2.4.6 under 3.11 and 3.13, and
   differs at 2.5.4.
5. **Owners and executables.** `numpy.mean`, `numpy.add` (ufunc), `numpy.linalg.norm` and
   `attrs.define` resolve to `numpy==2.5.4` and `attrs==26.1.0` from the module file. `math.sqrt`,
   `json.dumps`, `len` and `str.upper` resolve to the standard library. A `PYTHONPATH` checkout and
   a local package have no owner, which is the user-code path. Building a full file-to-distribution
   index over 65 distributions (4,957 files) took 1.08 s, which is too slow per process.
   `packages_distributions` (0.08–0.21 s), plus a `RECORD` check only for the candidates, is the
   way. Executables: `which` 0.05–1.1 ms, sha256 0.3–34 ms (0.1–40 MB), `--version` 3–70 ms. `ls`
   has no `--version`, so the version command has to be declared.
6. **Images.** Docker Hub anonymous resolution: `alpine:latest`, `alpine:3.20`, `busybox:latest`
   and `python:3.13-slim` returned OCI index digests, and `nipreps/fmriprep:latest` a single-platform
   manifest. Each took 61–87 ms for the token and 102–119 ms for the `HEAD`. Locally, `busybox:latest`
   has `RepoDigests` equal to the registry index digest (`a46257e0…`) and an `Id` that differs from
   it (`d2482869…`). The local `python:3.13-slim` digest (`09a17cad…`) differs from the registry's
   current one (`e48f59cd…`), so the tag has moved since the pull. Today's local-`Id` identity
   would report the stale image as current.
7. **Cost** (a loaded machine, load average 6.5): array digests 0.56 ms per 1 MB and 42 ms per
   100 MB with dtype, shape and no copy, against 1.15 ms and 111 ms today. File sha256 at 52 ms per
   100 MB from the page cache. A stat-keyed hit takes 2 µs. A trivial task checksum takes 0.21 ms.

Not checked: Linux or Windows (the encoding has no platform input, but the classification of §4.1
depends on the install layout), Singularity and Apptainer (not installed), private registries
with credentials, and Lmod.

## 9. Migration

- **One recompute.** Every checksum changes: the record format, the shell template, the typed
  digests and the library versions all enter it. Old entries are never found under the new
  checksums, and nothing is misread, because the addresses differ. `PickleStore` cleans them by
  age, as today. A `PortableStore` root records `"identity": "pydra-identity@1"` in its header
  (0003 §3.2). A root with another identity scheme raises `StoreLayoutMismatch`, because its
  checksums name other jobs.
- **Announced.** The release notes say so. The first `Submitter` run on a cache root writes
  `<root>/.pydra-identity` with the scheme name. A run that finds the marker missing while the root
  has entries, or finds it naming an older scheme, logs once: "this cache was written under an
  earlier identity scheme; its results will be recomputed once". That costs one `stat` per run.
- **Versioned after that.** `IDENTITY_SCHEME = "pydra-identity@1"` is the first field of every
  record. Any change to what is keyed or how a digest is computed bumps it; a new digest type adds
  a tag (`npy@2`) and leaves the old ones as they are. `ast@1` stays a separate version inside it
  (`hash.py:92`), so an AST encoding change is visible in the record.
- **Projects with `checksum=`** are unaffected: their checksums are theirs.
- **`pydracli check-versions` tables** (`pydra/scripts/cli.py:75-128`) store `code_digest`s. Those
  digests cover user code only (`hash.py:1705-1721`), so they do not change.

## 10. Delivery

The checksum-changing PRs (I1–I6) merge into a feature branch and ship together, so caches are
recomputed once, not six times.

| PR | Contents | Changes checksums | Depends on |
|---|---|---|---|
| I0 | `pydra/utils/digest.py`: typed digests (`json@1`, `npy@1`, `file@1`, `path@1`, `py@1`), `register_digest`, JCS. Unused. Tests: dtype, shape, byte order, non-contiguous arrays, relocation | no | — |
| I1 | identity record and `IDENTITY_SCHEME`; checksum = sha256 of its JCS; record kept with the result; `pydracli explain` | yes | I0 |
| I2 | inputs through I0; `Path` by content; stat cache for paths; persistent hits populate memory | yes | I1 |
| I3 | library references as `dist==version[@commit]`; one owner lookup shared with `context.py`; unowned modules; documentation of `PYDRA_USER_PACKAGES` across sites | yes | I1 |
| I4 | user-class instance inputs with their class's code; the §8.3 cases as regression tests; close #777 | yes | I1 |
| I5 | shell template in the record; `version=` and `version_command=` for shell; `version=` for workflows | yes | I1 |
| I6 | container index digests (given, `RepoDigests`, registry `HEAD`, SIF content), `resolve_images=`, unresolved rule; Lmod resolved names | yes | I5, G1 |
| I7 | `tools` context field; `ConsistentReuse` compares it; owning-package lookups behind a time limit | no (context) | I5 |
| I8 | `.pydra-identity` marker and the announcement; `PortableStore` header key (with RFC 0003 P2) | no | I1 |
| I9 | `pydracli check-reach` (runtime reach check under `sys.setprofile`) | no | I4 |

Each PR has tests on `debug` and `cf`. I3 and I6 have tests that run without network or Docker,
through fakes, plus one marked test each against a real registry and a real daemon.

## 11. Open questions for maintainers

- Should the identity record be the checksum's preimage, as proposed, or should the checksum stay
  as today's blake2b stream, with the record stored beside it for explanation only? The first is
  what RFC 0003 needs in order to recompute a checksum from an entry.
- Should the checksum be the full sha256 (64 hex), as RFC 0003 chose for fingerprints, or stay at
  128 bits (32 hex) to keep paths short?
- Should the local version segment (`torch 2.5.1+cu121` against `+cpu`) be keyed? This proposal
  records it, so a CPU and a GPU build of one release share a key and the policy decides.
- Should only directly reached distributions be keyed (proposed), or their requirement closure too?
- For a shell task, should the container index digest enter the key, as proposed? This reverses
  0001 §3.5 for shell tasks, and ADR-style arguments against environment keys ("almost no reuse")
  apply to Python tasks more than to tools whose image is their code.
- When a local image and its registry tag disagree, should the key use the local digest (proposed),
  or raise?
- Should `resolve_images` default to `"best-effort"` (proposed) or `"require"`?
- Is `version_command=` worth adding, or is `version=` enough for shell tasks?
- Should a `Path`-typed input that does not exist hash as its string (proposed), or raise?
- Should there be a one-release `PYDRA_IDENTITY=legacy` switch, or is one announced recompute
  acceptable during the 1.0 alphas?
