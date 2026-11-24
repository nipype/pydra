# Proposal 0004: portable automatic task identity

Status: proposed, for maintainer discussion; revision 4. Branch `rfc/portable-task-identity`, off
`enh/execution-extensions` at `3f734f9a`. Line references are to `3f734f9a` unless a branch is named.
Proposal 0001 names are used throughout: **K1** is a declared `version=` (0001 §3.11), **K2** a
supplied `checksum=` with the reuse hook (0001 §3.12), **G1** the environment checksum recorded and
never keyed (0001 §3.5). RFC 0003 is `rfc/portable-store` at `547f7b1d`; its
section numbers below are that revision's. The probes cited in §8 were run on one macOS arm64
machine under load. Timings are upper bounds. Revision 2 answers the review of #932, revision 3
its first re-review and revision 4 its second; §12 maps each blocking and important finding to the text that resolves it.

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
| A native executable is its name | `executable` is a `str` field (`compose/shell/builder.py:212-219`); `Native` has no fields, so its environment identity is `{}` (`environments/native.py:12-30`, `environments/base.py:53-69`). An upgraded tool changes neither key nor context |
| A container tag is not resolved | `Container._identity` returns `digest: None` when `_resolve_digest` cannot resolve one (`environments/base.py:142-156`); `Singularity` does not override it; `Docker` returns the local image id, or None (`environments/docker.py:15-33`). Probe §8.6: a local `python:3.13-slim` pulled earlier has digest `09a17cad…`, while the registry tag now points to `e48f59cd…` |
| Arrays are hashed without dtype or shape | class name, `size` and `tobytes()` (`hash.py:1790-1797`). Probe §8.2: int64 and float64 zeros of one size hash equal, as do shapes (2,3), (3,2) and (6,) of one buffer |
| A path is hashed as its string and its class's module | `bytes_repr_pathlike` (`hash.py:498-501`). The module is `pathlib` under 3.11 and `pathlib._local` under 3.13, so one `Path` hashes differently on the two. Only `FileSet` values are hashed by content (`hash.py:692-708`) |
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
  for the same code, the same library versions, the same tools, the same input values and content,
  and the same classification of installed code (§4.1, which the record shows);
- it changes when user code, a reached library's version, a shell command template, a tool's
  version or an image digest changes;
- inputs are identified by content where their type declares content (`FileSet`, arrays) and by
  value otherwise, through RFC 0003's output encoding (0003 §6.4), so an output a store has
  digested is the same digest as a downstream input (§6.2);
- what is keyed is what runs: an environment resolved for the key is executed in its resolved form,
  or the job fails (§5.3);
- what is keyed and what is only recorded follows one rule (§3.2);
- a checksum can be re-derived from a small, stored record, which is RFC 0003's key record (0003 §4);
- one recompute per cache when this lands, announced, and a versioned scheme after that.

**Non-goals:**
- Proving that a function is pure. Values read from mutable globals are hashed as they are at
  submission (`hash.py:1459-1461`), as now.
- Tracing dynamic dispatch (§4.2).
- Measuring how close two contexts are. That is the reuse policy's job (0003 §7).

**Defaults that do not change:** `code_hash="user"` (`hash.py:944-947`); `ast@1`; K1 and K2; the
context fields and the three reuse policies; `PersistentCache` and its location. **Checksums do
change**, once (§9). This reverses 0001 §2 ("checksums of tasks that use no environment, or
`Native`/`Lmod`" do not change), for the reasons in §1, and it reverses RFC 0003's "checksums are
unchanged" from the release that carries this scheme (§9.2).

## 3. The scheme

### 3.1 The identity record

A job's checksum is the digest of a small JSON document, its identity record:

```json
{
  "scheme": "pydra-identity@2",
  "kind": "python",
  "walk": {"code_hash": "user", "user_packages": [], "library_packages": []},
  "task": {"code": "code@1:9f2c…",
           "refs": ["lib:numpy==2.4.6:numpy.median", "…"],
           "values": ["dist:numpy==2.4.6"]},
  "outputs": "type@1:51ab…",
  "inputs": {"x": "json@1:3a7b…", "arr": "npy@1:c0de…", "img": "file@1:7d1e…"}
}
```

A shell job's record adds what it runs in, and a workflow's record what its nodes run in (§4.3):

```json
{
  "scheme": "pydra-identity@2",
  "kind": "shell",
  "walk": {"…": "…"},
  "task": {"template": "…", "version": "6.0.7"},
  "job": {"environment": "docker@1:sha256:e48f59cd…", "tool_version": null},
  "outputs": "type@1:…",
  "inputs": {"…": "…"}
}
```

- `checksum = f"{kind}-{sha256(JCS(record))}"`, where JCS is RFC 8785 canonical JSON, the encoding
  RFC 0003 uses for `pydra-context@1` (0003 §4). Numbers never reach JCS as JSON numbers (§6.2), so
  RFC 8785's double-precision number form does not apply.
- Every input is one typed digest string (§6.2). A value with no portable encoding falls back to
  today's `bytes_repr`, tagged `py@1:`, so a value that hashes today still hashes.
- `task` is the code part: §4 for Python tasks, §5.1-§5.2 for shell tasks, including a declared
  `version`. `job` is what the job runs in, which belongs to the job and not to the task: the
  resolved environment reference (§5.3) and the tool version a `version_command` reported
  (§5.2). Python jobs have no `job` field (G1). A workflow job has `nodes` instead (§4.3).
- **The job checksum is no longer `task._checksum` for shell and workflow tasks.** Today
  `Job.checksum` is the task's (`engine/job.py:236-246`) and the environment is the job's, the
  node's or the submitter's (`engine/job.py:133-135`, `engine/node.py:40`,
  `engine/submitter.py:319-321`). Under this proposal `Job.checksum` is the digest of the job's
  record, which is the task's record plus `job` (or `nodes`). `task._checksum` stays the digest of
  the task's record without them, and the task-level uses keep it: a task passed as an input value
  (`compose/base/task.py:706-715`) is a value and does not run, so its environment is not part of
  it; the element checksums of a split task (`task.py:531-550`) are supplied checksums (K2); the
  context's module collection (`engine/context.py:509-510`) and the hash-change message
  (`engine/lazy.py:195-208`) do not key anything. Cache directories, leases, the store and the
  success record all take `Job.checksum` already.
- `walk` is the classification
  that shaped it: the `code_hash` tier (`PYDRA_CODE_HASH`, `hash.py:944-947`) and the packages
  `PYDRA_USER_PACKAGES` moves (`hash.py:976-997`). Both already change today's checksum; in the
  record a mismatch between sites is visible in `pydracli explain` instead of hidden in a digest.
- A supplied `checksum=` (K2) replaces the whole record by `{"scheme": "pydra-identity@2",
  "supplied": "<checksum>"}`, and the checksum is the supplied string, as today (`task.py:482-508`).
- **The record is RFC 0003's key record.** This proposal adds a `StoreIdentity` (0003 §4) named
  `pydra-identity@2`. Its `key(job)` returns the record carried with the job, which is computed
  once, in the submitting process, with the checksum; a worker never recomputes it (#936). Upstream
  #961 instead makes the function hash itself stable under by-value pickling, and both mechanisms
  are kept. The carried record means a worker never walks code. The carried, code-stamped encoding
  (§4.2, "pickled by value") covers the processes that do hash a function they received by value,
  such as one that submits a task definition shipped to it. `main`'s `(module, qualname, mtime)`
  cache key is not adopted: it gives same-line lambdas, reassigned qualnames and `functools.wraps`
  wrappers the hash of another function. The cache stays keyed by the code object. Its `check_key(checksum, key)` is true when
  `key["scheme"]` is `pydra-identity@2` and the checksum equals the record's base checksum, followed
  by the absence suffix when the record has one. The base checksum is `key["supplied"]` for a
  supplied record and the record's digest as above otherwise. Its context and fingerprint are
  0003's defaults. A `PortableStore` root created under it writes the header key that 0003 §3.2
  defines:

  ```json
  "identity": {"name": "pydra-identity@2", "fingerprint": "pydra-context@1", "recorded_only": []}
  ```

  `entry.json` then holds the record (0003 §6.1) and every read checks it (0003 §6.2), and
  `pydracli explain <job>` shows the fields where two records differ.
- **The absence suffix of RFC 0002 §2.3(a).** For a task whose checksum cannot see an accepted or
  dropped `Absent` (a supplied checksum, or a field the checksum skips such as `container_path`),
  RFC 0002's runner appends `-absent-<hex>` to `job.checksum`, where `<hex>` is the digest of the
  sorted `(field, element index, cause)` triples of those `Absent`s, and keeps the unsuffixed
  checksum beside it. Without a rule for it, `check_key` would reject every such entry and 0003
  §6.2 would quarantine it on every read. So `key(job)` is built from the unsuffixed record and,
  when the job carries a suffix, adds the triples as `"absent": [[field, index, cause], …]`, sorted
  as RFC 0002 sorts them. The element index is the typed number form `{"$int": "<decimal>"}`, like
  every other integer in a record, so the record still holds no JSON numbers (§6.2, §9.2 item 9),
  and RFC 0002's suffix digest is computed over the same encoding of the triples. `check_key` recomputes the suffix from `key["absent"]`, with the one
  function the runner uses to make it, and compares the whole checksum. It never parses the suffix
  out of the checksum string. A record with no `"absent"` has no suffix, so a checksum carrying one
  under such a record is misfiled. The triples are what the suffix stands for, so they also show in
  `pydracli explain`.
- **Why `@2`.** RFC 0003 already names today's checksum, with no key record, `pydra-identity@1`
  (0003 §4). Reusing the name would make a root created under it open without error under this
  scheme and serve nothing, silently; a new name makes 0003's header check raise
  `StoreLayoutMismatch` naming `identity` (0003 §3.2).

### 3.2 What is keyed and what is recorded

One rule decides: **a fact enters the key if it names what is computed and is the same on every
platform that can run it. A fact that depends on the machine is recorded in the context and judged
by the reuse policy** (G1, 0003 §7). A machine-dependent fact that decides what runs (an image
digest, a module version) is keyed only in the form that is then executed (§5.3).

| Keyed (portable) | Recorded (`job_context`, `engine/context.py:489-522`) |
|---|---|
| user code, `ast@1`, and the classification that made it user code | Python full version, implementation, platform, C library |
| a reached library's distribution and version as installed from an index; the identity of a VCS, archive or local-directory install (§4.1) | the local version label of an index install (`+cu121`), the transitive requirements of reached distributions |
| the distributions owning the library types of input values | — |
| the shell command template | the resolved executable path and the sha256 of its file |
| a declared tool version, or the declared part of a version command's output | the owning OS or conda package of the executable |
| the digest of the image a container is run by | the local image id and the per-platform manifest digest |
| the full Lmod module names the environment loads | the Lmod tree, `MODULEPATH`, modules loaded as dependencies |
| input values and declared content, with dtype and shape | hardware, CUDA, environment variables |

## 4. Python tasks

### 4.1 Functions from installed libraries

Where the walk reaches library code today it writes `lib:{module.qualname}` (`hash.py:1385-1393`),
`module:{name}` (`hash.py:1366-1369`), or the reference of a builtin or descriptor
(`hash.py:1407-1431`). Under this proposal each such reference also carries its owner:

```
lib:{dist}=={identity}:{module.qualname}
```

- **The owner** is the distribution whose `RECORD` lists the module's file, which is
  `context._owners` (`engine/context.py:426-444`) moved to `pydra/utils/hash.py` so the walk and
  the context use one lookup. When no `RECORD` lists the file (Debian `dist-packages`, some conda
  packages), the owner is the one distribution that `packages_distributions()` names for the
  module's top-level package (`engine/context.py:318-323`), if exactly one does. The context uses
  the same lookup, so it reports a version for such a module too. It is memoised per module.
- **The identity** is the one `_identify` already computes for the context
  (`engine/context.py:246-288`), with one change for index installs:
  - installed from an index (no `direct_url.json`): the PEP 440 public version. The local segment
    (`2.5.1+cu121`) is recorded, not keyed: from an index it names a build for a platform;
  - from a VCS: `version@{commit_id}` (`:261-262`), which is portable;
  - from an archive: `version@{archive hash}` (`:263-270`). A compiled wheel has one hash per
    platform, so the key splits across platforms; that is a safe miss, and an index install is
    the portable form;
  - from a local directory, not editable (`pip install /src` in a Dockerfile): `dir:<digest>` over
    every hashed `RECORD` row outside `*.dist-info/` whose file is inside the install root, with no
    version prefix (today's context form is `version@record:…`, `:285-288`). The version string of
    such a package does not follow its code, and setuptools-scm or hatch-vcs put the commit and the
    dirty flag into its local segment, so neither the public version nor the local segment
    identifies it; the installed files' hashes do. The metadata files under `*.dist-info/`
    (`RECORD` itself, `INSTALLER`, `direct_url.json`) differ between installs of the same code and
    are left out;
  - editable, or a local checkout keyed as a library (an editable install inside `site-packages`
    that `PYDRA_USER_PACKAGES` marks as library code with `-name`): always `dir:<digest>` over every
    file in its package directories, data files included, with no version prefix, since the version
    string of an editable install does not follow its code. It is never `git:<commit>`, so the same
    code has the same key whether or not git is available and whether or not the tree is clean.
    Package data counts because a changed data file must never give a stale hit; a stray note in a
    package directory costs only a safe miss. An editable library over 256 MiB or 10 000 files is
    identified as unknown (`?`).

  An identity `_identify` cannot compute (`:246-260`) is keyed as `?` and the context gets
  `unknown`, as today for the context (`engine/context.py:520-521`).
- **The standard library and builtins** keep `lib:{module.qualname}` with no version. The Python
  minor is already compared by `ConsistentReuse` (`engine/store.py:246-260`). Keying it would bring
  back the false misses across minors that `ast@1` removed.
- **pydra's engine** is referenced the same way: a reference into `pydra.compose`, `pydra.engine`,
  `pydra.environments`, `pydra.utils`, `pydra.workers` or `pydra.scripts` is
  `lib:{module.qualname}` with no identity, and pydra's types never enter `task.values`. The
  context keeps pydra's identity, and `IDENTITY_SCHEME` versions what the engine contributes to a
  checksum. Keying pydra's version would re-key every workflow and split task at every pydra
  release, and at every edit of an editable pydra checkout, because their keys reach engine code,
  while a plain task's key does not. Task packages under `pydra.tasks.*` are libraries like any
  other and are keyed by their owning distribution.
- **C extensions** (`numpy.add`, a ufunc whose `__module__` is `numpy`) are owned like any other
  module. The owner is found from the module's file (§8.5). No source is read.
- **Editable installs** live outside `site-packages`, so `_path_is_library` already classifies them
  as user code (`hash.py:961-973`). Their source is walked under `ast@1`, so the version string,
  which an editable install does not change, is never used. An editable install inside
  `site-packages` is found through `direct_url.json` (`engine/context.py:195-205`) and walked the
  same way.
- **Library modules no distribution owns** (a vendored module, a `PYTHONPATH` entry inside a
  library root, a package neither `RECORD` nor `packages_distributions` claims): a pure-Python
  module is walked as user code. An extension module gets `lib:?:{module.qualname}`, and the
  context gets `unknown`, so the result is not reused by another process. That is today's
  behaviour for an unknown package.
- **Library types of input values are keyed.** The walk sees only the library code a function
  names. `def f(x: np.ndarray): return float(x.mean())` names none: the root's annotations are not
  encoded (`hash.py:802-805`) and `x.mean` is not a global. The context already collects the
  modules of input value types (`_value_modules`, `engine/context.py:366-387`). The record's
  `task.values` lists the owners of those types, as `dist:{dist}=={identity}`, so a numpy or pandas
  upgrade re-keys a task that only calls methods on its inputs. The owner lookup is memoised per
  type.
- **Only directly reached distributions are keyed.** A task that calls `scipy.signal.resample`
  keys scipy. numpy, which scipy requires, stays in the context's `packages` closure
  (`engine/context.py:448-486`) and is compared by `ConsistentReuse`, not by `AnyReuse` (§6.3).
  Keying the closure would make one unrelated upgrade in a large environment invalidate every task.

**Classification has to agree between sites.** A package installed as a wheel on a cluster is
library code, keyed by name and version. The same package in an editable checkout on a laptop is
user code, keyed by its AST. The two checksums differ although the code is the same. A project
that wants one checksum for both sets `PYDRA_USER_PACKAGES=name` (`hash.py:976-997`) on both,
which walks the package as user code everywhere. The `walk` field of the record (§3.1) carries the
tier and the overrides, so `pydracli explain` names this cause when two sites miss each other's
entries, and the documentation has to say it prominently.

### 4.2 User-defined functions

This part exists. The proposal keeps it and fixes two gaps.

- **What is traced.** The walk starts at the task function. It reads the globals named by the
  function's bytecode, with the attribute chains read from them (`hash.py:1100-1131`). It follows a
  chain through user modules and stops at the first library module (`hash.py:1322-1336`). It reads
  closure cells and the defaults of called functions (`hash.py:1311-1341`), and the bases and
  members of user classes (`hash.py:1343-1347`, `:1157-1236`). Each definition it reaches is
  encoded once under `ast@1` (`hash.py:1134-1154`), so cycles terminate. It is memoised per process
  and checked against source mtimes and weak references (`hash.py:1474-1642`).
- **Why it is portable.** `ast@1` drops positions, `ctx`, docstrings, and fields at their defaults,
  merges f-string parts, and for the root definition also drops its name, annotations and
  decorators (`hash.py:790-875`, `:802-805`). Probe §8.1: 14,986 functions encode identically under
  Python 3.11, 3.12, 3.13, 3.14 and 3.15. Probe §8.3: one task has one checksum under 3.11, 3.12 and
  3.13, and from two directories. The globals a function reads come from its bytecode
  (`hash.py:1100-1131`), whose opcodes change between minors; that reading was checked on 3.11 to
  3.13 only, so I4 runs the §8.3 cases on every minor pydra supports (`pyproject.toml:33-37`). The
  walk uses live objects, not a static import graph, so `from x import y`, aliases and re-exports
  resolve as Python resolves them.
- **Gap 1: inputs that are user-class instances.** The digest of a user-class instance carries its
  class's **class digest**, `class@1`, which this proposal defines once and uses in every place a
  user class enters the key (gap 1 here, `type@1` in §5.1, `$class` and `py@1` in §6.2): the
  sha256 of what the walk produces when it reaches the class as a definition, which is
  `_class_code` (`hash.py:1194-1236`), the `ast@1` encoding of the whole `ClassDef` with
  `__init__`, `__call__`, `__getitem__` and every other method, followed by the encoded references
  the walk yields for the class (`class_references`, `hash.py:1343-1347`): its bases, walked as
  classes in turn, and the closures, globals and defaults its methods read, walked as functions, so
  a module-level helper a method calls is in it (`hash.py:1311-1341`). The walk already does this
  for a user-class value it reads from a global (`hash.py:1460-1463`); this applies it to inputs.
  An edit to a base class's method or to a helper therefore changes the digest, as an edit to the
  class's own body does. `bytes_repr_type`, which the first draft used,
  leaves out every name starting with `__` (`hash.py:614-616`), and its `__dict__` bytes differ
  between Python minors for an `Enum` (`_hashable_values_` and `_unhashable_values_map_` appeared
  in 3.13). Probe §8.8: with `_class_code`, editing `__call__` changes the checksum, editing an
  unrelated class does not, and a task taking a user `Enum` member has one checksum under 3.11 and
  3.13. The probe covers a class body only; I4's tests add a method inherited from a user base
  class, a method that calls a module-level user helper, and an edit to each.
  A class pickled by value keeps its `class@1` (T2):
  - **Members.** A method whose copy was unpickled by value (`<dynamic-code>`) is still one of
    the class's members, so the walk reads the same references.
  - **Carried encoding.** A class hashed from its source carries its `ast@1` encoding as
    `(class stamp, encoding)`, checked against the class it is now. The details are in the
    "classes pickled by value" row below.
- **Gap 2: library references carry versions** (§4.1).

**What falls back, and how the user overrides it:**

| Case | Behaviour | Override |
|---|---|---|
| `getattr(module, name)`, `importlib.import_module(...)`, dispatch through a registry or string | the target is not seen (probe §8.3) | pass the callable as an input; or `version=`; or `checksum=` |
| a method called on an input object | its class's code, after gap 1; for a library type, its distribution (§4.1) | — |
| lambdas | walked as functions; lambdas that share a line are told apart by instruction positions (`hash.py:902-931`) | — |
| closures, factory-made functions | the closure cells are walked (probe §8.3) | — |
| decorators | a user decorator's wrapper is walked through its closure; a library wrapper with `__wrapped__` is walked through it (`hash.py:1385-1393`, `:1453-1458`); decorators of the task function itself are not encoded (`hash.py:803-810`) | `version=` |
| module constants, mutable globals | hashed by their value at submission (`hash.py:1459-1464`) | pass them as inputs |
| notebooks | cells registered in `linecache` (IPython does) are walked under `ast@1` (`hash.py:1186-1191`) | — |
| no source (REPL, `exec` without `linecache`) | bytecode tagged with the Python minor (`hash.py:1147-1150`): correct, but a miss across minors | define it in a file; or `version=` |
| pickled by value (cloudpickle >= 3.1.3 writes `co_filename="<dynamic-code>"`; #936) | a worker never hashes it (§3.1). Elsewhere, for example a process that receives a task definition by value and submits it, the encoding is taken from one of two places. The first is the encoding carried on the function as `(code stamp, ast@1 bytes)`, set whenever it is hashed, which travels in its `__dict__` (upstream #961's idea). It is taken before any file for code the process did not itself encode from its file, so cloudpickle 3.1.2 (real, possibly edited, filename) and 3.1.3 agree. The stamp covers the bytecode, names, argument counts, flags, cell and free variables, first line and constants, but not the file, so a `functools.wraps` copy or a replaced `__code__` does not take another code's encoding. The second, for `<dynamic-code>` code or code naming a file that `os.stat` cannot read (missing, or inside a zip archive: a zipapp, egg, PEX or shiv, whose source the module's loader gives), is the file of the loaded module its globals name. The match is by name, first line and instruction positions as above, and it is accepted only if the code compiled from that file at the same qualified name and first line has the function's stamp, so a different `__main__` (an SGE batch script) or a file edited since gives no match. Otherwise it is bytecode tagged with the minor, when it was pickled before it was hashed and no source is reachable. Readable-source encodings do not change | hash it before pickling (`Job` pickling always does) |
| classes pickled by value (gap 1 extended; T2; #936 G2) | Under cloudpickle 3.1.3, unpickling an instance of a class pickled by value (any `__main__` class, or one in a `register_pickle_by_value` module) runs cloudpickle's class setstate. It replaces the methods of the **live** class in the same process with copies whose `co_filename` is `<dynamic-code>`. Such a copy is still a member of the class: with a source file, when the code compiled from that file at its qualified name and first line has its code stamp; without one, when it (by value or with a real file) was compiled in a class body of the class's qualified name or wraps (`functools.wraps`) a function that was, so generated and library functions (dataclass `__repr__`, `__replace__`) are members on neither side. So the class's references, its `class@1` and the checksums that read it do not move in the process (`test_portable_submitter.py::test_default_portable_root_round_trips_any_object`). A class hashed from its source carries its `ast@1` encoding as `(class stamp, encoding)` in its `__dict__`, which travels when it is pickled by value. The stamp covers the class name, the code stamps of its functions and of the functions they wrap, and its constant attributes, so a replaced method, a changed wrapped function or a changed constant does not take the encoding. As for functions, the carried encoding is taken first for a class the process did not encode from its own file, and a class it did encode follows that file. A class whose body functions do not match the code compiled from its module's file (another `__main__`, an edited file) is not encoded from that file. cloudpickle drops a local class's qualified name; it is recovered from the class's by-value functions. Otherwise the class falls back to its members' encodings (the function row), so two local classes with one qualified name but different bodies stay distinct. `is_library` classifies a by-value function by the `__file__` its globals keep. A by-value class its module does not hold is classified by its by-value body functions, else by the functions generated in its module (dataclass methods), else as user code if it carries an encoding. So library code pickled by value stays library code, and a field-only user class stays user code in a process run from a library's `__main__`.  Readable-source encodings do not change. A copy unpickled by value (a class its module does not hold, other than a local class in the process that defines it) is also refused when an attribute holding a constant differs, in value or type, from the literal the last top-level assignment in the file's definition gives, so an edited dataclass default is caught. A library class carries no encoding. Residual: a class with no function of its own in its body (a plain constant class, or a field-only dataclass or attrs class, whose methods are all generated), pickled before it is hashed and unpickled where its module's file holds another definition, is checked against that file only through its literal constants; any other edit (an annotation, a default that is not a literal, a `field(...)` call) gives a false hit. The `!=3.1.3` cloudpickle pin is dropped with T2 | hash the class before pickling (`Job` pickling always does) |
| `code_hash="function"` or `"module"` | the walk stops early, by the user's choice (`hash.py:1300-1303`) | — |

Dynamic dispatch cannot be traced statically, and tracing it at run time costs too much for every
job. The proposal adds an opt-in check, `pydracli check-reach <module:task> [inputs]`. It runs the
task once under `sys.setprofile` and lists the user functions that ran but that the walk did not
reach, which are the places a `getattr` hides a dependency. It is a development tool and never
runs during a workflow.

**#777.** Close it, citing FH1–FH4 and the probe §8.3 cases as regression tests, with gap 1 in the
same PR.

### 4.3 Workflows

A workflow's constructor is walked like a function today (the `code_hash` documented at
`compose/workflow.py:141-153`), and a workflow job with a stored result returns it without
expanding its nodes (`engine/job.py:487-495`). The constructor's code and the workflow's inputs do
not hold what §5 keys for a node: a node's environment is given when it is added, and its tag, its
tool's version or its Lmod default can move while the constructor and inputs stay the same. Keyed
by code and inputs alone, the workflow would serve its old outputs after the image under one of
its nodes was re-pushed.

**The workflow record carries its nodes' run facts.** When a workflow job is keyed, the workflow is
constructed, as running it does first, through `Workflow.construct` (`engine/workflow.py:76-130`),
which runs the constructor without running a node and is memoised by the non-lazy inputs, so a
miss reuses the constructed graph. The record gets `nodes`: for each node, by name, its `job`
fields (§3.1), resolved as a shell job resolves them (§5.2, §5.3); a Python node has none and is
left out. A node that is itself a workflow contributes its own `nodes` when all its inputs are
concrete at keying; otherwise it is `"deferred"` (below).
Node checksums are not in the record: they depend on upstream outputs that do not exist at keying,
and they are already a function of the constructor's code, the workflow's inputs and the run facts.

**A workflow whose nodes cannot be resolved at keying is never served whole.** Two kinds of node
are recorded as `"deferred"`:
- a nested workflow any of whose inputs is lazy at keying (fed by an upstream output). Its shape
  may depend on that value, and a lazy construction does not reliably fail when it does:
  `LazyInField.__eq__` returns `False` (`engine/lazy.py:107-112`), so `if mode == "fast"`,
  `isinstance`, `is None` and truthiness all build one branch without error, which may not be the
  branch that runs. pydra documents a nested workflow branching on an upstream output as its way to
  write a conditional (`docs/source/tutorial/6-workflow.ipynb:245-246`). The nested job is still
  keyed when it runs, with its inputs then concrete, and is served whole from its own entry;
- a nested workflow node with a splitter. Its inputs may all be concrete (`Inner(x=x).split(mode=
  modes)` over a list), but at run each state element is constructed from its own task
  (`engine/submitter.py:1027-1036`), so one construction of the node's task does not show which
  branch each element takes. Each element job is keyed when it runs and is served whole from its
  own entry. Contributing `nodes` per state element, each constructed from the element's task as
  the run does, is future work;
- a node whose environment or tool version is `unresolved` (§5.2, §5.3).

A workflow record holding `"deferred"` at any depth of `nodes` is still computed and its entry
still written, but a lookup of it is a miss by rule, so a concrete nested workflow with a deferred
child defers its parent too: the job is expanded and each node is looked up under its own key, so
nodes whose run facts did not change are served from their own entries. **Cost.** An outer
workflow with a nested workflow fed by an upstream output is never served whole, only node by node:
one expansion and one lookup per node, each of which may hit. Offline, with an image that is
neither local nor given by digest, every node under it is `unresolved`, so a finished workflow is
not readable whole either, only node by node, and its shell nodes are not served to another
process (§5.3 step 5). Keying costs one construction per workflow job, which a miss would pay
anyway. The construction now runs when the workflow job's checksum is first computed, which is
before the job is dispatched and before its work directory exists (the store and lease calls take
`Job.checksum`, `engine/job.py:266`, `:415`, `:487`, and `__getstate__` computes it,
`engine/job.py:192-194`). A constructor error is therefore raised from that call, in the
submitting process, and is not recorded as the job's error in its work directory. For a nested
workflow that first use is `read_succeeded` or the de-duplication loop (`engine/submitter.py:571`,
`:705`), outside the per-job error handling (`:737-744`); I5 computes the checksum of each job of a
round inside that handling, so a constructor error at keying is recorded as that job's error and
the parent's loop continues under its `on_error` mode, as a constructor error at run is today.

A future alternative needs no construction at keying: the workflow's entry records, for each shell
job that ran under it, the reference it requested and what that resolved to, and a lookup
re-resolves the requested references and misses on any difference.

`version=` becomes available for workflows (§6.1). It replaces the constructor's code, not
`nodes`.

## 5. Shell tasks

### 5.1 The command template

The record of a shell task gets `task.template`. It holds the executable, then the definition of
every input field that shapes the command line: `argstr`, `position`, `sep`, `formatter`
(as a function reference, walked like any user function), `path_template`, `keep_extension`,
`copy_mode` and the field's type as `type@1`. Each is encoded through the existing `bytes_repr` of
attrs fields, as `_versioned_fields` already does for the `Outputs` class (`hash.py:602-609`,
`:1676-1690`), so input and output fields are treated the same way. `help` and other documentation
fields are left out. That needs a `hash_eq`-style metadata flag on them, because attrs `eq` cannot
be reused for this.

**`type@1`** is the digest of a type, used for field types here and for the record's `outputs`:
- a standard-library or builtin type: its qualified name;
- a library type: `lib:{dist}=={identity}:{qualname}` (§4.1);
- a class with pydra or attrs fields, which includes every task's `Outputs` class and every class
  built by `define`: the encoding of its fields only, not its qualified name, so two equivalent
  tasks defined in different ways (a decorated function, a class, a `define` call) share a
  checksum. The fields are encoded by `_versioned_fields`
  (`hash.py:602-609`, `:1676-1690`), names, types (as `type@1`), `path_template`s and the other
  metadata of §5.1, as today's branch for such classes does. `_class_code` is not used for them: a
  class built by `define` has no source, and its fallback holds only member functions
  (`hash.py:1229-1236`);
- any other user class: its qualified name and its `class@1` (§4.2), so a user `Enum` return type has one digest under every minor. Today's `bytes_repr_type`
  hashes its `__dict__` and differs between 3.11 and 3.13;
- a union, written `X | Y`, `Union[X, Y]` or `Optional[X]`: one tag, `union`, over the sorted,
  de-duplicated digests of its flattened members, with `None` as `NoneType`. The origin is not
  used: `get_origin(int | None)` is `types.UnionType` under 3.11-3.13 and `typing.Union` under 3.14,
  while `Optional[int]` is `typing.Union` on all of them (§8.8), and `int | str == str | int`;
- any other generic alias (`list[int]`, `dict[str, X]`, `Callable[[int], str]`): its origin's
  digest and its arguments' digests in order, with a list of argument types (`Callable`'s first
  argument) as a list and `...` as the token `...` (`tuple[int, ...]`);
- `Literal[...]`: the tag `literal` over the sorted digest forms (§6.2) of its values, since
  `Literal["a", 1] == Literal[1, "a"]`;
- `Annotated[T, *meta]`: `T`'s digest; metadata is not hashed (as upstream #887; field types are
  stripped at definition);
- a `TypeVar` or `ParamSpec`: its name and the digests of its bound and constraints.

### 5.2 The executable

- **Keyed:** the executable as written (`"bet"`, `["fslmaths", "-dt", "float"]`), and the tool's
  version if it is known portably:
  - `shell.define(..., version="6.0.7")`, K1 extended to shell tasks;
  - or `shell.define(..., version_command=["ffmpeg", "-version"],
    version_pattern=r"ffmpeg version (\S+)")`. The key carries the first group of the pattern's
    first match in stdout, never the whole output: `ffmpeg -version` also prints its compiler and
    install prefix, `git --version` an `(Apple Git-155)` suffix and `bash --version` its host
    triple (review run), so the whole output gives one tool release a key per platform. A command
    whose output the pattern does not match raises `VersionCommandError` at checksum time.

  A declared `version` belongs to the task and is `task.version` in the record. A version command's
  result depends on where it runs, so it is `job.tool_version` (§3.1), and a workflow carries it in
  `nodes` (§4.3).
- **Where the version command runs.** Only under `Native`, and only in the submitting process, once
  per process per resolved executable path, memoised in memory. It is not cached in
  `PersistentCache`: a shim (pyenv, conda wrappers, FSL wrapper scripts, `/usr/bin/git` on macOS)
  keeps its stat while the tool behind it changes, so a stat-keyed cache would serve a stale
  version to later processes. Under `Docker`, `Singularity` or `Lmod` the command is not run at
  checksum time, because the environment's keyed reference already names the tool (§5.3), and
  running it there would start, and for an image not present pull, a container per checksum.
- **Bound to the worker.** `Native.execute` runs the version command again in the worker before
  the task, memoised per worker process, and fails the job with both values when its version
  differs from the key. A remote worker with another build of the tool therefore fails instead of
  filing its result under the submitting host's version.
- **Recorded, not keyed** (a `tools` field in the context): the resolved path (`shutil.which`, then
  `realpath`), the sha256 of that file, and its owning package if one can be read cheaply (a
  Homebrew `Cellar/<name>/<version>` path, a conda `conda-meta` record, `dpkg -S` / `rpm -qf`
  behind a time limit). `ConsistentReuse` adds `tools` to the fields it compares
  (`engine/store.py:252`).
- **`tools` is built where the rest of the context is:** once, in the submitting process, and
  carried with the job (`engine/job.py:249-262`). An executable not found there is recorded as
  `{"executable": "bet", "found": false}`, which is the same on every lookup, so a lookup and the
  stored entry agree. A worker's own facts belong in RFC 0003's execute-phase context (0003 §4),
  under a `PortableStore` only; the lookup context is never filled in the worker.
- **Not found on the submitting host** with a `version_command`: the key carries the command and
  `unresolved`, and the context gets `unknown`, so the result is used within the process and not
  served to another; a warning recommends `version=`. With `version=`, or with neither, nothing is
  looked up and the key does not depend on the submitting host.
- **Why the binary digest is not keyed:** it differs between architectures for one release, so
  keying it would split every cross-platform cache. It also misses the shared libraries that do
  the work: the `ffmpeg` binary is 0.4 MB and its code is in `libav*` (probe §8.5). As context it
  is still exact on one platform.
- **Cost:** `which` plus `stat` take under 1 ms. A sha256 takes 0.3 ms for a 0.1 MB binary and
  34 ms for 40 MB, once per process per path, then served from `PersistentCache`, which is sound
  for a file's own digest. A version command takes 3 to 70 ms (probe §8.5), once in the submitting
  process and once per worker process.

### 5.3 Environments

The environment a shell task runs in is part of the tool, so for shell tasks its identity is
keyed. Python tasks keep G1 as it is: their code is keyed by §4, and the environment is context.

The checksum and the context are built once in the submitting process (`engine/job.py:249-262`),
and the command runs in the worker: `docker run image:tag` (`environments/docker.py:36`),
`singularity exec image:tag` (`singularity.py:16`), `lmod python load <modules>` (`lmod.py:32-41`).
A key resolved in one place and an execution resolved in another can disagree: a worker that pulls
a moved tag, a node with other module defaults. **The rule: the environment resolves its reference
once, in the submitting process, carries the resolved reference with the job, and the worker
executes that reference, never the tag or the short name.** A worker that cannot run it fails the
job. The resolved reference is the record's `job.environment` (§3.1); in a workflow it is the
node's entry in `nodes` (§4.3), so a moved tag under a node re-keys the workflow too. `Docker.execute` and `Singularity.execute` therefore stop appending `:{tag}` to a reference
that is already resolved (`docker.py:36`, `singularity.py:16`, `base.py:133-137`); today
`image="repo@sha256:…"` would run `repo@sha256:…:latest` and `image="x.sif"` would run
`x.sif:latest`.

- **Docker**, in order:
  1. `image@sha256:…` given, or `identity=` given (`environments/base.py:128-137`): key the digest and run
     the reference, with no lookup. `identity=` is the author's assertion and is keyed as written.
  2. The tag is in the submitting host's image store, with `RepoDigests`: `docker image inspect
     --format '{{json .RepoDigests}}'` (probe §8.6: about 30 ms, no network). The entry whose
     repository equals the image's repository is used; that is the digest the image was pulled by,
     the index digest when it was pulled by a multi-platform tag (probe §8.6: `busybox:latest`), a
     platform manifest digest when it was pulled by one. Key `docker@1:sha256:<digest>` and run
     `repo@sha256:<digest>`.
  3. The tag is in the image store, and step 2 does not apply: no `RepoDigests` (a local build or a
     `docker load`), no entry for the image's repository (an image pulled under another name and
     re-tagged, `docker tag ghcr.io/org/tool:1.2 tool:latest`), or several entries for it (pulled by
     several digests, and `inspect` does not say which pull the tag now names). Key
     `docker-local@1:<Id>` and run `<Id>`: this is the image that runs on this host, and asking the
     registry for `library/tool:latest` would key, and run, another publisher's image. The registry is never asked for such an image: a local
     rebuild of `nipreps/fmriprep:latest` would otherwise be keyed by the published image while
     `docker run` executes the build. The key is per image store, so it reuses on that host, and
     on a cluster whose workers load the same image, as today's local-`Id` identity does
     (`docker.py:15-33`); a worker without the image fails.
  4. The tag is not local: the registry, once per tag per `Submitter` run: an anonymous token
     request, then `HEAD /v2/<repo>/manifests/<tag>` with the index media types in `Accept`,
     reading `Docker-Content-Digest`. Probe §8.6: 60–90 ms for the token and 100–120 ms for the
     `HEAD` on Docker Hub. Never a pull. Key `docker@1:sha256:<digest>` and run
     `repo@sha256:<digest>`; the worker pulls that digest, which does not move.
- **The digest is keyed, the repository is carried.** A digest names the image's bytes wherever it
  is pulled from, so `org/fsl@sha256:d` and the same image through a site mirror,
  `mirror.site/org/fsl@sha256:d`, share one key. The repository goes with the job, in the
  environment, as the place the worker pulls from, and in the context.
  5. Offline and not local: the key carries `image:tag` and `unresolved`, the context carries
     `unknown` (`engine/context.py:520-521`), and the job runs the tag. The result is used within
     the process and never served to another. `Submitter(resolve_images="require")` raises
     instead. An air-gapped cluster loads or mirrors its images, so they resolve at step 2 or 3.
- **A local digest that is behind the registry** (the tag moved after the pull; probe §8.6 found
  one) is not a conflict any more: step 2 keys and runs the local digest, which is what runs on
  this host, and a remote worker pulls that same digest. A warning names both digests when step 4
  has been asked for another reason.
- **Singularity/Apptainer.** A `docker://` URI resolves through steps 1, 4 and 5 and is executed as
  `docker://repo@sha256:<digest>`, which Singularity pulls and converts; that is the portable form,
  and its key is the Docker key of the same image. A `.sif` path is keyed by its content sha256
  through `PersistentCache` (a full read the first time, at about 0.5 ms/MB, then a stat), which is
  exact for that file but not portable: two sites that build a SIF from one `docker://` image get
  different bytes and different keys, so a site that wants to share results names the `docker://`
  digest. The worker checks the `.sif` digest through its own stat cache before `exec` and fails
  the job when it differs from the key. A `library://` or `shub://` URI is unresolved, as step 5.
- **Lmod.** The key carries the full module names (`fsl/6.0.7.9`, not `fsl`), and `Lmod.execute`
  loads those full names, so a worker whose tree has another default loads the keyed version or
  fails. They are resolved in the submitting process, once per process per module list: in a
  child process, `lmod python purge` with the calling shell's modules still in place (so Lmod
  unloads them and unwinds the `MODULEPATH` a loaded compiler added), then `lmod python load`
  of the requested names in the purged environment, reading `LOADEDMODULES` from the output;
  `Lmod.execute` purges and loads the same way (`environments/lmod.py:32-41`
  shows the output's form). An entry is kept when its name part (before the last `/`) equals a
  requested name or the requested name is a prefix of it; the dependencies a module pulls in and
  the modules of the calling shell are left out of the key and recorded, with `MODULEPATH`. A
  submitting process without Lmod (`MODULESHOME` unset, `lmod.py:52-54`) keys the names as written
  with `unresolved`, as Docker step 5. This is one extra `lmod` process per module list per
  submitting process; `execute`'s call in the worker is a separate one. No probe covers it (§8),
  so I6 measures it on a cluster with Lmod and tests the name mapping against a fake `lmod`. Sites
  that name modules differently get different keys. `version=` on the task overrides this when an
  author knows the sites are equivalent.
- **Native** contributes nothing to the key beyond §5.2.
- **Venv (G2, not yet landed).** A Python task in a `Venv` must resolve the distribution versions of
  §4.1 inside the venv, not in the submitting interpreter. The lock digest is recorded. G2 has to
  carry this; it is noted here so the two proposals agree.

## 6. Inputs, overrides and context

### 6.1 Overrides

- `version=` (K1) replaces the `ast@1` code of user definitions with `qualname==version`, as today
  (`hash.py:1495-1501`, `compose/python.py:223-234`). Library references keep their distribution
  identities, so a declared task still reruns when numpy changes. The recorded code digest and
  `DeclaredVersionWarning` stay as they are (`engine/store.py:277-337`). `version=` is added to
  `shell.define` and `workflow.define`.
- `checksum=` (K2) replaces the whole record (`compose/base/builder.py:152-153`, `task.py:482-508`).
  pydra validates only its form. This is how a project with its own identity discipline drives
  pydra.
- `PYDRA_USER_PACKAGES` (`hash.py:96-99`, `:976-997`) moves a package between library and user
  code, and is shown in the record's `walk`.

### 6.2 Canonical typed encoding of inputs

**One encoding for outputs and inputs.** An input's digest is computed from its *digest form*:
RFC 0003's output encoding (0003 §6.4) of the value, with two substitutions so it can be hashed
exactly:

- **Numbers are typed strings.** An `int` is `{"$int": "<decimal>"}` and a `float` is
  `{"$float": float.hex(x)}`; `bool` stays JSON `true`/`false`. JCS (RFC 8785) writes numbers as
  ECMAScript doubles, so `1` and `1.0` would be one number and an integer above 2^53 would not be
  exact. Today `int:` and `float:` differ (`hash.py:517-533`), and a shell task rendering
  `--scale {scale}` runs `--scale 1` and `--scale 1.0` differently. `float.hex` gives `nan`,
  `inf` and `-inf` for the non-finite values, which are the strings 0003's `{"$float": …}` uses.
- **Tuples are tagged** `{"$tuple": [...]}`. 0003 decodes a JSON array to the field's declared
  type; an input digest has no declared type to lean on, and today a list and a tuple differ.

Everything else is 0003's form as it stands: `{"$set": …}`, `{"$complex": …}`, `{"$bytes":
"<sha256>"}`, `{"$dict": …}`, `{"$datetime": …}`, `{"$path": "<str>"}`, `{"$array": "<sha256>"}`,
`{"$codec": "<tag>", "value": …}`, and numpy scalars in 0003's scalar form (0003 §6.4) with the
value as a typed string. One addition applies at any depth: **a value whose type is user code
carries its class.** A `$codec` value of a user class (a project's `attrs` class registered with
`register_codec`, which it must be to be a `PortableStore` output at all) is
`{"$codec": "<tag>", "value": …, "$class": "class@1:<hex>"}` in the digest form, with the class
digest of §4.2, so `list[Params]` re-keys when `Params.scale`, a base class's method or a helper it
calls is edited, as a bare `Params` does (gap 1, §4.2). The
output encoding has no `$class`; it is identity, not data.

A digest is then one string `<tag>@<n>:<hex>`, from `pydra/utils/digest.py`, which a registry
extends (`register_digest(type, fn)`). The first row that applies wins:

| Value | Digest |
|---|---|
| a value whose type has a registered digest function | the function's own string. Inside a container it appears as `{"$digest": "<that string>"}` (below) |
| `numpy.ndarray` exactly (not a subclass), whose dtype holds no references | `npy@1:` the sha256 of its canonical `.npy` bytes (below). Inside a container it appears as `{"$array": "<same sha256>"}` |
| `FileSet` (`File`, `Directory`, any `fileformats` type) with a registered tag | `file@1:` the sha256 of the JCS of `{"$fileset": "<registered type tag>", "files": [[<key>, <content sha256>], …]}`, sorted by key, where the key is the one `FileSet.byte_chunks` yields today (`hash.py:703-708`). It does not depend on the directory or the file name, as today (probe §8.2) |
| a value whose digest form exists: `None`, `bool`, numbers, `str`, `Path`, every type with a codec in the registry, and containers of these (a user class with a codec carries `$class`) | `json@1:` the sha256 of the JCS of its digest form |
| anything else: a user-class instance without a codec, a container holding one, array subclasses, arrays of object or `StringDType`, a `FileSet` subclass with no registered tag | `py@1:` from today's `bytes_repr`. Gap 1 is made in the generic serialiser (`hash.py:417-437`), so a user-class instance carries its `class@1` (§4.2) at any depth of a `py@1` value; object arrays are digested element by element, as now (`hash.py:1794-1795`); an unregistered `FileSet` keeps today's content hash (`hash.py:692-708`) |

- **Registered digests.** A digest registered with `register_digest(type, fn)` wins over every
  built-in row: registering one for `int` replaces `$int`, and one for `numpy.ndarray` replaces
  `npy@1`. That is the hook for a project that wants normalisation, such as rounding floats. At top
  level the input's digest is the string the function returns. Nested at any depth in a container,
  the value's digest form is `{"$digest": "<that string>"}`. A user dict key `$digest` is written
  `$$digest`, as 0003 writes every key that starts with `$` (0003 §6.4), so the form stays
  unambiguous. 0003 never emits `$digest` in an output encoding and refuses it on decode. The
  function's result must match `[a-z][a-z0-9_]*@[0-9]+:[0-9a-f]{16,}`, a scheme, a version and at
  least 16 hex digits, and its scheme must not be one of the built-in ones (`json`, `npy`, `file`,
  `py`, `class`, `code`, `type`); a result that does not match raises when the digest is computed.
  Codec tags may contain dots, but digest schemes may not.
- **A value with no digest raises.** A `str`, a `Path` or a dict key that is not well-formed
  Unicode (a lone surrogate) has no JCS form (RFC 8785 §3.2.2.2), and today's `bytes_repr` raises
  `UnicodeEncodeError` for it too. There is no `py@1` fallback: the digest raises, and the error
  names the input field (I2).
- **Reference-dtype arrays at any depth.** An array whose dtype holds references (object,
  `StringDType`, a structured dtype with an object field) is digested element by element wherever
  it appears: at top level, in a list, tuple or dict, in an `attrs` instance, as an array subclass.
  Today `bytes_repr_numpy` walks elements only when `dtype == "object"` (`hash.py:1792-1797`) and
  hashes the pointer bytes otherwise. So two nested `StringDType` arrays of different strings
  collide (a false hit), and two structured arrays with object fields of equal content differ (a
  false miss). The fix is in the generic serialiser, `bytes_repr_numpy` walking elements whenever
  the dtype holds references, so every depth has one definition. It changes today's checksums for
  those arrays, so it lands with I2 or as a separate bug fix, not in I0, and a pydra issue tracks
  today's collision.
- **The canonical `.npy`** is the byte form of 0003's `.npy` member: format version 1.0 as
  `numpy.lib.format.write_array(..., version=(1, 0), allow_pickle=False)` writes it, with the dtype
  converted to little-endian and the data in C order. Its header carries `descr`, so two
  structured dtypes that both report `dtype.str == "|V8"` differ, and an array loaded from a
  big-endian file has the digest of the same values in little-endian order (probe §8.8). A
  non-contiguous array is copied to C order first.
- **Arrays that keep today's encoding.** An array subclass keeps `py@1`, which includes its class
  (`hash.py:1793`): `np.ma.MaskedArray` (whose mask the canonical `.npy` would drop), `np.matrix`.
  An array whose elements are references (object dtype, numpy 2's `StringDType`) is digested
  element by element; for `StringDType` that fixes a false hit today, because `tobytes()` holds
  references and two arrays of different strings give equal bytes.
- **A `Path` is a value.** A `Path` or `os.PathLike` input is its string, `{"$path": "<str>"}`, as
  0003 encodes a `Path` output, never its content, and never tested for existence. pydra tells
  users to type output locations as `Path` (`engine/job.py:676-678`), and an output location is
  missing before the first run and present after it; hashing it by content when it exists would
  miss on every rerun, and, for a task that also has a `File` input, raise "Input field hashes have
  changed during the execution" after the run (`job.py:653-698`). A `Path` naming a file inside a
  container would key the submitting host's file of that name, and a `Path` naming a dataset root
  would read the dataset at every checksum. Content identity is declared by the field's type: a
  field typed `File` or `Directory` is hashed by content, as today. The class's module is not in
  the digest form, so one `Path` has one digest under 3.11 and 3.13. An absolute path is still a
  per-host string: a relative path, or one under a layout the sites share, is portable, and the
  documentation says so.
- **Shared with RFC 0003: arrays.** `npy@1:<hex>` is, by construction, the name of the entry's array
  member `<hex>.npy` (0003 §6.1), because the preimage is the member's bytes.
- **Shared with RFC 0003: file sets, with their keys kept.** RFC 0003 stores a `FileSet` output under
  one directory, `files/<set>/<path>`, with the set's files laid out as the producer had them: a
  `File` or a multi-file set (a NIfTI `.hdr`/`.img` pair) side by side under their own names, a
  `Directory` with its tree (0003 §6.1). Decoding therefore gives a value whose `byte_chunks` keys
  are the producer's (`"."` for one file, `".hdr"` and `".img"` for the pair, paths relative to the
  root for a directory), `entry.json` records those keys and each member's sha256, and `<set>` is
  the set's `file@1`. The read check requires `<set>` to be the `file@1` of the recorded type tag,
  keys and member digests, and the decoded keys to be the recorded ones (0003 §6.2). Two
  consequences for inputs:
  - the digest is read from the entry, not recomputed: the store that decodes a verified `$fileset`
    puts `<set>` into the in-memory digest cache under `fileset_cache_key` of the decoded paths
    (`hash.py:658-689`), which saves reading the files. The in-memory entry skips the 2 s settle
    window, because the members are sealed read-only and were just checked; the persistent cache
    keeps the window;
  - recomputing it agrees: a process without that cache entry, such as the worker that runs the
    downstream job and re-hashes its `FileSet` inputs after the run (`engine/job.py:653-698`,
    `compose/base/task.py:524-529`), computes `file@1` from the decoded files and gets `<set>`,
    because the keys and the contents are the producer's. Without the key-preserving layout a
    two-file or `Directory` input read from a store would give another `file@1` there and the job
    would fail with "Input field hashes have changed".

  A file set that is not read through a store is digested from its files, as any input.
- **One tag registry.** The `$fileset` type tags and `$codec` tags are one registry, defined in
  `pydra/utils/digest.py` by I0 (`register_codec(type, tag, encode, decode)` and the `FileSet`
  tags), and used by 0003's entry format; 0003 does not define a second one (§9.2, §10).
- No value is normalised. NaN payloads and `-0.0` give misses, which are safe;
  senselab-style normalisation is left to a registered digest.
- The persistent stat-to-digest cache (`hash.py:148-255`) is used for `file@1`, keyed by
  `(path, mtime, size, inode, ctime)` with the 2 s settle window (`hash.py:50`, `:658-689`). A hit
  should also populate the in-memory dict, which it does not today (`hash.py:222-224`).

### 6.3 Context and reuse

Nothing in §3.2's right column enters the key. `job_context` gains `tools` (§5.2) and the
per-platform image digests (§5.3), and loses nothing. Whether an entry recorded elsewhere is served
remains the reuse policy's decision: `ReusePolicy` today, and RFC 0003's `Reuse` verdict later
(0003 §7).

What `AnyReuse` serves changes, but not to "never across a library upgrade". After this proposal it
does not serve across an upgrade of a distribution the task names (§4.1) or whose types its inputs
carry (§4.1), across a tool version that is declared, or across an image digest. It still serves
across an upgrade of a distribution reached only transitively (numpy under a task that names only
scipy), and of one reached only through dynamic dispatch. `ConsistentReuse`, the default
(`engine/store.py:465`), compares the whole `packages` closure and refuses those. Across platforms
and Python minors `AnyReuse` serves exactly the entries with the same record.

## 7. Cost

Per process, once:
- `importlib.metadata.packages_distributions()`: 78–213 ms (§8.5, §8.7). `job_context` already pays
  it (`engine/context.py:317-322`), so the walk adds one memoised dict lookup per reached module.
- An owner lookup from a module's file through the cached `RECORD` files: 0.25 ms uncached, then
  memoised. The same per library type of an input value.
- Images: 30 ms local, about 200 ms for a registry round trip, per tag per run. A `.sif` digest is
  a read of the file the first time and a stat after it, in the submitting process and in each
  worker process.
- Lmod: one extra `lmod` purge-and-load per module list in the submitting process, not measured
  (§5.3). `execute`'s own call in the worker is unchanged.
- A version command: once in the submitting process and once per worker process (§5.2).

Per workflow job: one construction of its graph at keying (§4.3), memoised by its non-lazy inputs
and reused by the run on a miss, plus the environment resolutions of its shell nodes, which are
memoised per tag or module list as above.

Per task definition, memoised per process: the walk (`hash.py:1601-1642`), unchanged. A checksum of
a trivial Python task on a fresh instance takes 0.21 ms (§8.7). The record adds one JCS
serialisation and one sha256 of a few hundred bytes.

Per item:
- arrays: 0.42 ms/MB for a contiguous array with dtype and shape and no copy, against 1.1 ms/MB
  today, which copies through `tobytes` (§8.7); the canonical `.npy` streamed into sha256 took
  45 ms for 100 MB against 42 ms for the bare buffer (§8.8). A non-contiguous or big-endian array
  adds one copy;
- files: 0.5 ms/MB on the first read, 2 µs on a `PersistentCache` hit;
- `Path` values and scalars: unchanged in order.

Nothing grows faster than linearly in the number of items. The extra work is per process (owners,
tags), per tool (one digest) or per byte of declared content. A `Path` input is no longer read.

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
   `importlib.import_module`. The base checksums were equal under 3.11, 3.12 and 3.13, and with the
   package copied to another directory.
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
   has no `--version`, so the version command has to be declared. Only the first line of
   `--version` was looked at; the review found per-platform text in later lines (§5.2).
6. **Images.** Docker Hub anonymous resolution: `alpine:latest`, `alpine:3.20`, `busybox:latest`
   and `python:3.13-slim` returned OCI index digests, and `nipreps/fmriprep:latest` a single-platform
   manifest. Each took 61–87 ms for the token and 102–119 ms for the `HEAD`. Locally, `busybox:latest`
   has `RepoDigests` equal to the registry index digest (`a46257e0…`) and an `Id` that differs from
   it (`d2482869…`). The local `python:3.13-slim` digest (`09a17cad…`) differs from the registry's
   current one (`e48f59cd…`), so the tag has moved since the pull. A locally built image shows
   digest `<none>` in `docker images --digests`, which is the step 3 case of §5.3.
7. **Cost** (a loaded machine, load average 6.5, contiguous arrays only): array digests 0.56 ms per
   1 MB and 42 ms per 100 MB with dtype, shape and no copy, against 1.15 ms and 111 ms today. File
   sha256 at 52 ms per 100 MB from the page cache. A stat-keyed hit takes 2 µs. A trivial task
   checksum takes 0.21 ms.
8. **Revision 2 checks** (CPython 3.11.15 and 3.13.11, numpy 2.4.6, load average about 4).
   - Gap 1 through `_class_code`, as a serialiser patch over the generic one: editing `Proc.__call__`
     from `x * self.k` to `x + self.k` changes the checksum of a task taking a `Proc` instance;
     editing an unrelated class does not; a task taking a user `Enum` member has one checksum under
     3.11 and 3.13 (`python-ade1e0fd…` on both), and the `Proc` task one checksum on both.
   - Canonical `.npy`: two structured dtypes reporting `|V8` get different digests; `<f8` and `>f8`
     arrays of one set of values get one digest; dtype and each of three shapes give different
     digests; a strided view equals its contiguous copy; a `MaskedArray` and a `StringDType` array
     are routed to `py@1`. Streaming `write_array` into sha256 took 45 ms for 100 MB, against 42 ms
     for the sha256 of the buffer alone.
   - Unions (revision 3, under 3.11, 3.13 and 3.14 through `uv`): `get_origin(int | None)` is
     `types.UnionType` under 3.11 and 3.13 and `typing.Union` under 3.14; `get_origin(Optional[int])`
     is `typing.Union` under all three; `int | str == str | int` and `Literal["a", 1] ==
     Literal[1, "a"]` on all three. Hence the order-free `union` and `literal` tags of `type@1`.

Not checked: Linux or Windows (the encoding has no platform input, but the classification of §4.1
depends on the install layout, and the version-pattern and image steps were run on macOS only),
Python 3.14 and 3.15 for the walk's bytecode reading (§4.2), Singularity and Apptainer (not
installed), private registries with credentials, big-endian hosts, and Lmod.

## 9. Migration

### 9.1 In pydra

- **One recompute.** Every checksum changes: the record format, the shell template, the typed
  digests and the library identities all enter it. Old entries are never found under the new
  checksums, and nothing is misread, because the addresses differ. `PickleStore` cleans them by
  age, as today. A `PortableStore` root created under `pydra-identity@1` (0003 §4) raises
  `StoreLayoutMismatch` naming `identity` when opened under `pydra-identity@2`, because its
  checksums name other jobs; new roots default to `@2`. The old checksum code is not kept, so `@1`
  roots are not readable after this lands; their results are recomputed into a new root.
- **No migration of entries across schemes.** The record needs the task's code and input content
  as they are at submission. An old entry does not hold them, and re-deriving a record from a
  `PickleStore` slot's `_job.pklz` would key the old result under the code that is installed now,
  which is the false hit this proposal exists to remove.
- **Announced.** The release notes say so. The first `Submitter` run on a `PickleStore` cache root
  writes `<root>/.pydra-identity` with the scheme name. A run that finds the marker missing while
  the root has entries, or finds it naming an older scheme, logs once: "this cache was written
  under an earlier identity scheme; its results will be recomputed once". That costs one `stat` per
  run. A `PortableStore` root uses its header instead.
- **Versioned after that.** `IDENTITY_SCHEME = "pydra-identity@2"` is the first field of every
  record. Any change to what is keyed or how a digest is computed bumps it; a new digest type adds
  a tag (`npy@2`) and leaves the old ones as they are. `ast@1` stays a separate version inside it
  (`hash.py:92`), so an AST encoding change is visible in the record.
- **Projects with `checksum=`** keep their checksums, and a `PickleStore` cache of theirs is read
  as before. A `PortableStore` root of theirs created under `@1` is not: its header names `@1`, so
  it raises `StoreLayoutMismatch` under `@2`, and an `@1` entry's key record is `None` (0003 §4), so
  a supplied entry cannot be told from an automatic one and `@2` cannot read only the former. Such
  a root is re-created and its jobs run once. `PortableStore` and this scheme both ship in the 1.0
  alphas (§10), so the roots that exist under `@1` are the ones made between 0003 P4 and I1, and
  0003 says so (§9.2).
- **`pydracli check-versions` tables** (`pydra/scripts/cli.py:75-128`) store `code_digest`s. Those
  digests cover user code only (`hash.py:1705-1721`), so they do not change.

### 9.2 What RFC 0003 carries

RFC 0003 at `547f7b1d` carries what this proposal needs from it; none of it changes 0003's
protocols. By section:

1. **`$array` is exactly `numpy.ndarray` whose dtype holds no references** (0003 §6.4), the set §6.2
   gives `npy@1`. Subclasses and reference dtypes have no default codec there and are `py@1` here.
2. **numpy scalars** have the form `{"$npscalar": ["<dtype>", "<value>"]}`, the value as a typed
   string (0003 §6.4), which §6.2's digest form uses as it stands.
3. **One tag registry**, in `pydra/utils/digest.py` (I0), shared by `$fileset` and `$codec` tags;
   an unregistered `FileSet` subclass is `UnencodableOutput` as an output there and `py@1` as an
   input here (0003 §6.4, §13).
4. **`migrate()` and `legacy=` under `@2`** (0003 §8.1): `migrate()` migrates only slots asserted to
   have supplied checksums (`supplied_checksums=True`) and publishes them with the key
   `{"scheme": "pydra-identity@2", "supplied": <checksum>}`, which `check_key` accepts (§3.1); a
   `legacy=` root is useful only to jobs with a supplied checksum.
5. **Checksums change once** (0003 §2, §4, §8.1): checksums are unchanged only under
   `pydra-identity@1`; from the release that carries this proposal `@2` is the default, `@1`
   checksums can no longer be produced, and an `@1` root raises `StoreLayoutMismatch` under `@2`.
6. **Order of delivery** (0003 §13): P3 after I0, P7 after I1 and I8.
7. **The canonical `.npy`** (0003 §6.1): the array member is §6.2's canonical `.npy`, named by its
   sha256, which is `npy@1`.
8. **File sets keep their keys** (0003 §6.1, §6.2): `files/<set>/<path>`, laid out as the producer
   had them, `<set>` named by the set's `file@1`, the keys recorded, and both checked on read
   (§6.2 above).
9. **Numbers.** `pydra-identity@2`'s record holds no JSON numbers (§6.2), so 0003's JCS of the key
   record is exact. 0003's output encoding keeps JSON numbers for Python `int` and `float`; nothing
   here depends on it.

## 10. Delivery

The checksum-changing PRs (I1–I6) merge into a feature branch and ship together with I8, so caches
are recomputed once, not six times, and the default `PortableStore` identity switches to `@2` in
the same release that changes the checksums. Shipped without I8, the default would still be `@1`
(0003 §4) while every checksum was an `@2` value, and an `@1` root would open without error and
serve nothing, which is the failure the `@2` name exists to prevent.

Order, with RFC 0003's PRs: I0 depends on nothing. 0003 P1 → P2 → P3, with P3 after I0 because it
uses I0's digests and tag registry. I1 needs I0 and P2. There is no cycle: I0 holds the registry,
and P3 uses it rather than defining it.

| PR | Contents | Changes checksums | Depends on |
|---|---|---|---|
| I0 | `pydra/utils/digest.py`: the digest form, typed digests (`json@1`, `npy@1` as canonical `.npy`, `file@1`, `py@1`), `register_digest` with its precedence, nested `$digest` form and result grammar, JCS, and the tag registry (`register_codec`, `FileSet` type tags) that 0003 P3 uses. Unused. Tests: dtype and structured dtypes, shape, byte order, non-contiguous arrays, subclasses and reference dtypes routed to `py@1`, `1` against `1.0`, integers above 2^53, relocation | no | — |
| I1 | identity record and `IDENTITY_SCHEME = "pydra-identity@2"`; checksum = sha256 of its JCS; `walk` field; the `pydra-identity@2` `StoreIdentity` with `key` and `check_key`; record kept with the result; `pydracli explain` | yes | I0, RFC 0003 P2 |
| I2 | inputs through I0, with the field named in a digest error; `Path` as a value; reference-dtype arrays element by element at any depth; persistent hits populate memory; decoded `$fileset` values carry their recorded `file@1` | yes | I1, 0003 P3 for the decoded digests |
| I3 | library references as `dist==identity`; owners of input value types; one owner lookup shared with `context.py`, with the `packages_distributions` fallback; unowned modules; documentation of `PYDRA_USER_PACKAGES` across sites | yes | I1 |
| I4 | `class@1` (`_class_code` and the walk's class references), used for user-class instance inputs at any depth, and `$class` for user-class codec values; `type@1` (fields classes, unions, `Literal`, `Annotated`); the §8.3 and §8.8 cases, plus a base-class method and a method's helper edited, as regression tests on every supported minor; close #777 | yes | I1 |
| I5 | shell template in the record; `job` field and `Job.checksum` as the digest of the job's record; workflow `nodes` and the `"deferred"` rule; checksums of a round computed inside the per-job error handling; `version=`, `version_command=` and `version_pattern=` for shell, with the worker re-check under `Native`; `version=` for workflows | yes | I1, I4 |
| I6 | resolved references for Docker (given, `RepoDigests`, local `Id` for local builds and re-tagged images, registry `HEAD`; digest keyed, repository carried), Singularity (`docker://` digest, `.sif` content with the worker check) and Lmod (full names); `execute` runs the resolved reference without `:{tag}`; `resolve_images=`; unresolved rule | yes | I5, G1 |
| I7 | `tools` context field, built in the submitting process; `ConsistentReuse` compares it; owning-package lookups behind a time limit | no (context) | I5 |
| I8 | `.pydra-identity` marker and the announcement; `pydra-identity@2` as the `PortableStore` default. Ships with I1–I6 | no (switches the default identity) | I1 |
| I9 | `pydracli check-reach` (runtime reach check under `sys.setprofile`) | no | I4 |

Each PR has tests on `debug` and `cf`. I3 and I6 have tests that run without network, Docker or
Lmod, through fakes, plus one marked test each against a real registry and a real daemon. I6's
Lmod resolution is measured on a cluster that has Lmod before it is merged.

## 11. Open questions for maintainers

- Should the identity record be the checksum's preimage, as proposed, or should the checksum stay
  as today's blake2b stream, with the record stored beside it for explanation only? The first is
  what lets RFC 0003's `check_key` bind an entry to its record.
- Should the checksum be the full sha256 (64 hex), as RFC 0003 chose for fingerprints, or stay at
  128 bits (32 hex) to keep paths short?
- Should the local version segment of an index install (`torch 2.5.1+cu121` against `+cpu`) be
  keyed? This proposal records it, so a CPU and a GPU build of one release share a key and the
  policy decides.
- Should only directly reached distributions be keyed (proposed), or their requirement closure too?
- For a shell task, should the container reference enter the key, as proposed? This reverses
  0001 §3.5 for shell tasks, and ADR-style arguments against environment keys ("almost no reuse")
  apply to Python tasks more than to tools whose image is their code.
- Should `resolve_images` default to `"best-effort"` (proposed) or `"require"`?
- Is `version_command=` with `version_pattern=` worth adding, or is `version=` enough for shell
  tasks?
- An editable install left as user code (not keyed as a library) is identified by the code the
  task reaches. A change to a data file inside that package, read at run time, is not seen, so a
  stale result is served (measured: changing `factor.txt` from 2 to 3 reuses the result computed
  with 2). Options: digest the package files of every editable the task's code reaches, as for
  libraries (cost: safe misses on stray files, and a size cap), or document that data read at run
  time must be a task input. Default in this proposal: unchanged; the first is recommended.
- Should there be a one-release `PYDRA_IDENTITY=legacy` switch, or is one announced recompute
  acceptable during the 1.0 alphas?

## 12. Changes in revisions 2 to 4

Amendment after revision 4 (T2): classes pickled by value keep their `class@1` (§4.2 gap 1 and the "classes pickled by value" row), and the `cloudpickle !=3.1.3` pin is dropped.

Amendment after revision 4 (T4): `type@1` of `Annotated[T, *meta]` is `T`'s digest and the metadata is not hashed, as upstream #887, which this RFC had contradicted; task checksums do not change, since field types are stripped at definition (§5.1).

Revision 4 answers the second re-review of #932 at `e98634c6`, made jointly with RFC 0003 at
`547f7b1d`:

| Finding | How it is addressed |
|---|---|
| R2-B1 a nested constructor branching on a lazy input builds the wrong branch silently | A nested workflow with any lazy input at keying is `"deferred"`, whether or not lazy construction raises; its job is keyed and served whole when it runs. The cost (never served whole, node by node) is stated, and the re-resolve alternative is named as future work (§4.3) |
| R2-B1 cost notes | Offline with a non-local image, a finished workflow is readable only node by node; the construction runs when the workflow job's checksum is first computed, so a constructor error raises there and is not a recorded job error (§4.3) |
| R2-I1 `$class` leaves out the walk's references | One definition, `class@1`: `_class_code` plus the walk's class references (bases, the functions and globals methods reach), used for gap 1, `type@1`, `$class` and `py@1`; I4 tests a base-class method and a helper (§4.2, §5.1, §6.2, §10) |
| R2-I2 0004 describes an RFC 0003 that no longer exists | Re-pinned to `547f7b1d`; the file-set passage rewritten on 0003's key-preserving `files/<set>/<path>` layout: the digest is read from the entry, and recomputing it in a worker agrees (§6.2); §9.2 reworded to what 0003 carries, by section |

| RFC 0002 review, round 4, nit j: entries under an absence-suffixed checksum fail `check_key` | The record carries the `"absent"` triples, and `check_key` compares the checksum with the base checksum plus the recomputed suffix of RFC 0002 §2.3(a) (§3.1) |
| R3-I1 a split nested workflow builds one branch | A nested workflow node with a splitter is `"deferred"`; per-element `nodes` named as future work (§4.3). With it, three nits: `"deferred"` at any depth is a miss, the per-job error isolation of constructor errors at keying (§4.3, §10 I5), the status line |

Nits of rounds 1 and 2 are not addressed in this revision.

Revision 3 answers the first re-review of #932 at `871c5d96`:

| Finding | How it is addressed |
|---|---|
| R1-B1 a workflow's key holds none of what §5 keys | Design change: the record names `task.version` and `job` (`environment`, `tool_version`); `Job.checksum` is the digest of the job's record and `task._checksum` keeps the environment-free task record, with the rule for each task-level use (§3.1). A workflow record carries `nodes`, the run facts of every node of the constructed graph, nested workflows constructed with lazy inputs; a node that cannot be resolved is `"deferred"` and makes the workflow a miss by rule, so its nodes are looked up under their own keys (§4.3, §5.2, §5.3) |
| R1-I1 (B1 residual) `@2` default outside the group; `checksum=` roots | I8 ships with I1–I6, and why (§10); `checksum=` projects' `@1` `PortableStore` roots are re-created, and why `@2` cannot read only their supplied entries (§9.1) |
| R1-I2 (B2 residual) `file@1` not recomputable from a decoded multi-file value | The writer records `file@1` per `$fileset`; a store that decodes a verified `$fileset` primes the in-memory digest cache with it, so the downstream input gets the producer's digest (§6.2, §9.2 item 8) |
| R1-I3 (I4 residual) `type@1` unions and `Outputs` classes | Fields classes (every `Outputs`, every `define` class) by `_versioned_fields`; unions under one order-free tag; rules for `Literal`, `Annotated`, `Callable`, `...`, type variables (§5.1); probe §8.8 |
| R1-I4 user-class instance inside a container with a codec | Precedence stated ("the first row that applies wins"); a user-class `$codec` value carries `$class` in the digest form; gap 1 lives in the generic serialiser, so it holds at any depth of `py@1` (§6.2) |
| R1-I5 (B4, I8 residual) Docker re-tagged images, step reference, registry name in the key | Step 3 now covers no `RepoDigests`, no entry for the repository and several entries, all keyed and run by `Id`; the registry is asked only when the tag is not local; the key is `docker@1:sha256:<digest>` and the repository is carried, so mirrors share keys (§5.3) |
| R1-I6 §9.2 incomplete; I0 ↔ 0003 P3 cycle | The tag registry moves into I0 and P3 uses it; the order is stated (I0 independent; P1 → P2 → P3 after I0; I1 after I0 and P2) (§10); §9.2 lists the four changes RFC 0003's next revision carries, by section, and the five further items (§9.2) |

Nits of round 1 are not addressed in this revision.

Revision 2 answers the review of #932 at `6a4d8f84`. Blocking and important findings:

| Finding | How it is addressed |
|---|---|
| B1 `pydra-identity@1` collides with RFC 0003's name; stale 0003 citations | Renamed `pydra-identity@2`, defined as a 0003 `StoreIdentity` with `key` and `check_key`, header shape taken from 0003 §3.2 (§3.1); `@1` roots raise `StoreLayoutMismatch`, no cross-scheme migration, and why (§9.1); order of delivery and the text 0003 must carry (§9.2, §10 I1 after 0003 P2); every 0003 citation re-pinned to `8be263c9` (now `547f7b1d`) |
| B2 `npy@1`/`file@1` are not 0003's member names | `npy@1` is the sha256 of the canonical `.npy` bytes that are 0003's member, so the names are equal by construction; `file@1` is computed from what `entry.json` records, with the per-member key 0003 has to add; the digest form is 0003's output encoding with two stated substitutions, so sets, complex, bytes, datetimes, `Path` and numpy scalars use 0003's form (§6.2, §9.2) |
| B3 `Path` by content breaks output locations | Design change: a `Path` is a value (`{"$path": …}`, as 0003), never tested for existence; content identity is declared by typing a field `File`/`Directory`. Why, and what stays per-host (§6.2) |
| B4 environment keys not bound to what runs | Design change: the reference is resolved once in the submitting process and the worker executes the resolved reference (`repo@sha256:…`, local `Id`, full Lmod names), or fails; `.sif` and `Native` version are re-checked in the worker; local builds never consult the registry (§5.2, §5.3) |
| I1 JCS conflates `1` and `1.0`, loses large ints | Numbers enter the digest form as typed strings (`$int` decimal, `$float` `float.hex`) (§6.2); the §3.1 example no longer uses an undefined tag |
| I2 `npy@1` over `dtype.str` | Preimage is the canonical `.npy` (header `descr`, little-endian, C order); subclasses and reference dtypes stay on `py@1`; non-contiguous copies and the cost limited to contiguous arrays stated; probe §8.8 (§6.2, §7) |
| I3 `AnyReuse` claim false | Owners of input value types keyed (`task.values`, §4.1); §6.3 rewritten to state what `AnyReuse` still serves (transitive, dynamic dispatch) and that `ConsistentReuse` refuses it |
| I4 gap 1 misses dunders, unstable across minors; `type@1` unspecified | Gap 1 uses `_class_code` (whole `ClassDef` under `ast@1`); `type@1` specified (§4.2, §5.1); probe §8.8: `__call__` edit seen, `Enum` input equal under 3.11 and 3.13 |
| I5 public version is not an identity for local, archive or VCS installs | The key uses `_identify`'s identity for direct-URL installs (`@commit`, `@archive hash`, `dir:`); public version only for index installs (§4.1) |
| I6 `version_command` stdout machine-dependent; stale cache for shims; container start | `version_pattern` keys only the declared part; memoised per process, not in `PersistentCache`; run under `Native` only, re-checked in the worker; not-found rule stated (§5.2) |
| I7 `tools` filled in the worker | `tools` is built in the submitting process with the rest of the lookup context; a missing executable is a stable `found: false`; worker facts only in 0003's execute-phase context (§5.2) |
| I8 local and offline images lose reuse; `RepoDigests` selection | Local builds keyed by `docker-local@1:<Id>` and run by it (reused per image store, as today); `RepoDigests` entry chosen by repository, several entries fall to the registry; air-gapped images resolve locally (§5.3) |
| I9 `.sif` not portable; `:tag` appended to resolved forms | `docker://` digest is the portable Singularity form; `.sif` content stated as exact but per-site, re-checked in the worker; `execute` stops appending `:{tag}` to resolved references (§5.3, I6) |
| I10 Lmod unprobed, extra call, ambient modules | Resolution specified (purge, load, name mapping, dependencies and shell modules recorded), the extra call costed, no-Lmod rule, measurement required before merge (§5.3, §7, §10) |
| I11 key depends on classification and env vars | Record carries `walk` (tier, overrides) so a mismatch is explainable; unowned library modules get an owner from `packages_distributions`; 3.14/3.15 bytecode reading listed as unchecked and required in I4's tests; goal 1 stated as conditional (§2, §3.1, §4.1, §4.2, §8) |

Nits are not addressed in this revision, except two citations corrected while re-checking lines:
the `executable` field is built at `compose/shell/builder.py:212-219`, and `compose/workflow.py:141-153`
is cited as the documentation of the constructor's `code_hash`.


## Amendment (2026-11-24): classes made by a factory (section 4.2, gap 1; owner decision D1)

``class@1`` also hashes the values of a class's plain attributes whose class-body assignment is
not a literal: an assignment ``offset = n`` inside ``def make(n): class Made: ...`` puts the same
code in every class ``make`` returns, so the code alone cannot tell ``make(1)`` from ``make(2)``.
A plain attribute is one that is not dunder, not callable and not a descriptor (``staticmethod``,
``classmethod``, ``property``). Each such attribute enters the digest as ``attr:<name>=`` and the
digest of its value, in name order. A class whose class-body assignments are all literals keeps its
digest. The names are found from the class's source; a class hashed in one process carries them
(``__pydra_computed_attributes__``), so a copy unpickled by value where the source cannot be read
hashes the same attributes.
