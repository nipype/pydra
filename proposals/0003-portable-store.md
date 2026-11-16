# Proposal 0003: a portable result store (proposal 0001, D2)

Status: proposed, for maintainer discussion. Branch `rfc/portable-store`, off
`enh/execution-extensions` at `6315ad7e`. This is the design of D2 (`PortableStore`), which proposal
0001 §3.3 step 3 names in one paragraph. Line references are to `6315ad7e` unless a PR or branch is
named: #929 is `enh/success-record` at `f803e8d5`; RFC 0002 is `rfc/graph-runner-executor` at
`3352353b`.

## 1. Motivation

D1 gave pydra a `ResultStore` protocol and a behaviour-identical `PickleStore`
(`engine/store.py:363-410`, `:413-537`). #924 and #928 replaced the run-long soft lock with a
`DirectoryLease` (`engine/lease.py:290`). #929 adds a success record so a cached job is read
without its lease. What is left is the layout and the entry format, and they have the problems that
0001 §1 lists for a shared cache:

| Problem | Evidence |
|---|---|
| Reading a cache runs its writers' code | results are cloudpickled (`engine/result.py:12`, `:306-307`) and loaded with `cp.load` (`:256-257`); the docstrings say so (`engine/store.py:416-419`, `engine/submitter.py:122-125`, `:139-142`) |
| The layout has no version | `<cache_root>/<checksum>/<ctxkey>/` with no header or version directory (`engine/store.py:456-457`); 0001 promises the layout is unchanged until a major version (0001 §2) |
| The slot name is a 64-bit hash of the whole context | `context_key` is the first 16 hex of the sha256 of the context as JSON (`engine/store.py:356-360`). A field recorded only for provenance would split one result into many slots, and a store cannot choose otherwise |
| A commit is correct only under the lease | the job runs in the slot (`engine/job.py:346-349`), `commit` writes in place and checks the lease first (`engine/store.py:528-537`). A process that does not hold the slot's lease, such as an executor whose job ran in its own scratch space (RFC 0002 §3.2), has no way to commit |
| Failures are stored | an errored run writes `_error.pklz` and an errored `_result.pklz` into the slot (`engine/job.py:440-450`, `engine/result.py:415-416`); the next run sees `errored` and reruns (`engine/job.py:410-415`, `:615-616`) |
| Paths are followed | `Path.is_dir()` and `iterdir()` in the walks (`engine/store.py:487-490`, `:500-512`) follow symlinks; file modes come from the umask (`utils/general.py:65`, `:78`) |
| Reuse is a yes or no | `ReusePolicy.accept` returns a bool (`engine/store.py:205`). A policy cannot say "serve this, but also recompute it and compare", cannot rank several admissible entries, and records nothing when it serves one |

D2 is opt-in and leaves `PickleStore` as it is. The goal is a store that any group can point at
shared storage, that never executes code on read, that survives a Python upgrade, and whose
identity, entry format and reuse rules a downstream system can replace without forking it.

**One motivating user.** senselab 2.x (sensein/senselab#617) keeps its own content-addressed store,
`store@6/<d[:2]>/<d>/<f>/` with `result.json`, `context.json` and arrays as `<sha256>.npy`, committed
by staging and rename (senselab `2.x` at `46608622`,
`packages/senselab-core/src/senselab_core/runtime/store.py:3-15`, `:503-552`), reached only through
descriptors (`runtime/storage.py:350-365`), and served through a serve/verify/refuse verdict
(`runtime/servable.py:25-53`). senselab has decided to move onto D2 before its 2.0.0 release, so this
design has to carry such a store with no loss of guarantees, and to move its existing entries by
translation, not recomputation (§7.2). The rest of this proposal does not refer to senselab.

## 2. Goals, non-goals, compatibility

**Goals:**
- a versioned layout with a header, that a reader checks before it reads an entry;
- the slot named by a context fingerprint that the store defines;
- commit by staging and one directory rename, first writer wins, no lease needed for correctness;
- no code executed on read from a shared store;
- every operation relative to a directory descriptor, never following a symlink;
- no stored failures;
- reuse decided by a verdict (serve, verify, refuse) over ranked candidates, with a reuse record;
- a migration from `PickleStore` that does not recompute.

**Non-goals:**
- Changing job checksums. A `PortableStore` slot is under the same checksum as today.
- A database or object-store backend. The layout is for POSIX filesystems, including NFS. An
  object-store backend can implement the same protocol later.
- Changing `PickleStore`, which stays the default until maintainers decide otherwise (§9).

**Defaults that do not change:** `PickleStore` and its layout; the `debug` worker; checksums;
`Submitter(cache_root=...)`; `ReusePolicy` and the three built-in policies.

## 3. Layout

### 3.1 Paths

```
<root>/pydra-store.json                                  layout header (§3.2)
<root>/<entries>/<shard>/<checksum>/<fingerprint>/       one committed entry, read-only
<root>/staging/<uuid>/                                   a writer's entry before its rename
<root>/quarantine/<uuid>                                 entries and wrong-kind objects moved aside
<root>/sidecars/<name>@<major>/...                       store-level records owned by a hook (§6.3)
<root>/leases/<checksum>-<fingerprint>.lock              optional de-duplication leases (§4.3)
<root>/work/<uuid>/ or <root>/work/<checksum>-<fingerprint>/   job work directories (§4.4), private stores only
```

- `<entries>` is the entry format's version directory, `results@1` for the default format (§5.1).
  It changes only when the entry format's major version changes.
- `<shard>` is `store.shard(checksum)`, one path component. The default is the first two hex
  characters of the checksum's hash part (`<task type>-<hash>`, `compose/base/task.py:508`); a
  supplied checksum (`__checksum__`, `compose/base/task.py:489-507`) uses its first two characters.
  A store may return `""` for no shard.
- `<checksum>` is `Job.checksum` unchanged (`engine/job.py:237-246`).
- `<fingerprint>` is `store.fingerprint(context)` (§3.3).
- Every name the store creates under `<root>` matches a fixed grammar: hex, a UUID, a member name of
  the entry format, or one of the fixed directory names above. Anything else under `<root>` is
  never read or removed. A checksum is already one path component (`compose/base/task.py:496-506`).

### 3.2 The layout header

`<root>/pydra-store.json`, written once with `O_CREAT | O_EXCL` by the first store that creates the
root, mode from the store's policy:

```json
{
  "format": "pydra-store@1.0",
  "entries": "results@1",
  "entry_format": "pydra-entry@1",
  "fingerprint": "pydra-context@1",
  "shard": "hex2",
  "pickle": false
}
```

- A store opening a root reads the header first. A `format` major it does not know raises
  `StoreLayoutUnknown`. An `entries`, `entry_format`, `fingerprint` or `shard` that differs from
  the store's own raises `StoreLayoutMismatch`, because two stores that name slots differently
  would misfile each other's entries.
- A root with entries but no header is a `PickleStore` cache or another tool's directory, and raises
  `LegacyLayout`, naming the migration (§7).
- `pickle` records what writers to this root may write (§5.3). It is a writer rule. A reader never
  decides to unpickle from the header, because anyone who can write the root can write the header.
- A minor version adds optional keys; readers ignore keys they do not know.

### 3.3 The context fingerprint

The slot is named by a fingerprint of the context, not by `context_key`
(`engine/store.py:356-360`).

```python
class Fingerprint(Protocol):
    name: str                                   # the header's "fingerprint", e.g. "pydra-context@1"
    def __call__(self, context: Context) -> str: ...   # 64 lower-case hex, one path component
```

- The default, `pydra-context@1`, is the full sha256 of the canonical JSON (RFC 8785) of the context
  without the fields the store declares recorded-only. By default none is recorded-only, so two
  contexts share a slot exactly when they are equal, as with `context_key` today, but with 256 bits
  instead of 64. A collision in a shared store would make a first writer's result serve another
  context, so the shortened form is not kept.
- A store declares recorded-only fields (`PortableStore(recorded_only=("installed",))`) or supplies
  its own `Fingerprint`. A recorded-only field is stored in `context.json` and visible to the reuse
  hook, but does not split slots. Two writers whose contexts differ only in recorded-only fields
  race for one slot, and the first wins.
- The reader binds an entry to its address (§5.2): it recomputes the fingerprint from the stored
  context and the checksum from the stored key, and requires both to equal the path.

## 4. Commit

### 4.1 Staging and rename

```python
@attrs.frozen
class Committed:
    won: bool                # False: another writer was first, and `entry` is theirs
    entry: EntryRef          # (checksum, fingerprint, root)
    output_digest: str       # the digest of the committed entry's encoded outputs

class PortableStore:
    def publish(self, checksum: str, context: Context, outputs: Mapping[str, Any],
                task: Task | None = None, files_from: os.PathLike | None = None) -> Committed: ...
```

1. The entry format encodes the outputs (§5) into a fresh `staging/<uuid>/`, made `0o700`. Files
   named by `files_from` (the job's work directory) are linked into it by descriptor where the
   filesystem allows, otherwise copied.
2. Each member is written with `O_CREAT | O_EXCL | O_NOFOLLOW` and `fsync`ed, then the staging
   directory is `fsync`ed.
3. The commit hook runs (§6.3). If it raises, the staging directory is removed and nothing is
   committed.
4. `rename(staging/<uuid>, <entries>/<shard>/<checksum>/<fingerprint>)`, both relative to directory
   descriptors.
5. If the rename succeeds, the entry's modes are set to the policy's committed modes through its
   descriptor, its parent is `fsync`ed, and `Committed(won=True, ...)` is returned.
6. If the rename fails and the target exists, another writer was first. The staging directory is
   removed and the store reads the existing entry: if it verifies (§5.2), `Committed(won=False,
   entry=<theirs>, output_digest=<theirs>)` is returned, so the caller can serve the winner's
   result as a reuse (§6). If it does not verify, it is quarantined and the commit is retried, a
   bounded number of times.

Other notes:
- On POSIX `rename(2)` onto an existing non-empty directory fails with `ENOTEMPTY` or `EEXIST`; onto
  an empty one it replaces it, which is harmless because an empty directory is not an entry. On
  Windows `os.rename` onto any existing directory fails. Both make the rename a first-writer-wins
  primitive. `staging/` is under `<root>`, so the rename never crosses filesystems.
- An entry is never written in place, rewritten or partly visible. Its existence means it is
  complete.
- `commit(lease, result, job)` of the `ResultStore` protocol (`engine/store.py:404-410`) is
  implemented as `publish` of `result.outputs` from `lease.cache_dir`. `job` is never written.

### 4.2 The #924 lease and the #929 success record

- **The lease is not needed for correctness.** A writer that lost its lease while running
  (`LeaseLost`, `engine/store.py:528-533`) still publishes: if the process that took over committed
  first, the rename fails and the writer gets the winner's entry. `PortableStore.commit` therefore
  never raises `LeaseLost`.
- **The lease is kept to avoid duplicate work** (§4.3), and for the work directory of a task that
  can resume (§4.4).
- **The success record is redundant.** #929 writes an empty file in `.pydra-succeeded/` after a
  successful commit (#929 `engine/store.py:35`, `:587-619`) and withdraws it before a slot is
  rewritten (`:569-571`, `engine/job.py:334-340`), because a `PickleStore` slot exists while a job
  runs in it and after it fails. A `PortableStore` entry exists only once committed, is never
  rewritten and is never a failure. Its directory is the success record. `PortableStore` implements
  `read_succeeded(checksum, context, task)` (#929 `engine/store.py:547-566`) as one `lstat` of the
  entry path relative to the shard descriptor, the same cost as #929's existence check (`:558`), and
  then the full lookup through the reuse hook (§6). `withdraw_success` is not defined, which #929
  already handles (`engine/job.py:338-340`, `engine/submitter.py:761-787`).
- **A record read is a lookup.** The success-record path returns a result only on a `serve` verdict.
  `verify` and `refuse` return None, so the job is dispatched and decided under the full path. This
  is the same rule #929 applies to cross-context hits (`engine/store.py:565`).

### 4.3 De-duplication leases

`PortableStore(dedupe=True)` (the default) returns a `DirectoryLease` from `acquire` at
`<root>/leases/<checksum>-<fingerprint>.lock`, with the #924 settings (`engine/store.py:517-523`). Two
submitters of one job then compute it once: the second waits, takes the lease, finds the entry and
serves it (`engine/job.py:410-415`). With `dedupe=False`, `acquire` returns a lease that is always
held, and concurrent submitters both compute; one wins the rename. A storage that cannot carry a
lease (`utils/storage.py:194`) can still host a `PortableStore` with `dedupe=False`.

### 4.4 Work directories

The job no longer runs in its slot. `ResultStore.cache_dir` (`engine/store.py:371-372`), which jobs
use for output templates and input copies (`engine/job.py:317-330`) and for `os.chdir`
(`engine/job.py:419-421`), returns a work directory:

- by default `<root>/work/<uuid>/`, one per attempt, removed after a successful publish and kept
  after a failure, for inspection;
- `<root>/work/<checksum>-<fingerprint>/`, stable, for a task with `can_resume`
  (`engine/job.py:282-284`, `:346-349`), under the de-duplication lease;
- anywhere else through `PortableStore(work_dir=callable)`, for executors that run jobs in their
  own scratch space. Such an executor calls `publish` with `files_from` set to that directory.

`work/` exists only under a store opened `private` or `trusted` (§5.3). A shared store must be given
a `work_dir` outside the root, because a work directory holds a job's pickles and crash files, and
these are not to be read by other users.

## 5. Entry format

### 5.1 The hook

```python
class EntryFormat(Protocol):
    name: str          # the header's "entry_format", e.g. "pydra-entry@1"
    entries: str       # the version directory, e.g. "results@1"
    def members(self) -> re.Pattern[str]: ...                        # names an entry may hold
    def stage(self, staging: StorageDir, checksum: str, context: Context,
              outputs: Mapping[str, Any], task: Task | None, files_from: StorageDir | None) -> str: ...
        # writes the members, returns the output digest
    def context(self, entry: StorageDir) -> Context: ...             # reads context.json only
    def read(self, entry: StorageDir, checksum: str, fingerprint: str,
             task: Task | None) -> StoredEntry: ...                  # outputs, context, output digest
```

The store owns the layout, staging, rename, sealing, quarantine and collection. The entry format
owns what is inside an entry. `read` raises `CorruptEntry` (quarantined), `MisfiledEntry` (a
`CorruptEntry`; quarantined), `EntryFormatUnknown` (another version's member; left in place, not
served) or `EntryNotTrusted` (a pickle member under a shared store, §5.3; left in place, not served).

The default, `pydra-entry@1` in `results@1/`:

```
<fingerprint>/entry.json        format, checksum, fingerprint, output digest, encoded outputs
<fingerprint>/context.json      format, the context
<fingerprint>/<sha256>.npy      one per array, named by the sha256 of its bytes
<fingerprint>/files/<sha256>/<name>   one per file or directory output, named by its content hash
<fingerprint>/<sha256>.pkl      private or trusted stores only (§5.3)
```

`Result`, `_job.pklz` and `_error.pklz` are never written to an entry.

### 5.2 Reading binds an entry to its address

A reader:
- reads `context.json`, recomputes the fingerprint and requires it to equal the directory name;
- reads `entry.json`, requires its `checksum` and `fingerprint` to equal the path and its `format` to
  be one it reads;
- reads each referenced member through the entry descriptor with `O_NOFOLLOW`, checks that it is a
  regular file and, for an array, that its bytes hash to its name; arrays load with
  `numpy.load(..., allow_pickle=False)`;
- recomputes the output digest over the encoded outputs and requires it to equal the recorded one.

A failed check quarantines the entry by renaming it to `quarantine/<uuid>`; an entry is never served
from a partial read. An `OSError` on read or write is raised as `StoreUnavailable` and moves
nothing.

### 5.3 Encoding and trust

The default encoding is JSON plus members, with no executable form:

| Output type | Encoded as |
|---|---|
| `None`, `bool`, `int`, `str` | JSON |
| `float` | JSON; NaN and infinities as `{"$float": "nan" \| "inf" \| "-inf"}` |
| `list`, `tuple` | JSON array; decoded to the field's declared type (`utils/typing.py` `TypeParser`) |
| `dict` with `str` keys | JSON object; a key starting with `$` is written `$$...` |
| `numpy.ndarray` (not object dtype) | `{"$array": "<sha256>"}` and a canonical `.npy` member |
| `File`, `Directory`, any `FileSet` | `{"$fileset": "<type>", "files": ["files/<sha256>/<name>", ...]}`; decoded to paths inside the read-only entry |
| `attrs` class, `enum`, other types with a codec | `{"$codec": "<tag>", "value": ...}` through `register_codec(type, tag, encode, decode)` |

A codec's `decode` builds a value from JSON and members only. Codecs ship in pydra for the types
pydra defines, and in plugins for domain types.

An output with no codec depends on how the store was opened. That is the reader's or writer's own
declaration, `PortableStore(trust=...)`, never something read from the root:

- **`trust="shared"`** (the default for a root passed explicitly). Writing an unencodable output
  raises `UnencodableOutput`, naming the field and its type; the job errors and nothing is committed.
  A `.pkl` member is never opened: an entry holding one raises `EntryNotTrusted` and is not served.
- **`trust="private"`.** The store checks at open that `<root>` is owned by the current user and not
  writable by group or others (`fstat` of the root descriptor), and refuses to open otherwise. An
  unencodable output is cloudpickled into `<sha256>.pkl`, tagged in `entry.json` with the
  cloudpickle version and the Python major.minor, and read back only by a matching interpreter. The
  header records `"pickle": true`.
- **`trust="trusted"`.** As `private`, without the ownership check: the operator asserts that
  everyone who can write the root is trusted with code execution, as `PickleStore` assumes today
  (`engine/store.py:416-419`).

When `Submitter` builds a `PortableStore` for its default cache directory, it opens it `private`, so
a pydra user whose task returns an arbitrary Python object keeps working. A cache pointed at shared
storage is `shared` unless declared otherwise.

## 6. Reuse

### 6.1 The hook

```python
class Verdict(Enum):
    SERVE = "serve"      # return the stored result
    VERIFY = "verify"    # run the job, then compare its result with the stored one
    REFUSE = "refuse"    # do not serve this entry; run the job

@attrs.frozen
class Candidate:
    entry: EntryRef
    context: Context       # the stored context
    rank: int              # 0 is the narrowest admission (an equal fingerprint)
    admitted_by: str       # e.g. "exact", "consistent", "equivalent", "tolerant"

class Reuse(Protocol):
    def rank(self, current: Context, stored: Sequence[tuple[EntryRef, Context]],
             task: Task | None) -> list[Candidate]: ...
    def verdict(self, candidate: Candidate, current: Context, task: Task | None) -> Verdict: ...
    def served(self, candidate: Candidate, current: Context, task: Task | None,
               entry: StoredEntry) -> None: ...        # the reuse record
    def compared(self, candidate: Candidate, fresh: Committed, task: Task | None) -> None: ...
```

- **Ranking.** `rank` returns the admitted candidates, best first. The store reads only
  `context.json` to rank, and reads the outputs of the first candidate only.
- **Lookup.** The store takes the first candidate and asks for its verdict. `SERVE`: the entry is
  read (§5.2), `served` is called, then the result is returned. A corrupt candidate is quarantined
  and the next one is taken. `VERIFY`: the lookup returns no result and marks the job to verify
  against that candidate. `REFUSE`: the lookup returns no result and the job runs. A refusal ends
  the lookup and does not try a lower-ranked candidate, because a reason to refuse, such as a
  license, usually applies to every candidate of the job.
- **The reuse record.** `served` is called before the result is returned. If it raises, the result
  is not served and the job errors, so no result is served without its record.
- **Verify.** A job marked to verify runs as usual, publishes its result to its own slot, and
  `compared` is called with the candidate and the new entry. What a comparison records, and where,
  is the hook's (§6.3).
- **A lost race.** When `publish` returns `won=False`, the job's result is the winner's entry. The
  store passes it through `verdict` and `served` as a rank-0 candidate, so the license and the record
  apply to a result served this way too. A `REFUSE` there raises, because the job's own result was
  not committed and the committed one may not be served.

### 6.2 Compatibility with `ReusePolicy`

`PortableStore(reuse=...)` takes the same values as `PickleStore` (`engine/store.py:427-434`,
`:260-264`). A `ReusePolicy` is wrapped as a `Reuse`: `rank` puts the current fingerprint first, then
every stored context that `accept`s (`engine/store.py:205`) in path order, which is the order
`PickleStore.lookup` uses (`:466-482`); `verdict` is always `SERVE`; `served` sets
`Result.cross_context` and `Result.reused_in` as today (`:478-480`); `compared` does nothing. The
declared-version check (`engine/store.py:281-327`) reads `stored_contexts`, which `PortableStore`
implements over `context.json`.

`PortableStore(readonly=[...])` takes further roots, searched after `<root>` as `readonly_caches`
are today (`engine/store.py:466-471`). Each must carry a header matching the store's (§3.2), is read
under the store's own `trust`, and is never written, collected or quarantined into: a corrupt entry
in a read-only root is skipped and logged. Candidates from all roots are ranked together.

### 6.3 The commit hook and sidecars

`PortableStore(on_commit=callable)` is called with the staging directory, checksum, context and
output digest after the members are written and before the rename (§4.1 step 3). It is the place to
write a provenance record that must exist before the result is visible; if it raises, nothing is
committed. A hook that keeps store-level records, such as verify comparisons, writes them under
`sidecars/<name>@<major>/` through `store.sidecar(name)`, which returns a `StorageDir` with the
same rules as the rest of the root. `collect` (§8) never touches `sidecars/`.

## 7. Migration

### 7.1 From `PickleStore`

`pydracli store migrate <cache_root> <root> [--trust-source]`:

- walks `<cache_root>/<checksum>/<ctxkey>/` without following symlinks;
- skips a slot with an errored `_result.pklz`, without one, or without `_context.json`
  (`engine/store.py:491-494`);
- unpickles `_result.pklz`, which needs `--trust-source` because it executes the old cache's code;
  the user already trusted that cache by reading from it;
- computes the fingerprint from `_context.json`, discards the `ctxkey`, and publishes the outputs
  through the target's entry format, with the slot's files as `files_from`;
- reports each output that the target cannot encode (a `shared` target), and leaves that slot
  unmigrated;
- copies no `_job.pklz`, `_error.pklz`, lease or `.pydra-succeeded/` record.

No job is recomputed: checksums are unchanged, and the outputs are decoded and encoded, not
produced. A run can also read an unmigrated cache in place: `PortableStore(legacy=[cache_root])`
reads old slots as a read-only fallback, only when the store is `private` or `trusted`, and
publishes every hit into `<root>` so the old cache is read once per result.

### 7.2 From an existing store already in this shape

A store laid out as `<version dir>/<shard>/<checksum>/<fingerprint>/` with its own members can adopt
D2 without rewriting an entry, provided that its checksum, its fingerprint and its encoded outputs
stay the same:

1. implement `EntryFormat` over its members, with `entries` set to its existing version directory
   name, and `Fingerprint` with its existing scheme;
2. write `pydra-store.json` naming them;
3. rename its staging, quarantine and store-level record directories to `staging/`, `quarantine/`
   and `sidecars/<name>@<major>/`, after its writers have stopped.

Each entry still verifies against its address (§5.2), so a translation that went wrong is caught on
read and quarantined. Only a change to checksums, fingerprints or the output encoding forces a
recompute, because it creates a new address space.

## 8. Storage rules and platforms

- **Descriptors.** Every directory below `<root>` is opened with `O_RDONLY | O_DIRECTORY |
  O_NOFOLLOW` relative to its parent's descriptor; files with `O_NOFOLLOW` (and `O_EXCL` when
  created); renames, `mkdir`, `unlink`, `rmdir` and `lstat` with `dir_fd`. `<root>` itself is opened
  as given. These are the operations `os.supports_dir_fd` lists on Linux and macOS (`rename`,
  `mkdir`, `open`, `lstat`, `unlink`, `rmdir`, `link`). Listing uses `os.scandir(fd)`.
- **Wrong kinds.** A symlink, or an object of the wrong kind, where the store expects a directory or
  member is moved to `quarantine/`; one that cannot be moved raises `WrongKindError`, naming the path
  and its owner.
- **Modes** come from `StorePolicy(file_mode, dir_mode, committed_file_mode, committed_dir_mode)`,
  set through descriptors after creation, never from the umask. `staging/` and `work/` entries are
  `0o700`. Today `atomic_write` creates files `0o644` adjusted by the umask (`utils/general.py:65`,
  `:89`).
- **Collection.** `store.collect(older_than)` removes `staging/` and `work/` entries and quarantined
  objects older than the policy's ages, by descriptor. It never removes an entry. `remove_from_cache`
  (`engine/store.py:125-198`) supplies the NFS and Windows retry behaviour.
- **NFS.** `rename` within one export is atomic; `staging/` is under `<root>` for that reason. Silly
  renames (`.nfs*`) are recognised as `remove_from_cache` does (`engine/store.py:45`, `:89-102`).
- **Windows.** `os.supports_dir_fd` is empty on Windows and `O_NOFOLLOW` does not exist, so the
  descriptor rules cannot hold. On Windows a `PortableStore` opens only as `private` or `trusted`,
  and `shared` raises `StorageUnsupported` (`utils/storage.py:40`). Under `private` it uses paths, and
  checks before each step that no component below `<root>` is a reparse point
  (`st_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT`); that check is not race-free, which is
  acceptable for a store only its owner can write. A directory rename fails while a file in it is
  open (sharing violation); the rename is retried as `remove_from_cache` retries (`engine/store.py:40-44`,
  `:149-198`). Committed modes are applied as the read-only attribute on files only.

## 9. Failures

A `PortableStore` stores no failure.

- An errored job publishes nothing. Its traceback goes to `<work dir>/_error.json` (JSON, not a
  pickle), where `record_error` writes `_error.pklz` today (`engine/job.py:440-443`,
  `engine/result.py:415-416`). `pydracli crash` reads it (`scripts/cli.py:31-33`).
- `_job.pklz` is written to the work directory, never to the entry, when it is persisted
  (`engine/job.py:350-353`, `:387-390`).
- **Rerun semantics.** Today the next run finds an errored result and reruns the job
  (`engine/job.py:410-415`, `:615-616`). Under `PortableStore` the next run finds no entry and runs
  the job. The observable behaviour is the same: an errored job runs again, a successful one does
  not.
- **Within a run.** `Job.errored` and the error summary on the submitter (0001 §3.1) are in memory
  and unchanged; downstream elements of an errored element stay unrunnable.
- **`rerun=True`.** An entry is never replaced in place. A rerun computes the job and publishes; the
  rename fails if the entry exists, and the existing entry is kept. `store.evict(entry)` moves an
  entry to `quarantine/` so that the next commit replaces it.

## 10. Delivery

| PR | Contents | Depends on |
|---|---|---|
| P1 | `pydra.utils.storagedir`: a descriptor-relative directory handle (`child`, `open_new`, `rename`, `lstat`, `scandir`, `set_mode`, `quarantine`), the wrong-kind rule, `StorePolicy` modes; Windows reparse-point checks. Unused by `PickleStore`. Tests: symlink swapped in at each level, wrong kinds, modes under a permissive umask | — |
| P2 | layout header, `Fingerprint` (`pydra-context@1`), shard hook, `StoreLayoutUnknown`/`Mismatch`/`LegacyLayout` | P1 |
| P3 | `EntryFormat` protocol, `pydra-entry@1`, the codec registry and the encoding table; `UnencodableOutput`; `trust=` and the private-root check | P2 |
| P4 | `PortableStore`: `publish`, first writer wins, sealing, quarantine, `collect`, de-duplication leases, work directories; `commit` over `publish`; errors to `_error.json`; `pydracli crash` reads it. Tests: two processes racing one checksum, SIGKILL between staging and rename, a lost lease that still commits, a read-only root, `debug` and `cf` parity | P3, #928 |
| P5 | `Reuse` protocol, the `ReusePolicy` wrapper, verify, the reuse record, the lost-race path, `on_commit` and sidecars | P4, K2 |
| P6 | `read_succeeded` for `PortableStore` through the reuse hook | P5, #929 |
| P7 | `pydracli store migrate` and `legacy=` | P4 |
| P8 | `Submitter(store="portable")` shortcut; documentation; Windows CI for `private` | P5 |

P1 to P3 change no behaviour. P4 is the first PR a user can opt into. RFC 0002 needs nothing more:
an executor that runs a job outside `Job.run` commits through `publish` (RFC 0002 §3.2).

## 11. Open questions for maintainers

- Should `installed` be recorded-only in the default fingerprint? Today `ConsistentReuse` ignores it
  (`engine/store.py:236-250`) but it splits slots. If it is recorded-only, should `ExactReuse`
  compare fingerprints rather than whole contexts (`engine/store.py:232-233`)?
- `rerun=True`: keep the existing entry (as proposed) or evict and replace it?
- Should `REFUSE` end the lookup, as proposed, or try the next candidate?
- Is a `shared` store on Windows worth supporting with path checks that are not race-free, or is
  `StorageUnsupported` right?
- Workflow outputs that are files: copy or link them into the workflow's entry (as `save` copies
  them today, `engine/result.py:300-304`), or refer to the inner job's entry?
- Should `ResultStore.commit(lease, result, job)` keep its signature, or should the protocol gain
  `publish` and `work_dir` so that executors do not need a lease to commit?
- Which codecs ship in pydra (attrs, enum, pathlib, numpy scalars) and which in plugins?
- Should `PortableStore` with `trust="private"` become the default at the next major version, with
  `PickleStore` kept for one release behind `store="pickle"`?
- Should a hit from a `readonly` root be recorded as cross-root reuse, as a hit from another context
  is recorded as cross-context (`engine/store.py:478-480`)?
