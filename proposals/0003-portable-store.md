# Proposal 0003: a portable result store (proposal 0001, D2)

Status: proposed, for maintainer discussion; revision 3. Branch `rfc/portable-store`, off
`enh/execution-extensions` at `3f734f9a` (the merge of #929). This is the design of D2
(`PortableStore`), which proposal 0001 §3.3 step 3 names in one paragraph. Line references are to
`3f734f9a` unless another commit is named. RFC 0002 is #930 at `bdce243d`. Revision 2 answers the
review of #931 and revision 3 its first re-review; §12 maps each finding to the text that resolves it.

## 1. Motivation

D1 gave pydra a `ResultStore` protocol and a behaviour-identical `PickleStore`
(`engine/store.py:379-425`, `:428-620`). #924 and #928 replaced the run-long soft lock with a
`DirectoryLease` (`engine/lease.py:290`). #929 added a success record so a cached job is read
without its lease (`engine/store.py:35`, `:547-567`). What is left is the layout and the entry
format, and they have the problems that 0001 §1 lists for a shared cache:

| Problem | Evidence |
|---|---|
| Reading a cache runs its writers' code | results are cloudpickled (`engine/result.py:12`, `:306-307`) and loaded with `cp.load` (`:256-257`); the docstrings say so (`engine/store.py:431-434`, `engine/submitter.py:122-125`, `:139-142`) |
| The layout has no version | `<cache_root>/<checksum>/<ctxkey>/` with no header or version directory (`engine/store.py:475-476`); 0001 promises the layout is unchanged until a major version (0001 §2) |
| The slot name is a 64-bit hash of the whole context | `context_key` is the first 16 hex of the sha256 of the context as JSON (`engine/store.py:371-375`). A field recorded only for provenance splits one result into many slots, and a store cannot choose otherwise |
| The context and the key are pydra's alone | the context is `job_context(task, environment)`, built once in the submitting process (`engine/job.py:249-262`, `engine/context.py:489-522`); a store sees the checksum as a string and nothing of the key behind it (`engine/store.py:389-404`) |
| A commit is correct only under the lease | the job runs in the slot (`engine/job.py:352-357`), `commit` writes in place and checks the lease first (`engine/store.py:587-600`). A process that does not hold the slot's lease, such as an executor whose job ran in its own scratch space (RFC 0002 §3.2, `checks_store = False`), has no way to commit |
| Failures are stored | an errored run writes `_error.pklz` and an errored `_result.pklz` into the slot (`engine/job.py:448-459`, `engine/result.py:415-416`); the next run sees `errored` and reruns (`engine/job.py:418-424`, `:625-626`) |
| Paths are followed | `Path.is_dir()` and `iterdir()` in the walks (`engine/store.py:503-513`, `:515-531`) follow symlinks; file modes come from the umask (`utils/general.py:65`, `:78`) |
| Reuse is a yes or no | `ReusePolicy.accept` returns a bool (`engine/store.py:215`). A policy cannot say "serve this, but also recompute it and compare", cannot rank several admissible entries, and records nothing when it serves one |

D2 is opt-in and leaves `PickleStore` as it is. The goal is a store that any group can point at
shared storage, that never executes code on read from storage others can write, that survives a
Python upgrade, and whose identity, entry format and reuse rules a downstream system can replace
without forking it.

**One motivating user.** senselab 2.x (sensein/senselab#617) keeps its own content-addressed store,
`store@6/<d[:2]>/<d>/<f>/` with `result.json`, `context.json` and arrays as `<sha256>.npy`, committed
by staging and rename (senselab `2.x` at `4472cfbb`,
`packages/senselab-core/src/senselab_core/runtime/store.py:3-15`, `:503-552`). Its key `d` is the
digest of a key record stored in `result.json`, and its fingerprint `f` is taken over a context of its
own (torch build, env lock, model identity) that pydra's context does not hold (`:299-307`,
`:532-538`). Its entries are made read-only after the rename (`:520`, `:548`, `:555-560`) inside
group-writable directories (`runtime/storage.py:57`, `:152-161`). It is reached only through
descriptors (`runtime/storage.py:350-365`), and serves through a serve/verify/refuse verdict
(`runtime/servable.py:25-53`). senselab has decided to move onto D2 before its 2.0.0 release, so this
design has to carry such a store with no loss of guarantees, and to move its existing entries by
translation, not recomputation (§8.2). The rest of this proposal does not refer to senselab.

## 2. Goals, non-goals, compatibility, threat model

**Goals:**
- a versioned layout with a header, which a store checks before it reads an entry;
- the key, the context and the slot's fingerprint supplied by hooks, with pydra's as defaults;
- commit by staging and one directory rename, first writer wins, no lease needed for correctness;
- no code executed on read from a root other users can write;
- every operation relative to a directory descriptor, never following a symlink below the root;
- no stored failures;
- reuse decided by a verdict (serve, verify, refuse) over ranked candidates, with a reuse record;
- a migration from `PickleStore`, as a function, that does not recompute.

**Non-goals:**
- Changing job checksums. Under `pydra-identity@1` a `PortableStore` slot is under the same
  checksum as today. RFC 0004 changes every automatic checksum once, in its own release (§4).
- A database or object-store backend. The layout is for POSIX filesystems, including NFS, and for
  private stores on Windows. An object-store backend can implement the same protocol later.
- Changing `PickleStore`, which stays the default until maintainers decide otherwise (§13).

**Defaults that do not change:** `PickleStore` and its layout; the `debug` worker; checksums under
`pydra-identity@1`;
`Submitter(cache_root=...)`; `ReusePolicy` and the three built-in policies.

**Threat model.** A user who can write a `shared` root can make other users' runs read a wrong
result: they can commit an entry with wrong outputs under a valid address. They cannot make a reader
execute code, read or write outside the root through the store, change a committed entry they do
not own, or make a reader serve an entry whose members do not match their recorded digests. A
`private` root admits writes only by its owner, so its entries may hold pickles (§6.4). Detecting
wrong-but-consistent results is the reuse hook's business (verify sampling, §7.1), not the
store's.

## 3. Layout

### 3.1 Paths

```
<root>/pydra-store.json                                    layout header (§3.2)
<root>/<entries>/<shard>/<cdir>/<fingerprint>/             one committed entry
<root>/<entries>/<shard>/<cdir>/q-<uuid>                   an entry its reader could not move to quarantine
                                                           (§6.3); the UUID is derived from the entry's inode
<root>/staging/<uuid>/                                     a writer's entry or directory before its rename
<root>/quarantine/<uuid>                                   entries and wrong-kind objects moved aside; an
                                                           entry's UUID is derived from its inode (§6.3)
<root>/sidecars/<name>@<major>/...                         store-level records owned by a hook (§7.4)
<root>/leases/<lease name>.lock                            optional de-duplication leases (§5.3)
```

The header may rename `staging`, `quarantine` and each sidecar directory (§3.2), so a store that
already has such directories keeps them (§8.2). Work directories are not under `<root>` unless the
root is `private` (§5.4).

- `<entries>` is the entry format's version directory, `results@1` for the default format (§6.1).
  It changes only when the entry format's major version changes.
- `<cdir>` is the checksum as a directory name. A checksum that matches
  `[a-z0-9][a-z0-9_-]{0,199}` is used as it is. Any other checksum (upper case, a dot, other
  characters, or longer) becomes `h-<sha256 of the checksum>`. pydra's own checksums,
  `<task type>-<hash>` (`compose/base/task.py:508`), are lower case and kept as they are. The
  real checksum is recorded in `entry.json` and compared on read. This rule makes the layout safe on
  case-insensitive filesystems: two checksums that differ only in case never share a directory,
  because neither is used raw.
- `<shard>` is two lower-case hex characters. By default they are the first two of the sha256 of the
  checksum. The header may name `prefix2` instead, the first two characters of `<cdir>`, for a store
  whose checksums are already hex digests (§8.2). `prefix2` is refused for a `<cdir>` that does not
  start with two hex characters, so `__checksum__` values such as `..x`, which the check at
  `compose/base/task.py:496-506` allows, never produce a shard outside `<entries>`.
- `<fingerprint>` is `identity.fingerprint(context)` (§4), 64 lower-case hex.
- Every name the store creates under `<root>` matches a fixed grammar: hex, `h-<hex>`, `q-<uuid>`,
  a UUID, a member name of the entry format, an aside name the entry format declares (§6.1), or one of
  the directory names in the header. Anything else under `<root>` is never read or removed.

### 3.2 The layout header

`<root>/pydra-store.json`:

```json
{
  "format": "pydra-store@1.0",
  "entries": "results@1",
  "entry_format": "pydra-entry@1",
  "identity": {"name": "pydra-identity@1", "fingerprint": "pydra-context@1", "recorded_only": []},
  "shard": "sha256-2",
  "dirs": {"staging": "staging", "quarantine": "quarantine", "sidecars": {}},
  "modes": {"file": "0o444", "entry": "0o755", "sealed": null, "container": "0o755",
            "side": "0o755", "staging": "0o700"},
  "pickle": false
}
```

- **Written atomically.** The first store to create the root writes the header to a private
  temporary in `<root>` (`.pydra-store.<uuid>.tmp`), `fsync`s it, and `link()`s it to
  `pydra-store.json`, which fails if the name exists; it then removes the temporary. A concurrent
  opener therefore sees either no header or a complete one. A kill before the link leaves only a
  temporary, which the next opener removes once it is older than the staging age (§9). On a
  filesystem without hard links the store uses `rename` without replacement where the platform
  provides it (Windows), and refuses to create a header otherwise.
- **Checked on open.** A `format` major the store does not know raises `StoreLayoutUnknown`. Any
  other key that differs from the store's configuration raises `StoreLayoutMismatch`, naming the key.
  That includes the `recorded_only` set: two stores that fingerprint different fields would each
  find the other's entries misfiled and quarantine them.
- **No header.** A root with entries but no header is a `PickleStore` cache or another tool's
  directory and raises `LegacyLayout`, naming the migration (§8).
- **`modes`** are the store policy's (§9). Recording them makes two users with different defaults
  fail at open rather than write entries with mixed modes. `file` is a committed member's mode;
  `entry` is the entry directory's mode, set before the rename; `sealed`, when not null, is the
  narrower mode it is given after the rename; `container` is the mode of `<entries>`, `<shard>` and
  `<cdir>`; `side` is the mode of the directories named in `dirs` and of `leases/` and `work/`;
  `staging` is the mode of a writer's own `staging/<uuid>/` while it writes. `file`, `entry` and
  `sealed` never carry group or other write; `container` and `side` may, and must for a root that
  several users write (§9).
- **`pickle`** says whether writers to this root may write pickle members. It is set when the root is
  created and is a writer rule: with `false`, no writer pickles, whatever trust it declared. A reader
  never decides to unpickle from the header, because anyone who can write the root can write the
  header (§6.4).
- A minor version adds optional keys; readers ignore keys they do not know.

## 4. Identity: the key, the context and the fingerprint

The store does not compute identity. It asks a `StoreIdentity`:

```python
Context = Mapping[str, JSON]          # opaque to the store: any JSON mapping
KeyRecord = Mapping[str, JSON] | None

class StoreIdentity(Protocol):
    name: str                                         # the header's identity.name
    fingerprint_name: str                             # the header's identity.fingerprint
    recorded_only: frozenset[str]                     # the header's identity.recorded_only
    def key(self, job: Job) -> KeyRecord: ...         # the record the checksum is the digest of, or None
    def check_key(self, checksum: str, key: KeyRecord) -> bool: ...   # does the record digest to the checksum
    def context(self, job: Job, phase: Literal["lookup", "execute"]) -> Context: ...
    def fingerprint(self, context: Context) -> str: ...               # 64 lower-case hex
```

- **The default, `pydra-identity@1`,** returns `None` as the key (pydra's checksum is not the digest
  of a stored record; `check_key` is then true), `job_context(job.task, job.environment)` as the
  context in both phases, which is what `Job.context` builds today (`engine/job.py:249-262`), and as
  fingerprint `pydra-context@1`: the full sha256 of the canonical
  JSON (RFC 8785) of the context without the `recorded_only` fields. By default none is recorded-only,
  so two contexts share a slot exactly when they are equal, as with `context_key` today, but with 256
  bits instead of 64. A collision in a shared store would make one context's result serve another,
  so the shortened form is not kept.
- **After RFC 0004.** RFC 0004 adds `pydra-identity@2`, whose key is the identity record that the
  checksum is the digest of. From the release that carries it, `pydra-identity@2` is the default,
  every automatic checksum changes once, and `@1` checksums can no longer be produced. A root
  created under `@1` raises `StoreLayoutMismatch` naming `identity` when opened under `@2`, also
  when it holds only supplied-checksum entries, whose `@1` key record is `None` and so cannot be
  told from automatic ones; such a root is re-created, or kept open under an explicit
  `identity=` of `pydra-identity@1` for as long as that code ships.
- **The two phases.** `context(job, "lookup")` runs in the process that looks the job up: the runner
  in `ready()` or `Job.run` under the lease. `context(job, "execute")` runs in the process that runs
  the job, after the task has run and before `publish`, so a context can record facts of the worker
  (the library build it loaded, the device it ran on). The entry is published under the execute
  context's fingerprint. If the two differ, the job's result is in another slot than the one it was
  looked up in. The runner never recomputes the slot from a context: it reads the job's result by
  the `EntryRef` that `publish` returns (§5.1, §7.2).
- **The key.** `publish` takes the key record with the checksum and passes both to the entry format,
  which may store the record. On read the store calls `check_key(checksum_in_entry, key_in_entry)` and
  treats a false as misfiled. This lets a downstream store whose checksum is the digest of its own key
  record bind each entry to that record, as §6.2 binds it to its context.
- **How it is called.** Under a store that has an `identity`, `Job.context` returns
  `store.identity.context(job, "lookup")`, built once in the submitting process and carried with the
  job as today. Every call that passes `job.context` therefore passes the identity's lookup context:
  `acquire` (`engine/job.py:415`), `lookup` (`:617-624`), `read_succeeded`
  (`engine/submitter.py:779`) and the declared-version check (`engine/job.py:259-261`). Without that
  delegation those calls would fingerprint pydra's context and look in slots that nothing is
  published under. The calls that need more than a context take the job: `find(job)` returns a typed
  lookup result (§7.1), and `work_dir(job)` reads the attempt from it (§5.4).
- **Where it is configured.** `PortableStore(identity=...)`. The identity object must be picklable,
  because `context(job, "execute")` runs in worker processes; it travels with the store.
- **`recorded_only`** fields are kept in `context.json` and reach the reuse hook, but do not split
  slots. Two writers whose contexts differ only in recorded-only fields race for one slot, and the
  first wins (§5.1).

## 5. Commit

### 5.1 Staging and rename

```python
@attrs.frozen
class EntryRef:
    root: str              # which root (§6.4) the entry is in
    cdir: str
    fingerprint: str

@attrs.frozen
class Committed:
    entry: EntryRef                  # the entry now at the address, ours or another writer's
    won: bool                        # our rename placed it
    own_digest: str                  # the digest of our encoded outputs
    committed_digest: str | None     # the digest of the entry at the address; None if it cannot be read (§5.2)
    stored: bool                     # our outputs, or outputs equal to them, are in the store
    kept: str | None                 # with stored=False: where our outputs are kept (§5.4)

class PortableStore:
    def publish(self, checksum: str, key: KeyRecord, context: Context, outputs: Mapping[str, Any],
                task: Task | None, files_from: os.PathLike | None, *,
                verify: Candidate | None = None, replace: bool = False) -> Committed: ...
```

1. The entry format encodes the outputs (§6) into a fresh `staging/<uuid>/`, created `0o700`. Every
   file output is **copied** into the staging directory, using a reflink (`FICLONE` on Linux,
   `clonefile` on macOS) where the filesystem supports one, never a hard link. The copy is hashed as
   it is written, and each file member's sha256 and size go into `entry.json`. Copying is why a
   committed entry cannot change when a user edits a dataset file that an input copy hard-linked
   into the work directory (`engine/job.py:318-324`), and why the store never changes the mode of a
   user's file: it only sets modes on its own copies.
2. Each member is written with `O_CREAT | O_EXCL | O_NOFOLLOW`, given its final mode from the policy
   through its descriptor (by default `0o444`), and `fsync`ed. `entry.json` records a random writer
   token (a UUID, nothing else).
3. If the job is marked to verify (§7.1), `Reuse.compare` runs now, before anything is visible.
4. The commit hook runs (§7.4). If it raises, the staging directory is removed and nothing is
   committed.
5. The staging directory gets the `entry` mode through its descriptor (by default `0o755`: the
   owner keeps write permission, because a directory without it cannot be renamed into another
   parent) and is `fsync`ed. The entry is readable by every user it is meant for before it is
   visible, so a writer killed after the rename leaves a readable entry.
6. `rename(staging/<uuid>, <entries>/<shard>/<cdir>/<fingerprint>)`, both relative to directory
   descriptors. The parent directories are created the same way: each is made in `staging/`, given the
   `container` mode, and renamed into place; a loser of that rename uses the existing directory. A
   kill can then never leave a shard or checksum directory with a private mode.
7. If the rename succeeds and the policy has a `sealed` mode, the entry directory is given it
   through a descriptor opened on the new name. A kill between the rename and the seal leaves an
   entry in the `entry` mode, which is readable and is sealed by its owner's next read (§6.2). The
   parent is `fsync`ed and `Committed(won=True, stored=True, committed_digest=own_digest)` is
   returned.
8. If the rename fails and the target exists, the store reads the target's `entry.json`. If it
   carries this writer's token, the rename did succeed and its reply was lost (NFS): the result is as
   in step 7. Otherwise another writer was first, and the target is read and verified (§6.2):
   - it verifies: `committed_digest` is its digest, `won=False`, and `stored` is true exactly when the
     two digests are equal;
   - it is corrupt or misfiled: it is quarantined (§6.3) and the commit is retried, a bounded number
     of times;
   - it cannot be read by this store (`EntryNotTrusted`, `EntryFormatUnknown`, `StoreUnavailable` on
     its members): `committed_digest=None`, `stored=False`.

   When `stored` is true the staging directory is removed. When it is false the staged entry is kept,
   unpublished, in the work directory (§5.4), and `Committed.kept` names it.
9. Once §7.3 has decided the job's result, the executing process writes `<work dir>/_committed.json`
   (to a temporary name, then renamed into place):
   `stored`, and where the result is read from, an `EntryRef` or the kept entry. This is how a process
   other than the executing one reads the result (§7.2).

**A parent this user cannot write.** If making `staging/<uuid>`, entering or creating `<shard>` or
`<cdir>`, or the rename fails with `EACCES` or `EPERM`, `publish` raises `StoreNotWritable`, a
`StoreUnavailable` naming the directory, its owner and mode, and the header's `container` and `side`
modes. A root whose `container` and `side` modes are not group-writable (the default) is written by
its creator only, and this is the error another user's first publish gets. It is a configuration
error, so the job errors; it is not taken for a lost race.

What the job's result is after a lost race is §7.3.

Other notes:
- On POSIX `rename(2)` onto an existing non-empty directory fails with `ENOTEMPTY` or `EEXIST`; onto
  an empty one it replaces it. On Windows `os.rename` onto any existing directory fails. Both make the
  rename a first-writer-wins primitive for entries. `staging/` is under `<root>`, so the rename never
  crosses filesystems.
- **Containers are not stable across another writer's rename.** Replacing an empty directory is not
  harmless for its containers: two writers can each create an empty `<shard>` or `<cdir>` and rename
  it into place, and a writer that already holds a descriptor on the replaced one keeps working in a
  directory that is no longer in the tree, so its own rename then fails with `ENOENT` (seen with six
  threads publishing one address on macOS). So after each rename of its own, and on `ENOENT` or
  `ESTALE` while staging or renaming, a writer compares the device and inode of each container it
  holds with those of the same path walked again from `<root>`. If any differs, it walks down again
  and retries the step, a bounded number of times. It never assumes a container it holds is still the
  one at its path.
- An entry is never written in place or partly visible. Its existence means it is complete.
- **File outputs.** An output that is itself a symlink to a file is resolved at staging and its target
  copied, because the outputs are the writer's own and the digest covers the copied content. A
  symlink to a directory inside a `Directory` output raises `UnencodableOutput`: `FileSet.byte_chunks`,
  and so RFC 0004's `file@1`, does not descend into it, so copying its target would store content
  that the entry's digest does not cover. A file output outside `files_from` is copied too. A special file
  (FIFO, socket, device) inside a `Directory` output raises `UnencodableOutput` (§6.4).
- **Cost.** Every file output is read once and written once, and hashed in the same pass. With a
  reflink the write is free. `fsync` of each member, the staging directory and the parent costs a
  round trip each on NFS. The policy's `durability` setting (§9) can drop the per-member `fsync` to
  one `syncfs`-style flush before the rename, or drop it entirely for a scratch store.
- `commit(lease, result, job)` of the `ResultStore` protocol (`engine/store.py:419-425`) is kept as
  an adapter: it calls `publish` with `result.outputs`, the job's key and execute context, and
  `lease.cache_dir` as `files_from`. `job` is never written to an entry.

### 5.2 The #924 lease and the #929 success record

- **The lease is not needed for correctness.** A writer that lost its lease while running (the check
  at `engine/store.py:590-595`) still publishes: if the process that took over committed first, the
  rename fails and §5.1 step 8 applies. `PortableStore` never raises `LeaseLost`, so the `LeaseLost`
  error entry that RFC 0002 §3.1 gives a workflow job does not arise under it.
- **The success record is redundant.** #929 writes an empty file in `.pydra-succeeded/` after a
  successful commit (`engine/store.py:35`, `:587-620`) and withdraws it before a slot is rewritten
  (`:569-571`, `engine/job.py:334-340`), because a `PickleStore` slot exists while a job runs in it and
  after it fails. A `PortableStore` entry exists only once committed, is replaced only by a rerun's
  evict-and-rename (§10), and is never a failure. Its directory is the success record.
  `PortableStore.read_succeeded(checksum, context, task)` (`engine/store.py:547-567`) is one `lstat`
  of the rank-0 entry path relative to the shard descriptor, the cost of #929's existence check
  (`:558`), followed by the full lookup through the reuse hook (§7.1). It returns a result only on a
  `SERVE` verdict, as #929 returns none for a cross-context hit (`:565`); `VERIFY` and `REFUSE` return
  `None`, and the job is dispatched. Because it returns a `Result | None`, it cannot carry the verify
  mark, so a runner over a `PortableStore` calls `find(job)` (§7.1) in its place, and a `Verify` sets
  `job.verify` before the job is dispatched (§13). `withdraw_success` is not defined, which `Job` and
  `Submitter` already allow (`engine/job.py:338-340`, `engine/submitter.py:774`).
### 5.3 De-duplication leases (optional)

`PortableStore(dedupe=...)` decides whether two submitters of one job compute it once.

- `dedupe=True` returns a `DirectoryLease` from `acquire`, with the #924 settings
  (`engine/store.py:579-585`), at `<root>/leases/<lease name>.lock`. `<lease name>` is the sha256 of
  `<cdir>/<fingerprint>`, so the name carries no checksum text. The second submitter waits, takes the
  lease, finds the entry and serves it (`engine/job.py:418-424`).
- `dedupe=False` returns a lease that is always held. Concurrent submitters both compute and one wins
  the rename. A storage that cannot carry a lease (`utils/storage.py:194`) can host a `PortableStore`
  only this way.
- **Default:** `True` for a `private` root, `False` for a `shared` one.
- **What it writes.** `DirectoryLease` writes `holder.json` with the owner's uid, the host name and
  the pid (`engine/lease.py:434-441`), which only messages use (`:239-248`). Under a `PortableStore`
  the lease is built with `LeaseSettings(holder="anonymous")`, a new setting under which the record
  holds only the random token and the start time, and messages say "another process". Nothing in
  `leases/` names a user, a host or a job.
- **Bounds.** `PortableStore(lease_settings=...)` takes a `LeaseSettings`, as `PickleStore` does,
  and every lease the store takes (de-duplication, resume and rerun) uses it. The default keeps the
  #924 bounds. The holder is forced to `"anonymous"` whatever is passed, and the setting is fixed at
  construction; a worker that unpickles the store uses the same one. Processes on one root may use
  different bounds. A holder records its `stale_after` in `holder.json` (it names no one), and a
  taker treats a lease as stale only after the larger of its own `stale_after` and the holder's,
  at most its `max_honoured_stale_after` (default four times the default `stale_after`, 1200 s). A
  record without a `stale_after`, written by an older pydra, counts as the default 300 s; a record
  that exists but cannot be read (another user's on a shared root) counts as the cap. So a process
  with shorter bounds never takes over a live holder whose bounds are within the cap, a dead holder
  that recorded an absurd bound is still taken over at the cap, and a short `stale_after` speeds up
  a resume after a crash only when the crashed run used it too.
- **Paths.** `DirectoryLease` works by path (`engine/lease.py:290`), not by descriptor. The store opens
  `leases/` by descriptor at start and refuses a symlink there (§9), and the lease directory holds no
  entry data, so a lease redirected by a race below `leases/` can block or duplicate work but cannot
  change a result. Moving the lease to descriptor-relative operation is 0001's M1 follow-up, not a
  prerequisite of this proposal.

### 5.4 Work directories

The job no longer runs in its slot. `ResultStore.cache_dir(checksum, context)`
(`engine/store.py:386-387`), which jobs use for output templates and input copies
(`engine/job.py:318-330`), for `os.chdir` (`:430`) and for the crash path (`engine/submitter.py:491-494`),
returns a work directory. It is called many times and in several processes (`engine/job.py:287-289`,
`engine/store.py:302`), so it has to be a pure function of values the job carries.

- **The attempt.** The runner gives each leaf job an `attempt` id (a UUID) when it hands the job out,
  and the job carries it to the executing process with its checksum and context (RFC 0002 §3.4). A
  workflow job is never handed out: it gets its attempt when the runner starts to expand it, and the
  outer job of a `Submitter` call when it is created (`engine/submitter.py:476-481`), so a workflow's
  `_error.json`, kept entry and `_committed.json` have a directory of their own. A job looked up but
  never handed out or expanded has none; a `workflow-stored` job, settled without expansion, has
  none.
- **The contract.** `work_dir(checksum: str, fingerprint: str, attempt: str) -> Path` is pure: the
  same arguments give the same path in every process. `fingerprint` is that of the job's lookup
  context (§4), which the runner and the executing process both hold; the execute context is not known
  until the task has run. `PortableStore.work_dir(job)` calls it with `job.checksum`,
  `identity.fingerprint(job.context)` and `job.attempt`, and `Job.cache_dir` (`engine/job.py:287-289`)
  calls `work_dir(job)` when the store defines it, because `ResultStore.cache_dir(checksum, context)`
  cannot reach the attempt. `check_declared_code`, which uses `cache_dir` only to ask whether the
  context's own slot holds a result (`engine/store.py:302`), asks `PortableStore` for the entry path
  instead. It must return an absolute path on storage the executing process can write, and outside
  `<root>` for a `shared` root. The path must also be readable by the runner's process and by every
  process that runs a downstream job, because `_committed.json` and kept outputs are read there
  (§7.2). For a `shared` root, and for any executor that runs jobs on other nodes, that means storage
  every such node sees, not node-local scratch; a store used with an out-of-process executor checks
  at hand-out that the runner's process can see the work directory. The store creates the directory
  `0o700` if it is absent and never removes one it did not create.
- **Attempts allocated outside the runner.** A worker pool (RFC 0005 §5.5) runs a job under the
  attempt `<pool id>.<g>`, where `g` is the job's own generation number: unique per `(checksum,
  fingerprint)`, not across jobs. A `work_dir` function receives such an attempt as it is and must
  use the job's address (`checksum`, `fingerprint`) to keep two jobs' directories apart. The default
  layout names a dotted attempt `<root>/work/<attempt>-<first 24 hex of the sha256 of
  "<cdir>/<fingerprint>">/`. `PortableStore.work_path(checksum, fingerprint, attempt)` gives the
  directory without a job. A pool's job payloads are not in any work directory (RFC 0005 §5.2), so
  `collect` never removes one.
- **The default** is `<root>/work/<attempt>/` for a `private` root. A `shared` root has no default:
  `PortableStore(work_dir=...)` is required, because a work directory holds a job's pickles and crash
  file, which other users must not read.
- **Resumable tasks.** A task with `can_resume` (`engine/job.py:282-284`, `:355-357`) gets
  `attempt="resume"`, so its work directory is stable across runs. It is then guarded by the
  de-duplication lease, which such a store needs whatever its `dedupe` default.
- **Kept outputs.** When `publish` returns `stored=False` (§5.1 step 8, §6.3, §7.3), the staged
  entry is moved to `<work dir>/_entry/`, by rename when the work directory is on the root's
  filesystem and otherwise by copy, each member re-hashed against `entry.json`; the staging directory
  is then removed. A kept entry is read with the checks of §6.2 except the address, its `checksum`
  and `fingerprint` compared with the job's instead, and as `private` when the work directory passes
  the ownership check of §6.4, as it does for the user who ran the job. An output that could not be
  encoded at all (§6.4) has no staged entry: it is cloudpickled to `<work dir>/_outputs.pklz`, the
  trust `_job.pklz` already has there, and read back only after the same ownership check. A kept
  result serves the run that produced it; the next run computes the job again.
- **Lifetime.** After a publish with `stored=True` the default work directory is emptied except for
  `_committed.json`, which the runner removes once it holds the job's outcome. After a failure, or
  `stored=False`, it is kept and reported. `collect` removes default work directories older than the
  policy's work age (§9), which must exceed the longest run that reads a kept result.
- **Crashes.** An errored job writes `_error.json` into its work directory (§11), so the crash path is
  `work_dir(checksum, fingerprint, job.attempt) / "_error.json"`, which the submitter's message can
  name (`engine/submitter.py:491-494`) and `pydracli crash` can read.

## 6. Entry format

### 6.1 The hook

```python
class EntryFormat(Protocol):
    name: str          # the header's entry_format, e.g. "pydra-entry@1"
    entries: str       # the version directory, e.g. "results@1"
    def members(self) -> re.Pattern[str]: ...         # names an entry may hold
    def aside_names(self) -> re.Pattern[str]: ...     # names of entries this format set aside in place
    def stage(self, staging: StorageDir, checksum: str, key: KeyRecord, context: Context,
              outputs: Mapping[str, Any], task: Task | None, files_from: StorageDir | None,
              writer: str) -> str: ...                # writes the members; returns the output digest
    def context(self, entry: StorageDir) -> tuple[Context, bytes]: ...   # reads context.json; its bytes
    def read(self, entry: StorageDir, checksum: str, fingerprint: str,
             task: Task | None, trust: Trust) -> StoredEntry: ...
```

The store owns the layout, staging, rename, modes, quarantine and collection. The entry format owns
what is inside an entry. `read` raises `CorruptEntry` (quarantined), `MisfiledEntry` (a
`CorruptEntry`; quarantined), `EntryFormatUnknown` (another version's member, or a codec tag this
process has not registered; left in place, not served) or `EntryNotTrusted` (a pickle member that the
root's or entry's trust does not admit, §6.4; left in place, not served). `StoredEntry` carries the
outputs, the context, the key record, the output digest and the entry's `EntryRef`.

The default, `pydra-entry@1` in `results@1/`:

```
<fingerprint>/entry.json            format, checksum, key record or null, fingerprint, writer token,
                                    output digest, encoded outputs, each member's sha256 and size
<fingerprint>/context.json          format, the context
<fingerprint>/<sha256>.npy          one per array, its canonical `.npy` (below), named by its sha256
<fingerprint>/<sha256>.bin          one per bytes output
<fingerprint>/files/<set>/<path>    one directory per file output, named by its `file@1` digest,
                                    holding the set's files laid out as the producer had them
<fingerprint>/<sha256>.pkl          pickle members, under a root whose header allows them (§6.4)
```

`Result`, `_job.pklz` and `_error.pklz` are never written to an entry.

- **Canonical `.npy`.** An array member is the canonical `.npy` of RFC 0004 §6.2: format version 1.0,
  as `numpy.lib.format.write_array(..., version=(1, 0), allow_pickle=False)` writes it, with the dtype
  converted to little-endian and the data copied to C order. Its name is the sha256 of those bytes,
  which is RFC 0004's `npy@1` digest. Both proposals compute it with `pydra/utils/digest.py` (§13).
- **File outputs keep their keys.** A `FileSet` output is stored under one directory, `files/<set>/`.
  The files of a `File` or of a multi-file set (a NIfTI `.hdr`/`.img` pair) sit side by side there
  under their own names, and a `Directory` is `files/<set>/<its name>/` with its tree. Decoding then
  gives a value whose `FileSet.byte_chunks` keys (`utils/hash.py:702-707`) are the producer's:
  `"."` for a single file, `".hdr"` and `".img"` for the pair, paths relative to the root for a
  directory. `entry.json` records the keys, and `<set>` is RFC 0004's `file@1` of the recorded
  type tag and `[key, content sha256]` pairs, so the digest of a decoded output is read from the
  entry, not recomputed from its files.

### 6.2 Reading binds an entry to its address and to its members

The store opens the entry directory once, with `O_DIRECTORY | O_NOFOLLOW`, and does every read below
through that descriptor:

- reads `context.json`, recomputes the fingerprint and requires it to equal the directory name;
- reads `entry.json`, requires its `checksum` to give this `<cdir>`, its `fingerprint` to equal the
  directory name, its `format` to be one it reads, and `identity.check_key` to accept its key record;
- for each `$fileset`, requires `<set>` to be the `file@1` of its recorded type tag, keys and member
  digests, and the decoded value's `byte_chunks` keys to be the recorded keys;
- opens each member with `O_NOFOLLOW`, requires a regular file, and checks its size and sha256
  against `entry.json`. File members are checked when the entry is read, not when the path is first
  used, so a reader never hands out a path to an unchecked file. The policy's `verify_files="size"`
  (§9) narrows the check to sizes for a `private` root. It is the owner's assertion that their own
  processes do not change a member in place, and it is refused for a `shared` root and unless
  `durability="member"`, because a crash under `"entry"` or `"none"` can leave a member with its
  recorded size and stale or zero-filled blocks;
- loads arrays with `numpy.load(..., allow_pickle=False)`;
- recomputes the output digest over the encoded outputs and requires it to equal the recorded one.

A committed entry is expected in the `sealed` mode when the header has one, and in the `entry` mode
otherwise; an entry in the `entry` mode under a `sealed` policy is one whose writer was killed before
the seal (§5.1 step 7). An entry whose modes differ from the expected ones is resealed when its reader
owns it, and read as it is otherwise. A failed check quarantines the entry (§6.3); an entry is never
served from a partial read. An `OSError` on read or write is raised as `StoreUnavailable` and moves
nothing.

### 6.3 Quarantine

Quarantine moves an entry aside so the address heals. Two readers of one corrupt entry, or a reader
and a writer, can race, so the move is a compare-and-move, and it is fenced:

1. Rename the entry to `quarantine/<uuid>`, relative to descriptors, where `<uuid>` is derived from
   the inode (`st_ino`, never `st_dev`, which differs between NFS clients) of the directory the
   mover read and found corrupt. The first mover moves that directory. A late mover that found the
   same directory corrupt, when the address now holds another, valid directory, renames that one
   onto the existing, non-empty `quarantine/<uuid>`, and the rename fails and moves nothing. No lock
   is taken. A store requires stable, non-zero inode numbers to publish, because its identity checks
   rely on them. Where a filesystem reports `st_ino == 0`, the quarantine name is a random UUID, which
   claims no identity, and the protection degrades to the fence of steps 2 and 3. An empty directory
   at the address, which a rename would replace rather than fail on, is removed (`rmdir`) before the
   move.
2. Re-verify the moved copy against the address it came from (§6.2).
3. If it verifies, it was not the corrupt entry: another writer committed a valid one at the address
   between this reader's check and its rename. Rename it back without replacement. If the address is
   occupied again, the moved copy stays in `quarantine/`. That is not only a cost in space: when the
   two entries' digests differ, a writer that was told `stored=True` for the moved entry would see
   its outputs replaced at the address by others. The inode-derived name of step 1 keeps a valid
   entry from being moved in the first place; this step remains for the fallback.
4. Otherwise stamp its mtime (so `collect` ages it from the move) and log the address and the check
   that failed.

**A reader that cannot move the entry.** Moving a directory into another parent needs write
permission on the directory itself. A user who does not own a corrupt entry cannot move it to
`quarantine/` and cannot change its mode. The store then renames it within its own parent to
`q-<uuid>`, with the UUID derived from the inode as in step 1, which needs write permission only on
the parent, and which is not a fingerprint and is never served. The parent is writable by other
users only when the `container` mode makes it so (§3.2, §9). The same fence applies. `collect` removes `q-` names when it can and logs the owner when
it cannot. If the parent is not writable either, the entry stays, the read raises `CorruptEntry` with
the path and its owner, and the job runs and keeps its outputs as `stored=False` (§5.4).

`store.evict(entry: EntryRef, expected_digest: str)` uses the same steps and restores the moved copy
when its digest is not `expected_digest`, so an evict never removes an entry committed after its
caller read the old one.

### 6.4 Encoding and trust

**Trust is a property of each root**, declared by whoever opens it, never read from the root:

```python
Trust = Literal["shared", "private", "trusted"]
PortableStore(root, trust="shared", readonly=[Root(path, trust="shared"), ...],
              legacy=[Root(path, trust="shared"), ...])
```

- **`shared`** (the default for every root passed explicitly, including each `readonly` and `legacy`
  root). A `.pkl` member is never opened: an entry holding one raises `EntryNotTrusted` and is not
  served. A writer never writes one.
- **`private`.** The root must be owned by the current user and not writable by group or others. This
  is checked when the root is opened (`fstat` of its descriptor) and, before any pickle member is
  opened, on every directory from the root to the entry and on the member itself, through the
  descriptors of the walk. A root at `0o755` whose shard directory is still group-writable from an
  earlier policy fails the second check for every entry below it. A failure raises `EntryNotTrusted`
  for that entry. A `private` root whose header has `"pickle": true` may hold pickle members.
- **`trusted`.** No ownership check: the operator asserts that everyone who can write the root is
  trusted with code execution, as `PickleStore` assumes today (`engine/store.py:431-434`). Writers
  pickle only when the header allows it, so a `trusted` open of a root created with
  `"pickle": false` behaves like `shared` for writes and like `trusted` for reads.
- **Windows.** `fstat` gives no owner and no group or other bits there, so `private` cannot be
  checked from modes. On Windows `private` is accepted only for a root under the user's local
  application data directory (`%LOCALAPPDATA%`), whose default ACL admits only the user, and is
  refused elsewhere with a message to declare `trusted` or `shared`. pydra does not read ACLs. A
  `shared` root on Windows raises `PlatformUnsupported` (§9).
- `Submitter` opens its default cache directory, which is the user's, as `private`, so a pydra user
  whose task returns an arbitrary Python object keeps working. A cache pointed at shared storage is
  `shared` unless declared otherwise, and the roots in `readonly_caches` are `shared` unless each is
  declared otherwise.

**The default encoding** is JSON plus members:

| Output type | Encoded as |
|---|---|
| `None`, `bool`, `int`, `str` | JSON |
| `float` | JSON; NaN and infinities as `{"$float": "nan" \| "inf" \| "-inf"}` |
| `complex` | `{"$complex": [re, im]}` |
| `bytes` | `{"$bytes": "<sha256>"}` and a `.bin` member |
| `list` | JSON array; decoded to the field's declared type (`utils/typing.py:87`, `TypeParser`) |
| `tuple` | `{"$tuple": [...]}`, so a tuple that no declared type fixes (an untyped or `Any` field, or one inside an untyped container) comes back a tuple; a declared type still decodes it |
| `set`, `frozenset` | `{"$set": [...]}`, elements sorted by their canonical JSON |
| `dict` with `str` keys | JSON object; a key starting with `$` is written `$$...` |
| `dict` with other keys | `{"$dict": [[key, value], ...]}`, sorted by the canonical JSON of the key |
| `datetime`, `date` | `{"$datetime": "<ISO 8601>"}`; a naive datetime is refused in a `shared` root |
| `pathlib.Path` that is not a `FileSet` | `{"$path": "<str>"}`; the path is a value, its file is not stored. A path to an existing file or directory inside the job's work directory (§5.4), which is emptied once the entry is published, is stored instead as a file set, `{"$path": {"$fileset": "generic/file" \| "generic/directory", "files": ...}}`, and decoded to the path of the entry's copy |
| `numpy.ndarray` exactly (not a subclass), whose dtype holds no references (not object, not `StringDType`, no such field in a structured dtype) | `{"$array": "<sha256>"}` and a canonical `.npy` member (§6.1). Subclasses (`numpy.ma.MaskedArray`, `numpy.matrix`) and reference dtypes have no default codec: they are pickled under a root that allows it and are otherwise unencodable, as in RFC 0004 they are `py@1` |
| numpy scalars | `{"$npscalar": ["<dtype>", "<value>"]}`: the dtype's `str` in little-endian form, and the value as a string: a decimal for integers, `float.hex` for floating types up to `float64`, `[re, im]` of those for complex, `"true"`/`"false"` for `bool_`, the integer count for `datetime64` and `timedelta64`, whose unit is in the dtype. `np.float32(1)` is `["<f4", "0x1.0000000000000p+0"]` and `np.float64(1)` is `["<f8", ...]`. `longdouble` and its complex form are unencodable, because `float.hex` would round them |
| `File`, `Directory`, any `FileSet` | `{"$fileset": "<registered type tag>", "files": [[<key>, "files/<set>/<path>"], ...]}`, sorted by key; decoded to paths inside the entry (§6.1) |
| `Absent` (RFC 0002 §2.3(a)) | `{"$absent": {"cause": ..., "source": ...}}` |
| `attrs` classes, `enum` members, other types with a codec | `{"$codec": "<tag>", "value": ...}` through `register_codec(type, tag, encode, decode)` |

- **One tag registry.** `$fileset` and `$codec` tags live in one registry in `pydra/utils/digest.py`,
  RFC 0004's I0, which `register_codec` here and `register_digest` there both fill, so a tag means the
  same type in an entry and in an input digest. It lands with I0, before P3 (§13). An unregistered
  `FileSet` subclass is unencodable as an output (`UnencodableOutput`, below); as an input RFC 0004
  digests it as `py@1`, so it keys as today and never reads as an entry member.
- **Decoding never imports.** A `$fileset` type tag and a `$codec` tag are looked up only in that
  registry as this process has filled it, never resolved as a module or class name. One exception:
  to map a `$fileset` tag that is not yet registered to a class, the store may import the installed
  `fileformats.*` namespace packages, and nothing outside `fileformats.*`; it never imports code that
  the entry itself names.
  An unknown tag raises `EntryFormatUnknown`; the entry is not served. This holds when `task` is
  `None`. A codec's `decode` builds a value from JSON and members only. Codecs for the standard
  library types above, `fileformats` types, `Absent`, numpy scalars, `attrs` classes and `enum`
  members are registered explicitly, by pydra for the types it defines and by packages for theirs.
- **`$digest` is reserved.** RFC 0004 §6.2 uses `{"$digest": ...}` for a nested registered digest in
  an input's digest form; no output codec emits it, and a decoder that meets it in an entry raises
  `CorruptEntry`.
- **A file set is never pickled**, even under a root that allows pickles. A pickled file set records
  paths, not content, and would be served after its files are gone. So a `FileSet` whose tag cannot
  be resolved back to a class and is not registered by the user, a mock file set, or one without
  `mime_like` raises `UnencodableOutput` at publish, and its value is kept as `_outputs.pklz` with
  `stored=False` (§5.4). On read, once every member's hash has verified, a file set whose recorded
  class or format does not match its members raises `EntryFormatUnknown` and the entry is left in
  place, not `CorruptEntry`: its bytes are intact and another process's registry may read it.
- **An output with no codec, checked before the run.** When a job is handed out to a store whose
  writes cannot pickle, the runner checks each declared output field's type against the registry.
  A field whose declared type has no codec raises `UnencodableOutput` then, naming the field and the
  type, before any computation.
- **An output with no codec, found at publish.** A field typed `Any`, or a value whose runtime type
  differs from its declaration, can only be found at publish. Then the job does not error:
  `Committed.stored` is false, the outputs are kept in the work directory as `_outputs.pklz` (§5.4),
  and a warning names the field and the type. The `Result` in the executing process holds them, and a
  runner in another process, such as one behind a Slurm job, reads them from the work directory
  (§7.2). Downstream jobs of the same run use the outputs. The next run computes the job again. Under
  a root that allows pickles, the value is cloudpickled into `<sha256>.pkl`, tagged in `entry.json`
  with the cloudpickle version and the Python major.minor, and read back only by a matching
  interpreter.

## 7. Reuse

### 7.1 The hook

```python
class Verdict(Enum):
    SERVE = "serve"      # return the stored result
    VERIFY = "verify"    # run the job, compare its outputs with the stored ones, then publish
    REFUSE = "refuse"    # do not serve this entry; run the job
    SKIP = "skip"        # do not serve this entry; consider the next candidate

@attrs.frozen
class Candidate:
    entry: EntryRef
    context: Context           # the stored context
    context_digest: str        # the sha256 of the context.json bytes ranked
    rank: int                  # 0 is the narrowest admission (an equal fingerprint)
    admitted_by: str           # e.g. "exact", "consistent", "equivalent", "tolerant"

class Reuse(Protocol):
    def bind(self, store: PortableStore) -> None: ...
    def rank(self, current: Context, stored: Sequence[tuple[EntryRef, Context, str]],
             task: Task | None) -> list[Candidate]: ...
    def verdict(self, candidate: Candidate, current: Context, task: Task | None) -> Verdict: ...
    def served(self, candidate: Candidate, current: Context, task: Task | None,
               entry: StoredEntry) -> None: ...                  # the reuse record
    def compare(self, candidate: Candidate, stored: StoredEntry, fresh: Mapping[str, Any],
                fresh_digest: str, task: Task | None) -> None: ...   # before the fresh commit

@attrs.frozen
class Hit:     result: Result
@attrs.frozen
class Verify:  candidate: Candidate
@attrs.frozen
class Miss:    pass
Lookup = Hit | Verify | Miss

class PortableStore:
    def find(self, job: Job) -> Lookup: ...
    def lookup(self, checksum: str, context: Context, task: Task | None, stored=None) -> Result | None: ...
```

`find(job)` is the lookup. It takes the job, so the identity and the attempt are reachable, and its
result says whether to verify. `lookup`, the `ResultStore` method (`engine/store.py:389-404`), is
kept as an adapter that returns the `Hit`'s result and `None` otherwise; a caller that holds a job
and finds `find` on the store (as `Submitter` finds `read_succeeded`, `engine/submitter.py:774`)
calls `find` instead, and sets `job.verify` from a `Verify`. In pydra those callers are `Job.run`
under the lease (`engine/job.py:418-423`) and the runner's `ready()` (§13).

- **Ranking.** `rank` returns the admitted candidates, best first. The store reads only
  `context.json` to rank.
- **Lookup.** The store takes the first candidate and asks for its verdict.
  - `SERVE`: the store opens the entry's descriptor, reads `context.json` through it and requires the
    sha256 of its bytes to equal `context_digest`. If it differs, the entry changed between ranking
    and reading (an evict, a quarantine and a new commit with other recorded-only fields), and the
    store ranks again. The verdict therefore always applies to the context of the members it reads.
    Then the entry is read (§6.2), `served` is called, and the result is returned. A corrupt candidate
    is quarantined and the next one is taken.
  - `VERIFY`: `find` returns `Verify(candidate)`, and the caller marks the job to verify against it.
  - `REFUSE`: `find` returns `Miss` and the job runs. It ends the lookup, because a reason to
    refuse, such as a licence recorded in the stored context, usually applies to every candidate of
    the job.
  - `SKIP`: the next candidate is considered.
- **The reuse record.** `served` is called before the result is returned. If it raises, the result is
  not served and the job runs, so no result is served without its record. The `Reuse` object is
  picklable and runs in whichever process looks up: the runner in `ready()`, or a worker under the
  lease in `Job.run`. A record it writes must be durable in that process.
- **The verify mark** is `job.verify: Candidate | None`. It is set from the `Verify` that `find`
  returned, in the runner or under the lease, and travels with the job to the executing process beside
  the checksum and context, which RFC 0002 §3.4 has to name (§13). `publish(..., verify=job.verify)`
  reads the candidate and calls `compare` with the stored entry and the job's fresh outputs and
  digest, after staging and before the commit hook and the rename (§5.1 step 3). What a comparison
  records, and where, is the hook's; it may write a sidecar (§7.4) or hand a record to the commit
  hook. If `compare` raises, the job errors and nothing is committed.
- **Verify at rank 0.** A rank-0 candidate is at the job's own address, so the fresh entry cannot be
  committed beside it. `compare` still runs against the fresh outputs, which is the point of the
  verify; the rename then fails as in §5.1 step 8, and the result is as in §7.3.
- **`bind`** gives the hook the store, so it can reach `store.sidecar(name)` (§7.4). The store calls
  it once at construction and again after unpickling in a worker.

### 7.2 Reading one's own result

A job's own result is not a reuse, and is not read through the hook. After `publish`, the executing
process holds `Committed`, and the job's `Result` is built from it. Every later read of that result
goes by `store.read(entry: EntryRef, task)`, which verifies the entry (§6.2) and calls neither `rank`,
`verdict` nor `served`:

- `Job.result()` reads by `job.committed` when it is set, and otherwise by
  `<work dir>/_committed.json` when that exists (`engine/job.py:617-624` today reads through
  `lookup`). A `_committed.json` that names an `EntryRef` is read by `store.read`; one that names the
  kept entry or `_outputs.pklz` is read from the work directory (§5.4). The executing process writes
  it to a temporary name in the work directory and renames it into place, so a reader sees no file or
  a whole one. When it is absent after the visibility retry, `Job.result()` reads the entry at the
  job's own address, under the lookup context's fingerprint, by `store.read`, which recovers every
  `stored=True` job published there; only when that read also finds nothing is the job reported
  `Failed`, naming the work directory;
- RFC 0002's `complete(job, outcome)` holds the `Result` on the job (RFC 0002 §3.1), so readers in
  the runner's process use it without a store read. RFC 0002's `Outcome` is `Result | Failed |
  NotRun` and has no place for the `Committed`; this proposal puts it on the `Result`
  (`Result.committed`), which leaves the union as it is and which `complete` has to keep (§13);
- an executor that reports from another process, such as the Slurm and SGE adapter, which reads the
  result with `job.result()` after the job returns `True` (RFC 0002 §3.3), reads through
  `_committed.json`, not by `lookup`, so a job whose outputs were not stored still reports a
  `Result`. The adapter's visibility retry applies to `_committed.json`;
- lazy fields that read an upstream job (`engine/lazy.py:189`), `runner.result()` (RFC 0002 §3.1) and
  the submitter's final read (`engine/submitter.py:503-513`) go through `Job.result()` and so read the
  same way, including for an outer job whose outputs were not stored.

`lookup` is therefore called only where a reuse can happen: in `ready()` (through `read_succeeded`, or
for every job under `lookup_before_dispatch`, RFC 0002 §3.1) and under the lease in `Job.run`.

`PortableStore` returns a `Result` with `outputs` decoded, `errored=False`, `context` the stored
context, `cross_context` and `reused_in` set as today for a hit from another context
(`engine/store.py:497-499`), `cache_dir` the entry's path (read-only), `task` the current job's task,
and `runtime` the recorded runtime when the entry format stores one, otherwise `None`. `FileSet`
outputs are paths inside the entry; an `evict` of that entry moves them, so a caller that keeps paths
across runs copies the files.

### 7.3 A lost race

When `publish` returns `won=False`:

| Case | The job's result | Recorded as |
|---|---|---|
| `stored=True` (equal digests) | its own outputs, read by `EntryRef` from the committed entry | an execution; `served` is not called |
| unequal digests, `replace=False`, and the hook gives `SERVE` for the committed entry as a rank-0 candidate | the committed entry's outputs | a reuse; `served` is called |
| unequal digests and `REFUSE` or `SKIP`, or `replace=True`; or the entry cannot be read (`committed_digest=None`) | its own outputs, uncommitted (`Result.stored=False`), read from the kept entry (§5.4) | an execution that was not stored; no error |

Equal digests are the common race, two writers of a deterministic task, and are not a reuse. The
last row means a job whose address holds an entry this user may not use still completes, every run,
from its own outputs. That repeats the computation on each run, which is the cost of the refusal; it
never raises. A publish with `replace=True` (§10) never serves the occupant: the hook is not asked,
and the job's result is its own outputs, stored only when the digests are equal.

### 7.4 Compatibility with `ReusePolicy`, read-only roots, the commit hook and sidecars

**`ReusePolicy`.** `PortableStore(reuse=...)` takes the same values as `PickleStore`
(`engine/store.py:446-453`, `:270-274`). A `ReusePolicy` is wrapped as a `Reuse`: `rank` puts the
current fingerprint first, then every stored context that `accept`s (`:215`) in path order, which is
the order `PickleStore.lookup` uses (`:478-501`); `verdict` is always `SERVE`; `served` sets
`Result.cross_context` and `Result.reused_in` as today (`:497-499`) and writes nothing;
`compare` does nothing. The declared-version check (`engine/store.py:291-337`) reads
`stored_contexts`, which `PortableStore` implements over `context.json`.

**Read-only roots.** `readonly=[Root(...)]` are searched after `<root>`, as `readonly_caches` are today
(`engine/store.py:485-490`). Each must carry a header matching the store's (§3.2), is read under its
own declared trust (§6.4), and is never written, collected or quarantined into: a corrupt entry there
is skipped and logged. Candidates from all roots are ranked together, and `EntryRef.root` reaches the
hook, so a hook can refuse or record a hit by its root.

**The commit hook.** `PortableStore(on_commit=callable)` is called with the staging directory, the
checksum, the key, the context and the output digest after the members are written and after
`compare`, before the rename (§5.1 step 4). It is the place for a provenance record that must exist
before the result is visible. If it raises, nothing is committed. It must not add members to the staging directory: the member list is in `entry.json` already, so the store refuses an entry the hook added a file to (a hook writes through `store.sidecar(name)` or the audit directory). It may run more than once for one entry (a restage, a lost race), so it must be idempotent.

**Sidecars.** A hook that keeps store-level records, such as verify comparisons, writes them under
`sidecars/<name>@<major>/` (or the directory the header names for it) through `store.sidecar(name)`,
which returns a `StorageDir` with the same rules as the rest of the root. `collect` never touches
sidecars.

## 8. Migration

### 8.1 From `PickleStore`

```python
def migrate(source: os.PathLike, target: PortableStore, *, trust_source: bool,
            on_slot: Callable[[MigratedSlot], None] | None = None) -> MigrationReport: ...
```

`pydracli store migrate <cache_root> <root> --trust-source` wraps it. A downstream system calls the
function from its own recorded step and gets one `MigratedSlot` per source slot through `on_slot`.

- It walks `<cache_root>/<checksum>/<ctxkey>/` by descriptor, without following symlinks.
- It takes each slot's `PickleStore` lease (`engine/store.py:533-534`, `:579-585`) with one
  non-blocking attempt. A slot whose lease is held is being written; it is skipped and reported, and a
  later call migrates it.
- It skips a slot with an errored `_result.pklz`, without one, or without `_context.json`
  (`engine/store.py:509-513`).
- It skips a slot whose context has `unknown`, the per-process nonce that `job_context` adds when a
  package identity is unknown (`engine/context.py:520-521`). Such a result is reused by no other
  process today, so migrating it would only fill a slot no lookup reaches.
- It unpickles `_result.pklz`, which needs `trust_source=True` because that executes the old cache's
  code. The user already trusted that cache by reading from it.
- It publishes the outputs under the stored context, through the target's identity and entry format,
  with the slot's files as `files_from`; the `ctxkey` is discarded. An output the target cannot encode
  is reported and leaves the slot unmigrated.
- It copies no `_job.pklz`, `_error.pklz`, lease or `.pydra-succeeded/` record.

No job is recomputed: under `pydra-identity@1` checksums are unchanged, and the outputs are decoded
and encoded, not produced. A root migrated under `@1` raises `StoreLayoutMismatch` under `@2` (§4).

**Scope under `pydra-identity@2`.** `migrate()` has no `Job` to build an `@2` record from, and an old
automatic checksum is not an `@2` checksum, so under `@2` it migrates only slots whose checksums were
supplied: it is called with `supplied_checksums=True`, the caller's assertion, and publishes each slot
with the key `{"scheme": "pydra-identity@2", "supplied": <checksum>}`. Without that flag it refuses
an `@2` target, naming the identity. The assertion is accepted only for a cache keyed before
`@2`: one whose `.pydra-identity` marker (RFC 0004 §9.1) is absent or names `pydra-identity@1`. A
slot of such a cache wrongly asserted to be supplied is harmless: its checksum is never produced under
`@2`, so no lookup reaches it. A cache whose marker names `pydra-identity@2` (every cache a current
`Submitter` writes) refuses `supplied_checksums=True`, pointing to `legacy=`: its checksums are
automatic `@2` digests, which lookups do reach, so a supplied-key record would be served in place of
the job's own record and would misstate its provenance. Rebuilding the record from `_job.pklz` is not
an option, since a `Submitter` writes that file only for remote or failed jobs. Such a cache is read
through `legacy=` below, which publishes each result with the looked-up job's own record. A marker
naming an unknown or newer scheme is refused, and so is an `@2` cache migrated into an `@1` target.
Under any other identity (`@1` kept explicitly, or a downstream one, §8.2) it behaves as above.

A run can also read an unmigrated cache in place: `PortableStore(legacy=[Root(cache_root,
trust="private")])`. Legacy roots are read only when declared `private` or `trusted`, and a `private`
one passes the ownership check of §6.4 on the cache root and on each slot directory it reads. A hit
there is published into `<root>` under the stored context's fingerprint, not the current one, so the
old cache is read once per result. Legacy roots are read when a lookup ends with nothing served or
verified (no candidate, every candidate skipped or unreadable), but not after a `REFUSE`, and once per
lookup. A hit is published with the current job's key record, so under `@2` a `legacy=` root serves
jobs whose checksums name its slots: supplied checksums in an `@1`-era cache, and every checksum in an
`@2`-marked one; for every other job it costs one `lstat` and finds nothing.

### 8.2 From an existing store already in this shape

A store laid out as `<version dir>/<shard>/<checksum>/<fingerprint>/` with its own members can adopt
D2 without rewriting an entry, provided its checksums, its fingerprints and its encoded outputs stay
the same. The steps:

1. Implement `StoreIdentity` with its key record, its context and its fingerprint (§4), and
   `EntryFormat` over its members, declaring the names under which it set entries aside in place as
   `aside_names` (§6.1), so `collect` treats them as its own.
2. **Fence the writers that predate the header.** Code that predates `pydra-store.json` never reads
   it, so a header alone does not stop it writing. Adoption therefore requires a new version
   directory: stop the writers, rename `<old version dir>` to the new `entries` name in one directory
   rename (entries are untouched), and write the header naming it. Old code that runs afterwards
   finds no version directory of its own name and creates a new, empty tree, which the adopted store
   never reads; the store logs a warning when it sees that directory.
3. In the header, name the existing staging, quarantine and store-level record directories (`dirs`,
   §3.2), so nothing else is moved, and state the modes the existing tree has: `sealed` equal to the
   mode of its committed entry directories and `container` and `side` to those of the directories that
   hold them. Entries then match the header and are not resealed on read; a header whose `sealed`
   differs would have each owner reseal every entry on first read. Records that old code adds to a
   sidecar directory after adoption are in the format the adopted hook reads, so they are not lost.

Each entry still verifies against its address, its key record and its context (§6.2, §4), so a
translation that went wrong is caught on read and quarantined. Only a change to checksums,
fingerprints or the output encoding forces a recompute, because it creates a new address space.

## 9. Storage rules and platforms

- **Descriptors.** Every directory below `<root>` is opened with `O_RDONLY | O_DIRECTORY |
  O_NOFOLLOW` relative to its parent's descriptor; files with `O_NOFOLLOW` (and `O_EXCL` when
  created); `rename`, `mkdir`, `unlink`, `rmdir`, `link` and `lstat` with `dir_fd`. `<root>` itself is
  opened as given. These are in `os.supports_dir_fd` on Linux and macOS. Listing uses
  `os.scandir(fd)`.
- **Wrong kinds.** A symlink, or an object of the wrong kind, where the store expects a directory or
  member is quarantined (§6.3); one that cannot be moved raises `WrongKindError`, naming the path and
  its owner.
- **The policy.** `StorePolicy(file_mode, entry_mode, sealed_mode, container_mode, side_mode,
  staging_mode, durability, verify_files, ages)`. Modes are recorded in the header (§3.2) and set
  through descriptors, never from the umask. Today `atomic_write` creates files `0o644` adjusted by
  the umask (`utils/general.py:65`, `:89`). `entry_mode` must include owner write and search (§5.1
  step 5); `file_mode`, `entry_mode` and `sealed_mode` must not include group or other write, which is
  what keeps a committed entry unchangeable by anyone but its owner (§2). The default
  (`0o444`, `0o755`, none, `0o755`, `0o755`, `0o700`) gives a root that one user writes and everyone
  reads. A root that a group writes needs group-writable containers and side directories, for
  example `file_mode=0o440, entry_mode=0o2750, sealed_mode=0o2550, container_mode=0o2770,
  side_mode=0o2770`: any member of the group can create shards, checksum directories and staging
  directories and can move a corrupt entry to `q-<uuid>` (§6.3), and each entry is read-only after
  its seal. `durability` is `"member"` (default: every member, the directory and the parent
  `fsync`ed), `"entry"` (one flush before the rename) or `"none"`. `verify_files` is `"digest"`
  (default) or `"size"`, which is accepted only for a `private` root with `durability="member"`
  (§6.2).
- **Collection.** `store.collect()` removes, by descriptor, `staging/` directories, default work
  directories, quarantined objects and `q-` and aside names older than the policy's ages. A
  quarantined object is aged from its move (§6.3 step 4). A writer with a large copy in progress
  touches its staging directory at least once per quarter of the staging age, so a live stage is never
  collected; a writer whose staging directory vanished stages again once, then raises. `collect` never
  removes an entry; retention of entries is an open question (§13). `remove_from_cache`
  (`engine/store.py:135-208`) supplies the NFS and Windows retry behaviour.
- **NFS.** `rename` within one export is atomic; `staging/` is under `<root>` for that reason. A lost
  rename reply is detected by the writer token (§5.1 step 8). Silly renames (`.nfs*`) are recognised
  as `remove_from_cache` does (`engine/store.py:55`, `:99-112`).
- **Windows.** `os.supports_dir_fd` is empty on Windows and `O_NOFOLLOW` does not exist, so the
  descriptor rules cannot hold. A `PortableStore` on Windows opens only as `private` (under
  `%LOCALAPPDATA%`, §6.4) or `trusted`, and `shared` raises `PlatformUnsupported`, a new subclass of
  `StorageUnsupported` (`utils/storage.py:40`), whose message says that the platform, not a storage
  primitive, is the reason. Under `private` and `trusted` the store uses paths and checks before each
  step that no component below `<root>` is a reparse point (`st_file_attributes &
  FILE_ATTRIBUTE_REPARSE_POINT`). That check is not race-free, which is acceptable only because no
  other user can write such a root. A directory rename fails while a file in it is open (sharing
  violation); it is retried as `remove_from_cache` retries (`engine/store.py:50-54`, `:159-208`).
  Committed modes are applied as the read-only attribute on files only.

## 10. Rerun, and the executor's contract per store

**`rerun=True` evicts and replaces.** A rerun takes the de-duplication lease (a resumable task's, or
this one's when `dedupe` is on), computes the job and publishes with `replace=True`: just before the
rename, the store evicts the entry at the address with `evict(entry, expected_digest)` (§6.3), where
`expected_digest` is the digest it read when the job started, and then renames. If another writer
commits between the evict and the rename, or the evict cannot move the entry (the rerun's user does
not own it, or the parent is not writable), the rerun loses the race. It then never serves the
occupant: its result is its own outputs, stored only if the digests are equal and otherwise kept as
`stored=False` (§5.4, §7.3). A rerun therefore always returns a fresh computation, and the replaced
entry goes to `quarantine/`, from which `collect` removes it.

**What an executor does, by store** (RFC 0002 §3.4 keeps the table for `PickleStore`):

| Step | `PickleStore` | `PortableStore` |
|---|---|---|
| Lease | always, for every leaf job (`engine/job.py:415`) | only with `dedupe`, a resumable task, or a rerun |
| Hit under the lease | `lookup` (`engine/job.py:418-424`) | `find(job)` through the reuse hook; a `Verify` sets `job.verify` |
| Where the job runs | the slot | `work_dir(checksum, fingerprint, job.attempt)` |
| Commit | `commit` under the lease; raises `LeaseLost` | `publish`; never raises `LeaseLost`; returns `Committed` |
| Success record | written last; withdrawn before a rerun (`engine/store.py:587-620`, `engine/job.py:427`) | none: the entry is the record |
| Failure | `_error.pklz` and an errored result in the slot | `_error.json` in the work directory; nothing in the store |
| Reporting | the `Result` | the `Result`, carrying the `Committed`; from another process, through `_committed.json` (§7.2) |

The runner writes workflow jobs, the executor leaf jobs, as RFC 0002 §3.4 says; both write through
`publish` under a `PortableStore`. The table is for leaf jobs; the workflow-job steps that RFC 0002
§3.1 gives the runner take the `PortableStore` form listed in §13.

**Provenance.** RFC 0002's audit hook emits `reused` for jobs settled from the store and `executed`
before a commit (RFC 0002 §3.4). Those are pydra's records. `Reuse.served` is a downstream system's
own record of the same event, and the default wrapper's `served` writes nothing, so by default each
reuse is recorded once. A system that records in `served` turns off the audit's `reused` messenger,
or reads the `EntryRef` the audit record carries to join the two.

## 11. Failures

A `PortableStore` stores no failure.

- An errored job publishes nothing. Its traceback goes to `<work dir>/_error.json` (JSON, not a
  pickle), where `record_error` writes `_error.pklz` today (`engine/job.py:448-454`,
  `engine/result.py:415-416`). `pydracli crash` reads it (`scripts/cli.py:31-33`), and the submitter's
  crash message names it (§5.4).
- `_job.pklz` is written to the work directory, never to the entry, when it is persisted
  (`engine/job.py:360-361`, `:395-398`).
- **Across runs.** Today the next run finds an errored result and reruns the job
  (`engine/job.py:418-424`, `:625-626`). Under `PortableStore` the next run finds no entry and runs the
  job. The observable behaviour is the same: an errored job runs again, a successful one does not.
- **Within a run.** `Job.errored`, RFC 0002's `Failed` outcome and the error summary on the submitter
  (0001 §3.1) are in memory and unchanged; downstream elements of an errored element stay unrunnable.
- A job whose result was not stored (§6.4, §7.3) is not a failure. Its downstream jobs run, reading
  its kept outputs in whichever process they run (§5.4, §7.2).

## 12. Changes in revisions 2 and 3

Revision 3 answers the first re-review:

| Finding | Resolved in |
|---|---|
| NI-1 `stored=False` across processes | §5.1 steps 8-9 (staged entry kept, `Committed.kept`, `_committed.json`); §5.4 (kept entry or `_outputs.pklz` in the work directory, read under the ownership check; work directory lifetime); §6.4; §7.2 (`Job.result()`, the Slurm and SGE adapter, `runner.result()`, the submitter's final read) |
| NI-2 one directory mode | §3.2, §9 (`file`, `entry`, `sealed`, `container`, `side`, `staging`; no group write on entries; a group-written example); §5.1 steps 5-7 (seal after the rename) and `StoreNotWritable`; §6.2 (expected modes, reseal after a kill before the seal); §6.3; §8.2 step 3 |
| NI-3 lookup calls carry no job | §4 (`Job.context` is the identity's lookup context; the calls it reaches); §5.2, §7.1 (`find(job)` returning `Hit`, `Verify` or `Miss`; `lookup` as an adapter); §5.4 (`work_dir(job)`, lookup fingerprint, `Job.cache_dir`) |
| I13 residual | §7.3, §10 (`replace=True` never serves the occupant) |
| RFC 0002 requirements | §13, one list; §7.2 no longer cites RFC 0002 §3.1 for a field it does not have |
| nit: `verify_files="size"` | §6.2, §9 (owner's assertion; requires `durability="member"`) |
| RFC 0004 §9.2 and its review | §2, §4 (checksums unchanged only under `pydra-identity@1`; `@2` default after RFC 0004; `@1` roots under `@2`); §8.1 (`migrate()` and `legacy=` under `@2`, `supplied_checksums`); §6.1 (canonical `.npy`; file outputs under `files/<set>/` keeping `byte_chunks` keys); §6.2 (set digest and keys checked on read); §6.4 (`$array` for exact reference-free `ndarray` only, `$npscalar`, one tag registry with RFC 0004 I0); §13 (P3 after I0, P7 after I1 and I8) |
| NI-4 work directory readers (second re-review) | §5.4 (readable by the runner and downstream processes; shared storage for shared roots and multi-node executors; checked at hand-out); §7.2 (fallback to `store.read` at the job's own address); §5.1 step 9, §7.2 (`_committed.json` written by temporary and rename) |

Revision 2 answered the first review:

| Finding | Resolved in |
|---|---|
| B1 hard-linked file members | §5.1 step 1 and notes (copy or reflink, never link; symlinks and outside files copied; special files refused); §6.2 (every member hashed on publish and checked on read; `size` only for private roots) |
| B2 trust per store | §6.4 (trust per root, `readonly` and `legacy` default `shared`, ownership checked on every directory to the entry and on the member); §8.1 |
| B3 unservable occupant | §5.1 step 8, §7.3 (own outputs returned uncommitted; equal digests are an execution, not a reuse); `Committed` carries both digests |
| B4 context and key | §4 (`StoreIdentity`: key record, context in two phases, fingerprint); `publish` and `stage` take the key |
| I1 verify | §7.1 (`compare` before the rename with the fresh outputs; `job.verify` travels with the job; rank 0) |
| I2 reads are not reuses | §7.2 (`store.read(EntryRef)`; where `lookup` is still called) |
| I3 quarantine race | §6.3 (re-verify the moved copy and restore; `evict` with an expected digest) |
| I4 non-owner quarantine | §6.3 (`q-<uuid>` beside the entry; the job still completes) |
| I5 modes after rename | §5.1 steps 2, 5, 6, 8 (modes before the rename; parents made by rename; writer token); §6.2 reseal |
| I6 verdict and members | §6.2 (one descriptor); §7.1 (`context_digest`, re-rank) |
| I7 header write | §3.2 (temporary, `fsync`, `link`) |
| I8 `recorded_only` | §3.2, §4 (in the header; mismatch raises) |
| I9 checksum names | §3.1 (`<cdir>` grammar, `h-<sha256>`, hashed shard, `prefix2` restricted) |
| I10 scope of `private` | §6.4 (walk and member checks; Windows; decoding never imports) |
| I11 unencodable after the run | §6.4 (declared types checked when handed out; late failures keep the outputs) |
| I12 work directories | §5.4 (`attempt` on the job; the `work_dir` contract; crash path) |
| I13 rerun and RFC 0002 | §10 (evict and replace; per-store executor table; one reuse record); `Absent` codec in §6.4 |
| I14 de-duplication lease | §5.3 (optional, default off for shared roots; anonymous holder; path-based, stated) |
| I15 adoption | §8.2 (new version directory fences old writers; header names directories; aside names) |
| I16 migration | §8.1 (a function; source leases; `unknown`; legacy hits under the stored fingerprint); §7.2 (`Result` fields) |
| N1, N2 | re-pinned to `3f734f9a`; line references rechecked |
| N3 | the default-store question is §13 |
| N4 | `PlatformUnsupported` (§9) |
| N5 | `entry.json` stores the key record; `check_key` (§4, §6.2) |
| N6 | `trusted` with `"pickle": false` (§6.4) |
| N7 | modes in the header (§3.2) |
| N8 | collection ages and live stages (§9) |
| N9 | paths into an entry (§7.2) |
| N10 | cost and `durability` (§5.1, §9) |
| N11 | `Reuse.bind` (§7.1) |
| N12 | encoding table and threat model (§6.4, §2) |

**Where this revision does not follow the review.** The review suggests linking a file member when
it is a regular file with one link, owned by the writer and inside `files_from`. This revision copies
always (with a reflink where possible). A file with one link at staging time can gain a second name
later from the same work directory if that directory outlives the publish (a resumable task's, a
caller's scratch), and the writer then owns a writable name for a committed member. A reflink gives
the same saving where the filesystem has one, without that case.

## 13. Delivery

| PR | Contents | Depends on |
|---|---|---|
| P1 | `pydra.utils.storagedir`: a descriptor-relative directory handle (`child`, `open_new`, `rename`, `link`, `lstat`, `scandir`, `set_mode`), the wrong-kind rule, `StorePolicy`, reflink copy with hashing; Windows reparse-point checks; `PlatformUnsupported`. Unused by `PickleStore`. Tests: a symlink swapped in at each level, wrong kinds, modes under a permissive umask, copy of a hard-linked input leaves the original's mode | — |
| P2 | layout header (atomic write, checks), `<cdir>` and shard rules, `StoreIdentity` and `pydra-identity@1`; `Job.context` through the identity | P1 |
| P3 | `EntryFormat`, `pydra-entry@1`, codecs over RFC 0004's tag registry and the encoding table, canonical `.npy` and member digests from `pydra/utils/digest.py`, file outputs under `files/<set>/` with their keys; trust per root and the ownership walk; `UnencodableOutput` at hand-out | P2, RFC 0004 I0 |
| P4 | `PortableStore`: `publish` and `Committed`, first writer wins with the writer token, modes before rename and the seal after it, `StoreNotWritable`, kept entries and `_committed.json`, fenced quarantine and `q-` names, `evict`, `collect`; `read(EntryRef)`; work directories and `attempt`; optional anonymous de-duplication lease; `commit` as an adapter; errors to `_error.json`; `pydracli crash`. Tests: two processes racing one checksum (equal and unequal digests), SIGKILL before and after the rename, a lost lease that still commits, a lost rename reply, two readers quarantining one entry while a writer commits, a non-owner reader, a second writer on a default-mode root (`StoreNotWritable`) and on a group-writable one, a SIGKILL between rename and seal, a `stored=False` result read by another process through `_committed.json`, a read-only root, case-colliding supplied checksums, `debug` and `cf` parity | P3, #928 |
| P5 | `Reuse` protocol, `find(job)` and `Hit`/`Verify`/`Miss`, the `ReusePolicy` wrapper, `job.verify` and `compare`, the reuse record, the lost-race table, `on_commit` and sidecars, read-only roots | P4, K2 |
| P6 | `read_succeeded` for `PortableStore` through the reuse hook; `rerun` as evict and replace | P5, #929 |
| P7 | `migrate()`, `pydracli store migrate` and `legacy=`; `supplied_checksums=` | P4, RFC 0004 I1 and I8 |
| P8 | `Submitter(store="portable")`; RFC 0002 R4's executor reporting of `Committed`; documentation; Windows CI for `private` | P5, RFC 0002 R4 |

P1 to P3 change no behaviour. P4 is the first PR a user can opt into.

**Order with RFC 0004.** RFC 0004's I0 (`pydra/utils/digest.py`: the digest form, the canonical
`.npy`, the tag registry) depends on nothing and lands first; P3 builds on it. P2 lands before
RFC 0004's I1, which needs `StoreIdentity` and the header. So the order is P1, P2, then I0 and I1 in
either order, then P3, with no cycle. P7 waits for RFC 0004's I1 and I8, which makes
`pydra-identity@2` the default; this assumes RFC 0004 ships I8 with I1. Otherwise a user who
migrates before them migrates into a root that the next release cannot open (§4, §8.1).

**What RFC 0002 needs.** None of these contradicts RFC 0002. All six are carried by RFC 0002 at
`bdce243d` (#930); the references below are to that revision.

1. **`attempt`** (carried: RFC 0002 §3.1). `ready()` gives each job in `Round.dispatch` an `attempt`
   id (§5.4). A workflow job gets one when the runner starts to expand it and the outer job when it
   is created; a `workflow-stored` job has none.
2. **`Committed` in the outcome** (carried: RFC 0002 §3.1, `complete()`, and R4). `Outcome = Result |
   Failed | NotRun` is unchanged; the `Committed` is carried as `Result.committed`, `complete()`
   keeps it with the `Result` it holds, and R4's reporting passes it through.
3. **What travels with a job** (carried: RFC 0002 §3.4, §3.5). Besides the checksum and the context,
   under a `PortableStore` the job carries `attempt` and `verify` (§7.1), and its context is the
   identity's lookup context (§4).
4. **Lookups in `ready()`** (carried: RFC 0002 §3.1). Through `read_succeeded` and under
   `lookup_before_dispatch`, a runner over a store that has `find` calls `find(job)`, settles a
   `Hit`, dispatches a `Miss`, and dispatches a `Verify` with `job.verify` set (§5.2, §7.1).
5. **Result reads** (carried: RFC 0002 §3.3, §3.1). The Slurm and SGE adapter (RFC 0002 §3.3),
   `runner.result()` (RFC 0002 §3.1) and `Submitter.__call__`'s final read
   (`engine/submitter.py:503-513`) read through `Job.result()`, which under a `PortableStore` reads
   `_committed.json`, so a job whose outputs were not stored reports its `Result` and its readers run
   (§5.4, §7.2). The visibility retry applies to `_committed.json`.
6. **Workflow jobs** (carried: RFC 0002 §3.1). The steps RFC 0002 §3.1 gives the runner for a
   workflow job take this form:
   - *Lease:* `try_take` on what `acquire` returns, which is always held unless `dedupe` is on, the
     task is resumable or the job is a rerun (§5.3, §10);
   - *Store check:* `find(job)`, before or under the lease; a `Hit` is `workflow-stored` and its
     nodes are not expanded, a `Verify` sets `job.verify` and the workflow is expanded;
   - *Expansion:* no success record is withdrawn, because there is none (§5.2);
   - *Commit:* `publish` of the `WorkflowOutputs` with the workflow job's key and execute context,
     returning `Committed`; it never raises `LeaseLost`, so the `phase="commit"` entry does not
     arise. An error in a node publishes nothing, writes `_error.json` to the workflow job's work
     directory and records the error entry as today (§11).

## 14. Open questions for maintainers

- Should `installed` be recorded-only in the default fingerprint? This proposal says no for
  `pydra-context@1`; making it recorded-only later is a new fingerprint name, and `ExactReuse` would
  then compare fingerprints rather than whole contexts (`engine/store.py:237-243`).
- Is `REFUSE` ending the lookup, with `SKIP` to try the next candidate, the right pair of verdicts?
- Should `ResultStore` gain `publish`, `read(EntryRef)` and `work_dir` as protocol methods, with
  `commit(lease, result, job)` kept as an adapter?
- Who supplies the context in each phase when the default is replaced (§4): is a store-level
  `StoreIdentity` right, or should it sit on the task or the environment?
- Is the late-unencodable rule (keep the outputs for the run, do not store them, do not error) right,
  or should a declared `shared` store error?
- Workflow outputs that are files: copied into the workflow's entry, as proposed, or referred to the
  inner job's entry, which would make `evict` reference-aware?
- Retention: nothing but `evict` and a rerun removes an entry. Should `collect` take an entry age, or
  should retention stay outside the store?
- Which codecs ship in pydra and which in plugins?
- Should `PortableStore` with `trust="private"` become the default at the next major version, with
  `PickleStore` kept for one release behind `store="pickle"`?
- Should `LeaseSettings(holder="anonymous")` become the default for every lease on shared storage,
  including `PickleStore`'s?


## Amendment (2026-11-24): lease settings and the visibility timeout (section 5.3)

The derivations below moved here from the ``LeaseSettings`` docstring.

| Setting | Default | Derivation |
|---|---|---|
| ``attribute_delay`` | 60 s | The NFS default ``acdirmax``: a client may see a directory's mtime this late. Use 0 for a cache on local disk. |
| ``stale_after`` | 300 s | With ``renew_fraction`` 0.1 a holder beats every 30 s; ``(1 + missed_beats) * 30 + 60 = 120 < 300`` leaves 180 s for a slow storage server or a clock step. A crashed holder is taken over 5 to 6 minutes after its last beat, at one ``mkdir`` and one ``rmdir`` per running job every 30 s. |
| ``timeout`` | 600 s | At least ``stale_after + attribute_delay = 360`` (a dead holder is always taken over before a waiter gives up), with room for the poll interval. |
| ``poll_min``, ``poll_max`` | 0.05 s, 2 s | The first retry of a held lease is quick (a holder that just released); later ones back off to one listing per two seconds per waiter. |
| ``missed_beats`` | 1 | A live holder may miss one beat (a slow or hung write) without being taken over; it enters the live-holder inequality. |
| ``release_patience`` | 5 s | A release retries a refused rename for at most ``min(release_patience, missed_beats * renew_interval)``, within which no waiter sees the stopped holder stale. |
| ``visibility_timeout`` (derived) | ``attribute_delay + poll_max`` = 62 s | A settled reader waits this long for another client's ``_committed.json`` or result before it reports a failure, so it never gives up sooner than the lease assumes a change can take to show. ``PortableStore.visibility_timeout`` and ``WorkerExecutor``'s default follow it. |

A regular file at a lease path is a lock left by a pydra that used ``filelock.SoftFileLock``. It is
never renewed, so it is taken over once older than ``stale_after`` even if the old process still
runs: do not run an older pydra and this one on the same cache at the same time.


## Amendment (2026-11-25): records, caps, the store protocol, legacy reads

**One reader per record kind.** `entry.json` and `context.json` are read through
`read_capped` with `JSON_CAP` (1 GiB: a bound on a hostile file only; at about 170 bytes a
member record it admits millions of members). Committed and error records are read with
`RECORD_CAP` (1 MiB: such a record is a few hundred bytes). A kept entry copied across
filesystems reads its member list through `EntryFormat.member_records`, not a second parser.

**Served hits are recorded when served (section 7.2).** A `Hit` read under the lease writes
`_committed.json` naming the entry (`record_served`), so a settled reader in another process
never waits out `visibility_timeout`.

**The store protocol declares its optional members.** `ResultStore` gives defaults for
`identity` (None), `holds`, `enumerates_contexts`, `declared_code_reuse`, `begin_rerun`,
`note_attempt`, `forget_committed` and `release_attempts`; an implementation subclasses it. The
declared-code warning asks the store's own reuse hook what `find` would serve. The store, not
the submitter, keeps the default work directories of attempts and removes them when empty.

**One store check before dispatch.** Runner (inline and off-thread) and submitter share
`find_before_dispatch` and `accept_found`. A job read without a result is not read again while
a cap holds it back. Off-thread reads run `readers` at a time (default `STORE_READERS = 4`, an
unmeasured choice that overlaps the latency of shared storage without flooding it).

**Lease records.** `DirectoryLease` and `PortableStore` read a holder record through one
function (`read_holder_record`). `LeaseSettings.to_json`/`from_json` land with the named lease
(RFC 0005 L0), their only user.

**Legacy roots (section 8.1).** A legacy candidate is offered to the reuse hook's `verdict`
before it is unpickled: `SKIP` takes the next, `REFUSE` ends the lookup with a `Miss`. A slot
this store already holds (published earlier and not served) is not read again. An empty or
garbled `.pydra-identity` marker reads as no marker (a root keyed under `pydra-identity@1`), as
the submitter reads it; the marker read is capped at `MARKER_CAP` (256 bytes; it names one
scheme). Which caches a store may read is a property of its identity (`StoreIdentity.scheme`,
`supplied_key`), never of its name.

**Numbers.** `open_layout` makes at most `_OPEN_PASSES = 8` passes (a pass that does not return
lost the link to another opener, whose header the next pass reads; the bound ends a loop
against a process that keeps removing the header). `adopt(sample=)` defaults to 16 entries
(enough to show a systematic mode or shard fault, few enough to read in seconds on network
storage).

**Entry codecs (section 6.4).** The encoder and the digest form are one walker
(`FormWalker`); the encoder writes plain numbers and members, the digest form typed numbers and
digests. A tuple is written `{"$tuple": [...]}`, which changes the output digest of tuple
outputs.

### Moved from docstrings

### 6.3 quarantine (from PortableStore._quarantine, B5)
The entry is renamed to `quarantine/<uuid>` (within its parent as `q-<uuid>` when the user cannot
move it across directories), the uuid built from the inode found corrupt, so the rename is a
compare-and-move: a late mover meeting a valid entry renames it onto a target already there and
moves nothing. An empty directory is removed with rmdir first, which fails on any entry (a rename
onto an empty directory would succeed). A store needs stable non-zero inode numbers; inode 0 gets
a random name. The moved copy is re-verified and put back if it verifies (another writer committed
between check and rename). The owner of an entry sealed without owner write gives itself write
first, which moving a directory needs.

The copy is stamped as soon as it is moved, so collect ages it from the move and spares it while it is judged (B6).

### 8.2 adopt (from adopt(), B10)
Make the store at ``root`` a `PortableStore` without rewriting an entry (RFC 0003
section 8.2). Stop the writers that predate the header first.

The header is checked against ``identity`` and ``entry_format`` (its ``identity``,
``entries`` and ``entry_format``) and against the tree: the old version directory and the
shard and checksum directories must have the ``container`` mode, the entry directories the
``sealed`` mode (or the ``entry`` mode, which a reader that owns them reseals), the
members the ``file`` mode, the side directories the header names, where they exist, the
``side`` mode, and each entry's shard must be the one the header's rule gives. The entries
``check`` covers are checked (``"none"``: the first entry's modes and shard only;
``"sample"``: the first ``sample`` in name order; ``"all"``), and those are also read back
through ``entry_format`` and ``identity``: the context must give the fingerprint directory
and ``identity.check_key`` must accept the key record for the checksum. Verifying the
checksum, the fingerprint and the key is also the entry format's duty in its ``read``,
which the store calls for every entry it serves.

``<root>/<old_entries>`` is then renamed to ``header.entries`` in one directory rename,
and the header written through `open_layout`. No entry is rewritten, moved or resealed;
an entry that failed the read back is reported in ``misfiled`` and quarantined by the
store when it reads it. If the header cannot be written, the rename is undone, unless
the header was published before the error. A process killed between the rename and the
header leaves ``header.entries`` and no header; ``adopt`` then refuses and says so, and
renaming ``header.entries`` back to ``old_entries`` lets it run again.

The header's ``dirs`` and ``shard`` pass `check_layout_names`, and ``old_entries`` must
be a single path component that is none of the directories the header names. A tree in
which no entry matches the layout's names is refused unless ``allow_empty``.

The header's ``identity`` names ``identity``, and it alone says how the root's entries
are keyed. A ``.pydra-identity`` marker, which only a `PickleStore` root reads and
writes (RFC 0004 section 9.1), is neither read nor written here, and a marker already in
the root is left as it is: no `PortableStore` reads it.



### 7.4 on_commit (from PortableStore docstring, B10)
on_commit may run more than once for one entry (a publish stages again if its staging directory was
collected, and a job that loses a race still ran it), so a record it writes can describe an entry
that never became visible. Adding members to staged raises CommitHookError. It runs after
Reuse.compare. A corrupt entry in a readonly root is skipped and logged; such roots are never
collected or quarantined into.
