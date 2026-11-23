Release notes: pydra-identity\@2
================================

This release makes ``pydra-identity@2`` (RFC 0004) the identity of every job. A job's
checksum is now the digest of its identity record: the task's code, its typed input
digests, its outputs class, and the facts of the job that decide what runs (the resolved
environment reference, a shell tool's version, the run facts of a workflow's nodes).

Every key changes: results are recomputed once
----------------------------------------------

No checksum computed before this release is computed again. The first run of each job
after upgrading recomputes it, and its results are reused from then on. Nothing old is
read, served or deleted.

* A ``PickleStore`` root (``Submitter(store="pickle")``) gets a ``.pydra-identity``
  marker naming the scheme it is keyed under. The first run on a root that holds slots
  and no marker, or whose marker names an older scheme, logs one line saying every job
  runs again once. The check is one read per ``Submitter`` call. A marker naming a newer
  or unknown scheme is left as it is, with one line.
* A ``PortableStore`` root names its identity in ``pydra-store.json``. A root created
  under ``pydra-identity@1`` is not readable under ``pydra-identity@2`` (RFC 0004 section
  9.1): it is refused with ``StoreLayoutMismatch`` naming ``identity``, even when it holds
  only supplied-checksum entries. Point the submitter at a new root and recompute.
* ``Result.record`` holds the key record for results read from a ``PortableStore``,
  first runs included, as it did under ``PickleStore``. ``pydra explain`` reads a
  portable entry directory as well as a JSON record or a pickle result directory.

Recorded-only context fields
----------------------------

``recorded_only`` names the context fields kept in ``context.json`` that never split a
slot. Under ``PydraIdentityV2`` it always holds, whatever else is given:

* ``environment_repository``: the repository an image came from (I6a);
* ``environment_facts``: what an environment's resolution found besides its key, such as
  a local image id, the per-platform manifest digest, and the Lmod tree, ``MODULEPATH``
  and dependency modules (I6b);
* ``pool``: the section a warm-pool worker adds to the execute context (RFC 0005, W11b).

A ``PortableStore`` root whose header lists a different ``recorded_only`` is refused with
``StoreLayoutMismatch("recorded_only")``, including roots created by pre-release builds
before the environment fields were added.

Environment keys
----------------

* Docker, and Singularity given a ``docker://`` image, key the image by its registry
  digest, ``docker@1:<digest>``; a tag is resolved once per run.
* A local ``.sif`` file is keyed by its content, ``sif@1:sha256:<digest>``, and the worker
  checks the digest again before it runs.
* Lmod modules are keyed by the full names Lmod loads, ``lmod@1:<name/version ...>``, in
  the order they were requested.
* A reference that cannot be resolved is recorded as unknown, or raises under
  ``resolve_images="require"``.

Errors at keying
----------------

``JobErrorEntry.phase`` has a new value, ``"keying"``: the job's checksum could not be
computed, because a shell tool's version command failed, a reference could not be
resolved under ``resolve_images="require"``, or an input value has no digest. The job
errors, its readers do not run, and the rest of the run continues under ``on_error``.
Code that switches on ``phase`` should handle it.

Workflows
---------

* A workflow whose run facts are not known when it is keyed (a nested workflow with a
  lazy input or a splitter, or a node whose environment or tool version is unresolved)
  is *deferred*: its entry is written, but it is never served whole; it is expanded and
  each node is looked up under its own key. A split task submitted through
  ``Submitter.__call__`` is wrapped in a split workflow, which is always deferred: one
  expansion, then one lookup per element.
* A workflow constructor runs when the workflow is keyed, to collect its nodes' facts.
  If it raises there, the error is swallowed and the node is recorded as deferred; the
  constructor runs again when the workflow is expanded, where the error is raised as that
  job's error (phase ``"run"``). A failing constructor therefore runs twice.

Absence suffixes
----------------

A job whose checksum carries an absence suffix (an ``Absent`` accepted or dropped by a
field the checksum does not see, RFC 0002) stores its triples in the key record's
``absent`` field, ``[field, {"$int": "<index>"}, cause]`` each. A store accepts the key
only if the digest of the record without them, plus the suffix of the triples, is the
checksum. ``pydra explain`` shows them as ``field[index]: cause``.
