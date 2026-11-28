Result stores
=============

A ``Submitter`` keeps the results of the jobs it runs in a *result store*, and serves a
stored result instead of running a job again. Two stores ship with Pydra:

* :class:`~pydra.engine.portable.store.PortableStore` (``store="portable"``, the default):
  each result is an *entry*, a directory of JSON and member files that any process, host or
  user with access to the root can read, published by an atomic rename;
* :class:`~pydra.engine.store.PickleStore` (``store="pickle"``): each job has a slot in the
  cache directory, holding its pickled result.

.. code-block:: python

    from pydra.engine.submitter import Submitter

    with Submitter(cache_root="/data/me/pydra-cache") as sub:   # a PortableStore
        result = sub(task)

    with Submitter(cache_root="/data/me/old-cache", store="pickle") as sub:
        result = sub(task)


The default store
-----------------

``Submitter`` opens its ``cache_root`` as a ``private`` portable root: owned by the current
user and writable by no one else (on Windows, under ``%LOCALAPPDATA%``), and allowed to hold
pickle members. A task that returns an arbitrary Python object therefore keeps working: what
no codec encodes is stored as a pickle, which a private root is trusted to read back. Without
``cache_root`` the root is ``store`` in the user cache directory
(:func:`~pydra.utils.general.default_store_root`, under ``PYDRA_CACHE_ROOT`` when that is set).

The submitter makes a root it creates ``0o700`` whatever the umask. A root that already
exists must belong to you and be writable by no one else (``chmod 700 <root>``); otherwise
pass ``store=PortableStore(root, trust="trusted")`` if everyone who can write it is trusted.

A portable root must be an empty directory or a portable root already. A directory holding
other names, such as a ``PickleStore`` cache, is refused with ``LegacyLayout``; give an empty
``cache_root``, or keep reading the old cache with ``store="pickle"``.

``readonly_caches`` are portable roots searched after ``cache_root``. Each is read as
``shared`` (its pickle members are never opened) unless it is given as a
:class:`~pydra.engine.portable.trust.Root` with another trust. Windows has no ``shared``
roots, so there a path is read as ``private``, which holds only under ``%LOCALAPPDATA%``: a
read-only cache elsewhere is skipped with a warning.

Entry files are read-only: on Windows, clear their read-only attribute before deleting a
root.


What is stored
--------------

Outputs are stored as JSON and member files. A ``File``, ``Directory`` or other file set of
the ``fileformats`` packages is copied into the entry and read back from it; a file set
class defined elsewhere is stored only once it is registered (see
:mod:`pydra.utils.digest`), and is otherwise kept for the run, not stored. A
``pathlib.Path`` output that names a file or directory the job wrote in its work directory
is copied into the entry too and read back as a path to the copy; any other path is stored
as a value, and its file is not. The work directory is emptied once the entry is published,
so a file a task writes and does not return as an output is gone after the run. A ``str``
output is text even when it names a file: one naming a file in the work directory is
stored as text, with a warning, and the file is not kept. A tuple
comes back a tuple, a list a list.

Shared roots
------------

A root on storage other users write is opened by passing the store itself:

.. code-block:: python

    from pydra.engine.portable.store import PortableStore

    store = PortableStore(
        "/shared/project/pydra-store",
        trust="shared",
        work_dir=lambda checksum, fingerprint, attempt: scratch / attempt,
    )
    with Submitter(store=store, cache_root=scratch / "submitter") as sub:
        result = sub(task)

A shared root never holds a pickle. An output that no codec encodes is not stored: the job
still succeeds, its outputs are kept in its work directory for the run
(``Result.stored`` is ``False``), and the jobs that read it run. The next run computes it
again. Work directories hold a job's pickles and crash file, so a shared root requires
``work_dir``, a pure function of the job's checksum, fingerprint and attempt that names
storage other users cannot read and that every node running jobs can reach.


Reuse, reruns and resumption
----------------------------

Before a job is handed to a worker, the runner asks the store whether the job's own address
holds an entry (one existence check). If it does, the store's reuse hook decides: the result
is served, the job is run and its fresh outputs compared with the stored ones (``VERIFY``),
or the job is run. Entries from other execution contexts and read-only roots are considered
when the job runs, under its lease. A runner built with ``lookup_before_dispatch=True`` asks
the full question for every job before handing it out.

``rerun=True`` computes the job again and replaces the stored entry, which moves to the
root's ``quarantine/`` directory; ``PortableStore.collect`` removes it later.

A run that was interrupted is resumed by running the same task again with the same store:
every job with a stored result is served, the rest run.


Reading results of jobs that ran elsewhere
------------------------------------------

A worker that runs jobs in other processes or on other hosts (Slurm, SGE) reports only that a
job ended. Its result is then read from ``_committed.json`` in the job's work directory,
which names the entry it published, or where its unstored outputs are kept; the read waits
for that file to become visible for up to the store's ``visibility_timeout``
(``LeaseSettings.visibility_timeout``: the lease's ``attribute_delay`` and one poll), then
reads the entry at the job's own address, and otherwise reports the job failed, naming its
work directory.


Collection
----------

A submitter removes the emptied work directories of the jobs it ran when it closes. What a
store keeps only for a while (old staging directories, replaced and quarantined entries,
work directories left by interrupted runs) is removed by ``Submitter.collect()``, or from the
shell by ``pydracli store collect <root>``, once it is older than the store policy's ages.
