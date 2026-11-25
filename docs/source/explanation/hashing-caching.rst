Caches and hashes
=================

In Pydra, each task is run within its own working directory. If a task completes
successfully, their outputs are stored within this working directory. Working directories
are created within a cache directory, which is specified when the task is executed, and
named according to the hash of the task's inputs. This means that if the same task is
executed with the same inputs, the same working directory will be used, and instead of the task
being rerun, the outputs from the previous run will be reused.

In this manner, incomplete workflows can be resumed from where they left off, and completed
workflows can be rerun without having to rerun all of the tasks. This is particularly useful
when working with datasets that are to be analysed in several different ways with
common intermediate steps, or when debugging workflows that have failed part way through.


Hash calculations
-----------------

Hashes are calculated for different types of objects in different ways. For example, the
hash of a string is simply the hash of the string itself, whereas the hash of a dictionary
is the hash of all the file names and contents within the directory. Implementations for
most common types are provided in the :mod:`pydra.utils.hash` module, but custom types
can be hashed by providing a custom ``bytes_repr`` function (see
:ref:`Registering custom bytes_repr functions`).

A cache dictionary, is passed each ``bytes_repr`` call that maps an objects id (i.e.
as returned by the built-in ``id()`` function) to the hash, to avoid infinite recursions
in the case of circular references.

The byte representation of each object is hashed using the BlakeB cryptographic algorithm,
and these hashes are then combined to create a hash of the entire inputs object.


Hashing the code of a task
--------------------------

The function of a Python task, and the constructor of a workflow, are hashed by a
version-neutral encoding of their syntax tree, so comments, formatting, docstrings and
signature annotations do not change the hash. Their ``code_hash`` tier sets how far the
hash follows the code they reach:

* ``"function"``: the function's own code, and the values it reads that are not
  inputs, the variables it closes over and the globals it reads, hashed by value as
  inputs are;
* ``"module"``: as ``"function"``, plus the functions and classes it reaches in its own
  module (their code, defaults and bases, and the values they read);
* ``"user"`` (the default): as ``"module"``, plus all reachable code that is not
  library code.

A function or class outside the tier is identified by its qualified name. Library code,
the standard library, packages installed in ``site-packages`` and pydra itself, is
always identified by its qualified name, never its code: its version is not part of the
hash. The ``PYDRA_USER_PACKAGES`` environment variable overrides that boundary with a
comma-separated list of packages, ``mypkg`` to hash an installed package as user code
and ``-mypkg`` to treat one (for instance an editable install) as library code. Set the
tier per task with ``python.define(code_hash=...)`` or ``workflow.define(code_hash=...)``,
or for all tasks with the ``PYDRA_CODE_HASH`` environment variable.

A library function or class is identified by name only when its module holds it at that
name; one that it does not (a closure made by a library factory, a generated function)
is hashed by its definition and the values it captures, as user code is.

The boundary has three consequences to be aware of:

* pydra is library code, including ``pydra.tasks.*`` packages, so authors of such a
  package set ``PYDRA_USER_PACKAGES=pydra.tasks.<theirs>`` for its code to be hashed
  while they develop it;
* code that is not installed in ``site-packages`` (vendored on ``PYTHONPATH``, installed
  with ``pip --target``, or run from a zipapp) counts as user code, so it is walked and
  hashed, which costs time; name it with ``-pkg`` to treat it as library code;
* state owned by a library (``os.environ``, a library's module-level registries) is
  identified by its name, not its value, so changing it does not change the hash;
  pass such values as inputs.

The walk of a function is memoised per process: it is reused while the source files it
reached are unchanged and every name it reads still refers to the same object, and the
values it reads are hashed again each time. Globals are read when the job is submitted,
so rebinding a name in a notebook changes the hash. A global that cannot be hashed raises the ordinary hashing error, naming it; pass
it as an input instead. The hash is only computed in the submitting process and carried
with the job, so workers never recompute it.


Declared versions
-----------------

``python.define(version="2")`` replaces the code part of the checksum with the
function's qualified name and that version. Only the code is replaced: the walk still
runs at the task's tier, and the values it reads (closure cells, globals and the
defaults of the functions it calls) are hashed as they are without a version, so two
closures made by one factory with different values still have different checksums. So
are the definitions the walk reaches and the library code it names, without their
code: a refactor that calls another helper or another library function changes the
checksum under the same version, a recompute rather than a reuse. A versioned task's
checksum can never equal a code-hashed one's, because the code part is tagged
``version:`` instead of ``ast@1:``. ``checksum=`` (``__checksum__``) remains the way to
replace the whole checksum.

Wherever a versioned task is hashed, it is hashed by its version: as a node class a
workflow's constructor reaches, and as a task passed as an input (as a split task is to
the workflow that runs its elements). Editing a versioned node therefore leaves the
workflow's checksum unchanged, as it leaves the node's.

The code the version stands for is not forgotten. Its digest,
:func:`pydra.utils.hash.code_digest` (the encoded code of every definition the walk
reaches, without the values), is recorded in the record beside each result, under
``code``, by ``module.qualname==version``, for every versioned task the checksum
reached: the task itself, a workflow's versioned nodes and versioned tasks passed as
inputs. When a job's context is built, in the submitting process, a result stored under
the same checksum, so of the same versions and values, recorded with another digest
issues a :class:`~pydra.engine.store.DeclaredVersionWarning` naming the job, each
versioned function whose code moved and both digests, with the remedy: bump
``version=`` or remove it. The result is in another slot, because the digests are part
of the record, so:

=================  ===========================================================
``"exact"``        it is not reused, and the task runs again
``"consistent"``   it is reused, as the version declares, and marked
                   ``cross_context``
``"any"``          it is reused, and marked ``cross_context``
=================  ===========================================================

A :class:`~pydra.engine.store.ReusePolicy` object sees ``code`` in both records and
decides as it compares them; the warning is issued either way. Under ``"exact"`` a
workflow, or a split, whose versioned node moved is rebuilt and the node recomputed.

The same digests serve a check in the defining project's CI, against a committed JSON
table of ``{"module.Task": {"version": ..., "digest": ...}}``:

.. code-block:: bash

    pydracli check-versions versions.json           # exits with 1 on a mismatch
    pydracli check-versions versions.json --update  # record bumped versions

A digest that moved while its version did not fails either way; a bumped version fails
until ``--update`` records it. A function without source is hashed by its byte-code,
whose digest differs between Python minors.


Execution contexts and reuse
----------------------------

The interpreter, the platform, the environment and the identities of installed
packages are not part of the checksum. They are recorded beside each result, as
``Result.context``, and select its slot within the checksum's directory, so results
recorded in different contexts never overwrite each other:

* ``python``: the full interpreter version, and ``implementation``, its name;
* ``platform``: the OS and machine, and the C library and its version where known, so
  updating glibc on a cluster ends ``"consistent"`` reuse of its earlier results;
* ``environment``: the checksum of the environment the job ran in;
* ``packages``: the identity of each distribution that installed a library module the
  task reaches, and of every distribution those require (their ``Requires-Dist``,
  under the markers of the running interpreter). The modules are those of its class,
  of the library code its function reaches (collected by the walk that computes its
  checksum, followed through user code) and of the types of its input values, through
  containers, attrs and dataclass instances. A module belongs to the distribution that
  installed its file, so one module of a namespace package (``google.protobuf``,
  ``pydra.tasks.x``) records only its own distribution;
* ``installed``: a digest of every installed distribution, those in ``packages`` by
  their identities and the others by their versions;
* ``code``: when the checksum reached tasks with declared versions, the digest of the
  code of each (see `Declared versions`_).

A distribution is identified by its version and, when it was not installed from an
index, by what it was installed from, which is read only for the distributions a task
reaches: the commit or archive hash it records; for an editable install, the commit
of its git checkout and a digest of the files modified or untracked there, or outside
git a digest of all its files, in both cases only those of its own packages and up to
``IDENTITY_MAX_FILES`` files and ``IDENTITY_MAX_BYTES`` bytes; for another install
from a local directory, a digest of the files it installed (its ``RECORD``). What
still cannot be identified (a module no distribution installed, an editable install
whose source is gone or that is over those limits) is unknown, and is logged with the
reason. A result
whose record holds an unknown is found within the process that computed it (and its
workers) but reused by no other process under ``"consistent"`` or ``"exact"``.

The record is built once in the submitting process and carried with the job, so a
worker records the same context. Whether a result recorded in another context is
reused is set by the reuse policy, ``Submitter(reuse=...)``:

=================  ===========================================================
``"consistent"``   (default) the same Python major.minor and implementation,
                   platform, environment, and identities of the distributions
                   the task reaches; other installed distributions and the Python
                   patch version may differ
``"exact"``        an equal record, including the full Python version and the
                   digest of every installed distribution
``"any"``          any record
=================  ===========================================================

A mismatch is a recompute, never an error: each slot's context is kept in
``_context.json``, written atomically, and compared before its result is loaded. A
reused result recorded in another context has ``cross_context`` set, keeps the context
it was recorded in as ``context`` and the context it was reused in as ``reused_in``.
A :class:`~pydra.engine.store.ReusePolicy` object can be passed instead, to compare
the records in another way.

File hash caching by mtime
--------------------------

To avoid having to recalculate the hash of large files between runs, file hashes themselves
are cached in a platform specific user directory. These hashes are stored within small
files named by yet another hash of the file-system path an mtime of the file. This means that
the contents of a file should only need to be hashed once unless it is modified.

.. note::

    Due to limitations in mtime resolution on different platforms (e.g. 1 second on Linux,
    potentially 2 seconds on Windows), it is conceivable that a file could be modified,
    hashed, and then modified again within resolution period, causing the hash to be
    invalid. Therefore, cached hashes are only used once the mtime resolution period
    has lapsed since it was last modified, and may be recalculated in some rare cases.


Registering custom bytes_repr functions
---------------------------------------

Work in progress...


Cache misses due to unstable hashes
-----------------------------------

Work in progress...
