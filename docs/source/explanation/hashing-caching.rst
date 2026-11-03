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


Execution contexts and reuse
----------------------------

The interpreter, the platform, the environment and the versions of installed packages
are not part of the checksum. They are recorded beside each result, as
``Result.context``, and select its slot within the checksum's directory, so results
recorded in different contexts never overwrite each other:

* ``python``: the full interpreter version, and ``implementation``, its name;
* ``platform``: the OS and machine, and the C library and its version where known;
* ``environment``: the checksum of the environment the job ran in;
* ``packages``: the version of each distribution providing the library code the task
  reaches: the modules of its class, of the library code its function reaches (followed
  through user code whatever the task's ``code_hash`` tier) and of the types of its
  input values. A distribution installed from a commit or an archive records that
  commit or hash too; one installed from a local directory (editable or not), and a
  package no installed distribution provides, have an unknown version;
* ``installed``: a digest of the names and versions of every installed distribution.

The record is built once in the submitting process and carried with the job, so a
worker records the same context. Whether a result recorded in another context is
reused is set by the reuse policy, ``Submitter(reuse=...)``:

=================  ===========================================================
``"consistent"``   (default) the same Python major.minor and implementation,
                   platform, environment, and versions of the distributions the
                   task reaches; other installed distributions and the Python
                   patch version may differ
``"exact"``        an equal record, including the full Python version and the
                   digest of every installed distribution
``"any"``          any record
=================  ===========================================================

A result recorded in an equal context is always reused. Otherwise a record with an
unknown version is never reused under ``"consistent"`` or ``"exact"``, and a mismatch
is a recompute, never an error. A reused result recorded in another context has
``cross_context`` set, keeps the context it was recorded in as ``context`` and the
context it was reused in as ``reused_in``. A
:class:`~pydra.engine.store.ReusePolicy` object can be passed instead, to compare the
records in another way.


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
