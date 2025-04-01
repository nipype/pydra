Typing and file-formats
=======================

Pydra implements strong(-ish) type-checking at workflow construction time so some errors
can be caught before workflows are run on potentially expensive computing resources.
Input and output fields of tasks can be typed using Python annotations.
Unlike how they are typically used, in Pydra these type annotations are not just for
documentation and linting purposes, but are used to enforce the types of the inputs
and outputs of tasks and workflows at workflow construction and runtime.

.. note::

    With the exception of fields containing file-system paths, which should be typed
    a FileFormats_ class, types don't need to be specified if not desired.

File formats
------------

The FileFormats_ package provides a way to specify the format of a file, or set of
files, by the extensible collection of file format classes. These classes can be
used to specify the format of a file in a task input or output, and can be used
to validate the format of a file at runtime.

It is important to use a FileFormats_ type instead of a ``str`` or ``pathlib.Path``,
when defining a field that take paths to file-system objects, because otherwise only
the file path, not the file contents, will be used in the hash used to locate the cache
(see :ref:`Caches and hashes`). However, in most cases, it is sufficient to use the
generic ``fileformats.generic.File``, ``fileformats.generic.Directory``, or the even
more generic ``fileformats.generic.FsObject`` or ``fileformats.generic.FileSet`` classes.

The only cases where it isn't sufficient to use generic classes, is when there are
implicit header or side cars assumed to be present adjacent to the primary file (e.g.
a NIfTI file `my_nifti.nii` with an associated JSON sidecar file `my_nifti.json`).
Because the header/sidecar file(s) will not be included in the hash calculation
by default and may be omitted if the "file set" is copied into a different work
directories. In such cases, a specific file format class, such as
``fileformats.nifti.NiftiGzX``, should be used instead.

Coercion
--------

Pydra will attempt to coerce the input to the correct type if it is not already, for example
if a tuple is provided to a field that is typed as a list, Pydra will convert the tuple to a list
before the task is run. By default the following coercions will be automatically
applied between the following types:

* ty.Sequence → ty.Sequence
* ty.Mapping → ty.Mapping
* Path → os.PathLike
* str → os.PathLike
* os.PathLike → Path
* os.PathLike → str
* ty.Any → MultiInputObj
* int → float
* field.Integer → float
* int → field.Decimal

In addition to this, ``fileformats.fields.Singular`` (see FileFormats_)
can be coerced to and from their primitive types and Numpy ndarrays and primitive types
can be coerced to and from Python sequences and built-in types, respectively.

Superclass auto-casting
-----------------------

Pydra is designed so that strict and specific typing can be used, but is not
unnecessarily strict, if it proves too burdensome. Therefore, upstream fields that are
typed as super classes  (or as ``typing.Any`` by default) of the task input they are
connected to will be automatically cast to the subclass when the task is run.
This allows workflows and tasks to be easily connected together
regardless of how specific typing is defined in the task definition. This includes
file format types, so a task that expects a ``fileformats.medimage.NiftiGz`` file can
be connected to a task that outputs a ``fileformats.generic.File`` file.
Therefore, the only cases where a typing error will be raised are when the upstream
field can't be cast or coerced to the downstream field, e.g. a ``fileformats.medimage.DicomSeries``
cannot be cast to a ``fileformats.medimage.Nifti`` file.


.. _FileFormats: https://arcanaframework.github.io/fileformats
