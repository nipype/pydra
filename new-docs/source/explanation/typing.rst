Typing and file-formats
=======================

Work in progress...

Pydra implements strong(-ish) type-checking at workflow construction time, which can
include file types.

Coercion
--------


File formats
------------

The FileFormats_ package provides a way to specify the format of a file, or set of
files, by the extensible collection of file format classes. These classes can be
used to specify the format of a file in a task input or output, and can be used
to validate the format of a file at runtime.

It is important to use a FileFormats_ type, when specifying fields that represent
a path to an existing file-system object (In most cases, it is sufficient to use the generic ``fileformats.generic.File``,
``fileformats.generic.File``, class

Superclass auto-casting
-----------------------

Not wanting the typing to get in the way by being unnecessarily strict,
upstream fields that are typed as super classes  (or as ``typing.Any`` by default)
of the task input they are connected to will be automatically cast to the subclass
when the task is run. This allows workflows and tasks to be easily connected together
regardless of how specific typing is defined in the task definition.


.. _FileFormats: https://arcanaframework.github.io/fileformats
