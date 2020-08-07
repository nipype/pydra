.. _Input Specification section:

Input Specification
===================

As it was mentioned in :ref:`shell_command_task`, the user can customize the input and output
for the `ShellCommandTask`.
In this section, more examples of the input specification will be provided.


Let's start from the previous example:

.. code-block:: python

    bet_input_spec = SpecInfo(
        name="Input",
        fields=[
        ( "in_file", File,
          { "help_string": "input file ...",
            "position": 1,
            "mandatory": True } ),
        ( "out_file", str,
          { "help_string": "name of output ...",
            "position": 2,
            "output_file_template":
                              "{in_file}_br" } ),
        ( "mask", bool,
          { "help_string": "create binary mask",
            "argstr": "-m", } ) ],
        bases=(ShellSpec,) )

    ShellCommandTask(executable="bet",
                     input_spec=bet_input_spec)



In order to create an input specification, a new `SpecInfo` object has to be created.
The field `name` specifiest the typo of the spec and it should be always "Input" for
the input specification.
The field `bases` specifies the "base specification" you want to use (can think about it as a
`parent class`) and it will usually contains `ShellSpec` only, unless you want to build on top of
your other specification (this will not be cover in this section).
The part that should be always customised is the `fields` part.
Each element of the `fields` is a separate input field that is added to the specification.
In this example, a three-elements tuples - with name, type and dictionary with additional
information - are used.
But this is only one of the supported syntax, more options will be described below.

Adding a New Field to the Spec
------------------------------

Pydra uses `attr` classes to represent the input specification, and the full syntax for each field
is:

.. code-block:: python

   field1 = ("field1_name", attr.ib(type=<'field1_type'>, metadata=<'dictionary with metadata'>)

However, we allow for shorter syntax, that does not include `attr.ib`:

- providing only name and the type

.. code-block:: python

   field1 = ("field1_name", <'field1_type'>)


- providing name, type and metadata (as in the example above)

.. code-block:: python

   field1 = ("field1_name", <'field1_type'>, <'dictionary with metadata'>))

- providing name, type and default value

.. code-block:: python

   field1 = ("field1_name", <'field1_type'>, <'default value'>)

- providing name, type, default value and metadata

.. code-block:: python

   field1 = ("field1_name", <'field1_type'>, <'default value', <'dictionary with metadata'>))


Each of the shorter versions will be converted to the `(name, attr.ib(...)`.

Type can be provided as a simple python type (e.g. `str`, `int`, `float`, etc.)
or can be more complex by using `typing.List`, `typing.Dict` and `typing.Union`.


Metadata
--------

In the example we used multiple keys in the metadata dictionary including `help_string`,
`position`, etc. In this section all allowed key will be described:

`help_string` (`str`, mandatory):
   A short description of the input field.

`mandatory` (`bool`, default: `False`):
   If `True` user has to provide a value for the field.

`sep` (`str`):
   A separator if a list is provided as a value.

`argstr` (`str`):
   A flag or string that is used in the command before the value, e.g. `-v` or `-v {inp_field}`,
   but it could be and empty string, `""`.
   If `...` are used, e.g. `-v...`, the flag is used before every element if a list is provided
   as a value.
   If no `argstr` is used the field is not part of the command.

`position` (`int`):
   Position of the field in the command, could be positive or negative integer.
   If nothing is provided the field will be inserted between all fields with positive positions
   and fields with negative positions.

`allowed_values` (`list`):
   List of allowed values for the field.

`requires` (`list`):
   List of field names that are required together with the field.

`xor` (`list`):
   List of field names that are mutually exclusive with the field.

`keep_extension` (`bool`, default: `True`):
   A flag that specifies if the file extension should be removed from the field value.

`copyfile` (`bool`, default: `False`):
   If `True`, a hard link is created for the input file in the output directory.
   If hard link not possible, the file is copied to the output directory.

`container_path` (`bool`, default: `False`, only for `ContainerTask`):
   If `True` a path will be consider as a path inside the container (and not as a local path).

`output_file_template` (`str`):
   If provided, the field is treated also as an output field and it is added to the output spec.
   The template can use other fields, e.g. `{file1}`.

`output_field_name` (`str`, used together with `output_file_template`)
   If provided the field is added to the output spec with changed name.

`readonly` (`bool`, default: `False`):
   If `True` the input field can't be provided by the user but it aggregates other input fields
   (for example the fields with `argstr: -o {fldA} {fldB}`).


Validators
----------
Pydra allows for using simple validator for types and `allowev_values`.
The validators are disabled by default, but can be enabled by calling
`pydra.set_input_validator(flag=True)`.
