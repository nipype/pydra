.. _Output Specification section:

Output Specification
====================

As it was mentioned in :ref:`shell_command_task`, the user can customize the input and output
for the `ShellCommandTask`.
In this section, the output specification will be covered.


Instead of using field with `output_file_template` in the customized `input_spec` to specify an output field,
a customized `output_spec` can be used, e.g.:


.. code-block:: python

    output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out1",
                attr.ib(
                    type=File,
                    metadata={
                        "output_file_template": "{inp1}",
                        "help_string": "output file",
                        "requires": ["inp1", "inp2"]
                    },
                ),
            )
        ],
        bases=(ShellOutSpec,),
    )

    ShellCommandTask(executable=executable,
                     output_spec=output_spec)



Similarly as for `input_spec`, in order to create an output specification,
a new `SpecInfo` object has to be created.
The field `name` specifies the type of the spec and it should be always "Output" for
the output specification.
The field `bases` specifies the "base specification" you want to use (can think about it as a
`parent class`) and it will usually contains `ShellOutSpec` only, unless you want to build on top of
your other specification (this will not be cover in this section).
The part that should be always customised is the `fields` part.
Each element of the `fields` is a separate output field that is added to the specification.
In this example, a three-elements tuple - with name, type and dictionary with additional
information - is used.
See :ref:`Input Specification section` for other recognized syntax for specification's fields
and possible types.



Metadata
--------

The metadata dictionary for `output_spec` can include:

`help_string` (`str`, mandatory):
   A short description of the input field. The same as in `input_spec`.

`output_file_template` (`str`):
   If provided the output file name (or list of file names) is created using the template.
   The template can use other fields, e.g. `{file1}`. The same as in `input_spec`.

`output_field_name` (`str`, used together with `output_file_template`)
   If provided the field is added to the output spec with changed name.
   The same as in `input_spec`.

`keep_extension` (`bool`, default: `True`):
   A flag that specifies if the file extension should be removed from the field value.
   The same as in `input_spec`.


`requires` (`list`):
   List of field names that are required to create a specific output.
   The fields do not have to be a part of the `output_file_template` and
   if any field from the list is not provided in the input, a `NOTHING` is returned for the specific output.
   This has a different meaning than the `requires` form the `input_spec`.

`callable` (`function`):
   If provided the output file name (or list of file names) is created using the function.
   The function can take `field` (the specific output field will be passed to the function),
   `output_dir` (task `output_dir` wil be used), `stdout`, `stderr` (`stdout` and `stderr` of
   the task will be sent) `inputs` (entire `inputs` will be passed) or any input field name
   (a specific input field will be sent).
