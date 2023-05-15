"""Decorators and helper functions to create ShellCommandTasks used in Pydra workflows"""
from __future__ import annotations
import typing as ty
import attrs


def shell_arg(
    help_string: str,
    default: ty.Any = attrs.NOTHING,
    argstr: str = None,
    position: int = None,
    mandatory: bool = False,
    sep: str = None,
    allowed_values: list = None,
    requires: list = None,
    xor: list = None,
    copyfile: bool = None,
    container_path: bool = False,
    output_file_template: str = None,
    output_field_name: str = None,
    keep_extension: bool = True,
    readonly: bool = False,
    formatter: ty.Callable = None,
):
    """
    Returns an attrs field with appropriate metadata for it to be added as an argument in
    a Pydra shell command task definition

    Parameters
    ------------
    help_string: str
        A short description of the input field.
    default : Any, optional
        the default value for the argument
    argstr: str, optional
        A flag or string that is used in the command before the value, e.g. -v or
        -v {inp_field}, but it could be and empty string, “”. If … are used, e.g. -v…,
        the flag is used before every element if a list is provided as a value. If no
        argstr is used the field is not part of the command.
    position: int, optional
        Position of the field in the command, could be nonnegative or negative integer.
        If nothing is provided the field will be inserted between all fields with
        nonnegative positions and fields with negative positions.
    mandatory: bool, optional
        If True user has to provide a value for the field, by default it is False
    sep: str, optional
        A separator if a list is provided as a value.
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        List of field names that are required together with the field.
    xor: list, optional
        List of field names that are mutually exclusive with the field.
    copyfile: bool, optional
        If True, a hard link is created for the input file in the output directory. If
        hard link not possible, the file is copied to the output directory, by default
        it is False
    container_path: bool, optional
        If True a path will be consider as a path inside the container (and not as a
        local path, by default it is False
    output_file_template: str, optional
        If provided, the field is treated also as an output field and it is added to
        the output spec. The template can use other fields, e.g. {file1}. Used in order
        to create an output specification.
    output_field_name: str, optional
        If provided the field is added to the output spec with changed name. Used in
        order to create an output specification. Used together with output_file_template
    keep_extension: bool, optional
        A flag that specifies if the file extension should be removed from the field value.
        Used in order to create an output specification, by default it is True
    readonly: bool, optional
        If True the input field can’t be provided by the user but it aggregates other
        input fields (for example the fields with argstr: -o {fldA} {fldB}), by default
        it is False
    formatter: function, optional
        If provided the argstr of the field is created using the function. This function
        can for example be used to combine several inputs into one command argument. The
        function can take field (this input field will be passed to the function),
        inputs (entire inputs will be passed) or any input field name (a specific input
        field will be sent).
    """

    metadata = {
        "help_string": help_string,
        "argstr": argstr,
        "position": position,
        "mandatory": mandatory,
        "sep": sep,
        "allowed_values": allowed_values,
        "requires": requires,
        "xor": xor,
        "copyfile": copyfile,
        "container_path": container_path,
        "output_file_template": output_file_template,
        "output_field_name": output_field_name,
        "keep_extension": keep_extension,
        "readonly": readonly,
        "formatter": formatter,
    }

    return attrs.field(
        default=default, metadata={k: v for k, v in metadata.items() if v is not None}
    )


def shell_out(
    help_string: str,
    mandatory: bool = False,
    output_file_template: str = None,
    output_field_name: str = None,
    keep_extension: bool = True,
    requires: list = None,
    callable: ty.Callable = None,
):
    """Returns an attrs field with appropriate metadata for it to be added as an output of
    a Pydra shell command task definition

    Parameters
    ----------
    help_string: str
        A short description of the input field. The same as in input_spec.
    mandatory: bool, default: False
        If True the output file has to exist, otherwise an error will be raised.
    output_file_template: str, optional
        If provided the output file name (or list of file names) is created using the
        template. The template can use other fields, e.g. {file1}. The same as in
        input_spec.
    output_field_name: str, optional
        If provided the field is added to the output spec with changed name. The same as
        in input_spec. Used together with output_file_template
    keep_extension: bool, default: True
        A flag that specifies if the file extension should be removed from the field
        value. The same as in input_spec.
    requires: list
        List of field names that are required to create a specific output. The fields
        do not have to be a part of the output_file_template and if any field from the
        list is not provided in the input, a NOTHING is returned for the specific output.
        This has a different meaning than the requires form the input_spec.
    callable: Callable
        If provided the output file name (or list of file names) is created using the
        function. The function can take field (the specific output field will be passed
        to the function), output_dir (task output_dir will be used), stdout, stderr
        (stdout and stderr of the task will be sent) inputs (entire inputs will be
        passed) or any input field name (a specific input field will be sent).
    """
    metadata = {
        "help_string": help_string,
        "mandatory": mandatory,
        "output_file_template": output_file_template,
        "output_field_name": output_field_name,
        "keep_extension": keep_extension,
        "requires": requires,
        "callable": callable,
    }

    return attrs.field(metadata={k: v for k, v in metadata.items() if v is not None})
