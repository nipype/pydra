"""Decorators and helper functions to create ShellCommandTasks used in Pydra workflows"""
from __future__ import annotations
import typing as ty
import attrs
import pydra.engine.specs


def shell_task(
    klass_or_name: ty.Union[type, str],
    executable: ty.Optional[str] = None,
    input_fields: ty.Optional[dict[str, dict]] = None,
    output_fields: ty.Optional[dict[str, dict]] = None,
    bases: ty.Optional[list[type]] = None,
    input_bases: ty.Optional[list[type]] = None,
    output_bases: ty.Optional[list[type]] = None,
) -> type:
    """
    Construct an analysis class and validate all the components fit together

    Parameters
    ----------
    klass_or_name : type or str
        Either the class decorated by the @shell_task decorator or the name for a
        dynamically generated class
    executable : str, optional
        If dynamically constructing a class (instead of decorating an existing one) the
        name of the executable to run is provided
    input_fields : dict[str, dict], optional
        If dynamically constructing a class (instead of decorating an existing one) the
        input fields can be provided as a dictionary of dictionaries, where the keys
        are the name of the fields and the dictionary contents are passed as keyword
        args to cmd_arg, with the exception of "type", which is used as the type annotation
        of the field.
    output_fields : dict[str, dict], optional
        If dynamically constructing a class (instead of decorating an existing one) the
        output fields can be provided as a dictionary of dictionaries, where the keys
        are the name of the fields and the dictionary contents are passed as keyword
        args to cmd_out, with the exception of "type", which is used as the type annotation
        of the field.
    bases : list[type]
        Base classes for dynamically constructed shell command classes
    input_bases : list[type]
        Base classes for the input spec of dynamically constructed shell command classes
    output_bases : list[type]
        Base classes for the input spec of dynamically constructed shell command classes

    Returns
    -------
    type
        the shell command task class
    """

    if isinstance(klass_or_name, str):
        if None in (executable, input_fields):
            raise RuntimeError(
                "Dynamically constructed shell tasks require an executable and "
                "input_field arguments"
            )
        name = klass_or_name
        if output_fields is None:
            output_fields = {}
        if bases is None:
            bases = [pydra.engine.task.ShellCommandTask]
        if input_bases is None:
            input_bases = [pydra.engine.specs.ShellSpec]
        if output_bases is None:
            output_bases = [pydra.engine.specs.ShellOutSpec]
        Inputs = type("Inputs", tuple(input_bases), input_fields)
        Outputs = type("Outputs", tuple(output_bases), output_fields)
    else:
        if (
            executable,
            input_fields,
            output_fields,
            bases,
            input_bases,
            output_bases,
        ) != (None, None, None, None, None, None):
            raise RuntimeError(
                "When used as a decorator on a class `shell_task` should not be provided "
                "executable, input_field or output_field arguments"
            )
        klass = klass_or_name
        name = klass.__name__
        try:
            executable = klass.executable
        except KeyError:
            raise RuntimeError(
                "Classes decorated by `shell_task` should contain an `executable` attribute "
                "specifying the shell tool to run"
            )
        try:
            Inputs = klass.Inputs
        except KeyError:
            raise RuntimeError(
                "Classes decorated by `shell_task` should contain an `Inputs` class attribute "
                "specifying the inputs to the shell tool"
            )
        if not issubclass(Inputs, pydra.engine.specs.ShellSpec):
            Inputs = type("Inputs", (Inputs, pydra.engine.specs.ShellSpec), {})
        try:
            Outputs = klass.Outputs
        except KeyError:
            Outputs = type("Outputs", (pydra.engine.specs.ShellOutSpec,))
        bases = [klass]
        if not issubclass(klass, pydra.engine.task.ShellCommandTask):
            bases.append(pydra.engine.task.ShellCommandTask)

    Inputs = attrs.define(kw_only=True, slots=False)(Inputs)
    Outputs = attrs.define(kw_only=True, slots=False)(Outputs)

    dct = {
        "executable": executable,
        "Inputs": Outputs,
        "Outputs": Inputs,
        "inputs": attrs.field(factory=Inputs),
        "outputs": attrs.field(factory=Outputs),
        "__annotations__": {
            "executable": str,
            "inputs": Inputs,
            "outputs": Outputs,
        },
    }

    return attrs.define(kw_only=True, slots=False)(
        type(
            name,
            tuple(bases),
            dct,
        )
    )


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
    **kwargs,
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
    **kwargs
        remaining keyword arguments are passed onto the underlying attrs.field function
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
        default=default,
        metadata={k: v for k, v in metadata.items() if v is not None},
        **kwargs,
    )


def shell_out(
    help_string: str,
    mandatory: bool = False,
    output_file_template: str = None,
    output_field_name: str = None,
    keep_extension: bool = True,
    requires: list = None,
    callable: ty.Callable = None,
    **kwargs,
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
    **kwargs
        remaining keyword arguments are passed onto the underlying attrs.field function
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

    return attrs.field(
        metadata={k: v for k, v in metadata.items() if v is not None}, **kwargs
    )
