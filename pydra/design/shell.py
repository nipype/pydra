"""Decorators and helper functions to create ShellCommandTasks used in Pydra workflows"""

from __future__ import annotations
import typing as ty
import re
from collections import defaultdict
import inspect
from copy import copy
import attrs
from fileformats.core import from_mime
from fileformats import generic, field
from fileformats.core.exceptions import FormatRecognitionError
from .base import (
    Arg,
    Out,
    check_explicit_fields_are_none,
    get_fields_from_class,
    collate_fields,
    Interface,
    make_interface,
)
from pydra.engine.task import ShellCommandTask


@attrs.define(kw_only=True)
class arg(Arg):
    """An input field that specifies a command line argument

    Parameters
    ----------
    help_string: str
        A short description of the input field.
    default : Any, optional
        the default value for the argument
    mandatory: bool, optional
        If True user has to provide a value for the field, by default it is False
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        List of field names that are required together with the field.
    xor: list, optional
        List of field names that are mutually exclusive with the field.
    copy_mode: File.CopyMode, optional
        The mode of copying the file, by default it is File.CopyMode.any
    copy_collation: File.CopyCollation, optional
        The collation of the file, by default it is File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition, optional
        The extension decomposition of the file, by default it is
        File.ExtensionDecomposition.single
    readonly: bool, optional
        If True the input field can’t be provided by the user but it aggregates other
        input fields (for example the fields with argstr: -o {fldA} {fldB}), by default
        it is False
    type: type, optional
        The type of the field, by default it is Any
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    argstr: str, optional
        A flag or string that is used in the command before the value, e.g. -v or
        -v {inp_field}, but it could be and empty string, “”, in which case the value is
        just printed to the command line. If … are used, e.g. -v…,
        the flag is used before every element if a list is provided as a value. If the
        argstr is None, the field is not part of the command.
    position: int, optional
        Position of the field in the command, could be nonnegative or negative integer.
        If nothing is provided the field will be inserted between all fields with
        nonnegative positions and fields with negative positions.
    sep: str, optional
        A separator if a list is provided as a value.
    container_path: bool, optional
        If True a path will be consider as a path inside the container (and not as a
        local path, by default it is False
    formatter: function, optional
        If provided the argstr of the field is created using the function. This function
        can for example be used to combine several inputs into one command argument. The
        function can take field (this input field will be passed to the function),
        inputs (entire inputs will be passed) or any input field name (a specific input
        field will be sent).
    """

    argstr: str | None = ""
    position: int | None = None
    sep: str | None = None
    allowed_values: list | None = None
    container_path: bool = False
    formatter: ty.Callable | None = None


@attrs.define(kw_only=True)
class out(Out):
    """An output field that specifies a command line argument

    Parameters
    ----------
    callable : Callable, optional
        If provided the output file name (or list of file names) is created using the
        function. The function can take field (the specific output field will be passed
        to the function), output_dir (task output_dir will be used), stdout, stderr
        (stdout and stderr of the task will be sent) inputs (entire inputs will be
        passed) or any input field name (a specific input field will be sent).
    """

    callable: ty.Callable | None = None


@attrs.define(kw_only=True)
class outarg(Out, arg):
    """An input field that specifies where to save the output file

    Parameters
    ----------
    help_string: str
        A short description of the input field.
    default : Any, optional
        the default value for the argument
    mandatory: bool, optional
        If True user has to provide a value for the field, by default it is False
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        List of field names that are required together with the field.
    xor: list, optional
        List of field names that are mutually exclusive with the field.
    copy_mode: File.CopyMode, optional
        The mode of copying the file, by default it is File.CopyMode.any
    copy_collation: File.CopyCollation, optional
        The collation of the file, by default it is File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition, optional
        The extension decomposition of the file, by default it is
        File.ExtensionDecomposition.single
    readonly: bool, optional
        If True the input field can’t be provided by the user but it aggregates other
        input fields (for example the fields with argstr: -o {fldA} {fldB}), by default
        it is False
    type: type, optional
        The type of the field, by default it is Any
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    argstr: str, optional
        A flag or string that is used in the command before the value, e.g. -v or
        -v {inp_field}, but it could be and empty string, “”. If … are used, e.g. -v…,
        the flag is used before every element if a list is provided as a value. If no
        argstr is used the field is not part of the command.
    position: int, optional
        Position of the field in the command, could be nonnegative or negative integer.
        If nothing is provided the field will be inserted between all fields with
        nonnegative positions and fields with negative positions.
    sep: str, optional
        A separator if a list is provided as a value.
    container_path: bool, optional
        If True a path will be consider as a path inside the container (and not as a
        local path, by default it is False
    formatter: function, optional
        If provided the argstr of the field is created using the function. This function
        can for example be used to combine several inputs into one command argument. The
        function can take field (this input field will be passed to the function),
        inputs (entire inputs will be passed) or any input field name (a specific input
        field will be sent).
    file_template: str, optional
        If provided, the field is treated also as an output field and it is added to
        the output spec. The template can use other fields, e.g. {file1}. Used in order
        to create an output specification.
    template_field: str, optional
        If provided the field is added to the output spec with changed name. Used in
        order to create an output specification. Used together with output_file_template

    """

    file_template: str | None = None
    template_field: str | None = None


def interface(
    wrapped: type | str | None = None,
    /,
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    auto_attribs: bool = True,
    args_last: bool = False,
    name: str | None = None,
) -> Interface:
    """Create a shell command interface

    Parameters
    ----------
    wrapped : type | str | None
        The class or command line template to create an interface for
    inputs : list[str | Arg] | dict[str, Arg | type] | None
        The input fields of the shell command
    outputs : list[str | Out] | dict[str, Out | type] | type | None
        The output fields of the shell command
    auto_attribs : bool
        Whether to use auto_attribs mode when creating the class
    args_last : bool
        Whether to put the executable argument last in the command line instead of first
        as they appear in the template
    name: str | None
        The name of the returned class
    """

    def make(
        wrapped: ty.Callable | type | None = None,
    ) -> Interface:

        if inspect.isclass(wrapped):
            klass = wrapped
            executable: str
            try:
                executable = attrs.fields(klass).executable.default
            except (AttributeError, attrs.exceptions.NotAnAttrsClassError):
                try:
                    executable = klass.executable
                except AttributeError:
                    raise AttributeError(
                        f"Shell task class {wrapped} must have an `executable` "
                        "attribute that specifies the command to run"
                    ) from None
            class_name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = get_fields_from_class(
                klass, arg, out, auto_attribs
            )
        else:
            if not isinstance(wrapped, str):
                raise ValueError(
                    f"wrapped must be a class or a string, not {wrapped!r}"
                )
            klass = None
            input_helps, output_helps = {}, {}

            executable, inferred_inputs, inferred_outputs = parse_command_line_template(
                wrapped,
                args_last=args_last,
                inputs=inputs,
                outputs=outputs,
            )

            parsed_inputs, parsed_outputs = collate_fields(
                arg_type=arg,
                out_type=out,
                inputs=inferred_inputs,
                outputs=inferred_outputs,
                input_helps=input_helps,
                output_helps=output_helps,
            )
            class_name = executable if not name else name

        # Update the inputs (overriding inputs from base classes) with the executable
        # and the output argument fields
        inputs_dict = {i.name: i for i in parsed_inputs}
        inputs_dict.update({o.name: o for o in parsed_outputs if isinstance(o, arg)})
        inputs_dict["executable"] = arg(
            name="executable", type=str, argstr="", position=0, default=executable
        )
        parsed_inputs = list(inputs_dict.values())

        # Set positions for the remaining inputs that don't have an explicit position
        position_stack = list(reversed(remaining_positions(parsed_inputs)))
        for inpt in parsed_inputs:
            if inpt.position is None:
                inpt.position = position_stack.pop()

        interface = make_interface(
            ShellCommandTask,
            parsed_inputs,
            parsed_outputs,
            name=class_name,
            klass=klass,
            bases=bases,
            outputs_bases=outputs_bases,
        )
        return interface

    # If a name is provided (and hence not being used as a decorator), check to see if
    # we are extending from a class that already defines an executable
    if wrapped is None and name is not None:
        for base in bases:
            try:
                wrapped = attrs.fields(base).executable.default
            except (AttributeError, attrs.exceptions.NotAnAttrsClassError):
                try:
                    wrapped = base.executable
                except AttributeError:
                    pass
            if wrapped:
                break
        if wrapped is None:
            raise ValueError(
                f"name ({name!r}) can only be provided when creating a class "
                "dynamically, i.e. not using it as a decorator. Check to see "
                "whether you have forgotten to provide the command line template"
            )
    # If wrapped is provided (i.e. this is not being used as a decorator), return the
    # interface class
    if wrapped is not None:
        if not isinstance(wrapped, (type, str)):
            raise ValueError(f"wrapped must be a class or a string, not {wrapped!r}")
        return make(wrapped)
    return make


def parse_command_line_template(
    template: str,
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
    args_last: bool = False,
) -> ty.Tuple[str, dict[str, Arg | type], dict[str, Out | type]]:
    """Parses a command line template into a name and input and output fields. Fields
    are inferred from the template if not provided, where inputs are specified with `<fieldname>`
    and outputs with `<out:fieldname>`. The types of the fields can be specified using their
    MIME like (see fileformats.core.from_mime), e.g.

    ```
    my_command <myinput> <out|myoutput2>
    ```

    The template can also specify options with `-` or `--`
    followed by the option name and arguments with `<argname:type>`. The type is optional and
    will default to `generic/fs-object` if not provided for arguments and `field/text` for
    options. The file-formats namespace can be dropped for generic and field formats, e.g.

    ```
    another-command <input1:directory> <input2:integer> --output <out|output1:text/csv>
    ```

    Parameters
    ----------
    template : str
        The command line template
    inputs : list[str | Arg] | dict[str, Arg | type] | None
        The input fields of the shell command
    outputs : list[str | Out] | dict[str, Out | type] | type | None
        The output fields of the shell command
    args_last : bool
        Whether to put the executable argument last in the command line instead of first
        as they appear in the template

    Returns
    -------
    executable : str
        The name of the command line template
    inputs : dict
        The input fields of the command line template
    outputs : dict
        The output fields of the command line template
    """
    if isinstance(inputs, list):
        inputs = {arg.name: arg for arg in inputs}
    elif isinstance(inputs, dict):
        inputs = copy(inputs)  # We don't want to modify the original
    else:
        inputs = {}
    if isinstance(outputs, list):
        outputs = {out.name: out for out in outputs}
    elif isinstance(outputs, dict):
        outputs = copy(outputs)  # We don't want to modify the original
    elif not outputs:
        outputs = {}
    parts = template.split(maxsplit=1)
    if len(parts) == 1:
        return template, inputs, outputs
    executable, args_str = parts
    tokens = re.split(r"\s+", args_str.strip())
    arg_re = re.compile(r"<([:a-zA-Z0-9\|\-\.\/\+]+)>")
    opt_re = re.compile(r"--?(\w+)")

    arguments = []
    options = []
    option = None
    position = 1

    def merge_or_create_field(name, field_type, type):
        """Merge the typing information with an existing field if it exists"""
        if isinstance(field_type, out):
            dct = outputs
        else:
            dct = inputs
        try:
            field = dct.pop(name)
        except KeyError:
            field = field_type(name=name, type=type_)
        else:
            if isinstance(field, dict):
                field = field_type(**field)
            elif not isinstance(field, field_type):  # If field type is outarg not out
                field = field_type(**attrs.asdict(field))
            field.name = name
            if field.type is ty.Any:
                field.type = type_

    def add_option(opt):
        name, field_type, type_ = opt
        if len(type_) > 1:
            type_ = tuple[tuple(type_)]
        else:
            type_ = type_[0]
        options.append(merge_or_create_field(name, field_type, type_))

    for token in tokens:
        if match := arg_re.match(token):
            name = match.group()
            if name.startswith("out|"):
                name = name[4:]
                field_type = outarg
            else:
                field_type = arg
            if ":" in name:
                name, type_str = name.split(":")
                if "/" in type_str:
                    type_ = from_mime(type_str)
                else:
                    try:
                        type_ = from_mime(f"field/{type_str}")
                    except FormatRecognitionError:
                        try:
                            type_ = from_mime(f"generic/{type_str}")
                        except FormatRecognitionError:
                            raise ValueError(f"Unknown type {type_str}")
            else:
                type_ = generic.FsObject if field_type is arg else field.Text
            type_ = from_mime(type_str) if type_str is not None else ty.Any
            if option is None:
                arguments.append(merge_or_create_field(name, field_type, type_))
                position += 1
            else:
                option[1].append((name, type_))
        elif match := opt_re.match(token):
            if option is not None:
                add_option(option)
                position += 1
            option = (match.group(1), field_type, [])
    if option is not None:
        add_option(option)

    inferred_inputs = []
    inferred_outputs = []

    all_args = options + arguments if args_last else arguments + options

    for i, argument in enumerate(all_args, start=1):
        argument.position = i
        if isinstance(argument, outarg):
            inferred_outputs.append(argument)
        else:
            inferred_inputs.append(argument)

    return executable, inferred_inputs, inferred_outputs


def remaining_positions(args: list[Arg], num_args: int | None = None) -> ty.List[int]:
    """Get the remaining positions for input fields

    Parameters
    ----------
    args : list[Arg]
        The list of input fields
    num_args : int, optional
        The number of arguments, by default it is the length of the args

    Returns
    -------
    list[int]
        The list of remaining positions

    Raises
    ------
    ValueError
        If multiple fields have the same position
    """
    if num_args is None:
        num_args = len(args)
    # Check for multiple positions
    positions = defaultdict(list)
    for arg in args:
        if arg.position is not None:
            if arg.position >= 0:
                positions[arg.position].append(arg)
            else:
                positions[num_args + arg.position].append(arg)
    if multiple_positions := {
        k: f"{v.name}({v.position})" for k, v in positions.items() if len(v) > 1
    }:
        raise ValueError(
            f"Multiple fields have the overlapping positions: {multiple_positions}"
        )
    return [i for i in range(num_args) if i not in positions]


# def interface(
#     klass_or_name: ty.Union[type, str],
#     executable: ty.Optional[str] = None,
#     input_fields: ty.Optional[dict[str, dict]] = None,
#     output_fields: ty.Optional[dict[str, dict]] = None,
#     bases: ty.Optional[list[type]] = None,
#     inputs_bases: ty.Optional[list[type]] = None,
#     outputs_bases: ty.Optional[list[type]] = None,
# ) -> type:
#     """
#     Construct an analysis class and validate all the components fit together

#     Parameters
#     ----------
#     klass_or_name : type or str
#         Either the class decorated by the @shell_task decorator or the name for a
#         dynamically generated class
#     executable : str, optional
#         If dynamically constructing a class (instead of decorating an existing one) the
#         name of the executable to run is provided
#     input_fields : dict[str, dict], optional
#         If dynamically constructing a class (instead of decorating an existing one) the
#         input fields can be provided as a dictionary of dictionaries, where the keys
#         are the name of the fields and the dictionary contents are passed as keyword
#         args to cmd_arg, with the exception of "type", which is used as the type annotation
#         of the field.
#     output_fields : dict[str, dict], optional
#         If dynamically constructing a class (instead of decorating an existing one) the
#         output fields can be provided as a dictionary of dictionaries, where the keys
#         are the name of the fields and the dictionary contents are passed as keyword
#         args to cmd_out, with the exception of "type", which is used as the type annotation
#         of the field.
#     bases : list[type]
#         Base classes for dynamically constructed shell command classes
#     inputs_bases : list[type]
#         Base classes for the input spec of dynamically constructed shell command classes
#     outputs_bases : list[type]
#         Base classes for the input spec of dynamically constructed shell command classes

#     Returns
#     -------
#     type
#         the shell command task class
#     """

#     annotations = {
#         "executable": str,
#         "Outputs": type,
#     }
#     dct = {"__annotations__": annotations}

#     if isinstance(klass_or_name, str):
#         # Dynamically created classes using shell_task as a function
#         name = klass_or_name

#         if executable is not None:
#             dct["executable"] = executable
#         if input_fields is None:
#             input_fields = {}
#         if output_fields is None:
#             output_fields = {}
#         bases = list(bases) if bases is not None else []
#         inputs_bases = list(inputs_bases) if inputs_bases is not None else []
#         outputs_bases = list(outputs_bases) if outputs_bases is not None else []

#         # Ensure base classes included somewhere in MRO
#         def ensure_base_included(base_class: type, bases_list: list[type]):
#             if not any(issubclass(b, base_class) for b in bases_list):
#                 bases_list.append(base_class)

#         # Get inputs and outputs bases from base class if not explicitly provided
#         for base in bases:
#             if not inputs_bases:
#                 try:
#                     inputs_bases = [base.Inputs]
#                 except AttributeError:
#                     pass
#             if not outputs_bases:
#                 try:
#                     outputs_bases = [base.Outputs]
#                 except AttributeError:
#                     pass

#         # Ensure bases are lists and can be modified
#         ensure_base_included(pydra.engine.task.ShellCommandTask, bases)
#         ensure_base_included(pydra.engine.specs.ShellSpec, inputs_bases)
#         ensure_base_included(pydra.engine.specs.ShellOutSpec, outputs_bases)

#         def convert_to_attrs(fields: dict[str, dict[str, ty.Any]], attrs_func):
#             annotations = {}
#             attrs_dict = {"__annotations__": annotations}
#             for name, dct in fields.items():
#                 kwargs = dict(dct)  # copy to avoid modifying input to outer function
#                 annotations[name] = kwargs.pop("type")
#                 attrs_dict[name] = attrs_func(**kwargs)
#             return attrs_dict

#         Inputs = attrs.define(kw_only=True, slots=False)(
#             type(
#                 "Inputs",
#                 tuple(inputs_bases),
#                 convert_to_attrs(input_fields, arg),
#             )
#         )

#         Outputs = attrs.define(kw_only=True, slots=False)(
#             type(
#                 "Outputs",
#                 tuple(outputs_bases),
#                 convert_to_attrs(output_fields, out),
#             )
#         )

#     else:
#         # Statically defined classes using shell_task as decorator
#         if (
#             executable,
#             input_fields,
#             output_fields,
#             bases,
#             inputs_bases,
#             outputs_bases,
#         ) != (None, None, None, None, None, None):
#             raise RuntimeError(
#                 "When used as a decorator on a class, `shell_task` should not be "
#                 "provided any other arguments"
#             )
#         klass = klass_or_name
#         name = klass.__name__

#         bases = [klass]
#         if not issubclass(klass, pydra.engine.task.ShellCommandTask):
#             bases.append(pydra.engine.task.ShellCommandTask)

#         try:
#             executable = klass.executable
#         except AttributeError:
#             raise RuntimeError(
#                 "Classes decorated by `shell_task` should contain an `executable` "
#                 "attribute specifying the shell tool to run"
#             )
#         try:
#             Inputs = klass.Inputs
#         except AttributeError:
#             raise RuntimeError(
#                 "Classes decorated by `shell_task` should contain an `Inputs` class "
#                 "attribute specifying the inputs to the shell tool"
#             )

#         try:
#             Outputs = klass.Outputs
#         except AttributeError:
#             Outputs = type("Outputs", (pydra.engine.specs.ShellOutSpec,), {})

#         # Pass Inputs and Outputs in attrs.define if they are present in klass (i.e.
#         # not in a base class)
#         if "Inputs" in klass.__dict__:
#             Inputs = attrs.define(kw_only=True, slots=False)(Inputs)
#         if "Outputs" in klass.__dict__:
#             Outputs = attrs.define(kw_only=True, slots=False)(Outputs)

#     if not issubclass(Inputs, pydra.engine.specs.ShellSpec):
#         Inputs = attrs.define(kw_only=True, slots=False)(
#             type("Inputs", (Inputs, pydra.engine.specs.ShellSpec), {})
#         )

#     template_fields = _gen_output_template_fields(Inputs, Outputs)

#     if not issubclass(Outputs, pydra.engine.specs.ShellOutSpec):
#         outputs_bases = (Outputs, pydra.engine.specs.ShellOutSpec)
#         add_base_class = True
#     else:
#         outputs_bases = (Outputs,)
#         add_base_class = False

#     if add_base_class or template_fields:
#         Outputs = attrs.define(kw_only=True, slots=False)(
#             type("Outputs", outputs_bases, template_fields)
#         )

#     dct["Inputs"] = Inputs
#     dct["Outputs"] = Outputs

#     task_klass = type(name, tuple(bases), dct)

#     if not hasattr(task_klass, "executable"):
#         raise RuntimeError(
#             "Classes generated by `shell_task` should contain an `executable` "
#             "attribute specifying the shell tool to run"
#         )

#     task_klass.input_spec = pydra.engine.specs.SpecInfo(
#         name=f"{name}Inputs", fields=[], bases=(task_klass.Inputs,)
#     )
#     task_klass.output_spec = pydra.engine.specs.SpecInfo(
#         name=f"{name}Outputs", fields=[], bases=(task_klass.Outputs,)
#     )

#     return task_klass


# def _gen_output_template_fields(Inputs: type, Outputs: type) -> dict:
#     """Auto-generates output fields for inputs that specify an 'output_file_template'

#     Parameters
#     ----------
#     Inputs : type
#         Inputs specification class
#     Outputs : type
#         Outputs specification class

#     Returns
#     -------
#     template_fields: dict[str, attrs._make_CountingAttribute]
#         the template fields to add to the output spec
#     """
#     annotations = {}
#     template_fields = {"__annotations__": annotations}
#     output_field_names = [f.name for f in attrs.fields(Outputs)]
#     for fld in attrs.fields(Inputs):
#         if "output_file_template" in fld.metadata:
#             if "output_field_name" in fld.metadata:
#                 field_name = fld.metadata["output_field_name"]
#             else:
#                 field_name = fld.name
#             # skip adding if the field already in the output_spec
#             exists_already = field_name in output_field_names
#             if not exists_already:
#                 metadata = {
#                     "help_string": fld.metadata["help_string"],
#                     "mandatory": fld.metadata["mandatory"],
#                     "keep_extension": fld.metadata["keep_extension"],
#                 }
#                 template_fields[field_name] = attrs.field(metadata=metadata)
#                 annotations[field_name] = str
#     return template_fields
