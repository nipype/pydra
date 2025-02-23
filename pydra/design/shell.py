"""Decorators and helper functions to create ShellTasks used in Pydra workflows"""

from __future__ import annotations
import typing as ty
import re
from collections import defaultdict
import inspect
from copy import copy
import attrs
import builtins
from typing import dataclass_transform
from fileformats.core import from_mime
from fileformats import generic
from fileformats.core.exceptions import FormatRecognitionError
from pydra.engine.helpers import attrs_values
from .base import (
    Arg,
    Out,
    check_explicit_fields_are_none,
    extract_fields_from_class,
    ensure_field_objects,
    make_task_def,
    NO_DEFAULT,
)
from pydra.utils.typing import (
    is_fileset_or_union,
    MultiInputObj,
    is_optional,
    optional_type,
)

if ty.TYPE_CHECKING:
    from pydra.engine.specs import ShellDef

__all__ = ["arg", "out", "outarg", "define"]

EXECUTABLE_HELP_STRING = (
    "the first part of the command, can be a string, "
    "e.g. 'ls', or a list, e.g. ['ls', '-l', 'dirname']"
)


@attrs.define(kw_only=True)
class arg(Arg):
    """An input field that specifies a command line argument

    Parameters
    ----------
    help: str
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
    sep: str | None = attrs.field()
    allowed_values: list | None = None
    container_path: bool = False  # IS THIS STILL USED??
    formatter: ty.Callable | None = None

    @sep.default
    def _sep_default(self):
        return " " if self.type is tuple or ty.get_origin(self.type) is tuple else None

    @sep.validator
    def _validate_sep(self, _, sep):
        if self.type is ty.Any:
            return
        if ty.get_origin(self.type) is MultiInputObj:
            tp = ty.get_args(self.type)[0]
        else:
            tp = self.type
        if is_optional(tp):
            tp = optional_type(tp)
        origin = ty.get_origin(tp) or tp

        if (
            inspect.isclass(origin)
            and issubclass(origin, ty.Sequence)
            and tp is not str
        ):
            if sep is None:
                raise ValueError(
                    f"A value to 'sep' must be provided when type is iterable {tp} "
                    f"for field {self.name!r}"
                )
        elif sep is not None:
            raise ValueError(
                f"sep ({sep!r}) can only be provided when type is iterable {tp} "
                f"for field {self.name!r}"
            )


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

    def __attrs_post_init__(self):
        # Set type from return annotation of callable if not set
        if self.type is ty.Any and self.callable:
            self.type = ty.get_type_hints(self.callable).get("return", ty.Any)


@attrs.define(kw_only=True)
class outarg(arg, Out):
    """An input field that specifies where to save the output file

    Parameters
    ----------
    help: str
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
        Position of the field in the command line, could be nonnegative or negative integer.
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
    path_template: str, optional
        The template used to specify where the output file will be written to can use
        other fields, e.g. {file1}. Used in order to create an output definition.
    """

    path_template: str | None = attrs.field(default=None)
    keep_extension: bool = attrs.field(default=False)

    @path_template.validator
    def _validate_path_template(self, attribute, value):
        if value and self.default not in (NO_DEFAULT, True, None):
            raise ValueError(
                f"path_template ({value!r}) can only be provided when no default "
                f"({self.default!r}) is provided"
            )
        if value and not (is_fileset_or_union(self.type) or self.type is ty.Any):
            raise ValueError(
                f"path_template ({value!r}) can only be provided when type is a FileSet, "
                f"or union thereof, not {self.type!r}"
            )

    @keep_extension.validator
    def _validate_keep_extension(self, attribute, value):
        if value and self.path_template is None:
            raise ValueError(
                f"keep_extension ({value!r}) can only be provided when path_template "
                f"is provided"
            )


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(out, outarg),
)
def outputs(wrapped):
    """Decorator to specify the output fields of a shell command is a dataclass-style type"""
    return wrapped


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(arg,),
)
def define(
    wrapped: type | str | None = None,
    /,
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    auto_attribs: bool = True,
    name: str | None = None,
) -> "ShellDef":
    """Create a task definition for a shell command. Can be used either as a decorator on
    the "canonical" dataclass-form of a task definition or as a function that takes a
    "shell-command template string" of the form

    ```
    shell.define("command <input1> <input2> --output <out|output1>")
    ```

    Fields are inferred from the template if not provided. In the template, inputs are
    specified with `<fieldname>` and outputs with `<out:fieldname>`.

    ```
    my_command <myinput> <out|myoutput2>
    ```

    The types of the fields can be specified using their MIME like (see fileformats.core.from_mime), e.g.

    ```
    my_command <myinput:text/csv> <out|myoutput2:image/png>
    ```

    The template can also specify options with `-` or `--` followed by the option name
    and arguments with `<argname:type>`. The type is optional and will default to
    `generic/fs-object` if not provided for arguments and `field/text` for
    options. The file-formats namespace can be dropped for generic and field formats, e.g.

    ```
    another-command <input1:directory> <input2:int> --output <out|output1:text/csv>
    ```

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

    Returns
    -------
    ShellDef
        The interface for the shell command
    """
    from pydra.engine.specs import ShellDef, ShellOutputs

    def make(
        wrapped: ty.Callable | type | None = None,
    ) -> ShellDef:

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
            if not isinstance(executable, str) and not (
                isinstance(executable, ty.Sequence)
                and all(isinstance(e, str) for e in executable)
            ):
                raise ValueError(
                    "executable must be a string or a sequence of strings"
                    f", not {executable!r}"
                )
            class_name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = extract_fields_from_class(
                ShellDef, ShellOutputs, klass, arg, out, auto_attribs
            )
        else:
            if not isinstance(wrapped, str):
                raise ValueError(
                    f"wrapped must be a class or a string, not {wrapped!r}"
                )
            klass = None
            input_helps, output_helps = {}, {}

            if isinstance(wrapped, list):
                wrapped = " ".join(wrapped)

            executable, inferred_inputs, inferred_outputs = parse_command_line_template(
                wrapped,
                inputs=inputs,
                outputs=outputs,
            )

            parsed_inputs, parsed_outputs = ensure_field_objects(
                arg_type=arg,
                out_type=out,
                inputs=inferred_inputs,
                outputs=inferred_outputs,
                input_helps=input_helps,
                output_helps=output_helps,
            )
            if name:
                class_name = name
            else:
                class_name = (
                    "_".join(executable) if isinstance(executable, list) else executable
                )
                class_name = re.sub(r"[^\w]", "_", class_name)
            if class_name[0].isdigit():
                class_name = f"_{class_name}"

            # Add in fields from base classes
            parsed_inputs.update({n: getattr(ShellDef, n) for n in ShellDef.BASE_NAMES})
            parsed_outputs.update(
                {n: getattr(ShellOutputs, n) for n in ShellOutputs.BASE_NAMES}
            )

        # Update the inputs (overriding inputs from base classes) with the executable
        # and the output argument fields
        parsed_inputs.update(
            {o.name: o for o in parsed_outputs.values() if isinstance(o, arg)}
        )
        parsed_inputs["executable"] = arg(
            name="executable",
            type=str | ty.Sequence[str],
            argstr="",
            position=0,
            default=executable,
            validator=attrs.validators.min_len(1),
            help=EXECUTABLE_HELP_STRING,
        )

        # Set positions for the remaining inputs that don't have an explicit position
        position_stack = remaining_positions(list(parsed_inputs.values()))
        for inpt in parsed_inputs.values():
            if inpt.name == "additional_args":
                continue
            if inpt.position is None:
                inpt.position = position_stack.pop(0)

        defn = make_task_def(
            ShellDef,
            ShellOutputs,
            parsed_inputs,
            parsed_outputs,
            name=class_name,
            klass=klass,
            bases=bases,
            outputs_bases=outputs_bases,
        )
        return defn

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
    outputs: list[str | Out] | dict[str, Out | type] | None = None,
) -> ty.Tuple[str, dict[str, Arg | type], dict[str, Out | type]]:
    """Parses a command line template into a name and input and output fields. Fields
    are inferred from the template if not explicitly provided.

    In the template, inputs are specified with `<fieldname>` and outputs with `<out:fieldname>`.
    The types of the fields can be specified using their MIME like (see fileformats.core.from_mime), e.g.

    ```
    my_command <myinput> <out|myoutput2>
    ```

    The template can also specify options with `-` or `--`
    followed by the option name and arguments with `<argname:type>`. The type is optional and
    will default to `generic/fs-object` if not provided for arguments and `field/text` for
    options. The file-formats namespace can be dropped for generic and field formats, e.g.

    ```
    another-command <input1:directory> <input2:int> --output <out|output1:text/csv>
    ```

    Parameters
    ----------
    template : str
        The command line template
    inputs : list[str | Arg] | dict[str, Arg | type] | None
        The input fields of the shell command
    outputs : list[str | Out] | dict[str, Out | type] | type | None
        The output fields of the shell command

    Returns
    -------
    executable : str
        The name of the command line template
    inputs : dict[str, Arg | type]
        The input fields of the command line template
    outputs : dict[str, Out | type]
        The output fields of the command line template

    Raises
    ------
    ValueError
        If an unknown token is found in the command line template
    TypeError
        If an unknown type is found in the command line template
    """
    if isinstance(inputs, list):
        inputs = {arg.name: arg for arg in inputs}
    elif isinstance(inputs, dict):
        inputs = copy(inputs)  # We don't want to modify the original
    else:
        assert inputs is None
        inputs = {}
    if isinstance(outputs, list):
        outputs = {out.name: out for out in outputs}
    elif isinstance(outputs, dict):
        outputs = copy(outputs)  # We don't want to modify the original
    else:
        assert outputs is None
        outputs = {}
    parts = template.split()
    executable = []
    start_args_index = 0
    for part in parts:
        if part.startswith("<") or part.startswith("-"):
            break
        executable.append(part)
        start_args_index += 1
    if not executable:
        raise ValueError(f"Found no executable in command line template: {template}")
    if len(executable) == 1:
        executable = executable[0]
    args_str = " ".join(parts[start_args_index:])
    if not args_str:
        return executable, inputs, outputs
    tokens = re.split(r"\s+", args_str.strip())
    arg_pattern = r"<([:a-zA-Z0-9_,\|\-\.\/\+\*]+(?:\?|=[^>]+)?)>"
    opt_pattern = r"--?[a-zA-Z0-9_]+"
    arg_re = re.compile(arg_pattern)
    opt_re = re.compile(opt_pattern)
    bool_arg_re = re.compile(f"({opt_pattern}){arg_pattern}")

    arguments = []
    option = None

    def add_arg(name, field_type, kwds):
        """Merge the typing information with an existing field if it exists"""
        if issubclass(field_type, Out):
            dct = outputs
        else:
            dct = inputs
        try:
            field = dct.pop(name)
        except KeyError:
            field = field_type(name=name, **kwds)
        else:
            if isinstance(field, dict):
                field = field_type(**field)
            elif isinstance(field, type) or ty.get_origin(field):
                kwds["type"] = field
                field = field_type(name=name, **kwds)
            elif not isinstance(field, field_type):  # If field type is outarg not out
                field = field_type(**attrs_values(field))
            field.name = name
            type_ = kwds.pop("type", field.type)
            if field.type is ty.Any:
                field.type = type_
            for k, v in kwds.items():
                setattr(field, k, v)
        dct[name] = field
        if issubclass(field_type, Arg):
            arguments.append(field)

    def from_type_str(type_str) -> type:
        types = []
        for tp in type_str.split(","):
            if "/" in tp:
                type_ = from_mime(tp)
            elif tp == "...":
                type_ = "..."
            else:
                if tp in ("int", "float", "str", "bool"):
                    type_ = getattr(builtins, tp)
                else:
                    try:
                        type_ = from_mime(f"generic/{tp}")
                    except FormatRecognitionError:
                        raise TypeError(
                            f"Found unknown type, {tp!r}, in command template: {template!r}"
                        ) from None
            types.append(type_)
        if len(types) == 2 and types[1] == "...":
            type_ = tuple[types[0], ...]
        elif len(types) > 1:
            type_ = tuple[*types]
        else:
            type_ = types[0]
        return type_

    for token in tokens:
        if match := arg_re.match(token):
            name = match.group(1)
            modify = False
            if name.startswith("out|"):
                name = name[4:]
                field_type = outarg
            elif name.startswith("modify|"):
                name = name[7:]
                field_type = arg
                modify = True
            else:
                field_type = arg
            # Identify type after ':' symbols
            kwds = {}
            is_multi = False
            optional = False
            if name.endswith("?"):
                assert "=" not in name
                name = name[:-1]
                optional = True
                kwds["default"] = None
            elif name.endswith("+"):
                is_multi = True
                name = name[:-1]
            elif name.endswith("*"):
                is_multi = True
                name = name[:-1]
                kwds["default"] = attrs.Factory(list)
            elif "=" in name:
                name, default = name.split("=")
                kwds["default"] = eval(default)
            if ":" in name:
                name, type_str = name.split(":")
                type_ = from_type_str(type_str)
                if ty.get_origin(type_) is tuple:
                    kwds["sep"] = " "
            else:
                type_ = generic.FsObject if option is None else str
            if is_multi:
                type_ = MultiInputObj[type_]
            if optional:
                type_ |= None  # Make the arguments optional
            kwds["type"] = type_
            if modify:
                kwds["copy_mode"] = generic.File.CopyMode.copy
                # Add field to outputs with the same name as the input
                add_arg(name, out, {"type": type_, "callable": _InputPassThrough(name)})
            # If name contains a '.', treat it as a file template and strip it from the name
            if field_type is outarg:
                path_template = name
                if is_fileset_or_union(type_):
                    if ty.get_origin(type_):
                        ext_type = next(a for a in ty.get_args(type_) if a is not None)
                    else:
                        ext_type = type_
                    if ext_type.ext is not None:
                        path_template = name + ext_type.ext
                kwds["path_template"] = path_template
            if option is None:
                add_arg(name, field_type, kwds)
            else:
                kwds["argstr"] = option
                add_arg(name, field_type, kwds)
                option = None

        elif match := bool_arg_re.match(token):
            argstr, var = match.groups()
            if "=" in var:
                var, default = var.split("=")
                default = eval(default)
            else:
                default = False
            add_arg(var, arg, {"type": bool, "argstr": argstr, "default": default})
        elif match := opt_re.match(token):
            option = token
        else:
            raise ValueError(
                f"Found unknown token '{token}' in command line template: {template}"
            )

    remaining_pos = remaining_positions(arguments, len(arguments) + 1, 1)

    for argument in arguments:
        if argument.position is None:
            argument.position = remaining_pos.pop(0)

    return executable, inputs, outputs


def remaining_positions(
    args: list[Arg], num_args: int | None = None, start: int = 0
) -> ty.List[int]:
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
        num_args = len(args) - 1  # Subtract 1 for the 'additional_args' field
    # Check for multiple positions
    positions = defaultdict(list)
    for arg in args:
        if arg.name == "additional_args":
            continue
        if arg.position is not None:
            if arg.position >= 0:
                positions[arg.position].append(arg)
            else:
                positions[num_args + arg.position].append(arg)
    if multiple_positions := {
        k: [f"{a.name}({a.position})" for a in v]
        for k, v in positions.items()
        if len(v) > 1
    }:
        raise ValueError(
            f"Multiple fields have the overlapping positions: {multiple_positions}"
        )
    return [i for i in range(start, num_args) if i not in positions]


@attrs.define
class _InputPassThrough:
    """A class that can be used to pass through an input to the output"""

    name: str

    def __call__(self, inputs: ShellDef) -> ty.Any:
        return getattr(inputs, self.name)
