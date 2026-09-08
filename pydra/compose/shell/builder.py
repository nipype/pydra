"""Decorators and helper functions to create ShellTasks used in Pydra workflows"""

from __future__ import annotations
import typing as ty
import re
import glob
from collections import defaultdict
import inspect
from copy import copy
import attrs
import builtins
from typing import dataclass_transform
from fileformats.core import from_mime
from fileformats import generic
from fileformats.core.exceptions import FormatRecognitionError
from pydra.utils.general import attrs_values
from pydra.compose.base import (
    Arg,
    Out,
    check_explicit_fields_are_none,
    extract_fields_from_class,
    ensure_field_objects,
    build_task_class,
    sanitize_xor,
    NO_DEFAULT,
)
from pydra.utils.typing import (
    is_fileset_or_union,
    MultiInputObj,
    TypeParser,
    is_optional,
)
from . import field
from .task import Task, Outputs


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(field.out, field.outarg),
)
def outputs(wrapped):
    """Decorator to specify the output fields of a shell command is a dataclass-style type"""
    return wrapped


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(field.arg,),
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
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]] = (),
) -> "Task":
    """Create a task for a shell command. Can be used either as a decorator on
    the "canonical" dataclass-form of a task or as a function that takes a
    "shell-command template string" of the form

    ```
    shell.define("command <input1> <input2> --output <out|output1>")
    ```

    Fields are inferred from the template if not provided. In the template, inputs are
    specified with `<fieldname>` and outputs with `<out|fieldname>`.

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
    xor: Sequence[str | None] | Sequence[Sequence[str | None]], optional
        Names of args that are exclusive mutually exclusive, which must include
        the name of the current field. If this list includes None, then none of the
        fields need to be set.

    Returns
    -------
    Task
        The interface for the shell command
    """

    def make(
        wrapped: ty.Callable | type | None = None,
    ) -> Task:

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
                Task,
                Outputs,
                klass,
                field.arg,
                field.out,
                auto_attribs,
                skip_fields=["executable"],
            )
        else:
            if not isinstance(wrapped, (str, list)):
                raise ValueError(
                    f"wrapped must be a class or a string, not {wrapped!r}"
                )
            klass = None
            input_helps, output_helps = {}, {}

            executable, inferred_inputs, inferred_outputs = parse_command_line_template(
                wrapped,
                inputs=inputs,
                outputs=outputs,
            )

            parsed_inputs, parsed_outputs = ensure_field_objects(
                arg_type=field.arg,
                out_type=field.out,
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
            parsed_inputs.update({n: getattr(Task, n) for n in Task.BASE_ATTRS})
            parsed_outputs.update({n: getattr(Outputs, n) for n in Outputs.BASE_ATTRS})

        if "executable" in parsed_inputs:
            raise ValueError(
                "The argument 'executable' is reserved for a field to hold the command "
                "to be run"
            )

        # Update the inputs (overriding inputs from base classes) with the executable
        # and the output argument fields
        parsed_inputs.update(
            {o.name: o for o in parsed_outputs.values() if isinstance(o, field.arg)}
        )
        parsed_inputs["executable"] = field.arg(
            name="executable",
            type=str | ty.Sequence[str],
            argstr="",
            position=0,
            default=executable,
            validator=attrs.validators.min_len(1),
            help=Task.EXECUTABLE_HELP,
        )

        # Set positions for the remaining inputs that don't have an explicit position
        position_stack = remaining_positions(list(parsed_inputs.values()), xor=xor)
        for inpt in parsed_inputs.values():
            if inpt.name == "append_args":
                continue
            if inpt.position is None:
                inpt.position = position_stack.pop(0)

        # Convert string default values to callables that glob the files in the cwd
        for outpt in parsed_outputs.values():
            if (
                isinstance(outpt, field.out)
                and isinstance(outpt.default, str)
                and TypeParser.contains_type(generic.FileSet, outpt.type)
            ):
                outpt.callable = GlobCallable(outpt.default)
                outpt.default = NO_DEFAULT

        defn = build_task_class(
            Task,
            Outputs,
            parsed_inputs,
            parsed_outputs,
            name=class_name,
            klass=klass,
            bases=bases,
            outputs_bases=outputs_bases,
            xor=xor,
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
        if not isinstance(wrapped, (type, str, list)):
            raise ValueError(
                f"wrapped must be a class, a string or a list, not {wrapped!r}"
            )
        return make(wrapped)
    return make


def parse_command_line_template(
    template: str,
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | None = None,
) -> ty.Tuple[str, dict[str, Arg | type], dict[str, Out | type]]:
    """Parses a command line template into a name and input and output fields. Fields
    are inferred from the template if not explicitly provided.

    In the template, inputs are specified with `<fieldname>` and outputs with `<out|fieldname>`.
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
        outputs = {o.name: o for o in outputs}
    elif isinstance(outputs, dict):
        outputs = copy(outputs)  # We don't want to modify the original
    else:
        assert outputs is None
        outputs = {}
    if isinstance(template, list):
        tokens = template
    else:
        tokens = template.split()
    executable = []
    start_args_index = 0
    for part in tokens:
        if part.startswith("<") or part.startswith("-"):
            break
        executable.append(part)
        start_args_index += 1
    if not executable:
        raise ValueError(f"Found no executable in command line template: {template}")
    if len(executable) == 1:
        executable = executable[0]
    tokens = tokens[start_args_index:]
    if not tokens:
        return executable, inputs, outputs
    arg_pattern = r"<([:a-zA-Z0-9_,\|\-\.\/\+\*]+(?:\?|(?:=|\$)[^>]+)?)>"
    opt_pattern = r"--?[a-zA-Z0-9_\-]+"
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
            fld = dct.pop(name)
        except KeyError:
            fld = field_type(name=name, **kwds)
        else:
            if isinstance(fld, dict):
                fld = field_type(**fld)
            elif isinstance(fld, type) or ty.get_origin(fld):
                kwds["type"] = fld
                fld = field_type(name=name, **kwds)
            elif not isinstance(fld, field_type):  # If fld type is outarg not out
                fld = field_type(**attrs_values(fld))
            fld.name = name
            type_ = kwds.pop("type", fld.type)
            if fld.type is ty.Any:
                fld.type = type_
            for k, v in kwds.items():
                setattr(fld, k, v)
        dct[name] = fld
        if issubclass(field_type, Arg):
            arguments.append(fld)

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
                field_type = field.outarg
            elif name.startswith("modify|"):
                name = name[7:]
                field_type = field.arg
                modify = True
            else:
                field_type = field.arg
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
                kwds["default"] = (
                    default[1:-1] if re.match(r"('|\").*\1", default) else eval(default)
                )
            elif "$" in name:
                name, path_template = name.split("$")
                kwds["path_template"] = path_template
                if field_type is not field.outarg:
                    raise ValueError(
                        f"Path templates can only be used with output fields, not {token}"
                    )
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
                add_arg(
                    name,
                    field.out,
                    {"type": type_, "callable": _InputPassThrough(name)},
                )
            # If name contains a '.', treat it as a file template and strip it from the name
            if field_type is field.outarg and "path_template" not in kwds:
                path_template = name
                if is_fileset_or_union(type_):
                    if ty.get_origin(type_):
                        ext_type = next(a for a in ty.get_args(type_) if a is not None)
                    else:
                        ext_type = type_
                    if ext_type.ext is not None:
                        path_template = name + ext_type.ext
                kwds["path_template"] = path_template
            # Set the default value to None if the field is optional and no default is
            # provided
            if is_optional(type_) and "default" not in kwds:
                kwds["default"] = None
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
            add_arg(
                var, field.arg, {"type": bool, "argstr": argstr, "default": default}
            )
            option = None
        elif match := opt_re.match(token):
            option = token
        else:
            raise ValueError(
                f"Found unknown token {token!r} in command line template: {template}"
            )

    if option:
        raise ValueError(f"Found an option without a field: {option!r}")

    remaining_pos = remaining_positions(arguments, len(arguments) + 1, 1)

    for argument in arguments:
        if argument.position is None:
            argument.position = remaining_pos.pop(0)

    return executable, inputs, outputs


def remaining_positions(
    args: list[Arg],
    num_args: int | None = None,
    start: int = 0,
    xor: set[frozenset[str]] | None = None,
) -> ty.List[int]:
    """Get the remaining positions for input fields

    Parameters
    ----------
    args : list[Arg]
        The list of input fields
    num_args : int, optional
        The number of arguments, by default it is the length of the args
    start : int, optional
        The starting position, by default 0
    xor : set[frozenset[str]], optional
        A set of mutually exclusive fields, by default None

    Returns
    -------
    list[int]
        The list of remaining positions

    Raises
    ------
    ValueError
        If multiple fields have the same position
    """
    xor = sanitize_xor(xor)
    if num_args is None:
        num_args = len(args) - 1  # Subtract 1 for the 'append_args' field
    # Check for multiple positions
    positions = defaultdict(list)
    for arg in args:
        if arg.name == "append_args":
            continue
        if arg.position is not None:
            if arg.position >= 0:
                positions[arg.position].append(arg)
            else:
                positions[num_args + arg.position].append(arg)
    if multiple_positions := {
        k: [f"{a.name}({a.position})" for a in v]
        for k, v in positions.items()
        if len(v) > 1 and not any(x.issuperset(a.name for a in v) for x in xor)
    }:
        raise ValueError(
            f"Multiple fields have the overlapping positions: {multiple_positions}"
        )
    return [i for i in range(start, num_args) if i not in positions]


@attrs.define
class _InputPassThrough:
    """A class that can be used to pass through an input to the output"""

    name: str

    def __call__(self, inputs: Task) -> ty.Any:
        return getattr(inputs, self.name)


class GlobCallable:
    """Callable that can be used to glob files"""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def __call__(self) -> generic.FileSet:
        matches = glob.glob(self.pattern)
        if not matches:
            raise FileNotFoundError(f"No files found matching pattern: {self.pattern}")
        return matches
