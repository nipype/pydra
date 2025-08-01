import typing as ty
import inspect
import attrs
import re
from copy import copy
from pydra.utils.typing import is_type, is_optional
from pydra.utils.general import get_fields
from .field import Field, Arg, Out, NO_DEFAULT


if ty.TYPE_CHECKING:
    from .task import Task, Outputs


def is_set(value: ty.Any) -> bool:
    """Check if a value has been set."""
    return value not in (attrs.NOTHING, NO_DEFAULT)


def ensure_field_objects(
    arg_type: type[Arg],
    out_type: type[Out],
    doc_string: str | None = None,
    inputs: dict[str, Arg | type] | None = None,
    outputs: dict[str, Out | type] | None = None,
    input_helps: dict[str, str] | None = None,
    output_helps: dict[str, str] | None = None,
) -> tuple[dict[str, Arg], dict[str, Out]]:
    """Converts dicts containing input/output types into input/output, including any
    help strings to the appropriate inputs and outputs

    Parameters
    ----------
    arg_type : type
        The type of the input fields
    out_type : type
        The type of the output fields
    doc_string : str, optional
        The docstring of the function or class
    inputs : dict[str, Arg | type], optional
        The inputs to the function or class
    outputs : dict[str, Out | type], optional
        The outputs of the function or class
    input_helps : dict[str, str], optional
        The help strings for the inputs
    output_helps : dict[str, str], optional
        The help strings for the outputs

    Returns
    -------
    inputs : dict[str, Arg]
        The input fields with help strings added
    outputs : dict[str, Out]
        The output fields with help strings added
    """

    for input_name, arg in list(inputs.items()):
        if isinstance(arg, Arg):
            if arg.name is None:
                arg.name = input_name
            elif arg.name != input_name:
                raise ValueError(
                    "Name of the argument must be the same as the key in the "
                    f"dictionary. The argument name is {arg.name} and the key "
                    f"is {input_name}"
                )
            else:
                arg.name = input_name
            if not arg.help:
                arg.help = input_helps.get(input_name, "") if input_helps else ""
        elif is_type(arg):
            inputs[input_name] = arg_type(
                type=arg,
                name=input_name,
                help=input_helps.get(input_name, ""),
            )
        elif isinstance(arg, dict):
            arg_kwds = copy(arg)
            if "help" not in arg_kwds:
                arg_kwds["help"] = input_helps.get(input_name, "")
            inputs[input_name] = arg_type(
                name=input_name,
                **arg_kwds,
            )
        else:
            raise ValueError(
                f"Input {input_name} must be an instance of {Arg}, a type, or a dictionary "
                f" of keyword arguments to pass to {Arg}, not {arg}"
            )

    for output_name, out in list(outputs.items()):
        if isinstance(out, Out):
            if out.name is None:
                out.name = output_name
            elif out.name != output_name:
                raise ValueError(
                    "Name of the argument must be the same as the key in the "
                    f"dictionary. The argument name is {out.name} and the key "
                    f"is {output_name}"
                )
            else:
                out.name = output_name
            if not out.help:
                out.help = output_helps.get(output_name, "") if output_helps else ""
        elif is_type(out):
            outputs[output_name] = out_type(
                type=out,
                name=output_name,
                help=output_helps.get(output_name, ""),
            )
            if is_optional(out):
                outputs[output_name].default = None
        elif isinstance(out, dict):
            out_kwds = copy(out)
            if "help" not in out_kwds:
                out_kwds["help"] = output_helps.get(output_name, "")
            if "path_template" in out_kwds:
                from pydra.compose.shell import outarg

                out_type_ = outarg
            else:
                out_type_ = out_type
            outputs[output_name] = out_type_(
                name=output_name,
                **out_kwds,
            )
        elif isinstance(out, ty.Callable) and hasattr(out_type, "callable"):
            outputs[output_name] = out_type(
                name=output_name,
                type=ty.get_type_hints(out).get("return", ty.Any),
                callable=out,
                help=re.split(r"\n\s*\n", out.__doc__)[0] if out.__doc__ else "",
            )
        else:
            raise ValueError(
                f"Unrecognised value provided to outputs ({arg}), can be either {out_type} "
                "type" + (" or callable" if hasattr(out_type, "callable") else "")
            )

    return inputs, outputs


def extract_function_inputs_and_outputs(
    function: ty.Callable,
    arg_type: type[Arg],
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
) -> tuple[dict[str, type | Arg], dict[str, type | Out]]:
    """Extract input output types and output names from the function source if they
    aren't explicitly

    Parameters
    ----------
    function : callable
        The function to extract the inputs and outputs from
    arg_type : type
        The type of the input fields
    out_type : type
        The type of the output fields
    inputs : list[str | Arg] | dict[str, Arg | type] | None
        The inputs to the function
    outputs : list[str | Out] | dict[str, Out | type] | type | None
        The outputs of the function

    Returns
    -------
    inputs : dict[str, Arg]
        The input fields extracted from the function
    outputs : dict[str, Out]
        The output fields extracted from the function
    """
    # if undefined_symbols := get_undefined_symbols(
    #     function, exclude_signature_type_hints=True, ignore_decorator=True
    # ):
    #     raise ValueError(
    #         f"The following symbols are not defined within the scope of the function "
    #         f"{function!r}, {undefined_symbols}. Ensure that all imports are "
    #         "defined within the function scope so it is portable"
    #     )
    sig = inspect.signature(function)
    type_hints = ty.get_type_hints(function)
    input_types = {}
    input_defaults = {}
    has_varargs = False
    for p in sig.parameters.values():
        if p.kind is p.VAR_POSITIONAL or p.kind is p.VAR_KEYWORD:
            has_varargs = True
            continue
        input_types[p.name] = type_hints.get(p.name, ty.Any)
        if p.default is not inspect.Parameter.empty:
            input_defaults[p.name] = p.default
    if inputs is not None:
        if not isinstance(inputs, dict):
            if non_named_args := [
                i for i in inputs if not isinstance(i, Arg) or i.name is None
            ]:
                raise ValueError(
                    "Only named Arg objects should be provided as inputs (i.e. not names or "
                    "other objects should not be provided when wrapping/decorating a "
                    f"function: found {non_named_args} when wrapping/decorating {function!r}"
                )
            inputs = {i.name: i for i in inputs}
        if not has_varargs:
            if unrecognised := set(inputs) - set(input_types):
                raise ValueError(
                    f"Unrecognised input names ({unrecognised}) not present in the signature "
                    f"of the function {function!r}"
                )
        for inpt_name, type_ in input_types.items():
            try:
                inpt = inputs[inpt_name]
            except KeyError:
                inputs[inpt_name] = type_
            else:
                if isinstance(inpt, Arg) and inpt.type is ty.Any:
                    inpt.type = type_
    else:
        inputs = input_types
    for inpt_name, default in input_defaults.items():
        inpt = inputs[inpt_name]
        if isinstance(inpt, arg_type):
            if inpt.mandatory:
                inpt.default = default
        elif inspect.isclass(inpt) or ty.get_origin(inpt):
            inputs[inpt_name] = arg_type(type=inpt, default=default)
        elif isinstance(inpt, dict):
            inputs[inpt_name] = arg_type(**inpt)
        else:
            raise ValueError(
                f"Unrecognised input type ({inpt}) for input {inpt_name} with default "
                f"value {default}"
            )
    return_type = type_hints.get("return", ty.Any)
    if outputs:
        if len(outputs) > 1:
            if return_type is not ty.Any:
                if ty.get_origin(return_type) is not tuple:
                    raise ValueError(
                        f"Multiple outputs specified ({outputs}) but non-tuple "
                        f"return value {return_type}"
                    )
                return_types = ty.get_args(return_type)
                if len(return_types) != len(outputs):
                    raise ValueError(
                        f"Length of the outputs ({outputs}) does not match that "
                        f"of the return types ({return_types})"
                    )
                output_types = dict(zip(outputs, return_types))
            else:
                output_types = {o: ty.Any for o in outputs}
            if isinstance(outputs, dict):
                for output_name, output in outputs.items():
                    if isinstance(output, Out) and output.type is ty.Any:
                        output.type = output_types[output_name]
            else:
                outputs = output_types
        else:
            if isinstance(outputs, dict):
                output_name, output = next(iter(outputs.items()))
            elif isinstance(outputs, list):
                output_name = outputs[0]
                output = ty.Any
            if isinstance(output, Out):
                if output.type is ty.Any:
                    output.type = return_type
            elif output is ty.Any:
                output = return_type
            outputs = {output_name: output}
    elif outputs == [] or return_type in (None, type(None)):
        outputs = {}
    else:
        outputs = {"out": return_type}
    return inputs, outputs


def parse_doc_string(doc_str: str) -> tuple[dict[str, str], dict[str, str] | list[str]]:
    """Parse the docstring to pull out the description of the parameters/args and returns

    Parameters
    -----------
    doc_string
        the doc string to parse

    Returns
    -------
    input_helps
        the documentation for each of the parameter/args of the class/function
    output_helps
        the documentation for each of the return values of the class function, if no
        names are provided then the help strings are returned as a list
    """
    input_helps = {}
    output_helps = {}
    if doc_str is None:
        return input_helps, output_helps
    for param, param_help in re.findall(r":param (\w+): (.*)", doc_str):
        input_helps[param] = param_help
    for return_val, return_help in re.findall(r":return (\w+): (.*)", doc_str):
        output_helps[return_val] = return_help
    google_args_match = re.match(
        r"(?:.*\n)?\s*Args:\n(.*)", doc_str, flags=re.DOTALL | re.MULTILINE
    )
    google_returns_match = re.match(
        r"(?:.*\n)?\s*Returns:\n(.*)", doc_str, flags=re.DOTALL | re.MULTILINE
    )
    if google_args_match:
        args_str = google_args_match.group(1)
        for arg_str in split_block(args_str):
            arg_name, arg_help = arg_str.split(":", maxsplit=1)
            arg_name = arg_name.strip()
            arg_help = white_space_re.sub(" ", arg_help).strip()
            input_helps[arg_name] = arg_help
    if google_returns_match:
        returns_str = google_returns_match.group(1)
        for return_str in split_block(returns_str):
            return_name, return_help = return_str.split(":", maxsplit=1)
            return_name = return_name.strip()
            return_help = white_space_re.sub(" ", return_help).strip()
            output_helps[return_name] = return_help
    numpy_args_match = re.match(
        r"(?:.*\n)?\s+Parameters\n\s*----------\s*\n(.*)",
        doc_str,
        flags=re.DOTALL | re.MULTILINE,
    )
    numpy_returns_match = re.match(
        r"(?:.*\n)?\s+Returns\n\s*-------\s*\n(.*)",
        doc_str,
        flags=re.DOTALL | re.MULTILINE,
    )
    if numpy_args_match:
        args_str = numpy_args_match.group(1)
        for arg_str in split_block(args_str):
            arg_decl, arg_help = arg_str.split("\n", maxsplit=1)
            arg_name = arg_decl.split(":")[0].strip()
            arg_help = white_space_re.sub(" ", arg_help).strip()
            input_helps[arg_name] = arg_help
    if numpy_returns_match:
        returns_str = numpy_returns_match.group(1)
        for return_str in split_block(returns_str):
            return_decl, return_help = return_str.split("\n", maxsplit=1)
            return_name = return_decl.split(":")[0].strip()
            return_help = white_space_re.sub(" ", return_help).strip()
            output_helps[return_name] = return_help
    return input_helps, output_helps


def split_block(string: str) -> ty.Generator[str, None, None]:
    """Split a block of text into groups lines"""
    indent_re = re.compile(r"^\s*")
    leading_indent = indent_re.match(string).group()
    leading_indent_len = len(leading_indent)
    block = ""
    for line in string.split("\n"):
        if not line.strip():
            break
        indent_len = len(indent_re.match(line).group())
        if block and indent_len == leading_indent_len:
            yield block.strip()
            block = ""
        block += line + "\n"
        if indent_len < leading_indent_len:
            raise ValueError(
                f"Indentation block is not consistent in docstring:\n{string}"
            )
    if block:
        yield block.strip()


def check_explicit_fields_are_none(klass, inputs, outputs):
    if inputs is not None:
        raise ValueError(
            f"inputs should not be provided to `python.task` ({inputs}) "
            f"explicitly when decorated a class ({klass})"
        )
    if outputs is not None:
        raise ValueError(
            f"outputs should not be provided to `python.task` ({outputs}) "
            f"explicitly when decorated a class ({klass})"
        )


def sanitize_xor(
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]],
) -> set[frozenset[str]]:
    """Convert a list of xor sets into a set of frozensets"""
    # Convert a single xor set into a set of xor sets
    if not xor:
        xor = frozenset()
    elif all(isinstance(x, str) or x is None for x in xor):
        xor = frozenset([frozenset(xor)])
    else:
        xor = frozenset(frozenset(x) for x in xor)
    return xor


def extract_fields_from_class(
    spec_type: type["Task"],
    outputs_type: type["Outputs"],
    klass: type,
    arg_type: type[Arg],
    out_type: type[Out],
    auto_attribs: bool,
    skip_fields: ty.Iterable[str] = (),
) -> tuple[dict[str, Arg], dict[str, Out]]:
    """Extract the input and output fields from an existing class

    Parameters
    ----------
    klass : type
        The class to extract the fields from
    arg_type : type
        The type of the input fields
    out_type : type
        The type of the output fields
    auto_attribs : bool
        Whether to assume that all attribute annotations should be interpreted as
        fields or not
    skip_fields : Iterable[str], optional
        The names of attributes to skip when extracting the fields, by default ()

    Returns
    -------
    inputs : dict[str, Arg]
        The input fields extracted from the class
    outputs : dict[str, Out]
        The output fields extracted from the class
    """

    input_helps, _ = parse_doc_string(klass.__doc__)

    def extract_fields(klass, field_type, auto_attribs, helps) -> dict[str, Field]:
        """Get the fields from a class"""
        fields_dict = {}
        # Get fields defined in base classes if present
        for field in get_fields(klass):
            if field.name not in skip_fields:
                fields_dict[field.name] = field
        type_hints = ty.get_type_hints(klass)
        for atr_name in dir(klass):
            if (
                atr_name == "Outputs"
                or atr_name in skip_fields
                or atr_name.startswith("__")
            ):
                continue
            try:
                atr = getattr(klass, atr_name)
            except Exception:
                continue
            if isinstance(atr, Field):
                atr.name = atr_name
                fields_dict[atr_name] = atr
                if atr_name in type_hints:
                    atr.type = type_hints[atr_name]
                if not atr.help:
                    atr.help = helps.get(atr_name, "")
            elif atr_name in type_hints:
                if atr_name.startswith("_"):
                    continue
                if atr_name in fields_dict:
                    fields_dict[atr_name].type = type_hints[atr_name]
                elif auto_attribs:
                    fields_dict[atr_name] = field_type(
                        name=atr_name,
                        type=type_hints[atr_name],
                        default=atr,
                        help=helps.get(atr_name, ""),
                    )
        if auto_attribs:
            for atr_name, type_ in type_hints.items():
                if atr_name.startswith("_") or atr_name in skip_fields:
                    continue
                if atr_name not in list(fields_dict) + ["Outputs"]:
                    fields_dict[atr_name] = field_type(
                        name=atr_name, type=type_, help=helps.get(atr_name, "")
                    )
        return fields_dict

    if not issubclass(klass, spec_type):
        raise ValueError(
            f"When using the canonical form for {spec_type.__module__.split('.')[-1]} "
            f"tasks, {klass} must inherit from {spec_type}"
        )

    inputs = extract_fields(klass, arg_type, auto_attribs, input_helps)

    try:
        outputs_klass = klass.Outputs
    except AttributeError:
        raise AttributeError(
            f"Nested Outputs class not found in {klass.__name__}"
        ) from None
    if not issubclass(outputs_klass, outputs_type):
        raise ValueError(
            f"When using the canonical form for {outputs_type.__module__.split('.')[-1]} "
            f"task outputs {outputs_klass}, you must inherit from {outputs_type}"
        )

    output_helps, _ = parse_doc_string(outputs_klass.__doc__)
    outputs = extract_fields(outputs_klass, out_type, auto_attribs, output_helps)

    return inputs, outputs


white_space_re = re.compile(r"\s+")
