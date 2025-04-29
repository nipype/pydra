import typing as ty
import re
import os
import inspect
from copy import copy
from pathlib import Path
from fileformats.generic import FileSet
from pydra.utils.general import attrs_values, get_fields
from pydra.utils.typing import is_lazy
from . import field

if ty.TYPE_CHECKING:
    from . import Task


def template_update(
    task,
    cache_dir: Path | None = None,
    map_copyfiles: dict[str, Path] | None = None,
):
    """
    Update all templates that are present in the input task.

    Should be run when all inputs used in the templates are already set.

    """

    values = attrs_values(task)
    if map_copyfiles is not None:
        values.update(map_copyfiles)

    # Collect templated inputs for which all requirements are satisfied.
    fields_templ = [
        fld
        for fld in get_fields(task)
        if isinstance(fld, field.outarg)
        and fld.path_template
        and getattr(task, fld.name)
        and all(req.satisfied(task) for req in fld.requires)
    ]

    dict_mod = {}
    for fld in fields_templ:
        dict_mod[fld.name] = template_update_single(
            fld=fld,
            task=task,
            values=values,
            cache_dir=cache_dir,
        )
    # adding elements from map_copyfiles to fields with templates
    if map_copyfiles:
        dict_mod.update(map_copyfiles)
    return dict_mod


def template_update_single(
    fld: "field.outarg",
    task: "Task",
    values: dict[str, ty.Any] = None,
    cache_dir: Path | None = None,
    spec_type: str = "input",
) -> Path | list[Path | None] | None:
    """Update a single template from the input_spec or output_spec
    based on the value from inputs_dict
    (checking the types of the fields, that have "output_file_template)"
    """
    # if input_dict_st with state specific value is not available,
    # the dictionary will be created from inputs object
    from pydra.utils.typing import TypeParser, OUTPUT_TEMPLATE_TYPES  # noqa

    if values is None:
        values = attrs_values(task)

    if spec_type == "input":
        field_value = values[fld.name]
        if isinstance(field_value, bool) and fld.type in (Path, str):
            raise TypeError(
                f"type of '{fld.name}' is Path, consider using Union[Path, bool]"
            )
        if field_value is not None and not is_lazy(field_value):
            field_value = TypeParser(ty.Union[OUTPUT_TEMPLATE_TYPES])(field_value)
    elif spec_type == "output":
        if not TypeParser.contains_type(FileSet, fld.type):
            raise TypeError(
                f"output {fld.name} should be file-system object, but {fld.type} "
                "set as the type"
            )
    else:
        raise TypeError(f"spec_type can be input or output, but {spec_type} provided")
    # for inputs that the value is set (so the template is ignored)
    if spec_type == "input":
        if isinstance(field_value, (Path, list)):
            return field_value
        if field_value is False:
            # if input fld is set to False, the fld shouldn't be used (setting NOTHING)
            return None
    # inputs_dict[fld.name] is True or spec_type is output
    value = _template_formatting(fld, task, values)
    if cache_dir and value is not None:
        # changing path so it is in the cache_dir
        # should be converted to str, it is also used for input fields that should be str
        if type(value) is list:
            value = [cache_dir / val.name for val in value]
        else:
            value = cache_dir / value.name
    return value


def _template_formatting(
    fld: "field.arg", task: "Task", values: dict[str, ty.Any]
) -> Path | list[Path] | None:
    """Formatting the fld template based on the values from inputs.
    Taking into account that the fld with a template can be a MultiOutputFile
    and the fld values needed in the template can be a list -
    returning a list of formatted templates in that case.
    Allowing for multiple input values used in the template as longs as
    there is no more than one file (i.e. File, PathLike or string with extensions)

    Parameters
    ----------
    fld : pydra.utils.general.Field
        field with a template
    task : pydra.compose.shell.Task
        the task
    values : dict
        dictionary with values from inputs object

    Returns
    -------
    formatted : Path or list[Path | None] or None
        formatted template
    """
    # if a template is a function it has to be run first with the inputs as the only arg
    template = fld.path_template
    if callable(template):
        template = template(task)

    # as default, we assume that keep_extension is True
    if isinstance(template, (tuple, list)):
        formatted = [_single_template_formatting(fld, t, values) for t in template]
        if any([val is None for val in formatted]):
            return None
    else:
        assert isinstance(template, str)
        formatted = _single_template_formatting(fld, template, values)
    return formatted


def _single_template_formatting(
    fld: "field.outarg",
    template: str,
    values: dict[str, ty.Any],
) -> Path | None:
    from pydra.utils.typing import MultiInputObj, MultiOutputFile

    inp_fields = re.findall(r"{\w+}", template)
    inp_fields_fl = re.findall(r"{\w+:[0-9.]+f}", template)
    inp_fields += [re.sub(":[0-9.]+f", "", el) for el in inp_fields_fl]

    # FIXME: This would be a better solution, and would allow you to explicitly specify
    # whether you want to use the extension of the input file or not, by referencing
    # the "ext" attribute of the input file. However, this would require a change in the
    # way the element formatting is done
    #
    # inp_fields = set(re.findall(r"{(\w+)(?:\.\w+)?(?::[0-9.]+f)?}", template))

    if len(inp_fields) == 0:
        return Path(template)

    val_dict = {}
    file_template = None

    for inp_fld in inp_fields:
        fld_name = inp_fld[1:-1]  # extracting the name form {field_name}
        if fld_name not in values:
            raise AttributeError(f"{fld_name} is not provided in the input")
        fld_value = values[fld_name]
        if fld_value is None:
            # if value is NOTHING, nothing should be added to the command
            return None
        # checking for fields that can be treated as a file:
        # have type File, or value that is path like (including str with extensions)
        if isinstance(fld_value, os.PathLike):
            if file_template:
                raise Exception(
                    f"can't have multiple paths in {fld.name} template,"
                    f" but {template} provided"
                )
            else:
                file_template = (fld_name, fld_value)
        else:
            val_dict[fld_name] = fld_value

    # if field is MultiOutputFile and some elements from val_dict are lists,
    # each element of the list should be used separately in the template
    # and return a list with formatted values
    if fld.type is MultiOutputFile and any(
        [isinstance(el, (list, MultiInputObj)) for el in val_dict.values()]
    ):
        # all fields that are lists
        keys_list = [
            k for k, el in val_dict.items() if isinstance(el, (list, MultiInputObj))
        ]
        if any(
            [len(val_dict[key]) != len(val_dict[keys_list[0]]) for key in keys_list[1:]]
        ):
            raise Exception(
                f"all fields used in {fld.name} template have to have the same length"
                f" or be a single value"
            )
        formatted_value = []
        for ii in range(len(val_dict[keys_list[0]])):
            val_dict_el = copy(val_dict)
            # updating values to a single element from the list
            for key in keys_list:
                val_dict_el[key] = val_dict[key][ii]

            formatted_value.append(
                _element_formatting(
                    template,
                    val_dict_el,
                    file_template,
                    keep_extension=fld.keep_extension,
                )
            )
    else:
        formatted_value = _element_formatting(
            template, val_dict, file_template, keep_extension=fld.keep_extension
        )
    if isinstance(formatted_value, list):
        return [Path(val) for val in formatted_value]
    elif isinstance(formatted_value, str):
        return Path(formatted_value)
    return None


def _element_formatting(
    template: str,
    values_template_dict: dict[str, ty.Any],
    file_template: str,
    keep_extension: bool,
):
    """Formatting a single template for a single element (if a list).
    Taking into account that a file used in the template (file_template)
    and the template itself could have file extensions
    (assuming that if template has extension, the field value extension is removed,
    if field has extension, and no template extension, than it is moved to the end).
    For values_template_dict the simple formatting can be used (no file values inside)
    """
    if file_template:
        fld_name_file, fld_value_file = file_template
        # splitting the filename for name and extension,
        # the final value used for formatting depends on the template and keep_extension flag
        name, *ext = Path(fld_value_file).name.split(".", maxsplit=1)
        filename = str(Path(fld_value_file).parent / name)
        # updating values_template_dic with the name of file
        values_template_dict[fld_name_file] = filename
        # if keep_extension is False, the extensions are removed
        if keep_extension is False:
            ext = []
    else:
        ext = []

    # if file_template is at the end of the template, the simplest formatting should work
    if file_template and template.endswith(f"{{{fld_name_file}}}"):
        # recreating fld_value with the updated extension
        values_template_dict[fld_name_file] = ".".join([filename] + ext)
        formatted_value = template.format(**values_template_dict)
    # file_template provided, but the template doesn't have its own extension
    elif file_template and "." not in template:
        # if the fld_value_file has extension, it will be moved to the end
        formatted_value = ".".join([template.format(**values_template_dict)] + ext)
    # template has its own extension or no file_template provided
    # the simplest formatting, if file_template is provided it's used without the extension
    else:
        formatted_value = template.format(**values_template_dict)
    return formatted_value


def parse_format_string(fmtstr: str) -> set[str]:
    """Parse a argstr format string and return all keywords used in it."""
    identifier = r"[a-zA-Z_]\w*"
    attribute = rf"\.{identifier}"
    item = r"\[\w+\]"
    # Example: var.attrs[key][0].attr2 (capture "var")
    field_with_lookups = (
        f"({identifier})(?:{attribute}|{item})*"  # Capture only the keyword
    )
    conversion = "(?:!r|!s)"
    nobrace = "[^{}]*"
    # Example: 0{pads[hex]}x (capture "pads")
    fmtspec = f"{nobrace}(?:{{({identifier}){nobrace}}}{nobrace})?"  # Capture keywords in definition
    full_field = f"{{{field_with_lookups}{conversion}?(?::{fmtspec})?}}"

    all_keywords = re.findall(full_field, fmtstr)
    return set().union(*all_keywords) - {""}


def fields_in_formatter(formatter: str | ty.Callable[..., str]) -> set[str]:
    """Extract all field names from a formatter string or function."""
    if isinstance(formatter, str):
        return parse_format_string(formatter)
    elif isinstance(formatter, ty.Sequence):
        return set().union(*[fields_in_formatter(f) for f in formatter])
    elif isinstance(formatter, ty.Callable):
        return set(inspect.signature(formatter).parameters.keys())
    else:
        raise ValueError(f"Unsupported formatter type: {type(formatter)} ({formatter})")


def argstr_formatting(argstr: str, values: dict[str, ty.Any]):
    """formatting argstr that have form {field_name},
    using values from inputs and updating with value_update if provided
    """
    # if there is a value that has to be updated (e.g. single value from a list)
    # getting all fields that should be formatted, i.e. {field_name}, ...
    inp_fields = parse_format_string(argstr)
    # formatting string based on the val_dict
    argstr_formatted = argstr.format(**{n: values.get(n, "") for n in inp_fields})
    # removing extra commas and spaces after removing the field that have NOTHING
    argstr_formatted = (
        argstr_formatted.replace("[ ", "[")
        .replace(" ]", "]")
        .replace("[,", "[")
        .replace(",]", "]")
        .strip()
    )
    return argstr_formatted
