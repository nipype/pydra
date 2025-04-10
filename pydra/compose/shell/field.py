from __future__ import annotations
import typing as ty
import attrs
from pydra.compose.base import (
    Arg,
    Out,
    NO_DEFAULT,
)
from pydra.utils.typing import is_optional
from pydra.utils.general import wrap_text


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
        A separator if a sequence type is provided as a value, by default " ".
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
    sep: str = " "
    allowed_values: list | None = None
    container_path: bool = False  # IS THIS STILL USED??
    formatter: ty.Callable | None = None

    def _additional_descriptors(self, as_input: bool = False, **kwargs) -> str:
        if not self.argstr or not as_input:
            return super()._additional_descriptors(as_input=as_input, **kwargs)
        descriptors = [f"{self.argstr!r}"]
        descriptors.extend(super()._additional_descriptors(as_input=as_input, **kwargs))
        return descriptors

    def __lt__(self, other: "arg") -> bool:
        """Compare two fields based on their position"""
        if self.position is None and other.position is None:
            return super().__lt__(other)
        elif self.position is None:
            return False
        elif other.position is None:
            return True
        else:
            assert self.position != other.position, "positions should be unique"
            if self.position < 0 and other.position < 0:
                return self.position > other.position
            elif self.position < 0:
                return False
            elif other.position < 0:
                return True
            return self.position < other.position


@attrs.define(kw_only=True)
class out(Out):
    """An output field that specifies a command line argument

    Parameters
    ----------
    callable : Callable, optional
        If provided the output file name (or list of file names) is created using the
        function. The function can take field (the specific output field will be passed
        to the function), cache_dir (task cache_dir will be used), stdout, stderr
        (stdout and stderr of the task will be sent) inputs (entire inputs will be
        passed) or any input field name (a specific input field will be sent).
    """

    callable: ty.Callable | None = attrs.field(default=None)

    def __attrs_post_init__(self):
        # Set type from return annotation of callable if not set
        if self.type is ty.Any and self.callable:
            self.type = ty.get_type_hints(self.callable).get("return", ty.Any)

    @callable.validator
    def _callable_validator(self, _, value):

        if value:
            if not callable(value):
                raise ValueError(f"callable must be a function, not {value!r}")
        elif (
            self.mandatory
            and not getattr(self, "path_template", None)
            and self.name
            not in [
                "return_code",
                "stdout",
                "stderr",
            ]
        ):  # shell.Outputs.BASE_ATTRS
            raise ValueError(
                "A shell output field must have either a callable or a path_template"
            )


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
    keep_extension: bool = attrs.field(default=True)

    @path_template.validator
    def _validate_path_template(self, attribute, value):
        if value:
            if self.default not in (NO_DEFAULT, True, None):
                raise ValueError(
                    f"path_template ({value!r}) can only be provided when there is no "
                    f"default value provided ({self.default!r})"
                )

    def markdown_listing(
        self,
        line_width: int = 79,
        help_indent: int = 4,
        as_input: bool = False,
        **kwargs,
    ):
        """Get the listing for the field in markdown-like format

        Parameters
        ----------
        line_width: int
            The maximum line width for the output, by default it is 79
        help_indent: int
            The indentation for the help text, by default it is 4
        as_input: bool
            Whether to format the field as an input or output if it can be both, by default
            it is False
        **kwargs: Any
            Additional arguments to allow it to be duck-typed with extension classes

        Returns
        -------
        str
            The listing for the field in markdown-like format
        """
        if not as_input:
            return super().markdown_listing(
                width=line_width, help_indent=help_indent, **kwargs
            )

        type_str = "Path | bool"
        if is_optional(self.type):
            type_str += " | None"
            default = "None"
            help_text = wrap_text(
                self.OPTIONAL_PATH_TEMPLATE_HELP,
                width=line_width,
                indent_size=help_indent,
            )
        else:
            default = True
            help_text = wrap_text(
                self.PATH_TEMPLATE_HELP, width=line_width, indent_size=help_indent
            )
        s = f"- {self.name}: {type_str}; default = {default}"
        if self._additional_descriptors(as_input=as_input):
            s += f" ({', '.join(self._additional_descriptors(as_input=as_input))})"
        s += "\n" + help_text
        return s

    PATH_TEMPLATE_HELP = (
        "The path specified for the output file, if True, the default "
        "'path template' will be used."
    )
    OPTIONAL_PATH_TEMPLATE_HELP = PATH_TEMPLATE_HELP + (
        "If False or None, the output file will not be saved."
    )
