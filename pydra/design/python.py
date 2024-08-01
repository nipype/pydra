import typing as ty
import attrs


@attrs.define(kw_only=True)
class arg:
    help_string: str
    default: ty.Any = attrs.NOTHING
    mandatory: bool = False
    allowed_values: list = None
    requires: list = None
    xor: list = None
    copyfile: bool = None
    keep_extension: bool = True
    readonly: bool = False
