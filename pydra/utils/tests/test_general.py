import typing as ty
import attrs
from collections.abc import Collection
from pydra.compose import python, workflow, shell
from fileformats.text import TextFile
from pydra.utils.general import (
    unstructure,
    structure,
    get_fields,
    filter_out_defaults,
)
from pydra.utils.tests.utils import Concatenate


def check_dict_fully_unstructured(dct: dict):
    """Checks if there are any Pydra objects or non list/dict containers in the dict."""
    stack = [dct]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            pass
        elif isinstance(item, Collection):
            raise ValueError(f"Unserializable container object {item} found in dict")
        elif type(item).__module__.split(".")[0] == "pydra":
            raise ValueError(f"Unserialized Pydra object {item} found in dict")


@python.define(outputs=["out_int"], xor=["b", "c"])
def Add(a: int, b: int | None = None, c: int | None = None) -> int:
    """
    Parameters
    ----------
    a: int
        the first arg
    b : int, optional
        the optional second arg
    c : int, optional
        the optional third arg

    Returns
    -------
    out_int : int
        the sum of a and b
    """
    return a + (b if b is not None else c)


def test_python_unstructure(tmp_path):

    assert Add(a=1, b=2)(cache_root=tmp_path / "cache1").out_int == 3

    dct = unstructure(Add)
    assert isinstance(dct, dict)
    check_dict_fully_unstructured(dct)
    Reloaded = structure(dct)
    assert get_fields(Add) == get_fields(Reloaded)

    assert Reloaded(a=1, b=2)(cache_root=tmp_path / "cache2").out_int == 3


def test_shell_unstructure():

    MyCmd = shell.define(
        "my-cmd <in_file> <out|out_file> --an-arg <an_arg:int=2> --a-flag<a_flag>"
    )

    dct = unstructure(MyCmd)
    assert isinstance(dct, dict)
    check_dict_fully_unstructured(dct)
    Reloaded = structure(dct)
    assert get_fields(MyCmd) == get_fields(Reloaded)


def test_workflow_unstructure(tmp_path):

    @workflow.define(outputs=["out_file"])
    def AWorkflow(in_file: TextFile, a_param: int) -> TextFile:
        concatenate = workflow.add(
            Concatenate(in_file1=in_file, in_file2=in_file, duplicates=a_param)
        )
        return concatenate.out_file

    dct = unstructure(AWorkflow)
    assert isinstance(dct, dict)
    check_dict_fully_unstructured(dct)
    Reloaded = structure(dct)
    assert get_fields(AWorkflow) == get_fields(Reloaded)

    foo_file = tmp_path / "file1.txt"
    foo_file.write_text("foo")

    outputs = Reloaded(in_file=foo_file, a_param=2)(cache_root=tmp_path / "cache")
    assert outputs.out_file.contents == "foo\nfoo\nfoo\nfoo"


def test_unstructure_with_value_serializer():

    @python.define
    def Identity(a: int) -> int:
        """
        Parameters
        ----------
        a: int
            the arg

        Returns
        -------
        out : int
            a returned as is
        """
        return a

    def type_to_str_serializer(
        klass: ty.Any, atr: attrs.Attribute, value: ty.Any
    ) -> ty.Any:
        if isinstance(value, type):
            return value.__module__ + "." + value.__name__
        return value

    dct = unstructure(Identity, value_serializer=type_to_str_serializer)
    assert isinstance(dct, dict)
    check_dict_fully_unstructured(dct)
    assert dct["inputs"] == {"a": {"type": "builtins.int", "help": "the arg"}}


def test_unstructure_with_filter():

    @python.define
    def Identity(a: int) -> int:
        """
        Parameters
        ----------
        a: int
            the arg

        Returns
        -------
        out : int
            a returned as is
        """
        return a

    def no_helps_filter(atr: attrs.Attribute, value: ty.Any) -> bool:
        if atr.name == "help":
            return False
        return filter_out_defaults(atr, value)

    dct = unstructure(Identity, filter=no_helps_filter)
    assert isinstance(dct, dict)
    check_dict_fully_unstructured(dct)
    assert dct["inputs"] == {"a": {"type": int}}
