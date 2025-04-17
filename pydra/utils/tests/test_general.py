import typing as ty
import attrs
from pydra.compose import python, workflow, shell
from fileformats.text import TextFile
from pydra.utils.general import task_class_as_dict, task_class_from_dict, task_fields
from pydra.utils.tests.utils import Concatenate


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


def test_python_task_class_as_dict(tmp_path):

    dct = task_class_as_dict(Add)
    Reloaded = task_class_from_dict(dct)
    assert task_fields(Add) == task_fields(Reloaded)

    add = Reloaded(a=1, b=2)
    assert add(cache_root=tmp_path / "cache").out_int == 3


def test_shell_task_class_as_dict():

    MyCmd = shell.define(
        "my-cmd <in_file> <out|out_file> --an-arg <an_arg:int=2> --a-flag<a_flag>"
    )

    dct = task_class_as_dict(MyCmd)
    Reloaded = task_class_from_dict(dct)
    assert task_fields(MyCmd) == task_fields(Reloaded)


def test_workflow_task_class_as_dict(tmp_path):

    @workflow.define(outputs=["out_file"])
    def AWorkflow(in_file: TextFile, a_param: int) -> TextFile:
        concatenate = workflow.add(
            Concatenate(in_file1=in_file, in_file2=in_file, duplicates=a_param)
        )
        return concatenate.out_file

    dct = task_class_as_dict(AWorkflow)
    Reloaded = task_class_from_dict(dct)
    assert task_fields(AWorkflow) == task_fields(Reloaded)

    foo_file = tmp_path / "file1.txt"
    foo_file.write_text("foo")

    outputs = Reloaded(in_file=foo_file, a_param=2)(cache_root=tmp_path / "cache")
    assert outputs.out_file.contents == "foo\nfoo\nfoo\nfoo"


def test_task_class_as_dict_with_value_serializer():

    def frozen_set_to_list_serializer(
        mock_class: ty.Any, atr: attrs.Attribute, value: ty.Any
    ) -> ty.Any:
        # This is just a dummy serializer
        if isinstance(value, frozenset):
            return list(
                frozen_set_to_list_serializer(mock_class, atr, v) for v in value
            )
        return value

    dct = task_class_as_dict(Add, value_serializer=frozen_set_to_list_serializer)
    assert dct["xor"] == [["b", "c"]] or dct["xor"] == [["c", "b"]]
