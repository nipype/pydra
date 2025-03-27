import os
import itertools
import sys
import typing as ty
from pathlib import Path
import tempfile
import traceback
from unittest.mock import Mock
import pytest
from pydra.compose import python
from fileformats.generic import File
from pydra.engine.lazy import LazyOutField
from pydra.compose import workflow
from pydra.utils.typing import TypeParser, MultiInputObj
from fileformats.application import Json, Yaml, Xml
from .utils import (
    GenericFuncTask,
    GenericShellTask,
    SpecificFuncTask,
    SpecificShellTask,
    OtherSpecificFuncTask,
    OtherSpecificShellTask,
    MyFormatX,
    MyOtherFormatX,
    MyHeader,
)
from pydra.utils.general import exc_info_matches


def exc_to_str(exc_info):
    return "".join(
        traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb)
    )


def lz(tp: ty.Type):
    """convenience method for creating a LazyField of type 'tp'"""
    node = Mock()
    return LazyOutField(node=node, field="boo", type=tp)


PathTypes = ty.Union[str, os.PathLike]


def test_type_check_basic1():
    TypeParser(float, coercible=[(int, float)])(lz(int))


def test_type_check_basic2():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(int, coercible=[(int, float)])(lz(float))
    assert exc_info_matches(exc_info, "doesn't match any of the explicit inclusion")


def test_type_check_basic3():
    TypeParser(int, coercible=[(ty.Any, int)])(lz(float))


def test_type_check_basic4():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(int, coercible=[(ty.Any, float)])(lz(float))
    assert exc_info_matches(exc_info, "doesn't match any of the explicit inclusion")


def test_type_check_basic5():
    assert TypeParser(float, not_coercible=[(ty.Any, str)])(lz(int))


def test_type_check_basic6():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(int, coercible=None, not_coercible=[(float, int)])(lz(float))
    assert exc_info_matches(exc_info, "explicitly excluded")


def test_type_check_basic7():
    path_coercer = TypeParser(Path, coercible=[(os.PathLike, os.PathLike)])

    path_coercer(lz(Path))

    with pytest.raises(TypeError) as exc_info:
        path_coercer(lz(str))

    assert exc_info_matches(exc_info, "doesn't match any of the explicit inclusion")


def test_type_check_basic8():
    TypeParser(Path, coercible=[(PathTypes, PathTypes)])(lz(str))


def test_type_check_basic8a():
    TypeParser(str, coercible=[(PathTypes, PathTypes)])(lz(Path))


def test_type_check_basic9():
    file_coercer = TypeParser(File, coercible=[(PathTypes, File)])
    file_coercer(lz(Path))
    file_coercer(lz(str))


def test_type_check_basic10():
    impotent_str_coercer = TypeParser(str, coercible=[(PathTypes, File)])

    with pytest.raises(TypeError) as exc_info:
        impotent_str_coercer(lz(File))
    assert exc_info_matches(exc_info, "doesn't match any of the explicit inclusion")


def test_type_check_basic11():
    TypeParser(str, coercible=[(PathTypes, PathTypes)])(lz(File))


def test_type_check_basic11a():
    TypeParser(File, coercible=[(PathTypes, PathTypes)])(lz(str))


def test_type_check_basic12():
    TypeParser(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )(lz(ty.Tuple[int, int, int]))


def test_type_check_basic13():
    TypeParser(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )(lz(ty.Tuple[int, ...]))


def test_type_check_basic14():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )(lz(str))
    assert exc_info_matches(exc_info, match="explicitly excluded")


def test_type_check_basic15():
    TypeParser(ty.Union[Path, File, float])(lz(int))


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_check_basic15a():
    TypeParser(Path | File | float)(lz(int))


def test_type_check_basic16():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.Union[Path, File, bool, int])(lz(float))
    assert exc_info_matches(
        exc_info, match="Cannot coerce <class 'float'> to any of the union types"
    )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_check_basic16a():
    with pytest.raises(
        TypeError,
        match="Incorrect type for lazy field: <class 'float'> is not a subclass of",
    ):
        TypeParser(Path | File | bool | int)(lz(float))


def test_type_check_basic17():
    TypeParser(ty.Sequence)(lz(ty.Tuple[int, ...]))


def test_type_check_nested1():
    TypeParser(ty.List[File])(lz(ty.List[Path]))


def test_type_check_nested2():
    TypeParser(ty.List[Path])(lz(ty.List[File]))


def test_type_check_nested3():
    TypeParser(ty.List[Path])(lz(ty.List[str]))


def test_type_check_nested4():
    TypeParser(ty.List[str])(lz(ty.List[File]))


def test_type_check_nested5():
    TypeParser(ty.Dict[str, ty.List[File]])(lz(ty.Dict[str, ty.List[Path]]))


def test_type_check_nested6():
    TypeParser(ty.Tuple[float, ...])(lz(ty.List[int]))


def test_type_check_nested7():
    TypeParser(ty.Tuple[float, float, float])(lz(ty.List[int]))


def test_type_check_nested7a():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.Tuple[float, float, float])(lz(ty.Tuple[int]))
    assert exc_info_matches(exc_info, "Wrong number of type arguments")


def test_type_check_nested8():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(
            ty.Tuple[int, ...],
            not_coercible=[(ty.Sequence, ty.Tuple)],
        )(lz(ty.List[float]))
    assert exc_info_matches(exc_info, "explicitly excluded")


def test_type_check_permit_superclass():
    # Typical case as Json is subclass of File
    TypeParser(ty.List[File])(lz(ty.List[Json]))
    # Permissive super class, as File is superclass of Json
    TypeParser(ty.List[Json], superclass_auto_cast=True)(lz(ty.List[File]))
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.List[Json], superclass_auto_cast=False)(lz(ty.List[File]))
    assert exc_info_matches(exc_info, "Cannot coerce")
    # Fails because Yaml is neither sub or super class of Json
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.List[Json], superclass_auto_cast=True)(lz(ty.List[Yaml]))
    assert exc_info_matches(exc_info, "Cannot coerce")


def test_type_check_fail1():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.Tuple[int, int, int])(lz(ty.Tuple[float, float, float, float]))
    assert exc_info_matches(exc_info, "Wrong number of type arguments in tuple")


def test_type_check_fail2():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.Union[Path, File])(lz(int))
    assert exc_info_matches(exc_info, "to any of the union types")


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_check_fail2a():
    with pytest.raises(TypeError, match="Incorrect type for lazy field: <class 'int'>"):
        TypeParser(Path | File)(lz(int))


def test_type_check_fail3():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.Sequence, coercible=[(ty.Sequence, ty.Sequence)])(
            lz(ty.Dict[str, int])
        )
    assert exc_info_matches(exc_info, "doesn't match any of the explicit inclusion")


def test_type_check_fail4():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.Sequence)(lz(ty.Dict[str, int]))
    assert exc_info_matches(
        exc_info,
        "Cannot coerce typing.Dict[str, int] into <class 'collections.abc.Sequence'>",
    )


def test_type_check_fail5():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.List[int])(lz(int))
    assert exc_info_matches(exc_info, "<class 'int'> doesn't match pattern")


def test_type_check_fail6():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.List[ty.Dict[str, str]])(lz(ty.Tuple[int, int, int]))
    assert exc_info_matches(exc_info, "<class 'int'> doesn't match pattern")


def test_type_coercion_basic():
    assert TypeParser(float, coercible=[(ty.Any, float)])(1) == 1.0


def test_type_coercion_basic1():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(float, coercible=[(ty.Any, int)])(1)
    assert exc_info_matches(exc_info, "doesn't match any of the explicit inclusion")


def test_type_coercion_basic2():
    assert (
        TypeParser(int, coercible=[(ty.Any, ty.Any)], not_coercible=[(ty.Any, str)])(
            1.0
        )
        == 1
    )


def test_type_coercion_basic3():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(int, coercible=[(ty.Any, ty.Any)], not_coercible=[(float, int)])(1.0)
    assert exc_info_matches(exc_info, "explicitly excluded")


def test_type_coercion_basic4():
    path_coercer = TypeParser(Path, coercible=[(os.PathLike, os.PathLike)])

    assert path_coercer(Path("/a/path")) == Path("/a/path")

    with pytest.raises(TypeError) as exc_info:
        path_coercer("/a/path")
    assert exc_info_matches(exc_info, "doesn't match any of the explicit inclusion")


def test_type_coercion_basic5():
    assert TypeParser(Path, coercible=[(PathTypes, PathTypes)])("/a/path") == Path(
        "/a/path"
    )


def test_type_coercion_basic6():
    assert TypeParser(str, coercible=[(PathTypes, PathTypes)])(Path("/a/path")) == str(
        Path("/a/path")
    )


@pytest.fixture
def a_file(tmp_path):
    fspath = tmp_path / "a-file.txt"
    Path.touch(fspath)
    return fspath


def test_type_coercion_basic7(a_file):
    file_coercer = TypeParser(File, coercible=[(PathTypes, File)])

    assert file_coercer(a_file) == File(a_file)
    assert file_coercer(str(a_file)) == File(a_file)


def test_type_coercion_basic8(a_file):
    impotent_str_coercer = TypeParser(str, coercible=[(PathTypes, File)])

    with pytest.raises(TypeError) as exc_info:
        impotent_str_coercer(File(a_file))
    assert exc_info_matches(exc_info, "doesn't match any of the explicit inclusion")


def test_type_coercion_basic9(a_file):
    assert TypeParser(str, coercible=[(PathTypes, PathTypes)])(File(a_file)) == str(
        a_file
    )


def test_type_coercion_basic10(a_file):
    assert TypeParser(File, coercible=[(PathTypes, PathTypes)])(str(a_file)) == File(
        a_file
    )


def test_type_coercion_basic11():
    assert TypeParser(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )((1, 2, 3)) == [1, 2, 3]


def test_type_coercion_basic12():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )("a-string")
    assert exc_info_matches(exc_info, "explicitly excluded")
    assert TypeParser(ty.Union[Path, File, int], coercible=[(ty.Any, ty.Any)])(1.0) == 1


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_coercion_basic12a():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )("a-string")
    assert exc_info_matches(exc_info, "explicitly excluded")
    assert TypeParser(Path | File | int, coercible=[(ty.Any, ty.Any)])(1.0) == 1


def test_type_coercion_basic13():
    assert (
        TypeParser(ty.Union[Path, File, bool, int], coercible=[(ty.Any, ty.Any)])(1.0)
        is True
    )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_coercion_basic13a():
    assert (
        TypeParser(Path | File | bool | int, coercible=[(ty.Any, ty.Any)])(1.0) is True
    )


def test_type_coercion_basic14():
    assert TypeParser(ty.Sequence, coercible=[(ty.Any, ty.Any)])((1, 2, 3)) == (
        1,
        2,
        3,
    )


@pytest.fixture
def another_file(tmp_path):
    fspath = tmp_path / "another-file.txt"
    Path.touch(fspath)
    return fspath


@pytest.fixture
def yet_another_file(tmp_path):
    fspath = tmp_path / "yet-another-file.txt"
    Path.touch(fspath)
    return fspath


def test_type_coercion_nested1(a_file, another_file, yet_another_file):
    assert TypeParser(ty.List[File], coercible=[(PathTypes, PathTypes)])(
        [a_file, another_file, yet_another_file]
    ) == [File(a_file), File(another_file), File(yet_another_file)]


def test_type_coercion_nested3(a_file, another_file, yet_another_file):
    assert TypeParser(ty.List[Path], coercible=[(PathTypes, PathTypes)])(
        [File(a_file), File(another_file), File(yet_another_file)]
    ) == [a_file, another_file, yet_another_file]


def test_type_coercion_nested4(a_file, another_file, yet_another_file):
    assert TypeParser(ty.Dict[str, ty.List[File]], coercible=[(PathTypes, PathTypes)])(
        {
            "a": [a_file, another_file, yet_another_file],
            "b": [a_file, another_file],
        }
    ) == {
        "a": [File(a_file), File(another_file), File(yet_another_file)],
        "b": [File(a_file), File(another_file)],
    }


def test_type_coercion_nested5(a_file, another_file, yet_another_file):
    assert TypeParser(ty.List[File], coercible=[(PathTypes, PathTypes)])(
        [a_file, another_file, yet_another_file]
    ) == [File(a_file), File(another_file), File(yet_another_file)]


def test_type_coercion_nested6():
    assert TypeParser(ty.Tuple[int, int, int], coercible=[(ty.Any, ty.Any)])(
        [1.0, 2.0, 3.0]
    ) == (1, 2, 3)


def test_type_coercion_nested7():
    assert TypeParser(ty.Tuple[int, ...], coercible=[(ty.Any, ty.Any)])(
        [1.0, 2.0, 3.0]
    ) == (1, 2, 3)


def test_type_coercion_nested8():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(
            ty.Tuple[int, ...],
            coercible=[(ty.Any, ty.Any)],
            not_coercible=[(ty.Sequence, ty.Tuple)],
        )([1.0, 2.0, 3.0])
    assert exc_info_matches(exc_info, "explicitly excluded")


def test_type_coercion_fail1():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.Tuple[int, int, int], coercible=[(ty.Any, ty.Any)])(
            [1.0, 2.0, 3.0, 4.0]
        )
    assert exc_info_matches(exc_info, "Incorrect number of items")


def test_type_coercion_fail2():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.Union[Path, File], coercible=[(ty.Any, ty.Any)])(1)
    assert exc_info_matches(exc_info, "to any of the union types")


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_coercion_fail2a():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(Path | File, coercible=[(ty.Any, ty.Any)])(1)
    assert exc_info_matches(exc_info, "to any of the union types")


def test_type_coercion_fail3():
    with pytest.raises(TypeError, match="Incorrect type for field"):
        TypeParser(ty.Sequence, coercible=[(ty.Sequence, ty.Sequence)])(
            {"a": 1, "b": 2}
        )


def test_type_coercion_fail4():
    with pytest.raises(TypeError, match="Incorrect type for field"):
        TypeParser(ty.Sequence, coercible=[(ty.Any, ty.Any)])({"a": 1})


def test_type_coercion_fail5():
    with pytest.raises(TypeError) as excinfo:
        TypeParser(ty.List[int], coercible=[(ty.Any, ty.Any)])(1)
    assert "as 1 is not iterable" in exc_to_str(excinfo)


def test_type_coercion_fail6():
    with pytest.raises(TypeError) as excinfo:
        TypeParser(ty.List[ty.Dict[str, str]], coercible=[(ty.Any, ty.Any)])((1, 2, 3))
    assert "is not a mapping type" in exc_to_str(excinfo)


def test_type_coercion_realistic():
    tmpdir = Path(tempfile.mkdtemp())
    a_file = tmpdir / "a-file.txt"
    another_file = tmpdir / "another-file.txt"
    yet_another_file = tmpdir / "yet-another-file.txt"
    Path.touch(a_file)
    Path.touch(another_file)
    Path.touch(yet_another_file)
    file_list = [File(p) for p in (a_file, another_file, yet_another_file)]

    @python.define(outputs={"a": ty.List[File], "b": ty.List[str]})
    def f(x: ty.List[File], y: ty.Dict[str, ty.List[File]]):
        return list(itertools.chain(x, *y.values())), list(y.keys())

    defn = f(x=file_list, y={"a": file_list[1:]})
    outputs = defn()

    TypeParser(ty.List[str])(outputs.a)  # pylint: disable=no-member
    with pytest.raises(
        TypeError,
        match=r"Incorrect type for field:",
    ) as exc_info:
        TypeParser(ty.List[int])(outputs.a)  # pylint: disable=no-member

    with pytest.raises(TypeError) as exc_info:
        defn.x = "bad-value"
    assert exc_info_matches(
        exc_info, match="Cannot coerce 'bad-value' into <class 'list'>"
    )


def test_check_missing_type_args():
    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.List[int]).check_type(list)
    assert exc_info_matches(exc_info, "wasn't declared with type args required")

    with pytest.raises(TypeError) as exc_info:
        TypeParser(ty.List[int]).check_type(dict)
    assert exc_info_matches(exc_info, "doesn't match pattern")


def test_matches_type_union():
    assert TypeParser.matches_type(ty.Union[int, bool, str], ty.Union[int, bool, str])
    assert TypeParser.matches_type(ty.Union[int, bool], ty.Union[int, bool, str])
    assert not TypeParser.matches_type(ty.Union[int, bool, str], ty.Union[int, bool])


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_matches_type_union_a():
    assert TypeParser.matches_type(int | bool | str, int | bool | str)
    assert TypeParser.matches_type(int | bool, int | bool | str)
    assert not TypeParser.matches_type(int | bool | str, int | bool)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_matches_type_union_b():
    assert TypeParser.matches_type(int | bool | str, ty.Union[int, bool, str])
    assert TypeParser.matches_type(int | bool, ty.Union[int, bool, str])
    assert not TypeParser.matches_type(int | bool | str, ty.Union[int, bool])


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_matches_type_union_c():
    assert TypeParser.matches_type(ty.Union[int, bool, str], int | bool | str)
    assert TypeParser.matches_type(ty.Union[int, bool], int | bool | str)
    assert not TypeParser.matches_type(ty.Union[int, bool, str], int | bool)


def test_matches_type_dict():
    COERCIBLE = [(str, Path), (Path, str), (int, float)]

    assert TypeParser.matches_type(
        ty.Dict[Path, int], ty.Dict[str, int], coercible=COERCIBLE
    )
    assert TypeParser.matches_type(
        ty.Dict[Path, int], ty.Dict[str, float], coercible=COERCIBLE
    )
    assert not TypeParser.matches_type(
        ty.Dict[Path, int], ty.Dict[str, int], coercible=[]
    )
    assert not TypeParser.matches_type(
        ty.Dict[Path, int], ty.Dict[str, float], coercible=[]
    )
    assert not TypeParser.matches_type(
        ty.Dict[Path, float], ty.Dict[str, int], coercible=COERCIBLE
    )
    assert not TypeParser.matches_type(
        ty.Tuple[str, int], ty.Dict[str, int], coercible=COERCIBLE
    )


def test_matches_type_type():
    assert TypeParser.matches_type(type, type)
    assert not TypeParser.matches_type(int, type)


def test_matches_type_tuple():
    assert TypeParser.matches_type(ty.Tuple[int], ty.Tuple[int])
    assert TypeParser.matches_type(
        ty.Tuple[int], ty.Tuple[float], coercible=[(int, float)]
    )
    assert not TypeParser.matches_type(
        ty.Tuple[float], ty.Tuple[int], coercible=[(int, float)]
    )
    assert TypeParser.matches_type(ty.Tuple[int, int], ty.Tuple[int, int])
    assert not TypeParser.matches_type(ty.Tuple[int, int], ty.Tuple[int])
    assert not TypeParser.matches_type(ty.Tuple[int], ty.Tuple[int, int])


def test_matches_type_tuple_ellipsis1():
    assert TypeParser.matches_type(ty.Tuple[int], ty.Tuple[int, ...])


def test_matches_type_tuple_ellipsis2():
    assert TypeParser.matches_type(ty.Tuple[int, int], ty.Tuple[int, ...])


def test_matches_type_tuple_ellipsis3():
    assert not TypeParser.matches_type(ty.Tuple[int, float], ty.Tuple[int, ...])


def test_matches_type_tuple_ellipsis4():
    assert TypeParser.matches_type(ty.Tuple[int, ...], ty.Tuple[int])


def test_matches_type_tuple_ellipsis5():
    assert TypeParser.matches_type(
        ty.Tuple[int], ty.List[int], coercible=[(tuple, list)]
    )


def test_matches_type_tuple_ellipsis6():
    assert TypeParser.matches_type(
        ty.Tuple[int, ...], ty.List[int], coercible=[(tuple, list)]
    )


def test_contains_type_in_dict():
    assert TypeParser.contains_type(int, ty.Dict[str, ty.List[ty.Tuple[int, ...]]])
    assert not TypeParser.contains_type(
        int, ty.Dict[str, ty.List[ty.Tuple[float, ...]]]
    )


def test_any_union():
    """Check that the superclass auto-cast matches if any of the union args match instead
    of all"""
    # The Json type within the Union matches File as it is a subclass as `match_any_of_union`
    # is set to True. Otherwise, all types within the Union would have to match
    TypeParser(File, match_any_of_union=True).check_type(ty.Union[ty.List[File], Json])


def test_union_superclass_check_type():
    """Check that the superclass auto-cast matches if any of the union args match instead
    of all"""
    # In this case, File matches Json due to the `superclass_auto_cast=True` flag being set
    TypeParser(ty.Union[ty.List[File], Json], superclass_auto_cast=True)(lz(File))


def test_type_matches():
    assert TypeParser.matches([1, 2, 3], ty.List[int])
    assert TypeParser.matches((1, 2, 3), ty.Tuple[int, ...])

    assert TypeParser.matches((1, 2, 3), ty.List[int])
    assert not TypeParser.matches((1, 2, 3), ty.List[int], coercible=[])


@pytest.fixture(params=["func", "shell"])
def GenericTask(request):
    if request.param == "func":
        return GenericFuncTask
    elif request.param == "shell":
        return GenericShellTask
    else:
        assert False


@pytest.fixture(params=["func", "shell"])
def SpecificTask(request):
    if request.param == "func":
        return SpecificFuncTask
    elif request.param == "shell":
        return SpecificShellTask
    else:
        assert False


@pytest.fixture(params=["func", "shell"])
def OtherSpecificTask(request):
    if request.param == "func":
        return OtherSpecificFuncTask
    elif request.param == "shell":
        return OtherSpecificShellTask
    else:
        assert False


def test_typing_implicit_cast_from_super(tmp_path, GenericTask, SpecificTask):
    """Check the casting of lazy fields and whether specific file-sets can be recovered
    from generic `File` classes"""

    @workflow.define(outputs=["out_file"])
    def Workflow(in_file: MyFormatX) -> MyFormatX:
        specific1 = workflow.add(SpecificTask(in_file=in_file))
        generic = workflow.add(GenericTask(in_file=specific1.out))  # Generic task
        specific2 = workflow.add(SpecificTask(in_file=generic.out), name="specific2")
        return specific2.out

    in_file = MyFormatX.sample()

    outputs = Workflow(in_file=in_file)()

    out_file: MyFormatX = outputs.out_file
    assert type(out_file) is MyFormatX
    assert out_file.parent != in_file.parent
    assert type(out_file.header) is MyHeader
    assert out_file.header.parent != in_file.header.parent


@pytest.mark.flaky(reruns=5)
def test_typing_cast(tmp_path, SpecificTask, OtherSpecificTask):
    """Check the casting of lazy fields and whether specific file-sets can be recovered
    from generic `File` classes"""

    @workflow.define(outputs=["out_file"])
    def Workflow(in_file: MyFormatX) -> MyFormatX:
        entry = workflow.add(SpecificTask(in_file=in_file))

        with pytest.raises(TypeError) as exc_info:
            # No cast of generic task output to MyFormatX
            workflow.add(OtherSpecificTask(in_file=entry.out))  # Generic task
        assert exc_info_matches(exc_info, "Cannot coerce")

        inner = workflow.add(  # Generic task
            OtherSpecificTask(in_file=workflow.cast(entry.out, MyOtherFormatX))
        )

        with pytest.raises(TypeError) as exc_info:
            # No cast of generic task output to MyFormatX
            workflow.add(SpecificTask(in_file=inner.out))

        assert exc_info_matches(exc_info, "Cannot coerce")

        exit = workflow.add(
            SpecificTask(in_file=workflow.cast(inner.out, MyFormatX)), name="exit"
        )

        return exit.out

    in_file = MyFormatX.sample()

    outputs = Workflow(in_file=in_file)()

    out_file: MyFormatX = outputs.out_file
    assert type(out_file) is MyFormatX
    assert out_file.parent != in_file.parent
    assert type(out_file.header) is MyHeader
    assert out_file.header.parent != in_file.header.parent


@pytest.mark.parametrize(
    ("sub", "super"),
    [
        (ty.Type[File], type),
        (ty.Type[Json], ty.Type[File]),
        (ty.Union[Json, Yaml], ty.Union[Json, Yaml, Xml]),
        (Json, ty.Union[Json, Yaml]),
        (ty.List[int], list),
        (None, ty.Union[int, None]),
        (ty.Tuple[int, None], ty.Tuple[int, None]),
        (None, None),
        (None, type(None)),
        (type(None), None),
        (type(None), type(None)),
        (type(None), type(None)),
    ],
)
def test_subclass(sub, super):
    assert TypeParser.is_subclass(sub, super)


@pytest.mark.parametrize(
    ("sub", "super"),
    [
        (ty.Type[File], ty.Type[Json]),
        (ty.Union[Json, Yaml, Xml], ty.Union[Json, Yaml]),
        (ty.Union[Json, Yaml], Json),
        (list, ty.List[int]),
        (ty.List[float], ty.List[int]),
        (None, ty.Union[int, float]),
        (None, int),
        (int, None),
    ],
)
def test_not_subclass(sub, super):
    assert not TypeParser.is_subclass(sub, super)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_union_is_subclass1a():
    assert TypeParser.is_subclass(Json | Yaml, Json | Yaml | Xml)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_union_is_subclass1b():
    assert TypeParser.is_subclass(Json | Yaml, ty.Union[Json, Yaml, Xml])


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_union_is_subclass1c():
    assert TypeParser.is_subclass(ty.Union[Json, Yaml], Json | Yaml | Xml)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_union_is_subclass2a():
    assert not TypeParser.is_subclass(Json | Yaml | Xml, Json | Yaml)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_union_is_subclass2b():
    assert not TypeParser.is_subclass(ty.Union[Json, Yaml, Xml], Json | Yaml)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_union_is_subclass2c():
    assert not TypeParser.is_subclass(Json | Yaml | Xml, ty.Union[Json, Yaml])


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_union_is_subclass3a():
    assert TypeParser.is_subclass(Json, Json | Yaml)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_union_is_subclass4a():
    assert not TypeParser.is_subclass(Json | Yaml, Json)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_none_is_subclass1a():
    assert TypeParser.is_subclass(None, int | None)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_none_is_subclass2a():
    assert not TypeParser.is_subclass(None, int | float)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Cannot subscript tuple in < Py3.9"
)
def test_generic_is_subclass4():
    class MyTuple(tuple):
        pass

    class A:
        pass

    class B(A):
        pass

    assert TypeParser.is_subclass(MyTuple[A], ty.Tuple[A])
    assert TypeParser.is_subclass(ty.Tuple[B], ty.Tuple[A])
    assert TypeParser.is_subclass(MyTuple[B], ty.Tuple[A])
    assert not TypeParser.is_subclass(ty.Tuple[A], ty.Tuple[B])
    assert not TypeParser.is_subclass(ty.Tuple[A], MyTuple[A])
    assert not TypeParser.is_subclass(MyTuple[A], ty.Tuple[B])
    assert TypeParser.is_subclass(MyTuple[A, int], ty.Tuple[A, int])
    assert TypeParser.is_subclass(ty.Tuple[B, int], ty.Tuple[A, int])
    assert TypeParser.is_subclass(MyTuple[B, int], ty.Tuple[A, int])
    assert TypeParser.is_subclass(MyTuple[int, B], ty.Tuple[int, A])
    assert not TypeParser.is_subclass(MyTuple[B, int], ty.Tuple[int, A])
    assert not TypeParser.is_subclass(MyTuple[int, B], ty.Tuple[A, int])
    assert not TypeParser.is_subclass(MyTuple[B, int], ty.Tuple[A])
    assert not TypeParser.is_subclass(MyTuple[B], ty.Tuple[A, int])


@pytest.mark.parametrize(
    ("tp", "obj"),
    [
        (File, ty.Type[File]),
        (Json, ty.Type[File]),
        (Json, type),
        (None, None),
        (None, type(None)),
        (None, ty.Union[int, None]),
        (1, ty.Union[int, None]),
    ],
)
def test_type_is_instance(tp, obj):
    assert TypeParser.is_instance(tp, obj)


@pytest.mark.parametrize(
    ("tp", "obj"),
    [
        (File, ty.Type[Json]),
        (None, int),
        (1, None),
        (None, ty.Union[int, str]),
    ],
)
def test_type_is_not_instance(tp, obj):
    assert not TypeParser.is_instance(tp, obj)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_is_instance9a():
    assert TypeParser.is_instance(None, int | None)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_is_instance10a():
    assert TypeParser.is_instance(1, int | None)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="No UnionType < Py3.10")
def test_type_is_instance11a():
    assert not TypeParser.is_instance(None, int | str)


@pytest.mark.parametrize(
    ("typ", "obj", "result"),
    [
        (MultiInputObj[str], "a", ["a"]),
        (MultiInputObj[str], ["a"], ["a"]),
        (MultiInputObj[ty.List[str]], ["a"], [["a"]]),
        (MultiInputObj[ty.Union[int, ty.List[str]]], ["a"], [["a"]]),
        (MultiInputObj[ty.Union[int, ty.List[str]]], [["a"]], [["a"]]),
        (MultiInputObj[ty.Union[int, ty.List[str]]], [1], [1]),
    ],
)
def test_multi_input_obj_coerce(typ, obj, result):
    assert TypeParser(typ)(obj) == result


def test_multi_input_obj_coerce4a():
    with pytest.raises(TypeError):
        TypeParser(MultiInputObj[ty.Union[int, ty.List[str]]])([[1]])


@pytest.mark.parametrize(
    ("reference", "to_be_checked"),
    [
        (MultiInputObj[str], str),
        (MultiInputObj[str], ty.List[str]),
        (MultiInputObj[ty.List[str]], ty.List[str]),
        (MultiInputObj[ty.Union[int, ty.List[str]]], ty.List[str]),
        (MultiInputObj[ty.Union[int, ty.List[str]]], ty.List[ty.List[str]]),
        (MultiInputObj[ty.Union[int, ty.List[str]]], ty.List[int]),
    ],
)
def test_multi_input_obj_check_type(reference, to_be_checked):
    TypeParser(reference)(lz(to_be_checked))


@pytest.mark.parametrize(
    ("reference", "to_be_checked"),
    [
        (MultiInputObj[ty.Union[int, ty.List[str]]], ty.List[ty.List[int]]),
    ],
)
def test_multi_input_obj_check_type_fail(reference, to_be_checked):
    with pytest.raises(TypeError):
        TypeParser(reference)(lz(to_be_checked))
