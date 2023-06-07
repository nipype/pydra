import os
import itertools
import typing as ty
from pathlib import Path
import tempfile
import pytest
from pydra import mark
from ...engine.specs import File, LazyField
from ..typing import TypeChecker


def lz(tp: ty.Type):
    """convenience method for creating a LazyField of type 'tp'"""
    return LazyField(name="foo", field="boo", attr_type="output", type=tp)


PathTypes = ty.Union[str, os.PathLike]


def test_type_check_basic1():
    TypeChecker(float, coercible=[(int, float)])(lz(int))


def test_type_check_basic2():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(int, coercible=[(int, float)])(lz(float))


def test_type_check_basic3():
    TypeChecker(int, coercible=[(ty.Any, int)])(lz(float))


def test_type_check_basic4():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(int, coercible=[(ty.Any, float)])(lz(float))


def test_type_check_basic5():
    assert TypeChecker(float, not_coercible=[(ty.Any, str)])(lz(int))


def test_type_check_basic6():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(int, coercible=None, not_coercible=[(float, int)])(lz(float))


def test_type_check_basic7():
    path_coercer = TypeChecker(Path, coercible=[(os.PathLike, os.PathLike)])

    path_coercer(lz(Path))

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        path_coercer(lz(str))


def test_type_check_basic8():
    TypeChecker(Path, coercible=[(PathTypes, PathTypes)])(lz(str))
    TypeChecker(str, coercible=[(PathTypes, PathTypes)])(lz(Path))


def test_type_check_basic9():
    file_coercer = TypeChecker(File, coercible=[(PathTypes, File)])

    file_coercer(lz(Path))
    file_coercer(lz(str))


def test_type_check_basic10():
    impotent_str_coercer = TypeChecker(str, coercible=[(PathTypes, File)])

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        impotent_str_coercer(lz(File))


def test_type_check_basic11():
    TypeChecker(str, coercible=[(PathTypes, PathTypes)])(lz(File))
    TypeChecker(File, coercible=[(PathTypes, PathTypes)])(lz(str))


def test_type_check_basic12():
    TypeChecker(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )(lz(ty.Tuple[int, int, int]))


def test_type_check_basic13():
    TypeChecker(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )(lz(ty.Tuple[int, ...]))


def test_type_check_basic14():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )(lz(str))


def test_type_check_basic15():
    TypeChecker(ty.Union[Path, File, float])(lz(int))


def test_type_check_basic16():
    with pytest.raises(
        TypeError, match="Cannot coerce <class 'float'> to any of the union types"
    ):
        TypeChecker(ty.Union[Path, File, bool, int])(lz(float))


def test_type_check_basic17():
    TypeChecker(ty.Sequence)(lz(ty.Tuple[int, ...]))


def test_type_check_nested1():
    TypeChecker(ty.List[File])(lz(ty.List[Path]))


def test_type_check_nested2():
    TypeChecker(ty.List[Path])(lz(ty.List[File]))


def test_type_check_nested3():
    TypeChecker(ty.List[Path])(lz(ty.List[str]))


def test_type_check_nested4():
    TypeChecker(ty.List[str])(lz(ty.List[File]))


def test_type_check_nested5():
    TypeChecker(ty.Dict[str, ty.List[File]])(lz(ty.Dict[str, ty.List[Path]]))


def test_type_check_nested6():
    TypeChecker(ty.Tuple[float, ...])(lz(ty.List[int]))


def test_type_check_nested7():
    with pytest.raises(TypeError, match="Wrong number of type arguments"):
        TypeChecker(ty.Tuple[float, float, float])(lz(ty.List[int]))


def test_type_check_nested8():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(
            ty.Tuple[int, ...],
            not_coercible=[(ty.Sequence, ty.Tuple)],
        )(lz(ty.List[float]))


def test_type_check_fail1():
    with pytest.raises(TypeError, match="Wrong number of type arguments in tuple"):
        TypeChecker(ty.Tuple[int, int, int])(lz(ty.Tuple[float, float, float, float]))


def test_type_check_fail2():
    with pytest.raises(TypeError, match="to any of the union types"):
        TypeChecker(ty.Union[Path, File])(lz(int))


def test_type_check_fail3():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(ty.Sequence, coercible=[(ty.Sequence, ty.Sequence)])(
            lz(ty.Dict[str, int])
        )


def test_type_check_fail4():
    with pytest.raises(TypeError, match="Cannot coerce <class 'dict'> into"):
        TypeChecker(ty.Sequence)(lz(ty.Dict[str, int]))


def test_type_check_fail5():
    with pytest.raises(TypeError, match="<class 'int'> doesn't match pattern"):
        TypeChecker(ty.List[int])(lz(int))


def test_type_check_fail6():
    with pytest.raises(TypeError, match="<class 'int'> doesn't match pattern"):
        TypeChecker(ty.List[ty.Dict[str, str]])(lz(ty.Tuple[int, int, int]))


def test_type_coercion_basic():
    assert TypeChecker(float, coercible=[(ty.Any, float)])(1) == 1.0


def test_type_coercion_basic1():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(float, coercible=[(ty.Any, int)])(1)


def test_type_coercion_basic2():
    assert (
        TypeChecker(int, coercible=[(ty.Any, ty.Any)], not_coercible=[(ty.Any, str)])(
            1.0
        )
        == 1
    )


def test_type_coercion_basic3():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(int, coercible=[(ty.Any, ty.Any)], not_coercible=[(float, int)])(
            1.0
        )


def test_type_coercion_basic4():
    path_coercer = TypeChecker(Path, coercible=[(os.PathLike, os.PathLike)])

    assert path_coercer(Path("/a/path")) == Path("/a/path")

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        path_coercer("/a/path")


def test_type_coercion_basic5():
    assert TypeChecker(Path, coercible=[(PathTypes, PathTypes)])("/a/path") == Path(
        "/a/path"
    )


def test_type_coercion_basic6():
    assert (
        TypeChecker(str, coercible=[(PathTypes, PathTypes)])(Path("/a/path"))
        == "/a/path"
    )


@pytest.fixture
def a_file(tmp_path):
    fspath = tmp_path / "a-file.txt"
    Path.touch(fspath)
    return fspath


def test_type_coercion_basic7(a_file):
    file_coercer = TypeChecker(File, coercible=[(PathTypes, File)])

    assert file_coercer(a_file) == File(a_file)
    assert file_coercer(str(a_file)) == File(a_file)


def test_type_coercion_basic8(a_file):
    impotent_str_coercer = TypeChecker(str, coercible=[(PathTypes, File)])

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        impotent_str_coercer(File(a_file))


def test_type_coercion_basic9(a_file):
    assert TypeChecker(str, coercible=[(PathTypes, PathTypes)])(File(a_file)) == str(
        a_file
    )


def test_type_coercion_basic10(a_file):
    assert TypeChecker(File, coercible=[(PathTypes, PathTypes)])(str(a_file)) == File(
        a_file
    )


def test_type_coercion_basic11():
    assert TypeChecker(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )((1, 2, 3)) == [1, 2, 3]


def test_type_coercion_basic12():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )("a-string")

    assert (
        TypeChecker(ty.Union[Path, File, int], coercible=[(ty.Any, ty.Any)])(1.0) == 1
    )


def test_type_coercion_basic13():
    assert (
        TypeChecker(ty.Union[Path, File, bool, int], coercible=[(ty.Any, ty.Any)])(1.0)
        is True
    )


def test_type_coercion_basic14():
    assert TypeChecker(ty.Sequence, coercible=[(ty.Any, ty.Any)])((1, 2, 3)) == (
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
    assert TypeChecker(ty.List[File], coercible=[(PathTypes, PathTypes)])(
        [a_file, another_file, yet_another_file]
    ) == [File(a_file), File(another_file), File(yet_another_file)]


def test_type_coercion_nested3(a_file, another_file, yet_another_file):
    assert TypeChecker(ty.List[Path], coercible=[(PathTypes, PathTypes)])(
        [File(a_file), File(another_file), File(yet_another_file)]
    ) == [a_file, another_file, yet_another_file]


def test_type_coercion_nested4(a_file, another_file, yet_another_file):
    assert TypeChecker(ty.Dict[str, ty.List[File]], coercible=[(PathTypes, PathTypes)])(
        {
            "a": [a_file, another_file, yet_another_file],
            "b": [a_file, another_file],
        }
    ) == {
        "a": [File(a_file), File(another_file), File(yet_another_file)],
        "b": [File(a_file), File(another_file)],
    }


def test_type_coercion_nested5(a_file, another_file, yet_another_file):
    assert TypeChecker(ty.List[File], coercible=[(PathTypes, PathTypes)])(
        [a_file, another_file, yet_another_file]
    ) == [File(a_file), File(another_file), File(yet_another_file)]


def test_type_coercion_nested6():
    assert TypeChecker(ty.Tuple[int, int, int], coercible=[(ty.Any, ty.Any)])(
        [1.0, 2.0, 3.0]
    ) == (1, 2, 3)


def test_type_coercion_nested7():
    assert TypeChecker(ty.Tuple[int, ...], coercible=[(ty.Any, ty.Any)])(
        [1.0, 2.0, 3.0]
    ) == (1, 2, 3)


def test_type_coercion_nested8():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(
            ty.Tuple[int, ...],
            coercible=[(ty.Any, ty.Any)],
            not_coercible=[(ty.Sequence, ty.Tuple)],
        )([1.0, 2.0, 3.0])


def test_type_coercion_fail1():
    with pytest.raises(TypeError, match="Incorrect number of items"):
        TypeChecker(ty.Tuple[int, int, int], coercible=[(ty.Any, ty.Any)])(
            [1.0, 2.0, 3.0, 4.0]
        )


def test_type_coercion_fail2():
    with pytest.raises(TypeError, match="to any of the union types"):
        TypeChecker(ty.Union[Path, File], coercible=[(ty.Any, ty.Any)])(1)


def test_type_coercion_fail3():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(ty.Sequence, coercible=[(ty.Sequence, ty.Sequence)])(
            {"a": 1, "b": 2}
        )


def test_type_coercion_fail4():
    with pytest.raises(TypeError, match="Cannot coerce {'a': 1} into"):
        TypeChecker(ty.Sequence, coercible=[(ty.Any, ty.Any)])({"a": 1})


def test_type_coercion_fail5():
    with pytest.raises(TypeError, match="as 1 is not iterable"):
        TypeChecker(ty.List[int], coercible=[(ty.Any, ty.Any)])(1)


def test_type_coercion_fail6():
    with pytest.raises(TypeError, match="is not a mapping type"):
        TypeChecker(ty.List[ty.Dict[str, str]], coercible=[(ty.Any, ty.Any)])((1, 2, 3))


def test_type_coercion_realistic():
    tmpdir = Path(tempfile.mkdtemp())
    a_file = tmpdir / "a-file.txt"
    another_file = tmpdir / "another-file.txt"
    yet_another_file = tmpdir / "yet-another-file.txt"
    Path.touch(a_file)
    Path.touch(another_file)
    Path.touch(yet_another_file)
    file_list = [File(p) for p in (a_file, another_file, yet_another_file)]

    @mark.task
    @mark.annotate({"return": {"a": ty.List[File], "b": ty.List[str]}})
    def f(x: ty.List[File], y: ty.Dict[str, ty.List[File]]):
        return list(itertools.chain(x, *y.values())), list(y.keys())

    task = f(x=file_list, y={"a": file_list[1:]})

    TypeChecker(ty.List[str])(task.lzout.a)  # pylint: disable=no-member
    with pytest.raises(
        TypeError,
        match="Cannot coerce <class 'fileformats.generic.File'> into <class 'int'>",
    ):
        TypeChecker(ty.List[int])(task.lzout.a)  # pylint: disable=no-member

    with pytest.raises(
        TypeError, match="Cannot coerce 'bad-value' into <class 'list'>"
    ):
        task.inputs.x = "bad-value"
