import os
import itertools
import typing as ty
from pathlib import Path
import tempfile
import pytest
from pydra import mark
from ..specs import File, LazyField
from ..type_checking import TypeChecker


def lz(tp: ty.Type):
    """convenience method for creating a LazyField of type 'tp'"""
    return LazyField(name="foo", field="boo", attr_type="input", type=tp)


def test_type_check_basic():
    TypeChecker(float, coercible=[(int, float)])(lz(int))
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(int, coercible=[(int, float)])(lz(float))
    TypeChecker(int, coercible=[(ty.Any, int)])(lz(float))
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(int, coercible=[(ty.Any, float)])(lz(float))
    assert TypeChecker(float, not_coercible=[(ty.Any, str)])(lz(int))
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(int, coercible=None, not_coercible=[(float, int)])(lz(float))

    path_coercer = TypeChecker(Path, coercible=[(os.PathLike, os.PathLike)])

    path_coercer(lz(Path))

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        path_coercer(lz(str))

    PathTypes = ty.Union[str, os.PathLike]

    TypeChecker(Path, coercible=[(PathTypes, PathTypes)])(lz(str))
    TypeChecker(str, coercible=[(PathTypes, PathTypes)])(lz(Path))

    file_coercer = TypeChecker(File, coercible=[(PathTypes, File)])

    file_coercer(lz(Path))
    file_coercer(lz(str))

    impotent_str_coercer = TypeChecker(str, coercible=[(PathTypes, File)])

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        impotent_str_coercer(lz(File))

    TypeChecker(str, coercible=[(PathTypes, PathTypes)])(lz(File))
    TypeChecker(File, coercible=[(PathTypes, PathTypes)])(lz(str))

    TypeChecker(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )(lz(ty.Tuple[int, int, int]))
    TypeChecker(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )(lz(ty.Tuple[int, ...]))

    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )(lz(str))

    TypeChecker(ty.Union[Path, File, float])(lz(int))
    with pytest.raises(
        TypeError, match="Cannot coerce <class 'float'> to any of the union types"
    ):
        TypeChecker(ty.Union[Path, File, bool, int])(lz(float))
    TypeChecker(ty.Sequence)(lz(ty.Tuple[int, ...]))


def test_type_check_nested():
    TypeChecker(ty.List[File])(lz(ty.List[Path]))
    TypeChecker(ty.List[Path])(lz(ty.List[File]))
    TypeChecker(ty.List[Path])(lz(ty.List[str]))
    TypeChecker(ty.List[str])(lz(ty.List[File]))
    TypeChecker(ty.Dict[str, ty.List[File]])(lz(ty.Dict[str, ty.List[Path]]))
    TypeChecker(ty.Tuple[float, ...])(lz(ty.List[int]))
    with pytest.raises(TypeError, match="Wrong number of type arguments"):
        TypeChecker(ty.Tuple[float, float, float])(lz(ty.List[int]))
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(
            ty.Tuple[int, ...],
            not_coercible=[(ty.Sequence, ty.Tuple)],
        )(lz(ty.List[float]))


def test_type_check_fail():
    with pytest.raises(TypeError, match="Wrong number of type arguments in tuple"):
        TypeChecker(ty.Tuple[int, int, int])(lz(ty.Tuple[float, float, float, float]))

    with pytest.raises(TypeError, match="to any of the union types"):
        TypeChecker(ty.Union[Path, File])(lz(int))

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(ty.Sequence, coercible=[(ty.Sequence, ty.Sequence)])(
            lz(ty.Dict[str, int])
        )

    with pytest.raises(TypeError, match="Cannot coerce <class 'dict'> into"):
        TypeChecker(ty.Sequence)(lz(ty.Dict[str, int]))

    with pytest.raises(TypeError, match="<class 'int'> doesn't match pattern"):
        TypeChecker(ty.List[int])(lz(int))

    with pytest.raises(TypeError, match="<class 'int'> doesn't match pattern"):
        TypeChecker(ty.List[ty.Dict[str, str]])(lz(ty.Tuple[int, int, int]))


def test_type_coercion_basic():
    assert TypeChecker(float, coercible=[(ty.Any, float)])(1) == 1.0
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(float, coercible=[(ty.Any, int)])(1)
    assert (
        TypeChecker(int, coercible=[(ty.Any, ty.Any)], not_coercible=[(ty.Any, str)])(
            1.0
        )
        == 1
    )
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(int, coercible=[(ty.Any, ty.Any)], not_coercible=[(float, int)])(
            1.0
        )

    path_coercer = TypeChecker(Path, coercible=[(os.PathLike, os.PathLike)])

    assert path_coercer(Path("/a/path")) == Path("/a/path")

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        path_coercer("/a/path")

    PathTypes = ty.Union[str, os.PathLike]

    assert TypeChecker(Path, coercible=[(PathTypes, PathTypes)])("/a/path") == Path(
        "/a/path"
    )
    assert (
        TypeChecker(str, coercible=[(PathTypes, PathTypes)])(Path("/a/path"))
        == "/a/path"
    )
    tmpdir = Path(tempfile.mkdtemp())
    a_file = tmpdir / "a-file.txt"
    Path.touch(a_file)

    file_coercer = TypeChecker(File, coercible=[(PathTypes, File)])

    assert file_coercer(a_file) == File(a_file)
    assert file_coercer(str(a_file)) == File(a_file)

    impotent_str_coercer = TypeChecker(str, coercible=[(PathTypes, File)])

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        impotent_str_coercer(File(a_file))

    assert TypeChecker(str, coercible=[(PathTypes, PathTypes)])(File(a_file)) == str(
        a_file
    )
    assert TypeChecker(File, coercible=[(PathTypes, PathTypes)])(str(a_file)) == File(
        a_file
    )

    assert TypeChecker(
        list,
        coercible=[(ty.Sequence, ty.Sequence)],
        not_coercible=[(str, ty.Sequence)],
    )((1, 2, 3)) == [1, 2, 3]

    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )("a-string")

    assert (
        TypeChecker(ty.Union[Path, File, int], coercible=[(ty.Any, ty.Any)])(1.0) == 1
    )
    assert (
        TypeChecker(ty.Union[Path, File, bool, int], coercible=[(ty.Any, ty.Any)])(1.0)
        is True
    )
    assert TypeChecker(ty.Sequence, coercible=[(ty.Any, ty.Any)])((1, 2, 3)) == (
        1,
        2,
        3,
    )


def test_type_coercion_nested():
    tmpdir = Path(tempfile.mkdtemp())
    a_file = tmpdir / "a-file.txt"
    another_file = tmpdir / "another-file.txt"
    yet_another_file = tmpdir / "yet-another-file.txt"
    Path.touch(a_file)
    Path.touch(another_file)
    Path.touch(yet_another_file)

    PathTypes = ty.Union[str, bytes, os.PathLike]

    assert TypeChecker(ty.List[File], coercible=[(PathTypes, PathTypes)])(
        [a_file, another_file, yet_another_file]
    ) == [File(a_file), File(another_file), File(yet_another_file)]

    assert TypeChecker(ty.List[Path], coercible=[(PathTypes, PathTypes)])(
        [File(a_file), File(another_file), File(yet_another_file)]
    ) == [a_file, another_file, yet_another_file]

    assert TypeChecker(ty.Dict[str, ty.List[File]], coercible=[(PathTypes, PathTypes)])(
        {
            "a": [a_file, another_file, yet_another_file],
            "b": [a_file, another_file],
        }
    ) == {
        "a": [File(a_file), File(another_file), File(yet_another_file)],
        "b": [File(a_file), File(another_file)],
    }

    assert TypeChecker(ty.List[File], coercible=[(PathTypes, PathTypes)])(
        [a_file, another_file, yet_another_file]
    ) == [File(a_file), File(another_file), File(yet_another_file)]

    assert TypeChecker(ty.Tuple[int, int, int], coercible=[(ty.Any, ty.Any)])(
        [1.0, 2.0, 3.0]
    ) == (1, 2, 3)
    assert TypeChecker(ty.Tuple[int, ...], coercible=[(ty.Any, ty.Any)])(
        [1.0, 2.0, 3.0]
    ) == (1, 2, 3)
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeChecker(
            ty.Tuple[int, ...],
            coercible=[(ty.Any, ty.Any)],
            not_coercible=[(ty.Sequence, ty.Tuple)],
        )([1.0, 2.0, 3.0])


def test_type_coercion_fail():
    with pytest.raises(TypeError, match="Incorrect number of items"):
        TypeChecker(ty.Tuple[int, int, int], coercible=[(ty.Any, ty.Any)])(
            [1.0, 2.0, 3.0, 4.0]
        )

    with pytest.raises(TypeError, match="to any of the union types"):
        TypeChecker(ty.Union[Path, File], coercible=[(ty.Any, ty.Any)])(1)

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeChecker(ty.Sequence, coercible=[(ty.Sequence, ty.Sequence)])(
            {"a": 1, "b": 2}
        )

    with pytest.raises(TypeError, match="Cannot coerce {'a': 1} into"):
        TypeChecker(ty.Sequence, coercible=[(ty.Any, ty.Any)])({"a": 1})

    with pytest.raises(TypeError, match="as 1 is not iterable"):
        TypeChecker(ty.List[int], coercible=[(ty.Any, ty.Any)])(1)

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
