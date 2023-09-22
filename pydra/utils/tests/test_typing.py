import os
import itertools
import typing as ty
from pathlib import Path
import tempfile
import pytest
from pydra import mark
from ...engine.specs import File, LazyOutField
from ..typing import TypeParser
from pydra import Workflow
from fileformats.application import Json
from .utils import (
    generic_func_task,
    GenericShellTask,
    specific_func_task,
    SpecificShellTask,
    MyFormatX,
    MyHeader,
)


def lz(tp: ty.Type):
    """convenience method for creating a LazyField of type 'tp'"""
    return LazyOutField(name="foo", field="boo", type=tp)


PathTypes = ty.Union[str, os.PathLike]


def test_type_check_basic1():
    TypeParser(float, coercible=[(int, float)])(lz(int))


def test_type_check_basic2():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeParser(int, coercible=[(int, float)])(lz(float))


def test_type_check_basic3():
    TypeParser(int, coercible=[(ty.Any, int)])(lz(float))


def test_type_check_basic4():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeParser(int, coercible=[(ty.Any, float)])(lz(float))


def test_type_check_basic5():
    assert TypeParser(float, not_coercible=[(ty.Any, str)])(lz(int))


def test_type_check_basic6():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeParser(int, coercible=None, not_coercible=[(float, int)])(lz(float))


def test_type_check_basic7():
    path_coercer = TypeParser(Path, coercible=[(os.PathLike, os.PathLike)])

    path_coercer(lz(Path))

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        path_coercer(lz(str))


def test_type_check_basic8():
    TypeParser(Path, coercible=[(PathTypes, PathTypes)])(lz(str))
    TypeParser(str, coercible=[(PathTypes, PathTypes)])(lz(Path))


def test_type_check_basic9():
    file_coercer = TypeParser(File, coercible=[(PathTypes, File)])

    file_coercer(lz(Path))
    file_coercer(lz(str))


def test_type_check_basic10():
    impotent_str_coercer = TypeParser(str, coercible=[(PathTypes, File)])

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        impotent_str_coercer(lz(File))


def test_type_check_basic11():
    TypeParser(str, coercible=[(PathTypes, PathTypes)])(lz(File))
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
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeParser(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )(lz(str))


def test_type_check_basic15():
    TypeParser(ty.Union[Path, File, float])(lz(int))


def test_type_check_basic16():
    with pytest.raises(
        TypeError, match="Cannot coerce <class 'float'> to any of the union types"
    ):
        TypeParser(ty.Union[Path, File, bool, int])(lz(float))


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
    with pytest.raises(TypeError, match="Wrong number of type arguments"):
        TypeParser(ty.Tuple[float, float, float])(lz(ty.List[int]))


def test_type_check_nested8():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeParser(
            ty.Tuple[int, ...],
            not_coercible=[(ty.Sequence, ty.Tuple)],
        )(lz(ty.List[float]))


def test_type_check_fail1():
    with pytest.raises(TypeError, match="Wrong number of type arguments in tuple"):
        TypeParser(ty.Tuple[int, int, int])(lz(ty.Tuple[float, float, float, float]))


def test_type_check_fail2():
    with pytest.raises(TypeError, match="to any of the union types"):
        TypeParser(ty.Union[Path, File])(lz(int))


def test_type_check_fail3():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeParser(ty.Sequence, coercible=[(ty.Sequence, ty.Sequence)])(
            lz(ty.Dict[str, int])
        )


def test_type_check_fail4():
    with pytest.raises(TypeError, match="Cannot coerce <class 'dict'> into"):
        TypeParser(ty.Sequence)(lz(ty.Dict[str, int]))


def test_type_check_fail5():
    with pytest.raises(TypeError, match="<class 'int'> doesn't match pattern"):
        TypeParser(ty.List[int])(lz(int))


def test_type_check_fail6():
    with pytest.raises(TypeError, match="<class 'int'> doesn't match pattern"):
        TypeParser(ty.List[ty.Dict[str, str]])(lz(ty.Tuple[int, int, int]))


def test_type_coercion_basic():
    assert TypeParser(float, coercible=[(ty.Any, float)])(1) == 1.0


def test_type_coercion_basic1():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeParser(float, coercible=[(ty.Any, int)])(1)


def test_type_coercion_basic2():
    assert (
        TypeParser(int, coercible=[(ty.Any, ty.Any)], not_coercible=[(ty.Any, str)])(
            1.0
        )
        == 1
    )


def test_type_coercion_basic3():
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeParser(int, coercible=[(ty.Any, ty.Any)], not_coercible=[(float, int)])(1.0)


def test_type_coercion_basic4():
    path_coercer = TypeParser(Path, coercible=[(os.PathLike, os.PathLike)])

    assert path_coercer(Path("/a/path")) == Path("/a/path")

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        path_coercer("/a/path")


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

    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        impotent_str_coercer(File(a_file))


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
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeParser(
            list,
            coercible=[(ty.Sequence, ty.Sequence)],
            not_coercible=[(str, ty.Sequence)],
        )("a-string")

    assert TypeParser(ty.Union[Path, File, int], coercible=[(ty.Any, ty.Any)])(1.0) == 1


def test_type_coercion_basic13():
    assert (
        TypeParser(ty.Union[Path, File, bool, int], coercible=[(ty.Any, ty.Any)])(1.0)
        is True
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
    with pytest.raises(TypeError, match="explicitly excluded"):
        TypeParser(
            ty.Tuple[int, ...],
            coercible=[(ty.Any, ty.Any)],
            not_coercible=[(ty.Sequence, ty.Tuple)],
        )([1.0, 2.0, 3.0])


def test_type_coercion_fail1():
    with pytest.raises(TypeError, match="Incorrect number of items"):
        TypeParser(ty.Tuple[int, int, int], coercible=[(ty.Any, ty.Any)])(
            [1.0, 2.0, 3.0, 4.0]
        )


def test_type_coercion_fail2():
    with pytest.raises(TypeError, match="to any of the union types"):
        TypeParser(ty.Union[Path, File], coercible=[(ty.Any, ty.Any)])(1)


def test_type_coercion_fail3():
    with pytest.raises(TypeError, match="doesn't match any of the explicit inclusion"):
        TypeParser(ty.Sequence, coercible=[(ty.Sequence, ty.Sequence)])(
            {"a": 1, "b": 2}
        )


def test_type_coercion_fail4():
    with pytest.raises(TypeError, match="Cannot coerce {'a': 1} into"):
        TypeParser(ty.Sequence, coercible=[(ty.Any, ty.Any)])({"a": 1})


def test_type_coercion_fail5():
    with pytest.raises(TypeError, match="as 1 is not iterable"):
        TypeParser(ty.List[int], coercible=[(ty.Any, ty.Any)])(1)


def test_type_coercion_fail6():
    with pytest.raises(TypeError, match="is not a mapping type"):
        TypeParser(ty.List[ty.Dict[str, str]], coercible=[(ty.Any, ty.Any)])((1, 2, 3))


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

    TypeParser(ty.List[str])(task.lzout.a)  # pylint: disable=no-member
    with pytest.raises(
        TypeError,
        match="Cannot coerce <class 'fileformats.generic.File'> into <class 'int'>",
    ):
        TypeParser(ty.List[int])(task.lzout.a)  # pylint: disable=no-member

    with pytest.raises(
        TypeError, match="Cannot coerce 'bad-value' into <class 'list'>"
    ):
        task.inputs.x = "bad-value"


def test_check_missing_type_args():
    with pytest.raises(TypeError, match="wasn't declared with type args required"):
        TypeParser(ty.List[int]).check_type(list)
    with pytest.raises(TypeError, match="doesn't match pattern"):
        TypeParser(ty.List[int]).check_type(dict)


def test_matches_type_union():
    assert TypeParser.matches_type(ty.Union[int, bool, str], ty.Union[int, bool, str])
    assert TypeParser.matches_type(ty.Union[int, bool], ty.Union[int, bool, str])
    assert not TypeParser.matches_type(ty.Union[int, bool, str], ty.Union[int, bool])


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


def test_matches_type_tuple_ellipsis():
    assert TypeParser.matches_type(ty.Tuple[int], ty.Tuple[int, ...])
    assert TypeParser.matches_type(ty.Tuple[int, int], ty.Tuple[int, ...])
    assert not TypeParser.matches_type(ty.Tuple[int, float], ty.Tuple[int, ...])
    assert not TypeParser.matches_type(ty.Tuple[int, ...], ty.Tuple[int])
    assert TypeParser.matches_type(
        ty.Tuple[int], ty.List[int], coercible=[(tuple, list)]
    )
    assert TypeParser.matches_type(
        ty.Tuple[int, ...], ty.List[int], coercible=[(tuple, list)]
    )


def test_contains_type_in_dict():
    assert TypeParser.contains_type(int, ty.Dict[str, ty.List[ty.Tuple[int, ...]]])
    assert not TypeParser.contains_type(
        int, ty.Dict[str, ty.List[ty.Tuple[float, ...]]]
    )


def test_type_matches():
    assert TypeParser.matches([1, 2, 3], ty.List[int])
    assert TypeParser.matches((1, 2, 3), ty.Tuple[int, ...])

    assert TypeParser.matches((1, 2, 3), ty.List[int])
    assert not TypeParser.matches((1, 2, 3), ty.List[int], coercible=[])


@pytest.fixture(params=["func", "shell"])
def generic_task(request):
    if request.param == "func":
        return generic_func_task
    elif request.param == "shell":
        return GenericShellTask
    else:
        assert False


@pytest.fixture(params=["func", "shell"])
def specific_task(request):
    if request.param == "func":
        return specific_func_task
    elif request.param == "shell":
        return SpecificShellTask
    else:
        assert False


def test_typing_cast(tmp_path, generic_task, specific_task):
    """Check the casting of lazy fields and whether specific file-sets can be recovered
    from generic `File` classes"""

    wf = Workflow(
        name="test",
        input_spec={"in_file": MyFormatX},
        output_spec={"out_file": MyFormatX},
    )

    wf.add(
        specific_task(
            in_file=wf.lzin.in_file,
            name="specific1",
        )
    )

    wf.add(  # Generic task
        generic_task(
            in_file=wf.specific1.lzout.out,
            name="generic",
        )
    )

    with pytest.raises(TypeError, match="Cannot coerce"):
        # No cast of generic task output to MyFormatX
        wf.add(
            specific_task(
                in_file=wf.generic.lzout.out,
                name="specific2",
            )
        )

    wf.add(
        specific_task(
            in_file=wf.generic.lzout.out.cast(MyFormatX),
            name="specific2",
        )
    )

    wf.set_output(
        [
            ("out_file", wf.specific2.lzout.out),
        ]
    )

    my_fspath = tmp_path / "in_file.my"
    hdr_fspath = tmp_path / "in_file.hdr"
    my_fspath.write_text("my-format")
    hdr_fspath.write_text("my-header")
    in_file = MyFormatX([my_fspath, hdr_fspath])

    result = wf(in_file=in_file, plugin="serial")

    out_file: MyFormatX = result.output.out_file
    assert type(out_file) is MyFormatX
    assert out_file.parent != in_file.parent
    assert type(out_file.header) is MyHeader
    assert out_file.header.parent != in_file.header.parent


def test_type_is_subclass1():
    assert TypeParser.is_subclass(ty.Type[File], type)


def test_type_is_subclass2():
    assert not TypeParser.is_subclass(ty.Type[File], ty.Type[Json])


def test_type_is_subclass3():
    assert TypeParser.is_subclass(ty.Type[Json], ty.Type[File])


def test_type_is_instance1():
    assert TypeParser.is_instance(File, ty.Type[File])


def test_type_is_instance2():
    assert not TypeParser.is_instance(File, ty.Type[Json])


def test_type_is_instance3():
    assert TypeParser.is_instance(Json, ty.Type[File])


def test_type_is_instance4():
    assert TypeParser.is_instance(Json, type)
