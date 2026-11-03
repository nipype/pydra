import pickle
from pathlib import Path
from fileformats.text import TextFile
from pydra.compose import python
from pydra.engine.result import Result, Runtime, copyfile_workflow
from pydra.utils.general import get_fields


@python.define(outputs=["d", "e", "f"])
def MockTask(
    a: TextFile, b: TextFile, c: TextFile
) -> tuple[TextFile, TextFile, TextFile]:
    return a, b, c


def test_runtime():
    runtime = Runtime()
    assert hasattr(runtime, "rss_peak_gb")
    assert hasattr(runtime, "vms_peak_gb")
    assert hasattr(runtime, "cpu_peak_percent")


def test_result(tmp_path: Path):
    result = Result(cache_dir=tmp_path)
    assert hasattr(result, "runtime")
    assert hasattr(result, "outputs")
    assert hasattr(result, "errored")
    assert getattr(result, "errored") is False


def test_copyfile_workflow_conflicting_filenames(tmp_path: Path) -> None:
    """Copy outputs to the workflow output directory with conflicting filenames.
    The filenames should be disambiguated to avoid clashes"""
    file1 = TextFile.sample(stem="out")
    file2 = TextFile.sample(stem="out")
    file3 = TextFile.sample(stem="out")

    workflow_dir = tmp_path / "output"
    mock = MockTask(a=file1, b=file2, c=file3)
    outputs = mock()
    workflow_dir.mkdir()

    copyfile_workflow(workflow_dir, outputs)

    assert sorted(p.stem for p in workflow_dir.iterdir()) == [
        "out",
        "out (1)",
        "out (2)",
    ]


def test_result_keeps_classes_pickled_by_value(tmp_path: Path):
    """Loading a result does not replace the attributes of a task class this process
    pickled by value, such as its function and the globals that function reads"""
    namespace = {"__name__": "__main__"}
    exec("SCALE = 2\n\ndef scale(a: int) -> int:\n    return a * SCALE\n", namespace)
    Scale = python.define(namespace["scale"])
    task = Scale(a=1)
    loaded = pickle.loads(pickle.dumps(Result(cache_dir=tmp_path, task=task)))
    assert type(loaded.task) is Scale
    assert get_fields(Scale).function.default is namespace["scale"]


def test_result_keeps_enums_pickled_by_value(tmp_path: Path):
    """Loading a result does not replace the members or methods of an enum this
    process pickled by value"""
    namespace = {"__name__": "__main__"}
    exec(
        "import enum\n\nclass Mode(enum.Enum):\n    A = 1\n\n"
        "    def scale(self):\n        return 2\n",
        namespace,
    )
    Mode = namespace["Mode"]
    member, scale = Mode.A, vars(Mode)["scale"]
    result = Result(cache_dir=tmp_path, task=MockTask(), outputs=member)
    loaded = pickle.loads(pickle.dumps(result))
    assert loaded.outputs is member
    assert vars(Mode)["scale"] is scale
