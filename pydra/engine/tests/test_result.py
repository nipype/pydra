from pathlib import Path
from fileformats.text import TextFile
from pydra.compose import python
from pydra.engine.result import Result, Runtime, copyfile_workflow


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
    outputs = MockTask.Outputs(d=file1, e=file2, f=file3)
    workflow_dir.mkdir()

    copyfile_workflow(workflow_dir, outputs)

    assert sorted(p.stem for p in workflow_dir.iterdir()) == [
        "out",
        "out (1)",
        "out (2)",
    ]
