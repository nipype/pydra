import json
import logging
import shlex
import stat
import sys
from pathlib import Path

from fileformats.text import TextFile

from pydra.compose import python
from pydra.engine.result import Result, Runtime, copyfile_workflow, record_error


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


def test_record_error_writes_crash_artifacts(tmp_path: Path) -> None:
    record_error(tmp_path, ["traceback"])

    notebook = json.loads((tmp_path / "crash.ipynb").read_text())
    assert len(notebook["cells"]) == 2
    assert "_job.pklz" in notebook["cells"][0]["source"]
    assert notebook["cells"][1]["source"] == "job.run(rerun=True)"

    shell_launcher = tmp_path / "launch-crash.sh"
    assert shlex.quote(sys.executable) in shell_launcher.read_text()
    assert shell_launcher.stat().st_mode & stat.S_IXUSR

    batch_launcher = (tmp_path / "launch-crash.bat").read_text()
    assert f'"{sys.executable}"' in batch_launcher


def test_crash_artifact_error_is_nonfatal(tmp_path: Path, monkeypatch, caplog) -> None:
    def raise_oserror(error_path):
        raise OSError("not supported")

    monkeypatch.setattr("pydra.engine.result._write_crash_artifacts", raise_oserror)

    with caplog.at_level(logging.WARNING, logger="pydra"):
        error_file = record_error(tmp_path, ["traceback"])

    assert error_file.exists()
    assert "Could not create Jupyter crash artifacts" in caplog.text
