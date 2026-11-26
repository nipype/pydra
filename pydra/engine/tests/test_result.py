import json
import logging
import shlex
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import attrs
from fileformats.text import TextFile

from pydra.compose import python
from pydra.engine.result import (
    Result,
    Runtime,
    _write_crash_artifacts,
    copyfile_workflow,
    record_error,
)


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
    assert len(notebook["cells"]) == 4
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "_job.pklz" in notebook["cells"][1]["source"]
    assert notebook["cells"][2]["cell_type"] == "markdown"
    assert notebook["cells"][3]["source"] == "job.run(rerun=True)"

    shell_launcher = tmp_path / "launch-crash.sh"
    assert shlex.quote(sys.executable) in shell_launcher.read_text()
    if sys.platform != "win32":
        assert shell_launcher.stat().st_mode & stat.S_IXUSR

    batch_launcher = (tmp_path / "launch-crash.bat").read_text()
    assert f'"{sys.executable}"' in batch_launcher


def test_crash_artifact_error_is_nonfatal(tmp_path: Path, monkeypatch, caplog) -> None:
    def raise_oserror(error_path, job=None):
        raise OSError("not supported")

    monkeypatch.setattr("pydra.engine.result._write_crash_artifacts", raise_oserror)

    with caplog.at_level(logging.WARNING, logger="pydra"):
        error_file = record_error(tmp_path, ["traceback"])

    assert error_file.exists()
    assert "Could not create Jupyter crash artifacts" in caplog.text


def test_crash_notebook_displays_successes_and_reruns_failures(
    tmp_path: Path,
) -> None:
    failed_job = SimpleNamespace(
        result=lambda: SimpleNamespace(errored=True),
    )
    queued_job = SimpleNamespace(result=lambda: None)
    node = SimpleNamespace(
        name="split_task",
        _tasks={0: object(), 1: failed_job, 2: queued_job},
        successful={0: object()},
        errored={},
        queued={1: failed_job, 2: queued_job},
        running={},
    )
    graph = SimpleNamespace(sorted_nodes=[node])
    job = SimpleNamespace(return_values={"exec_graph": graph})

    _write_crash_artifacts(tmp_path, job=job)

    cells = json.loads((tmp_path / "crash.ipynb").read_text())["cells"]
    sources = [cell["source"] for cell in cells]
    completed = next(source for source in sources if source.startswith("## Completed"))
    successful = next(source for source in sources if "task_job.result()" in source)
    failed = next(source for source in sources if source.startswith("## Failed"))
    rerun = next(source for source in sources if "failed_job.run" in source)

    assert "`split_task[0]`" in completed
    assert "graph['split_task']._tasks[0]" in successful
    assert "rerun=True" not in successful
    assert "`split_task[1]`" in failed
    assert "graph['split_task']._tasks[1]" in rerun
    assert "failed_job.run(rerun=True)" in rerun
    assert not any("_tasks[2]" in source for source in sources)


def test_result_display_formats_outputs_as_table(
    tmp_path: Path, monkeypatch
) -> None:
    displayed = []
    ipython = ModuleType("IPython")
    ipython_display = ModuleType("IPython.display")

    class HTML(str):
        pass

    ipython_display.HTML = HTML
    ipython_display.display = displayed.append
    ipython.display = ipython_display
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    @attrs.define
    class Outputs:
        answer: int
        text: str

    Result(
        cache_dir=tmp_path,
        outputs=Outputs(answer=42, text="<expected>"),
    ).display()

    assert len(displayed) == 1
    assert "<th>answer</th>" in displayed[0]
    assert "<code>42</code>" in displayed[0]
    assert "&lt;expected&gt;" in displayed[0]
