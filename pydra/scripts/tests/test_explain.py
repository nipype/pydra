import json

from click.testing import CliRunner

from pydra.compose import python
from pydra.scripts.cli import cli, explain_records
from pydra.utils.identity import task_record


@python.define
def Add(a: int, b: int = 2) -> int:
    return a + b


def test_explain_shows_differing_fields():
    one, two = task_record(Add(a=1)), task_record(Add(a=2))
    lines = explain_records(one, two)
    assert len(lines) == 1
    assert lines[0].startswith("inputs.a: ")
    assert one["inputs"]["a"] in lines[0] and two["inputs"]["a"] in lines[0]


def test_explain_identical_records():
    record = task_record(Add(a=1))
    assert explain_records(record, record) == []


def test_explain_names_walk_difference():
    one = task_record(Add(a=1))
    two = {**one, "walk": {**one["walk"], "user_packages": ["mylib"]}}
    (line,) = explain_records(one, two)
    assert line.startswith("walk.user_packages: ")
    assert "PYDRA_USER_PACKAGES" in line


def test_explain_shows_absent_triples_and_missing_fields():
    one = task_record(Add(a=1))
    two = {**one, "absent": [["out", 0, "dropped"]], "job": {"environment": "x"}}
    lines = explain_records(one, two)
    assert any(line.startswith("absent: ") and "dropped" in line for line in lines)
    assert any(line.startswith("job.environment: (missing) -> ") for line in lines)


def test_explain_command_reads_json_records(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(task_record(Add(a=1))))
    b.write_text(json.dumps(task_record(Add(a=2))))
    result = CliRunner().invoke(cli, ["explain", str(a), str(b)])
    assert result.exit_code == 1
    assert "inputs.a" in result.output
    same = CliRunner().invoke(cli, ["explain", str(a), str(a)])
    assert same.exit_code == 0
    assert "identical" in same.output


def test_explain_command_reads_result_directories(tmp_path):
    from pydra.engine.job import Job
    from pydra.engine.submitter import Submitter

    dirs = []
    for n in (1, 2):
        job = Job(
            name="t",
            task=Add(a=n),
            submitter=Submitter(store="pickle", cache_root=tmp_path / str(n)),
        )
        job.run()
        dirs.append(job.cache_dir)
    result = CliRunner().invoke(cli, ["explain", str(dirs[0]), str(dirs[1])])
    assert result.exit_code == 1
    assert "inputs.a" in result.output


def test_explain_old_result_without_record(tmp_path, monkeypatch):
    import cloudpickle as cp

    from pydra.engine.result import Result

    old = tmp_path / "old"
    old.mkdir()
    result = Result(cache_dir=old)
    # a result pickled before the record existed leaves the slot unset
    full = Result.__getstate__

    def without_record(self):
        state = full(self)
        del state["record"]
        return state

    monkeypatch.setattr(Result, "__getstate__", without_record)
    (old / "_result.pklz").write_bytes(cp.dumps(result))
    monkeypatch.undo()
    out = CliRunner().invoke(cli, ["explain", str(old), str(old)])
    assert out.exit_code == 1
    assert "no identity record" in out.output
    assert "Traceback" not in out.output


def test_explain_result_with_none_record(tmp_path):
    import cloudpickle as cp

    from pydra.engine.result import Result

    d = tmp_path / "none"
    d.mkdir()
    (d / "_result.pklz").write_bytes(cp.dumps(Result(cache_dir=d)))
    out = CliRunner().invoke(cli, ["explain", str(d), str(d)])
    assert out.exit_code == 1
    assert "no identity record" in out.output


def test_explain_directory_without_result(tmp_path):
    out = CliRunner().invoke(cli, ["explain", str(tmp_path), str(tmp_path)])
    assert out.exit_code == 1
    assert "_result.pklz" in out.output
    assert not isinstance(out.exception, FileNotFoundError)


def test_explain_renders_values():
    one = {"a": "x"}
    assert explain_records(one, {"a": "y"}) == ['a: "x" -> "y"']
    assert explain_records(one, {}) == ['a: "x" -> (missing)']
