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
            name="t", task=Add(a=n), submitter=Submitter(cache_root=tmp_path / str(n))
        )
        job.run()
        dirs.append(job.cache_dir)
    result = CliRunner().invoke(cli, ["explain", str(dirs[0]), str(dirs[1])])
    assert result.exit_code == 1
    assert "inputs.a" in result.output
