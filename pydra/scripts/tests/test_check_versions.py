import json
import sys
from pydra.compose import python
from pydra.scripts.cli import check_versions_command
from pydra.utils.hash import code_digest


def inc(a: int) -> int:
    return a + 1


def inc_edited(a: int) -> int:
    return a + 2


Inc = python.define(inc, version="1")
Plain = python.define(inc)


def _table(tmp_path, rows):
    path = tmp_path / "versions.json"
    path.write_text(json.dumps(rows))
    return path


def _name(attr):
    return f"{__name__}.{attr}"


def test_check_versions(cli_runner, tmp_path, monkeypatch):
    module = sys.modules[__name__]
    table = _table(
        tmp_path, {_name("Inc"): {"version": "1", "digest": code_digest(Inc)}}
    )
    assert cli_runner(check_versions_command, [str(table)]).exit_code == 0

    # the code moved under the same version: fails, and no update fixes it
    inc_edited.__qualname__ = "inc"
    monkeypatch.setattr(module, "Inc", python.define(inc_edited, version="1"))
    for args in ([], ["--update"]):
        result = cli_runner(check_versions_command, [str(table), *args])
        assert result.exit_code == 1
        assert "bump version= or remove it" in result.output

    # bumped: fails until the table is updated
    monkeypatch.setattr(module, "Inc", python.define(inc_edited, version="2"))
    result = cli_runner(check_versions_command, [str(table)])
    assert result.exit_code == 1 and "update the table" in result.output
    assert cli_runner(check_versions_command, [str(table), "--update"]).exit_code == 0
    assert json.loads(table.read_text()) == {
        _name("Inc"): {"version": "2", "digest": code_digest(module.Inc)}
    }
    assert cli_runner(check_versions_command, [str(table)]).exit_code == 0


def test_check_versions_needs_a_version(cli_runner, tmp_path):
    table = _table(tmp_path, {_name("Plain"): {"version": "1", "digest": "x"}})
    result = cli_runner(check_versions_command, [str(table)])
    assert result.exit_code == 1 and "declares no version" in result.output
