import os
import subprocess
import sys
import pytest

INNER_TESTS = """
import os
import uuid
from pathlib import Path
from pydra.compose import python


@python.define
def Stamp(log: str) -> int:
    Path(log, uuid.uuid4().hex).touch()
    return len(os.listdir(log))


def test_first():
    Stamp(log=os.environ["STAMP_LOG"])()


def test_second():
    Stamp(log=os.environ["STAMP_LOG"])()
"""


@pytest.mark.parametrize("isolated", [True, False])
def test_tests_do_not_see_each_others_results(tmp_path, isolated):
    """Two tests that run the same task in the default cache each run it when the
    suite's fixture isolates their caches, and share one result when it does not"""
    session = tmp_path / "session"
    session.mkdir()
    (session / "test_inner.py").write_text(INNER_TESTS)
    (session / "conftest.py").write_text(
        "from pydra.conftest import isolated_user_cache  # noqa: F401\n"
        if isolated
        else ""
    )
    log = tmp_path / "log"
    log.mkdir()
    shared_root = tmp_path / "shared-root"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-o",
            "addopts=",
            "-q",
            str(session),
        ],
        cwd=session,
        env={**os.environ, "STAMP_LOG": str(log), "PYDRA_CACHE_ROOT": str(shared_root)},
        check=True,
    )
    assert len(list(log.iterdir())) == (2 if isolated else 1)
    assert (shared_root / "run-cache").exists() is not isolated
