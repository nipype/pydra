import os
import stat
from pathlib import Path
import pytest
from pydra.engine.result import record_error, save, Result
from pydra.utils.general import atomic_write


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def umask():
    old = os.umask(0)
    yield os.umask
    os.umask(old)


def test_atomic_write_creates_file(tmp_path: Path):
    target = tmp_path / "f.bin"
    with atomic_write(target) as fp:
        fp.write(b"data")
    assert target.read_bytes() == b"data"
    assert list(tmp_path.iterdir()) == [target]


def test_failure_while_writing_leaves_nothing(tmp_path: Path):
    target = tmp_path / "f.bin"
    with pytest.raises(RuntimeError):
        with atomic_write(target) as fp:
            fp.write(b"par")
            raise RuntimeError("interrupted")
    assert list(tmp_path.iterdir()) == []


def test_failure_while_writing_keeps_old_file(tmp_path: Path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"old")
    with pytest.raises(RuntimeError):
        with atomic_write(target) as fp:
            fp.write(b"par")
            raise RuntimeError("interrupted")
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("existing", [None, b"old"])
def test_failed_replace_never_exposes_partial_file(
    tmp_path: Path, monkeypatch, existing
):
    target = tmp_path / "f.bin"
    if existing is not None:
        target.write_bytes(existing)

    def boom(src, dst):
        # the temp file is complete here, but must not be under the final name
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        with atomic_write(target) as fp:
            fp.write(b"new content")
    if existing is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == existing
    assert [p for p in tmp_path.iterdir() if p != target] == []


def test_final_name_absent_until_replace(tmp_path: Path, monkeypatch):
    target = tmp_path / "f.bin"
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        seen["target_exists_before"] = Path(dst).exists()
        seen["src_content"] = Path(src).read_bytes()
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    with atomic_write(target) as fp:
        fp.write(b"abc")
        fp.flush()
        assert not target.exists()
    assert seen == {"target_exists_before": False, "src_content": b"abc"}


@pytest.mark.parametrize(
    "mask, expected", [(0o077, 0o600), (0o022, 0o644), (0o002, 0o644), (0o000, 0o644)]
)
def test_mode_follows_umask(tmp_path: Path, umask, mask, expected):
    umask(mask)
    target = tmp_path / "f.bin"
    with atomic_write(target) as fp:
        fp.write(b"x")
    assert mode(target) == expected


@pytest.mark.parametrize("mask, expected", [(0o077, 0o600), (0o000, 0o644)])
def test_cache_pickles_have_stated_mode(tmp_path: Path, umask, mask, expected):
    umask(mask)
    cache_dir = tmp_path / "cache" / "abc"
    save(cache_dir, result=Result(cache_dir=cache_dir), job=None)
    record_error(cache_dir, "benign error text")
    for name in ("_result.pklz", "_error.pklz"):
        assert mode(cache_dir / name) == expected
    assert sorted(p.name for p in cache_dir.iterdir()) == [
        "_error.pklz",
        "_result.pklz",
    ]


def test_save_failure_keeps_previous_result(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "abc"
    save(cache_dir, result=Result(cache_dir=cache_dir))
    before = (cache_dir / "_result.pklz").read_bytes()

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        save(cache_dir, result=Result(cache_dir=cache_dir))
    assert (cache_dir / "_result.pklz").read_bytes() == before
