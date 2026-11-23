import errno
import os
import time
import pytest
from pydra.utils import storage
from pydra.utils.storage import (
    PROBE_PREFIX,
    StorageCapabilities,
    StorageUnsupported,
    probe_capabilities,
    require_lease_storage,
    storage_now,
)


def scratch_left(root):
    return [p for p in root.iterdir() if p.name.startswith(PROBE_PREFIX)]


def test_probe_passes_on_a_normal_filesystem(tmp_path):
    caps = probe_capabilities(tmp_path)
    assert caps.atomic_mkdir
    assert caps.atomic_rename
    assert caps.server_clock
    assert abs(caps.clock_offset) < 5  # local disk: the storage clock is this host's
    assert caps.missing() == []
    caps.require(tmp_path)
    assert scratch_left(tmp_path) == []


def test_probe_of_a_missing_root_uses_its_ancestor(tmp_path):
    caps = probe_capabilities(tmp_path / "not" / "yet")
    assert caps.missing() == []
    assert not (tmp_path / "not").exists()
    assert scratch_left(tmp_path) == []


def test_probe_fails_cleanly_without_atomic_mkdir(tmp_path, monkeypatch):
    real_mkdir = os.mkdir

    def mkdir_without_exclusion(path, *args, **kwargs):
        try:
            real_mkdir(path, *args, **kwargs)
        except FileExistsError:
            pass  # the second maker succeeds too

    monkeypatch.setattr(os, "mkdir", mkdir_without_exclusion)
    caps = probe_capabilities(tmp_path)
    assert not caps.atomic_mkdir
    assert caps.missing() == ["atomic_mkdir"]
    with pytest.raises(StorageUnsupported, match="atomic_mkdir") as excinfo:
        caps.require(tmp_path)
    assert excinfo.value.errno == errno.ENOTSUP
    assert excinfo.value.filename == str(tmp_path)
    assert scratch_left(tmp_path) == []


def test_probe_reports_mkdir_refused_as_unsupported(tmp_path, monkeypatch):
    def unsupported(path, *args, **kwargs):
        raise OSError(errno.ENOTSUP, "Operation not supported", str(path))

    monkeypatch.setattr(os, "mkdir", unsupported)
    caps = probe_capabilities(tmp_path)
    assert caps == StorageCapabilities(False, False, False, None)
    assert scratch_left(tmp_path) == []


def test_require_lease_storage_probes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_REQUIRED", {})
    calls = []
    real = storage.probe_capabilities

    def counting(root):
        calls.append(root)
        return real(root)

    monkeypatch.setattr(storage, "probe_capabilities", counting)
    require_lease_storage(tmp_path)
    require_lease_storage(tmp_path)
    assert len(calls) == 1


def test_storage_now_reads_the_storage_clock(tmp_path):
    before = time.time()
    now = storage_now(tmp_path)
    assert before - 5 < now < time.time() + 5
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def clock_calls(monkeypatch):
    """Record the flags of ``os.open`` and refuse every ``os.unlink`` of a clock probe, as
    when a scanner holds the new file open; the calls are recorded."""
    real_open, calls = os.open, {"flags": [], "unlink": []}
    fake_temporary = 1 << 29

    def open_(path, flags, *args, **kwargs):
        calls["flags"].append(flags)
        return real_open(path, flags & ~fake_temporary, *args, **kwargs)

    def unlink(path, *args, **kwargs):
        calls["unlink"].append(path)
        raise PermissionError(errno.EACCES, "The process cannot access the file", path)

    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "unlink", unlink)
    calls["temporary"] = fake_temporary
    return calls


def test_storage_now_probe_is_deleted_on_close_on_windows(
    tmp_path, monkeypatch, clock_calls
):
    """Where ``os.O_TEMPORARY`` exists (Windows) the probe is opened delete-on-close and
    never unlinked, so another process holding it open neither fails the read nor
    leaves it behind."""
    monkeypatch.setattr(os, "O_TEMPORARY", clock_calls["temporary"], raising=False)
    storage_now(tmp_path)
    (flags,) = clock_calls["flags"]
    assert flags & clock_calls["temporary"]
    assert clock_calls["unlink"] == []


def test_storage_now_raises_a_refused_removal_elsewhere(
    tmp_path, monkeypatch, clock_calls
):
    monkeypatch.delattr(os, "O_TEMPORARY", raising=False)
    with pytest.raises(PermissionError):
        storage_now(tmp_path)
    assert len(clock_calls["unlink"]) == 1
