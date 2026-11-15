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
