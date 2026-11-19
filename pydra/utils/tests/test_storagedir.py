"""Tests for `pydra.utils.storagedir` (RFC 0003 section 9)."""

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from pydra.utils import storagedir
from pydra.utils.storage import PlatformUnsupported, StorageUnsupported
from pydra.utils.storagedir import (
    StorageDir,
    StoreAges,
    StorePolicy,
    StoreUnavailable,
    WrongKindError,
    copy_hashed,
)

IS_WINDOWS = sys.platform == "win32"
POSIX_ONLY = pytest.mark.skipif(IS_WINDOWS, reason="needs symlinks and POSIX modes")


@pytest.fixture(params=["descriptor", "path"])
def backend(request, monkeypatch):
    """Both implementations: descriptor-relative (POSIX) and the path-and-checks one that
    Windows uses, which runs here too so that its logic is exercised on every platform.
    """
    if request.param == "descriptor":
        if not storagedir.DESCRIPTORS_SUPPORTED:
            pytest.skip("no descriptor-relative calls on this platform")
    else:
        monkeypatch.setattr(storagedir, "DESCRIPTORS_SUPPORTED", False)
    return request.param


@pytest.fixture
def policy():
    return StorePolicy()


@pytest.fixture
def root(tmp_path, policy, backend):
    d = tmp_path / "root"
    d.mkdir()
    # a store's roots are the operator's; on Windows only private/trusted open
    return StorageDir.open_root(d, policy, trust="private")


@pytest.fixture
def permissive_umask():
    old = os.umask(0)
    yield
    os.umask(old)


def _mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_exceptions_are_oserrors():
    assert issubclass(StoreUnavailable, OSError)
    assert issubclass(WrongKindError, StoreUnavailable)
    assert issubclass(PlatformUnsupported, StorageUnsupported)


def test_defaults_are_the_rfc_values():
    p = StorePolicy()
    assert (p.file_mode, p.entry_mode, p.sealed_mode) == (0o444, 0o755, None)
    assert (p.container_mode, p.side_mode, p.staging_mode) == (0o755, 0o755, 0o700)
    assert (p.durability, p.verify_files) == ("member", "digest")
    assert p.ages == StoreAges()


def test_child_create_and_roundtrip(root):
    sub = root.child("a", create=True)
    sub.child("b", create=True, mode=0o700)
    again = root.child("a").child("b")
    assert os.path.isdir(Path(root.path) / "a" / "b")
    assert again.fstat() is not None
    with pytest.raises(FileNotFoundError):
        root.child("missing")


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\0b"])
def test_names_are_single_components(root, name):
    with pytest.raises(ValueError):
        root.child(name, create=True)
    with pytest.raises(ValueError):
        root.open_new(name, 0o444)


@POSIX_ONLY
def test_symlink_swapped_in_at_each_level_refused(root, tmp_path):
    outside = tmp_path / "outside"
    (outside / "x" / "y").mkdir(parents=True)
    (outside / "x" / "y" / "secret").write_text("s")
    real = Path(root.path)
    for level, parts in enumerate([("l1",), ("l1", "l2"), ("l1", "l2", "l3")]):
        base = real / f"case{level}"
        (base / "l1" / "l2" / "l3").mkdir(parents=True)
        handle = root.child(base.name)
        target = base.joinpath(*parts)
        aside = target.with_name(target.name + "-aside")
        target.rename(aside)
        target.symlink_to(outside / "x", target_is_directory=True)
        # walking down from the root handle meets the symlink at exactly this level
        cur = handle
        names = ["l1", "l2", "l3"]
        with pytest.raises(WrongKindError):
            for n in names:
                cur = cur.child(n)
        # and a symlink where a file is expected is refused for reading and creating
    (real / "f").symlink_to(outside / "x" / "y" / "secret")
    with pytest.raises(WrongKindError):
        root.open_read("f")
    with pytest.raises(FileExistsError):
        root.open_new("f", 0o444)
    assert (outside / "x" / "y" / "secret").read_text() == "s"


@POSIX_ONLY
def test_handle_keeps_its_directory_when_the_path_is_swapped(tmp_path, policy):
    if not storagedir.DESCRIPTORS_SUPPORTED:
        pytest.skip("descriptor handles only")
    (tmp_path / "r" / "d").mkdir(parents=True)
    root = StorageDir.open_root(tmp_path / "r", policy)
    held = root.child("d")
    (tmp_path / "r" / "d").rename(tmp_path / "r" / "moved")
    (tmp_path / "evil").mkdir()
    (tmp_path / "r" / "d").symlink_to(tmp_path / "evil", target_is_directory=True)
    with held.open_new("m", 0o444) as f:
        f.write(b"x")
    assert (tmp_path / "r" / "moved" / "m").exists()
    assert not (tmp_path / "evil" / "m").exists()


def test_wrong_kind_raises(root):
    real = Path(root.path)
    (real / "afile").write_bytes(b"x")
    (real / "adir").mkdir()
    with pytest.raises(WrongKindError) as e:
        root.child("afile")
    assert "afile" in str(e.value)
    with pytest.raises(WrongKindError) as e:
        root.open_read("adir")
    assert "adir" in str(e.value)
    with pytest.raises(WrongKindError):
        root.child("afile", create=True)


@POSIX_ONLY
def test_wrong_kind_fifo_and_symlink_message_names_owner(root):
    real = Path(root.path)
    os.mkfifo(real / "pipe")
    with pytest.raises(WrongKindError) as e:
        root.open_read("pipe")
    assert "pipe" in str(e.value)
    assert "owner" in str(e.value)


def test_open_new_is_exclusive(root):
    with root.open_new("m", 0o444) as f:
        f.write(b"abc")
    with pytest.raises(FileExistsError):
        root.open_new("m", 0o444)
    with root.open_read("m") as f:
        assert f.read() == b"abc"


@POSIX_ONLY
def test_modes_ignore_permissive_umask(root, permissive_umask):
    real = Path(root.path)
    sub = root.child("c", create=True, mode=0o750)
    with root.open_new("m", 0o440) as f:
        f.write(b"x")
    assert _mode(real / "c") == 0o750
    assert _mode(real / "m") == 0o440
    root.set_mode(0o500, "c")
    assert _mode(real / "c") == 0o500
    sub.set_mode(0o700)
    assert _mode(real / "c") == 0o700
    root.set_mode(0o600, "m")
    assert _mode(real / "m") == 0o600


@POSIX_ONLY
def test_modes_ignore_restrictive_umask(root):
    old = os.umask(0o277)
    try:
        root.child("c", create=True, mode=0o755)
        with root.open_new("m", 0o444) as f:
            f.write(b"x")
    finally:
        os.umask(old)
    assert _mode(Path(root.path) / "c") == 0o755
    assert _mode(Path(root.path) / "m") == 0o444


@POSIX_ONLY
def test_child_create_mode_defaults_to_policy_container_mode(
    tmp_path, permissive_umask, backend
):
    r = tmp_path / "r"
    r.mkdir()
    root = StorageDir.open_root(r, StorePolicy(container_mode=0o2770), trust="private")
    root.child("c", create=True)
    assert _mode(r / "c") == 0o2770


@POSIX_ONLY
def test_existing_child_mode_is_not_changed_by_create(root):
    real = Path(root.path)
    (real / "c").mkdir(mode=0o700)
    root.child("c", create=True, mode=0o777)
    assert _mode(real / "c") == 0o700


def test_rename_link_lstat_scandir(root):
    a = root.child("a", create=True)
    b = root.child("b", create=True)
    with a.open_new("m", 0o444) as f:
        f.write(b"hello")
    a.link("m", "m2")
    assert a.lstat("m").st_ino == a.lstat("m2").st_ino
    a.rename("m", b, "n")
    assert sorted(e.name for e in a.scandir()) == ["m2"]
    assert sorted(e.name for e in b.scandir()) == ["n"]
    with b.open_read("n") as f:
        assert f.read() == b"hello"
    with pytest.raises(FileNotFoundError):
        a.lstat("m")
    assert stat.S_ISREG(b.lstat("n").st_mode)


def test_rename_does_not_replace_a_nonempty_directory(root):
    a = root.child("a", create=True)
    b = root.child("b", create=True)
    a.child("d", create=True)
    full = b.child("d", create=True)
    with full.open_new("m", 0o444):
        pass
    with pytest.raises(OSError):
        a.rename("d", b, "d")


def test_policy_rejects_group_write_on_entries():
    for kw in (
        {"file_mode": 0o464},
        {"file_mode": 0o446},
        {"entry_mode": 0o775},
        {"entry_mode": 0o757},
        {"sealed_mode": 0o575},
        {"sealed_mode": 0o552},
    ):
        with pytest.raises(ValueError, match="write"):
            StorePolicy(**kw)
    # an entry directory must keep owner write and search so that it can be renamed
    for kw in ({"entry_mode": 0o555}, {"entry_mode": 0o655}, {"entry_mode": 0o455}):
        with pytest.raises(ValueError, match="owner"):
            StorePolicy(**kw)
    with pytest.raises(ValueError):
        StorePolicy(staging_mode=0o500)
    with pytest.raises(ValueError):
        StorePolicy(durability="sometimes")
    with pytest.raises(ValueError):
        StorePolicy(verify_files="md5")
    with pytest.raises(ValueError):
        StorePolicy(file_mode=0o10444)


def test_policy_accepts_the_group_writable_example():
    p = StorePolicy(
        file_mode=0o440,
        entry_mode=0o2750,
        sealed_mode=0o2550,
        container_mode=0o2770,
        side_mode=0o2770,
    )
    assert p.entry_mode == 0o2750


def test_ages_are_validated():
    assert StoreAges(staging=60, work=120, quarantine=30).work == 120
    for bad in (
        {"staging": 0},
        {"work": -1},
        {"quarantine": float("inf")},
        {"staging": float("nan")},
    ):
        with pytest.raises(ValueError):
            StoreAges(**bad)


def test_size_verification_needs_a_private_root_and_member_durability(
    tmp_path, backend
):
    sized = StorePolicy(verify_files="size")
    with pytest.raises(ValueError, match="private"):
        StorageDir.open_root(tmp_path, sized, trust="trusted")
    with pytest.raises(ValueError, match="member"):
        StorageDir.open_root(
            tmp_path,
            StorePolicy(verify_files="size", durability="entry"),
            trust="private",
        )
    assert StorageDir.open_root(tmp_path, sized, trust="private") is not None


def test_windows_shared_platform_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(storagedir, "DESCRIPTORS_SUPPORTED", False)
    monkeypatch.setattr(storagedir, "IS_WINDOWS", True)
    with pytest.raises(PlatformUnsupported) as e:
        StorageDir.open_root(tmp_path, StorePolicy(), trust="shared")
    assert isinstance(e.value, StorageUnsupported)
    msg = str(e.value)
    assert "platform" in msg and str(tmp_path) in msg
    # the default trust is shared
    with pytest.raises(PlatformUnsupported):
        StorageDir.open_root(tmp_path, StorePolicy())
    # private and trusted open
    assert StorageDir.open_root(tmp_path, StorePolicy(), trust="private") is not None
    assert StorageDir.open_root(tmp_path, StorePolicy(), trust="trusted") is not None


def test_shared_opens_where_descriptors_exist(tmp_path):
    if not storagedir.DESCRIPTORS_SUPPORTED:
        pytest.skip("descriptor-relative calls needed")
    assert StorageDir.open_root(tmp_path, StorePolicy()) is not None


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows reparse points")
def test_windows_reparse_point_refused(tmp_path):
    root_path = tmp_path / "root"
    (root_path / "real").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    root = StorageDir.open_root(root_path, StorePolicy(), trust="private")
    # a junction below the root, made without privileges
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root_path / "junction"), str(outside)],
        capture_output=True,
    )
    assert done.returncode == 0, done.stderr
    with pytest.raises(WrongKindError):
        root.child("junction")
    # swapped in below a handle that was opened before the swap
    held = root.child("real")
    (root_path / "real").rmdir()
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root_path / "real"), str(outside)],
        capture_output=True,
    )
    assert done.returncode == 0, done.stderr
    with pytest.raises(WrongKindError):
        held.open_new("m", 0o444)
    assert not (outside / "m").exists()


def test_copy_hashed_digest_and_size(root, tmp_path):
    data = os.urandom(3 * 1024 * 1024 + 17)
    src = tmp_path / "src.bin"
    src.write_bytes(data)
    dst = root.child("d", create=True)
    digest, size = copy_hashed(src, dst, "out.bin", 0o444)
    assert digest == hashlib.sha256(data).hexdigest()
    assert size == len(data)
    with dst.open_read("out.bin") as f:
        assert f.read() == data
    if not IS_WINDOWS:
        assert _mode(Path(dst.path) / "out.bin") == 0o444
    with pytest.raises(FileExistsError):
        copy_hashed(src, dst, "out.bin", 0o444)


def test_copy_hashed_empty_file(root, tmp_path):
    src = tmp_path / "empty"
    src.write_bytes(b"")
    digest, size = copy_hashed(src, root, "e", 0o444)
    assert (digest, size) == (hashlib.sha256(b"").hexdigest(), 0)


def test_copy_hashed_without_reflink(root, tmp_path, monkeypatch):
    monkeypatch.setattr(storagedir, "_reflink", lambda *a, **k: False)
    data = b"plain copy" * 1000
    src = tmp_path / "s"
    src.write_bytes(data)
    digest, size = copy_hashed(src, root, "o", 0o444)
    assert (digest, size) == (hashlib.sha256(data).hexdigest(), len(data))


def test_copy_hashed_refuses_a_directory_source(root, tmp_path):
    (tmp_path / "dir").mkdir()
    with pytest.raises((WrongKindError, IsADirectoryError, PermissionError)):
        copy_hashed(tmp_path / "dir", root, "o", 0o444)


@POSIX_ONLY
@pytest.mark.parametrize("reflink", [True, False])
def test_copy_of_hardlinked_input_leaves_original_mode(
    root, tmp_path, monkeypatch, reflink
):
    if not reflink:
        monkeypatch.setattr(storagedir, "_reflink", lambda *a, **k: False)
    dataset = tmp_path / "dataset.txt"
    dataset.write_bytes(b"precious")
    os.chmod(dataset, 0o644)
    # what an input copy in a work directory is: a hard link to the user's file
    linked = tmp_path / "work-input.txt"
    os.link(dataset, linked)
    digest, size = copy_hashed(linked, root, "member", 0o444)
    assert _mode(dataset) == 0o644
    assert _mode(linked) == 0o644
    assert _mode(Path(root.path) / "member") == 0o444
    assert os.lstat(Path(root.path) / "member").st_ino != os.lstat(dataset).st_ino
    assert os.lstat(dataset).st_nlink == 2
    assert dataset.read_bytes() == b"precious"
    assert (digest, size) == (hashlib.sha256(b"precious").hexdigest(), 8)
    # editing the user's file afterwards does not reach the stored copy
    os.chmod(dataset, 0o644)
    dataset.write_bytes(b"changed!")
    with root.open_read("member") as f:
        assert f.read() == b"precious"
