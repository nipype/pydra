"""Tests for `pydra.utils.storagedir` (RFC 0003 section 9)."""

import errno
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
TRUST = "trusted" if IS_WINDOWS else "private"
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
    # on Windows a private root must sit under LOCALAPPDATA, which a temp directory may not
    return StorageDir.open_root(d, policy, trust=TRUST)


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
    os.chmod(r, 0o755)
    root = StorageDir.open_root(r, StorePolicy(container_mode=0o2770), trust=TRUST)
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


def test_size_verification_needs_member_durability():
    with pytest.raises(ValueError, match="member"):
        StorePolicy(verify_files="size", durability="none")
    with pytest.raises(ValueError, match="member"):
        StorePolicy(verify_files="size", durability="entry")
    assert StorePolicy(verify_files="size").durability == "member"


@POSIX_ONLY
def test_size_verification_needs_a_private_root(tmp_path, backend):
    sized = StorePolicy(verify_files="size")
    with pytest.raises(ValueError, match="private"):
        StorageDir.open_root(tmp_path, sized, trust="trusted")
    assert StorageDir.open_root(tmp_path, sized, trust="private") is not None


@POSIX_ONLY
def test_private_root_is_checked_on_open(tmp_path, backend):
    mine = tmp_path / "mine"
    mine.mkdir()
    os.chmod(mine, 0o755)
    assert StorageDir.open_root(mine, StorePolicy(), trust="private") is not None
    for bad in (0o775, 0o757, 0o777):
        os.chmod(mine, bad)
        with pytest.raises(StoreUnavailable, match="private"):
            StorageDir.open_root(mine, StorePolicy(), trust="private")
        # the declaration alone does not unlock size verification
        with pytest.raises(StoreUnavailable):
            StorageDir.open_root(
                mine, StorePolicy(verify_files="size"), trust="private"
            )
        # shared and trusted do not check
        assert StorageDir.open_root(mine, StorePolicy(), trust="trusted") is not None


@POSIX_ONLY
def test_private_root_owned_by_someone_else_refused(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(tmp_path).st_uid + 1)
    with pytest.raises(StoreUnavailable, match="owner"):
        StorageDir.open_root(tmp_path, StorePolicy(), trust="private")


def test_windows_private_root_only_under_local_app_data(tmp_path, monkeypatch):
    monkeypatch.setattr(storagedir, "DESCRIPTORS_SUPPORTED", False)
    monkeypatch.setattr(storagedir, "IS_WINDOWS", True)
    home = tmp_path / "local"
    (home / "pydra").mkdir(parents=True)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(home))
    assert (
        StorageDir.open_root(home / "pydra", StorePolicy(), trust="private") is not None
    )
    with pytest.raises(StoreUnavailable, match="trusted"):
        StorageDir.open_root(other, StorePolicy(), trust="private")


def test_windows_shared_platform_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(storagedir, "DESCRIPTORS_SUPPORTED", False)
    monkeypatch.setattr(storagedir, "IS_WINDOWS", True)
    with pytest.raises(PlatformUnsupported) as e:
        StorageDir.open_root(tmp_path, StorePolicy(), trust="shared")
    assert isinstance(e.value, StorageUnsupported)
    assert "platform" in str(e.value)
    assert e.value.filename == str(tmp_path)
    # the default trust is shared
    with pytest.raises(PlatformUnsupported):
        StorageDir.open_root(tmp_path, StorePolicy())
    # private and trusted open
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
    with pytest.raises(
        (WrongKindError, PermissionError) if IS_WINDOWS else WrongKindError
    ):
        copy_hashed(tmp_path / "dir", root, "o", 0o444)
    assert not (Path(root.path) / "o").exists()


@POSIX_ONLY
def test_copy_hashed_refuses_a_fifo_without_blocking(root, tmp_path):
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(WrongKindError):
        copy_hashed(tmp_path / "pipe", root, "o", 0o444)
    assert not (Path(root.path) / "o").exists()


@POSIX_ONLY
@pytest.mark.skipif(sys.platform != "darwin", reason="BSD file flags")
def test_copy_of_an_immutable_source_leaves_a_removable_member(root, tmp_path):
    src = tmp_path / "locked"
    src.write_bytes(b"data")
    os.chflags(src, stat.UF_IMMUTABLE)
    try:
        digest, size = copy_hashed(src, root, "m", 0o444)
        assert (digest, size) == (hashlib.sha256(b"data").hexdigest(), 4)
        assert os.lstat(Path(root.path) / "m").st_flags & stat.UF_IMMUTABLE == 0
        root.unlink("m")
    finally:
        os.chflags(src, 0)
    assert not (Path(root.path) / "m").exists()


@POSIX_ONLY
def test_reflink_clones_the_checked_descriptor_not_the_path(root, tmp_path):
    src = tmp_path / "src"
    src.write_bytes(b"the checked file")
    other = tmp_path / "other"
    other.write_bytes(b"something else")
    with open(src, "rb") as f:
        src.rename(tmp_path / "moved")
        src.symlink_to(other)
        if not storagedir._reflink(f, root, "c"):
            pytest.skip("no reflink on this filesystem")
    assert (Path(root.path) / "c").read_bytes() == b"the checked file"


@POSIX_ONLY
def test_ficlone_success_path(root, tmp_path, monkeypatch, backend):
    if backend != "descriptor":
        pytest.skip("the descriptor backend clones")
    fcntl = pytest.importorskip("fcntl")
    seen = []

    def fake_ioctl(fd, request, arg):
        seen.append(request)
        os.write(fd, os.pread(arg, 1 << 20, 0))
        return 0

    monkeypatch.setattr(storagedir, "_DARWIN", False)
    monkeypatch.setattr(fcntl, "ioctl", fake_ioctl)
    data = b"cloned bytes" * 100
    src = tmp_path / "s"
    src.write_bytes(data)
    digest, size = copy_hashed(src, root, "o", 0o440)
    assert seen == [getattr(fcntl, "FICLONE", 0x40049409)]
    assert (digest, size) == (hashlib.sha256(data).hexdigest(), len(data))
    assert _mode(Path(root.path) / "o") == 0o440


@POSIX_ONLY
def test_ficlone_refused_falls_back_and_leaves_no_partial(
    root, tmp_path, monkeypatch, backend
):
    if backend != "descriptor":
        pytest.skip("the descriptor backend clones")
    fcntl = pytest.importorskip("fcntl")

    def refuse(fd, request, arg):
        raise OSError(errno.EOPNOTSUPP, "no reflink")

    monkeypatch.setattr(storagedir, "_DARWIN", False)
    monkeypatch.setattr(fcntl, "ioctl", refuse)
    src = tmp_path / "s"
    src.write_bytes(b"abc")
    assert copy_hashed(src, root, "o", 0o444)[1] == 3
    assert (Path(root.path) / "o").read_bytes() == b"abc"


@POSIX_ONLY
def test_fsync_failure_after_a_clone_propagates(root, tmp_path, monkeypatch, backend):
    if backend != "descriptor":
        pytest.skip("the descriptor backend clones")
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(storagedir, "_DARWIN", False)
    monkeypatch.setattr(
        fcntl, "ioctl", lambda fd, r, a: os.write(fd, os.pread(a, 100, 0)) and 0
    )

    def broken(fd):
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(os, "fsync", broken)
    src = tmp_path / "s"
    src.write_bytes(b"abc")
    with pytest.raises(OSError) as e:
        copy_hashed(src, root, "o", 0o444)
    assert e.value.errno == errno.EIO


def test_reflink_on_this_filesystem(tmp_path):
    """Passes only where the filesystem has reflinks (btrfs, xfs reflink=1, APFS); run with
    --basetemp on one to record the proof. Elsewhere it skips with the reason."""
    root = StorageDir.open_root(tmp_path, StorePolicy(), trust=TRUST)
    src = tmp_path / "s"
    src.write_bytes(os.urandom(1 << 20))
    with open(src, "rb") as f:
        if not storagedir._reflink(f, root, "c"):
            pytest.skip("the filesystem refused a reflink")
    assert (tmp_path / "c").read_bytes() == src.read_bytes()


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


def test_use_after_close_raises(root):
    held = root.child("d", create=True)
    held.close()
    held.close()
    real = Path(root.path)
    for call in (
        lambda: held.open_new("m", 0o444),
        lambda: held.open_read("m"),
        lambda: held.child("x"),
        lambda: held.lstat("m"),
        lambda: held.fstat(),
        lambda: held.unlink("m"),
        lambda: held.set_mode(0o700),
        lambda: held.scandir(),
    ):
        with pytest.raises(ValueError, match="closed"):
            call()
    assert not (real / "d" / "m").exists()


@POSIX_ONLY
def test_closed_handle_never_follows_a_swapped_symlink(root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("s")
    held = root.child("d", create=True)
    held.close()
    (Path(root.path) / "d").rename(Path(root.path) / "d-aside")
    (Path(root.path) / "d").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        held.open_new("m", 0o444)
    with pytest.raises(ValueError):
        held.open_read("secret")
    assert not (outside / "m").exists()


@POSIX_ONLY
def test_root_that_is_a_symlink_opens_and_works(tmp_path, backend):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    root = StorageDir.open_root(link, StorePolicy(), trust=TRUST)
    sub = root.child("t", create=True)
    with sub.open_new("m", 0o444) as f:
        f.write(b"x")
    with sub.open_read("m") as f:
        assert f.read() == b"x"
    assert (real / "t" / "m").exists()
    assert stat.S_ISDIR(root.fstat().st_mode)


def test_path_below_a_root_that_is_swapped_is_still_refused(root, tmp_path):
    (Path(root.path) / "d").mkdir()
    held = root.child("d")
    if IS_WINDOWS:
        pytest.skip("symlink needs privilege; the junction test covers Windows")
    (Path(root.path) / "d").rename(Path(root.path) / "d-aside")
    (Path(root.path) / "d").symlink_to(tmp_path, target_is_directory=True)
    if storagedir.DESCRIPTORS_SUPPORTED:
        with held.open_new("m", 0o444):
            pass
        assert (Path(root.path) / "d-aside" / "m").exists()
    else:
        with pytest.raises(WrongKindError):
            held.open_new("m", 0o444)


@POSIX_ONLY
def test_scandir_listings_are_independent(root):
    for i in range(5):
        with root.open_new(f"f{i}", 0o444):
            pass
    first = root.scandir()
    next(first)
    assert sorted(e.name for e in root.scandir()) == [f"f{i}" for i in range(5)]
    nested = []
    for e in root.scandir():
        nested.append(sorted(x.name for x in root.scandir()))
    assert nested == [[f"f{i}" for i in range(5)]] * 5
    assert len(list(first)) == 4


def test_names_that_escape_on_windows_are_refused_in_path_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(storagedir, "DESCRIPTORS_SUPPORTED", False)
    root = StorageDir.open_root(tmp_path, StorePolicy(), trust="trusted")
    for name in ("D:x", "f:stream", "trailing.", "trailing "):
        with pytest.raises(ValueError):
            root.child(name, create=True)
        with pytest.raises(ValueError):
            root.open_new(name, 0o444)


def test_unlink(root):
    with root.open_new("m", 0o444) as f:
        f.write(b"x")
    root.unlink("m")
    with pytest.raises(FileNotFoundError):
        root.lstat("m")
    with pytest.raises(FileNotFoundError):
        root.unlink("m")
    with pytest.raises(ValueError):
        root.unlink("a/b")


@POSIX_ONLY
def test_unlink_removes_a_symlink_not_its_target(root, tmp_path):
    target = tmp_path / "t"
    target.write_text("keep")
    (Path(root.path) / "l").symlink_to(target)
    root.unlink("l")
    assert target.read_text() == "keep"
    assert not os.path.lexists(Path(root.path) / "l")


def test_wrong_kind_directory_where_a_member_is_expected_never_permission_error(root):
    (Path(root.path) / "adir").mkdir()
    with pytest.raises(WrongKindError):
        root.open_read("adir")


def test_a_file_open_for_reading_can_be_removed(root):
    """A reader shares deletion: the file it holds can be unlinked under it, on every platform."""
    with root.open_new("m", 0o600) as f:
        f.write(b"x")
    with root.open_read("m") as f:
        root.unlink("m")
        assert f.read() == b"x"
    assert "m" not in root.names()


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows sharing modes")
def test_open_read_while_another_handle_holds_delete_access(root):
    """A file just renamed into place is still open, with DELETE access, by the call that renamed
    it; a reader that does not share deletion meets a sharing violation (EACCES) then.
    """
    import _winapi

    with root.open_new("m", 0o600) as f:
        f.write(b"x")
    delete, share_all = 0x00010000, 0x7
    handle = _winapi.CreateFile(
        os.path.join(root.path, "m"),
        delete,
        share_all,
        _winapi.NULL,
        _winapi.OPEN_EXISTING,
        0,
        _winapi.NULL,
    )
    try:
        with root.open_read("m") as f:
            assert f.read() == b"x"
    finally:
        _winapi.CloseHandle(handle)


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows read-only attribute")
def test_windows_read_only_attribute(root):
    real = Path(root.path)
    with root.open_new("m", 0o444) as f:
        f.write(b"x")
    assert not os.access(real / "m", os.W_OK)
    root.set_mode(0o644, "m")
    assert os.access(real / "m", os.W_OK)
    root.child("d", create=True)
    root.set_mode(0o555, "d")
    assert os.access(real / "d", os.W_OK)
    root.unlink("m")
    assert not (real / "m").exists()


@POSIX_ONLY
@pytest.mark.skipif(sys.platform != "darwin", reason="HFS/APFS transparent compression")
def test_copy_of_a_compressed_source_keeps_its_bytes(root, tmp_path):
    plain = tmp_path / "plain"
    plain.write_bytes(b"compressible " * 20000)
    comp = tmp_path / "comp"
    done = subprocess.run(
        ["ditto", "--hfsCompression", str(plain), str(comp)], capture_output=True
    )
    if done.returncode != 0 or not os.lstat(comp).st_flags & stat.UF_COMPRESSED:
        pytest.skip("this volume does not compress")
    digest, size = copy_hashed(comp, root, "m", 0o444)
    assert size == len(plain.read_bytes())
    assert digest == hashlib.sha256(plain.read_bytes()).hexdigest()
    assert (Path(root.path) / "m").read_bytes() == plain.read_bytes()


@POSIX_ONLY
@pytest.mark.skipif(sys.platform != "darwin", reason="BSD file flags")
def test_only_immutable_and_append_flags_are_cleared(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"x")
    os.chflags(f, stat.UF_IMMUTABLE | stat.UF_HIDDEN)
    try:
        with open(f, "rb") as h:
            storagedir._clear_flags(h.fileno())
        assert os.lstat(f).st_flags == stat.UF_HIDDEN
    finally:
        os.chflags(f, 0)


def test_a_source_that_changes_size_during_the_copy_is_refused(
    root, tmp_path, monkeypatch
):
    src = tmp_path / "s"
    src.write_bytes(b"abcdef")
    real = storagedir._open_source

    class Growing:
        def __init__(self, f):
            self.f = f
            self.grown = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

        def fileno(self):
            return self.f.fileno()

        def read(self, n):
            if not self.grown:
                self.grown = True
                with open(src, "ab") as w:
                    w.write(b"more")
            return self.f.read(n)

    monkeypatch.setattr(storagedir, "_open_source", lambda path: Growing(real(path)))
    monkeypatch.setattr(storagedir, "_reflink", lambda *a, **k: False)
    with pytest.raises(StoreUnavailable, match="size"):
        copy_hashed(src, root, "o", 0o444)
    assert not os.path.lexists(Path(root.path) / "o")


def test_a_failed_copy_leaves_no_partial_member(root, tmp_path, monkeypatch):
    src = tmp_path / "s"
    src.write_bytes(os.urandom(3 * 1024 * 1024))
    monkeypatch.setattr(storagedir, "_reflink", lambda *a, **k: False)
    calls = []

    def broken(fd):
        calls.append(fd)
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(OSError):
        copy_hashed(src, root, "o", 0o444)
    assert not os.path.lexists(Path(root.path) / "o")
    monkeypatch.undo()
    assert copy_hashed(src, root, "o", 0o444)[1] == 3 * 1024 * 1024


def test_a_failed_copy_does_not_remove_an_existing_member(root, tmp_path, monkeypatch):
    src = tmp_path / "s"
    src.write_bytes(b"new")
    monkeypatch.setattr(storagedir, "_reflink", lambda *a, **k: False)
    with root.open_new("o", 0o444) as f:
        f.write(b"old")
    with pytest.raises(FileExistsError):
        copy_hashed(src, root, "o", 0o444)
    assert (Path(root.path) / "o").read_bytes() == b"old"


def test_scandir_entries_stay_valid_after_the_listing_ends(root):
    sub = root.child("a", create=True)
    with root.open_new("x", 0o444) as f:
        f.write(b"x" * 22)
    with sub.open_new("x", 0o444) as f:
        f.write(b"y" * 5)
    entries = {e.name: e for e in root.scandir()}
    again = root.child("a")  # may take the listing's old descriptor number
    assert entries["x"].stat().st_size == 22
    assert (
        entries["x"].is_file()
        and not entries["x"].is_dir()
        and not entries["x"].is_symlink()
    )
    assert entries["a"].is_dir()
    assert again.lstat("x").st_size == 5


@POSIX_ONLY
def test_scandir_entries_do_not_follow_symlinks(root, tmp_path):
    (Path(root.path) / "l").symlink_to(tmp_path, target_is_directory=True)
    (entry,) = list(root.scandir())
    assert entry.is_symlink() and not entry.is_dir()
    assert stat.S_ISLNK(entry.stat().st_mode)


@POSIX_ONLY
def test_scandir_never_started_leaks_no_descriptor(root):
    if not os.path.isdir("/dev/fd"):
        pytest.skip("no /dev/fd")
    before = len(os.listdir("/dev/fd"))
    for _ in range(50):
        root.scandir()
    assert len(os.listdir("/dev/fd")) <= before + 1
    it = root.scandir()
    list(it)
    it.close()


@POSIX_ONLY
def test_scandir_supports_closing_early(root):
    for i in range(3):
        with root.open_new(f"f{i}", 0o444):
            pass
    it = root.scandir()
    next(it)
    it.close()
    assert len(list(root.scandir())) == 3


def test_relative_root_survives_chdir(tmp_path, monkeypatch):
    monkeypatch.setattr(storagedir, "DESCRIPTORS_SUPPORTED", False)
    (tmp_path / "store").mkdir()
    (tmp_path / "job").mkdir()
    monkeypatch.chdir(tmp_path)
    root = StorageDir.open_root("store", StorePolicy(), trust="trusted")
    sub = root.child("c", create=True)
    os.chdir(tmp_path / "job")
    with sub.open_new("m", 0o444):
        pass
    root.child("d", create=True)
    assert (tmp_path / "store" / "c" / "m").exists()
    assert (tmp_path / "store" / "d").is_dir()
    assert not (tmp_path / "job" / "c").exists()


def test_windows_private_root_on_another_drive_is_refused_not_a_valueerror(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storagedir, "DESCRIPTORS_SUPPORTED", False)
    monkeypatch.setattr(storagedir, "IS_WINDOWS", True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    def other_drive(paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(os.path, "commonpath", other_drive)
    with pytest.raises(StoreUnavailable, match="trusted"):
        StorageDir.open_root(tmp_path, StorePolicy(), trust="private")


@pytest.mark.parametrize(
    "name", ["NUL", "nul", "CON", "con.txt", "PRN", "AUX.x.y", "COM1", "lpt9.log"]
)
def test_windows_device_names_are_refused_in_path_mode(tmp_path, monkeypatch, name):
    monkeypatch.setattr(storagedir, "DESCRIPTORS_SUPPORTED", False)
    root = StorageDir.open_root(tmp_path, StorePolicy(), trust="trusted")
    with pytest.raises(ValueError):
        root.open_new(name, 0o444)
    for ok in ("null", "console", "com10", "COM0", "a.con", "xcon", "con-1"):
        root.open_new(ok, 0o444).close()


def test_rmdir(root):
    root.child("d", create=True)
    with root.child("d").open_new("m", 0o444):
        pass
    with pytest.raises(OSError):
        root.rmdir("d")
    root.child("d").unlink("m")
    root.rmdir("d")
    with pytest.raises(FileNotFoundError):
        root.lstat("d")
    with pytest.raises(ValueError):
        root.rmdir("a/b")


@POSIX_ONLY
def test_rmdir_does_not_follow_a_symlink(root, tmp_path):
    (Path(root.path) / "l").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(WrongKindError):
        root.rmdir("l")
    assert tmp_path.is_dir()
    assert os.path.islink(Path(root.path) / "l")


def test_policy_requires_owner_read_and_search():
    for kw in (
        {"file_mode": 0o040},
        {"file_mode": 0o200},
        {"entry_mode": 0o300},
        {"entry_mode": 0o500},
        {"sealed_mode": 0o100},
        {"sealed_mode": 0o300},
        {"container_mode": 0o055},
        {"container_mode": 0o500},
        {"side_mode": 0o650},
        {"side_mode": 0o070},
    ):
        with pytest.raises(ValueError, match="owner"):
            StorePolicy(**kw)
    StorePolicy(file_mode=0o400, entry_mode=0o700, sealed_mode=0o500)


def test_scandir_after_close_raises_on_either_backend(root):
    held = root.child("d", create=True)
    for i in range(3):
        with held.open_new(f"f{i}", 0o444):
            pass
    with root.child("d") as d:
        unstarted = d.scandir()
        started = d.scandir()
        next(started)
    with pytest.raises(ValueError, match="closed"):
        next(unstarted)
    with pytest.raises(ValueError, match="closed"):
        next(started)


def test_names_lists_without_a_stat(root, monkeypatch):
    """`names` lists a directory's names with no per-entry stat, so an entry that vanishes or
    answers ESTALE between the listing and a stat can never fail or empty the listing; a symlink
    is listed by name, never followed."""
    d = root.child("d", create=True)
    for name in ("x", "y"):
        with d.open_new(name, 0o444):
            pass
    d.child("sub", create=True).close()
    if hasattr(os, "symlink") and not storagedir.IS_WINDOWS:
        os.symlink("x", os.path.join(d.path, "link"))
    expected = sorted(n for n in os.listdir(d.path))

    real_stat, real_lstat = os.stat, os.lstat

    def stale_entries(real):
        def call(path, *args, **kwargs):
            if os.path.basename(os.fspath(path)) in expected:
                raise OSError(errno.ESTALE, "stale file handle")
            return real(path, *args, **kwargs)

        return call

    monkeypatch.setattr(os, "stat", stale_entries(real_stat))
    monkeypatch.setattr(os, "lstat", stale_entries(real_lstat))
    assert sorted(d.names()) == expected
    d.close()
    with pytest.raises(ValueError):
        d.names()
