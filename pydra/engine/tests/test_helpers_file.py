import os
import time
import warnings
import pytest
from pathlib import Path

from ..helpers_file import (
    split_filename,
    copyfile,
    copyfiles,
    on_cifs,
    get_related_files,
    ensure_list,
    _cifs_table,
    _parse_mount_table,
)


def _ignore_atime(stat):
    return stat[:7] + stat[8:]


@pytest.mark.parametrize(
    "filename, split",
    [
        ("foo.nii", ("", "foo", ".nii")),
        ("foo.nii.gz", ("", "foo", ".nii.gz")),
        ("foo.niml.dset", ("", "foo", ".niml.dset")),
        ("/usr/local/foo.nii.gz", ("/usr/local", "foo", ".nii.gz")),
        ("../usr/local/foo.nii", ("../usr/local", "foo", ".nii")),
        ("/usr/local/foo.a.b.c.d", ("/usr/local", "foo.a.b.c", ".d")),
        ("/usr/local/", ("/usr/local", "", "")),
    ],
)
def test_split_filename(filename, split):
    res = split_filename(filename)
    assert res == split


@pytest.fixture()
def _temp_analyze_files(tmpdir):
    """Generate temporary analyze file pair."""
    orig_img = tmpdir.join("orig.img")
    orig_hdr = tmpdir.join("orig.hdr")
    orig_img.open("w+").close()
    orig_hdr.open("w+").close()
    return Path(orig_img), Path(orig_hdr)


@pytest.fixture()
def _temp_analyze_files_prime(tmpdir):
    """Generate temporary analyze file pair."""
    orig_img = tmpdir.join("orig_prime.img")
    orig_hdr = tmpdir.join("orig_prime.hdr")
    orig_img.open("w+").close()
    orig_hdr.open("w+").close()
    return Path(orig_img.strpath), Path(orig_hdr.strpath)


def test_copyfile(_temp_analyze_files):
    orig_img, orig_hdr = _temp_analyze_files
    pth, fname = os.path.split(orig_img)
    new_img = os.path.join(pth, "newfile.img")
    new_hdr = os.path.join(pth, "newfile.hdr")
    copyfile(orig_img, new_img)
    assert os.path.exists(new_img)
    assert os.path.exists(new_hdr)


def test_copyfile_true(_temp_analyze_files):
    orig_img, orig_hdr = _temp_analyze_files
    pth, fname = os.path.split(orig_img)
    new_img = os.path.join(pth, "newfile.img")
    new_hdr = os.path.join(pth, "newfile.hdr")
    # Test with copy=True
    copyfile(orig_img, new_img, copy=True)
    assert os.path.exists(new_img)
    assert os.path.exists(new_hdr)


def test_copyfiles(_temp_analyze_files, _temp_analyze_files_prime):
    orig_img1, orig_hdr1 = _temp_analyze_files
    orig_img2, orig_hdr2 = _temp_analyze_files_prime
    pth, fname = os.path.split(orig_img1)
    new_img1 = os.path.join(pth, "newfile.img")
    new_hdr1 = os.path.join(pth, "newfile.hdr")
    pth, fname = os.path.split(orig_img2)
    new_img2 = os.path.join(pth, "secondfile.img")
    new_hdr2 = os.path.join(pth, "secondfile.hdr")
    copyfiles([orig_img1, orig_img2], [new_img1, new_img2])
    assert os.path.exists(new_img1)
    assert os.path.exists(new_hdr1)
    assert os.path.exists(new_img2)
    assert os.path.exists(new_hdr2)


def test_linkchain(_temp_analyze_files):
    if os.name != "posix":
        return
    orig_img, orig_hdr = _temp_analyze_files
    pth, fname = os.path.split(orig_img)
    new_img1 = os.path.join(pth, "newfile1.img")
    new_hdr1 = os.path.join(pth, "newfile1.hdr")
    new_img2 = os.path.join(pth, "newfile2.img")
    new_hdr2 = os.path.join(pth, "newfile2.hdr")
    new_img3 = os.path.join(pth, "newfile3.img")
    new_hdr3 = os.path.join(pth, "newfile3.hdr")
    copyfile(orig_img, new_img1, use_hardlink=False)
    assert os.path.islink(new_img1)
    assert os.path.islink(new_hdr1)
    copyfile(new_img1, new_img2, copy=True, use_hardlink=False)
    assert not os.path.islink(new_img2)
    assert not os.path.islink(new_hdr2)
    assert not os.path.samefile(orig_img, new_img2)
    assert not os.path.samefile(orig_hdr, new_hdr2)
    copyfile(new_img1, new_img3, copy=True, use_hardlink=True)
    assert not os.path.islink(new_img3)
    assert not os.path.islink(new_hdr3)
    assert os.path.samefile(orig_img, new_img3)
    assert os.path.samefile(orig_hdr, new_hdr3)


def test_recopy(_temp_analyze_files):
    # Re-copying with the same parameters on an unchanged file should be
    # idempotent
    #
    # Test for copying from regular files and symlinks
    orig_img, orig_hdr = _temp_analyze_files
    pth, fname = os.path.split(orig_img)
    img_link = os.path.join(pth, "imglink.img")
    new_img = os.path.join(pth, "newfile.img")
    new_hdr = os.path.join(pth, "newfile.hdr")
    copyfile(orig_img, img_link)
    for copy in (True, False):
        for use_hardlink in (True, False):
            kwargs = {"copy": copy, "use_hardlink": use_hardlink}

            copyfile(orig_img, new_img, **kwargs)
            img_stat = _ignore_atime(os.stat(new_img))
            hdr_stat = _ignore_atime(os.stat(new_hdr))
            copyfile(orig_img, new_img, **kwargs)
            err_msg = "Regular - OS: {}; Copy: {}; Hardlink: {}".format(
                os.name, copy, use_hardlink
            )
            assert img_stat == _ignore_atime(os.stat(new_img)), err_msg
            assert hdr_stat == _ignore_atime(os.stat(new_hdr)), err_msg
            os.unlink(new_img)
            os.unlink(new_hdr)

            copyfile(img_link, new_img, **kwargs)
            img_stat = _ignore_atime(os.stat(new_img))
            hdr_stat = _ignore_atime(os.stat(new_hdr))
            copyfile(img_link, new_img, **kwargs)
            err_msg = "Symlink - OS: {}; Copy: {}; Hardlink: {}".format(
                os.name, copy, use_hardlink
            )
            assert img_stat == _ignore_atime(os.stat(new_img)), err_msg
            assert hdr_stat == _ignore_atime(os.stat(new_hdr)), err_msg
            os.unlink(new_img)
            os.unlink(new_hdr)


def test_get_related_files(_temp_analyze_files):
    orig_img, orig_hdr = _temp_analyze_files

    related_files = get_related_files(orig_img)
    assert orig_img in related_files
    assert orig_hdr in related_files

    related_files = get_related_files(orig_hdr)
    assert orig_img in related_files
    assert orig_hdr in related_files


def test_get_related_files_noninclusive(_temp_analyze_files):
    orig_img, orig_hdr = _temp_analyze_files

    related_files = get_related_files(orig_img, include_this_file=False)
    assert orig_img not in related_files
    assert orig_hdr in related_files

    related_files = get_related_files(orig_hdr, include_this_file=False)
    assert orig_img in related_files
    assert orig_hdr not in related_files


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("foo.nii", ["foo.nii"]),
        (["foo.nii"], ["foo.nii"]),
        (("foo", "bar"), ["foo", "bar"]),
        (12.34, None),
    ],
)
def test_ensure_list(filename, expected):
    x = ensure_list(filename)
    assert x == expected


@pytest.mark.parametrize(
    "file, length, expected_files",
    [
        (
            "/path/test.img",
            3,
            [Path("/path/test.hdr"), Path("/path/test.img"), Path("/path/test.mat")],
        ),
        (
            "/path/test.hdr",
            3,
            [Path("/path/test.hdr"), Path("/path/test.img"), Path("/path/test.mat")],
        ),
        ("/path/test.BRIK", 2, [Path("/path/test.BRIK"), Path("/path/test.HEAD")]),
        ("/path/test.HEAD", 2, [Path("/path/test.BRIK"), Path("/path/test.HEAD")]),
        ("/path/foo.nii", 2, [Path("/path/foo.nii"), Path("/path/foo.mat")]),
    ],
)
def test_related_files(file, length, expected_files):
    related_files = get_related_files(file)

    assert len(related_files) == length

    for ef in expected_files:
        assert ef in related_files


MOUNT_OUTPUTS = (
    # Linux, no CIFS
    (
        r"""sysfs on /sys type sysfs (rw,nosuid,nodev,noexec,relatime)
proc on /proc type proc (rw,nosuid,nodev,noexec,relatime)
udev on /dev type devtmpfs (rw,nosuid,relatime,size=8121732k,nr_inodes=2030433,mode=755)
devpts on /dev/pts type devpts (rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=000)
tmpfs on /run type tmpfs (rw,nosuid,noexec,relatime,size=1628440k,mode=755)
/dev/nvme0n1p2 on / type ext4 (rw,relatime,errors=remount-ro,data=ordered)
securityfs on /sys/kernel/security type securityfs (rw,nosuid,nodev,noexec,relatime)
tmpfs on /dev/shm type tmpfs (rw,nosuid,nodev)
tmpfs on /sys/fs/cgroup type tmpfs (ro,nosuid,nodev,noexec,mode=755)
cgroup on /sys/fs/cgroup/systemd type cgroup (rw,nosuid,nodev,noexec,relatime,xattr,release_agent=/lib/systemd/systemd-cgroups-agent,name=systemd)
pstore on /sys/fs/pstore type pstore (rw,nosuid,nodev,noexec,relatime)
efivarfs on /sys/firmware/efi/efivars type efivarfs (rw,nosuid,nodev,noexec,relatime)
cgroup on /sys/fs/cgroup/cpu,cpuacct type cgroup (rw,nosuid,nodev,noexec,relatime,cpu,cpuacct)
cgroup on /sys/fs/cgroup/freezer type cgroup (rw,nosuid,nodev,noexec,relatime,freezer)
cgroup on /sys/fs/cgroup/pids type cgroup (rw,nosuid,nodev,noexec,relatime,pids)
cgroup on /sys/fs/cgroup/cpuset type cgroup (rw,nosuid,nodev,noexec,relatime,cpuset)
systemd-1 on /proc/sys/fs/binfmt_misc type autofs (rw,relatime,fd=26,pgrp=1,timeout=0,minproto=5,maxproto=5,direct)
hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime)
debugfs on /sys/kernel/debug type debugfs (rw,relatime)
mqueue on /dev/mqueue type mqueue (rw,relatime)
fusectl on /sys/fs/fuse/connections type fusectl (rw,relatime)
/dev/nvme0n1p1 on /boot/efi type vfat (rw,relatime,fmask=0077,dmask=0077,codepage=437,iocharset=iso8859-1,shortname=mixed,errors=remount-ro)
/dev/nvme0n1p2 on /var/lib/docker/aufs type ext4 (rw,relatime,errors=remount-ro,data=ordered)
gvfsd-fuse on /run/user/1002/gvfs type fuse.gvfsd-fuse (rw,nosuid,nodev,relatime,user_id=1002,group_id=1002)
""",
        0,
        [],
    ),
    # OS X, no CIFS
    (
        r"""/dev/disk2 on / (hfs, local, journaled)
devfs on /dev (devfs, local, nobrowse)
map -hosts on /net (autofs, nosuid, automounted, nobrowse)
map auto_home on /home (autofs, automounted, nobrowse)
map -fstab on /Network/Servers (autofs, automounted, nobrowse)
/dev/disk3s2 on /Volumes/MyBookData (hfs, local, nodev, nosuid, journaled)
afni:/elrond0 on /Volumes/afni (nfs)
afni:/var/www/INCOMING on /Volumes/INCOMING (nfs)
afni:/fraid on /Volumes/afni (nfs, asynchronous)
boromir:/raid.bot on /Volumes/raid.bot (nfs)
elros:/volume2/AFNI_SHARE on /Volumes/AFNI_SHARE (nfs)
map -static on /Volumes/safni (autofs, automounted, nobrowse)
map -static on /Volumes/raid.top (autofs, automounted, nobrowse)
/dev/disk1s3 on /Volumes/Boot OS X (hfs, local, journaled, nobrowse)
""",
        0,
        [],
    ),
    # Non-zero exit code
    ("", 1, []),
    # Variant of Linux example with CIFS added manually
    (
        r"""sysfs on /sys type sysfs (rw,nosuid,nodev,noexec,relatime)
proc on /proc type proc (rw,nosuid,nodev,noexec,relatime)
udev on /dev type devtmpfs (rw,nosuid,relatime,size=8121732k,nr_inodes=2030433,mode=755)
devpts on /dev/pts type devpts (rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=000)
tmpfs on /run type tmpfs (rw,nosuid,noexec,relatime,size=1628440k,mode=755)
/dev/nvme0n1p2 on / type ext4 (rw,relatime,errors=remount-ro,data=ordered)
securityfs on /sys/kernel/security type securityfs (rw,nosuid,nodev,noexec,relatime)
tmpfs on /dev/shm type tmpfs (rw,nosuid,nodev)
tmpfs on /sys/fs/cgroup type tmpfs (ro,nosuid,nodev,noexec,mode=755)
cgroup on /sys/fs/cgroup/systemd type cgroup (rw,nosuid,nodev,noexec,relatime,xattr,release_agent=/lib/systemd/systemd-cgroups-agent,name=systemd)
pstore on /sys/fs/pstore type pstore (rw,nosuid,nodev,noexec,relatime)
efivarfs on /sys/firmware/efi/efivars type efivarfs (rw,nosuid,nodev,noexec,relatime)
cgroup on /sys/fs/cgroup/cpu,cpuacct type cgroup (rw,nosuid,nodev,noexec,relatime,cpu,cpuacct)
cgroup on /sys/fs/cgroup/freezer type cgroup (rw,nosuid,nodev,noexec,relatime,freezer)
cgroup on /sys/fs/cgroup/pids type cgroup (rw,nosuid,nodev,noexec,relatime,pids)
cgroup on /sys/fs/cgroup/cpuset type cgroup (rw,nosuid,nodev,noexec,relatime,cpuset)
systemd-1 on /proc/sys/fs/binfmt_misc type autofs (rw,relatime,fd=26,pgrp=1,timeout=0,minproto=5,maxproto=5,direct)
hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime)
debugfs on /sys/kernel/debug type debugfs (rw,relatime)
mqueue on /dev/mqueue type mqueue (rw,relatime)
fusectl on /sys/fs/fuse/connections type fusectl (rw,relatime)
/dev/nvme0n1p1 on /boot/efi type vfat (rw,relatime,fmask=0077,dmask=0077,codepage=437,iocharset=iso8859-1,shortname=mixed,errors=remount-ro)
/dev/nvme0n1p2 on /var/lib/docker/aufs type ext4 (rw,relatime,errors=remount-ro,data=ordered)
gvfsd-fuse on /run/user/1002/gvfs type fuse.gvfsd-fuse (rw,nosuid,nodev,relatime,user_id=1002,group_id=1002)
""",
        0,
        [],
    ),
    # Variant of OS X example with CIFS added manually
    (
        r"""/dev/disk2 on / (hfs, local, journaled)
devfs on /dev (devfs, local, nobrowse)
afni:/elrond0 on /Volumes/afni (cifs)
afni:/var/www/INCOMING on /Volumes/INCOMING (nfs)
afni:/fraid on /Volumes/afni/fraid (nfs, asynchronous)
boromir:/raid.bot on /Volumes/raid.bot (nfs)
elros:/volume2/AFNI_SHARE on /Volumes/AFNI_SHARE (nfs)
""",
        0,
        [("/Volumes/afni/fraid", "nfs"), ("/Volumes/afni", "cifs")],
    ),
    # From Windows: docker run --rm -it -v C:\:/data busybox mount
    (
        r"""overlay on / type overlay (rw,relatime,lowerdir=/var/lib/docker/overlay2/l/26UTYITLF24YE7KEGTMHUNHPPG:/var/lib/docker/overlay2/l/SWGNP3T2EEB4CNBJFN3SDZLXHP,upperdir=/var/lib/docker/overlay2/a4c54ab1aa031bb5a14a424abd655510521e183ee4fa4158672e8376c89df394/diff,workdir=/var/lib/docker/overlay2/a4c54ab1aa031bb5a14a424abd655510521e183ee4fa4158672e8376c89df394/work)
proc on /proc type proc (rw,nosuid,nodev,noexec,relatime)
tmpfs on /dev type tmpfs (rw,nosuid,size=65536k,mode=755)
devpts on /dev/pts type devpts (rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=666)
sysfs on /sys type sysfs (ro,nosuid,nodev,noexec,relatime)
tmpfs on /sys/fs/cgroup type tmpfs (ro,nosuid,nodev,noexec,relatime,mode=755)
cpuset on /sys/fs/cgroup/cpuset type cgroup (ro,nosuid,nodev,noexec,relatime,cpuset)
cpu on /sys/fs/cgroup/cpu type cgroup (ro,nosuid,nodev,noexec,relatime,cpu)
cpuacct on /sys/fs/cgroup/cpuacct type cgroup (ro,nosuid,nodev,noexec,relatime,cpuacct)
blkio on /sys/fs/cgroup/blkio type cgroup (ro,nosuid,nodev,noexec,relatime,blkio)
memory on /sys/fs/cgroup/memory type cgroup (ro,nosuid,nodev,noexec,relatime,memory)
devices on /sys/fs/cgroup/devices type cgroup (ro,nosuid,nodev,noexec,relatime,devices)
freezer on /sys/fs/cgroup/freezer type cgroup (ro,nosuid,nodev,noexec,relatime,freezer)
net_cls on /sys/fs/cgroup/net_cls type cgroup (ro,nosuid,nodev,noexec,relatime,net_cls)
perf_event on /sys/fs/cgroup/perf_event type cgroup (ro,nosuid,nodev,noexec,relatime,perf_event)
net_prio on /sys/fs/cgroup/net_prio type cgroup (ro,nosuid,nodev,noexec,relatime,net_prio)
hugetlb on /sys/fs/cgroup/hugetlb type cgroup (ro,nosuid,nodev,noexec,relatime,hugetlb)
pids on /sys/fs/cgroup/pids type cgroup (ro,nosuid,nodev,noexec,relatime,pids)
cgroup on /sys/fs/cgroup/systemd type cgroup (ro,nosuid,nodev,noexec,relatime,name=systemd)
mqueue on /dev/mqueue type mqueue (rw,nosuid,nodev,noexec,relatime)
//10.0.75.1/C on /data type cifs (rw,relatime,vers=3.02,sec=ntlmsspi,cache=strict,username=filo,domain=MSI,uid=0,noforceuid,gid=0,noforcegid,addr=10.0.75.1,file_mode=0755,dir_mode=0755,iocharset=utf8,nounix,serverino,mapposix,nobrl,mfsymlinks,noperm,rsize=1048576,wsize=1048576,echo_interval=60,actimeo=1)
/dev/sda1 on /etc/resolv.conf type ext4 (rw,relatime,data=ordered)
/dev/sda1 on /etc/hostname type ext4 (rw,relatime,data=ordered)
/dev/sda1 on /etc/hosts type ext4 (rw,relatime,data=ordered)
shm on /dev/shm type tmpfs (rw,nosuid,nodev,noexec,relatime,size=65536k)
devpts on /dev/console type devpts (rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=666)
proc on /proc/bus type proc (ro,relatime)
proc on /proc/fs type proc (ro,relatime)
proc on /proc/irq type proc (ro,relatime)
proc on /proc/sys type proc (ro,relatime)
proc on /proc/sysrq-trigger type proc (ro,relatime)
tmpfs on /proc/kcore type tmpfs (rw,nosuid,size=65536k,mode=755)
tmpfs on /proc/timer_list type tmpfs (rw,nosuid,size=65536k,mode=755)
tmpfs on /proc/sched_debug type tmpfs (rw,nosuid,size=65536k,mode=755)
tmpfs on /proc/scsi type tmpfs (ro,relatime)
tmpfs on /sys/firmware type tmpfs (ro,relatime)
""",
        0,
        [("/data", "cifs")],
    ),
    # From @yarikoptic - added blank lines to test for resilience
    (
        r"""/proc on /proc type proc (rw,relatime)
sysfs on /sys type sysfs (rw,nosuid,nodev,noexec,relatime)
tmpfs on /dev/shm type tmpfs (rw,relatime)
devpts on /dev/pts type devpts (rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=666)

devpts on /dev/ptmx type devpts (rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=666)

""",
        0,
        [],
    ),
)


@pytest.mark.parametrize("output, exit_code, expected", MOUNT_OUTPUTS)
def test_parse_mount_table(output, exit_code, expected):
    assert _parse_mount_table(exit_code, output) == expected


def test_cifs_check():
    assert isinstance(_cifs_table, list)
    assert isinstance(on_cifs("/"), bool)
    fake_table = [("/scratch/tmp", "ext4"), ("/scratch", "cifs")]
    cifs_targets = [
        ("/scratch/tmp/x/y", False),
        ("/scratch/tmp/x", False),
        ("/scratch/x/y", True),
        ("/scratch/x", True),
        ("/x/y", False),
        ("/x", False),
        ("/", False),
    ]

    orig_table = _cifs_table[:]
    _cifs_table[:] = []

    for target, _ in cifs_targets:
        assert on_cifs(target) is False

    _cifs_table.extend(fake_table)
    for target, expected in cifs_targets:
        assert on_cifs(target) is expected

    _cifs_table[:] = []
    _cifs_table.extend(orig_table)
