import typing as ty
import sys
from pathlib import Path
import attr
from unittest.mock import Mock
import pytest
from fileformats.generic import File
from ..specs import SpecInfo, ShellSpec
from ..task import ShellTask
from ..helpers_file import (
    ensure_list,
    MountIndentifier,
    copy_nested_files,
    template_update_single,
)


def _ignore_atime(stat):
    return stat[:7] + stat[8:]


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


def test_copy_nested_files_copy(tmp_path: Path):
    # Test copying files from within nested data structures
    src_dir = tmp_path / "src"

    src_dir.mkdir()

    # Create temporary files
    files = []
    for x in "abcde":
        p = src_dir / (x + ".txt")
        p.write_text(x)
        files.append(File(p))
    a, b, c, d, e = files

    nested_files = [{"a": a}, b, [(c, a), (d, e)]]

    dest_dir = tmp_path / "dest"
    nested_files_copy = copy_nested_files(
        nested_files, dest_dir, mode=File.CopyMode.copy
    )
    assert sorted(p.relative_to(src_dir) for p in src_dir.glob("**/*.txt")) == sorted(
        p.relative_to(dest_dir) for p in dest_dir.glob("**/*.txt")
    )
    copied_files = []
    for x in "abcde":
        copied_files.append(File(dest_dir / (x + ".txt")))
    a, b, c, d, e = copied_files
    assert nested_files_copy == [{"a": a}, b, [(c, a), (d, e)]]


def test_copy_nested_files_hardlink(tmp_path: Path):
    src_dir = tmp_path / "src"

    src_dir.mkdir()

    # Create temporary files
    files = []
    for x in "abcde":
        p = src_dir / (x + ".txt")
        p.write_text(x)
        files.append(File(p))
    a, b, c, d, e = files

    nested_files = [{"a": a}, b, [(c, a), (d, e)]]

    dest_dir = tmp_path / "dest"
    nested_files_copy = copy_nested_files(
        nested_files, dest_dir, mode=File.CopyMode.hardlink
    )
    assert sorted(p.relative_to(src_dir) for p in src_dir.glob("**/*.txt")) == sorted(
        p.relative_to(dest_dir) for p in dest_dir.glob("**/*.txt")
    )
    copied_files = []
    for x in "abcde":
        copied_files.append(File(dest_dir / (x + ".txt")))
    a, b, c, d, e = copied_files
    assert nested_files_copy == [{"a": a}, b, [(c, a), (d, e)]]


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="symlinks not supported on Windows"
)
def test_copy_nested_files_symlink(tmp_path: Path):
    src_dir = tmp_path / "src"

    src_dir.mkdir()

    # Create temporary files
    files = []
    for x in "abcde":
        p = src_dir / (x + ".txt")
        p.write_text(x)
        files.append(File(p))
    a, b, c, d, e = files

    nested_files = [{"a": a}, b, [(c, a), (d, e)]]

    dest_dir = tmp_path / "dest"
    nested_files_copy = copy_nested_files(
        nested_files, dest_dir, mode=File.CopyMode.symlink
    )
    assert sorted(p.relative_to(src_dir) for p in src_dir.glob("**/*.txt")) == sorted(
        p.relative_to(dest_dir) for p in dest_dir.glob("**/*.txt")
    )
    copied_files: ty.List[File] = []
    for x in "abcde":
        copied_files.append(File(dest_dir / (x + ".txt")))
    assert all(f.fspath.is_symlink() for f in copied_files)
    a, b, c, d, e = copied_files
    assert nested_files_copy == [{"a": a}, b, [(c, a), (d, e)]]


def test_copy_nested_files_leave(tmp_path: Path):
    src_dir = tmp_path / "src"

    src_dir.mkdir()

    # Create temporary files
    files = []
    for x in "abcde":
        p = src_dir / (x + ".txt")
        p.write_text(x)
        files.append(File(p))
    a, b, c, d, e = files

    nested_files = [{"a": a}, b, [(c, a), (d, e)]]

    dest_dir = tmp_path / "dest"  # not used

    nested_files_copy = copy_nested_files(
        nested_files, dest_dir, mode=File.CopyMode.leave
    )
    assert nested_files_copy == nested_files


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
    assert MountIndentifier.parse_mount_table(exit_code, output) == expected


def test_cifs_check():
    assert isinstance(MountIndentifier.get_mount_table(), list)
    assert isinstance(MountIndentifier.on_cifs("/"), bool)
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

    with MountIndentifier.patch_table([]):
        for target, _ in cifs_targets:
            assert MountIndentifier.on_cifs(target) is False

    with MountIndentifier.patch_table(fake_table):
        for target, expected in cifs_targets:
            assert MountIndentifier.on_cifs(target) is expected


def test_output_template(tmp_path):
    filename = str(tmp_path / "file.txt")
    with open(filename, "w") as f:
        f.write("hello from pydra")
    in_file = File(filename)

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "in_file",
                attr.ib(
                    type=File,
                    metadata={
                        "mandatory": True,
                        "position": 1,
                        "argstr": "",
                        "help_string": "input file",
                    },
                ),
            ),
            (
                "optional",
                attr.ib(
                    type=ty.Union[Path, bool],
                    default=False,
                    metadata={
                        "position": 2,
                        "argstr": "--opt",
                        "output_file_template": "{in_file}.out",
                        "help_string": "optional file output",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    class MyCommand(ShellTask):
        executable = "my"
        input_spec = my_input_spec

    task = MyCommand(in_file=filename)
    assert task.cmdline == f"my {filename}"
    task.inputs.optional = True
    assert task.cmdline == f"my {filename} --opt {task.output_dir / 'file.out'}"
    task.inputs.optional = False
    assert task.cmdline == f"my {filename}"
    task.inputs.optional = "custom-file-out.txt"
    assert task.cmdline == f"my {filename} --opt custom-file-out.txt"


def test_template_formatting(tmp_path):
    field = Mock()
    field.name = "grad"
    field.argstr = "--grad"
    field.metadata = {"output_file_template": ("{in_file}.bvec", "{in_file}.bval")}
    inputs = Mock()
    inputs_dict = {"in_file": "/a/b/c/file.txt", "grad": True}

    assert template_update_single(
        field,
        inputs,
        inputs_dict_st=inputs_dict,
        output_dir=tmp_path,
        spec_type="input",
    ) == [str(tmp_path / "file.bvec"), str(tmp_path / "file.bval")]
