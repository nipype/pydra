import json
import glob as glob
from pydra.compose import python, shell, workflow
from pydra.utils.messenger import FileMessenger, PrintMessenger, collect_messages
from pydra.engine.audit import AuditFlag
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job
from fileformats.generic import File
from pydra.utils.hash import hash_function


def test_audit_prov(
    tmpdir,
):
    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    # printing the audit message
    funky = TestFunc(a=1)
    funky(cache_root=tmpdir, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())

    # saving the audit message into the file
    funky = TestFunc(a=2)
    funky(cache_root=tmpdir, audit_flags=AuditFlag.PROV, messengers=FileMessenger())
    # this should be the default loctaion
    message_path = tmpdir / funky._checksum / "messages"
    assert (tmpdir / funky._checksum / "messages").exists()

    collect_messages(tmpdir / funky._checksum, message_path, ld_op="compact")
    assert (tmpdir / funky._checksum / "messages.jsonld").exists()


def test_audit_task(tmpdir):
    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    from glob import glob

    funky = TestFunc(a=2)
    funky(
        cache_root=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = tmpdir / funky._checksum / "messages"

    for file in glob(str(message_path) + "/*.jsonld"):
        with open(file) as f:
            data = json.load(f)
            if "@type" in data:
                if "AssociatedWith" in data:
                    assert "main" in data["Label"]

            if "@type" in data:
                if data["@type"] == "input":
                    assert None is data["Label"]
            if "AssociatedWith" in data:
                assert None is data["AssociatedWith"]

    # assert any(json_content)


def test_audit_shellcommandtask(tmpdir):
    Shelly = shell.define("ls -l<long=True>")

    from glob import glob

    shelly = Shelly()

    shelly(
        cache_root=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = tmpdir / shelly._checksum / "messages"
    # go through each jsonld file in message_path and check if the label field exists

    command_content = []

    for file in glob(str(message_path) + "/*.jsonld"):
        with open(file) as f:
            data = json.load(f)

            if "@type" in data:
                if "AssociatedWith" in data:
                    assert "main" == data["Label"]

            if "@type" in data:
                if data["@type"] == "input":
                    assert data["Label"] is None

            if "Command" in data:
                command_content.append(True)
                assert "ls -l" == data["Command"]

    assert any(command_content)


def test_audit_shellcommandtask_file(tmp_path):
    # sourcery skip: use-fstring-for-concatenation
    # create test.txt file with "This is a test" in it in the tmpdir
    # create txt file in cwd
    test1_file = tmp_path / "test.txt"
    test2_file = tmp_path / "test2.txt"
    with open(test1_file, "w") as f:
        f.write("This is a test")

    with open(test2_file, "w") as f:
        f.write("This is a test")

    cmd = "cat"
    file_in = File(test1_file)
    file_in_2 = File(test2_file)
    test_file_hash = hash_function(file_in)
    test_file_hash_2 = hash_function(file_in_2)
    Shelly = shell.define(
        cmd,
        inputs={
            "in_file": shell.arg(
                type=File,
                position=1,
                argstr="",
                help="text",
            ),
            "in_file_2": shell.arg(
                type=File,
                position=2,
                argstr="",
                help="text",
            ),
        },
    )
    shelly = Shelly(
        in_file=file_in,
        in_file_2=file_in_2,
    )
    shelly(
        cache_root=tmp_path,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = tmp_path / shelly._hash / "messages"
    for file in glob.glob(str(message_path) + "/*.jsonld"):
        with open(file) as x:
            data = json.load(x)
            if "@type" in data:
                if data["@type"] == "input":
                    if data["Label"] == "in_file":
                        assert data["AtLocation"] == str(file_in)
                        assert data["digest"] == test_file_hash
                    if data["Label"] == "in_file_2":
                        assert data["AtLocation"] == str(file_in_2)
                        assert data["digest"] == test_file_hash_2


def test_audit_shellcommandtask_version(tmpdir):
    import subprocess as sp

    version_cmd = sp.run("less --version", shell=True, stdout=sp.PIPE).stdout.decode(
        "utf-8"
    )
    version_cmd = version_cmd.splitlines()[0]
    cmd = "less test_task.py"
    Shelly = shell.define(cmd)
    shelly = Shelly()

    import glob

    shelly(
        cache_root=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = tmpdir / shelly._checksum / "messages"
    # go through each jsonld file in message_path and check if the label field exists
    version_content = []
    for file in glob.glob(str(message_path) + "/*.jsonld"):
        with open(file) as f:
            data = json.load(f)
            if "AssociatedWith" in data:
                if version_cmd in data["AssociatedWith"]:
                    version_content.append(True)

    assert any(version_content)


def test_audit_prov_messdir_1(
    tmpdir,
):
    """customized messenger dir"""

    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    # printing the audit message
    funky = TestFunc(a=1)
    funky(cache_root=tmpdir, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())

    # saving the audit message into the file
    funky = TestFunc(a=2)
    # user defined path
    message_path = tmpdir / funky._checksum / "my_messages"
    # providing messenger_dir for audit
    funky_task = Job(
        task=funky,
        submitter=Submitter(
            cache_root=tmpdir, audit_flags=AuditFlag.PROV, messengers=FileMessenger()
        ),
        name="funky",
    )
    funky_task.audit.messenger_args = dict(message_dir=message_path)
    funky_task.run()
    assert (tmpdir / funky._checksum / "my_messages").exists()

    collect_messages(tmpdir / funky._checksum, message_path, ld_op="compact")
    assert (tmpdir / funky._checksum / "messages.jsonld").exists()


def test_audit_prov_messdir_2(
    tmpdir,
):
    """customized messenger dir in init"""

    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    # printing the audit message
    funky = TestFunc(a=1)
    funky(cache_root=tmpdir, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())

    # user defined path (doesn't depend on checksum, can be defined before init)
    message_path = tmpdir / "my_messages"
    # saving the audit message into the file
    funky = TestFunc(a=2)
    # providing messenger_dir for audit
    funky(
        cache_root=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
        messenger_args=dict(message_dir=message_path),
    )
    assert (tmpdir / "my_messages").exists()

    collect_messages(tmpdir, message_path, ld_op="compact")
    assert (tmpdir / "messages.jsonld").exists()


def test_audit_prov_wf(
    tmpdir,
):
    """FileMessenger for wf"""

    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    @workflow.define
    def Workflow(x: int):
        test_func = workflow.add(TestFunc(a=x))
        return test_func.out

    wf = Workflow(x=2)

    wf(
        cache_root=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    # default path
    message_path = tmpdir / wf._checksum / "messages"
    assert message_path.exists()

    collect_messages(tmpdir / wf._checksum, message_path, ld_op="compact")
    assert (tmpdir / wf._checksum / "messages.jsonld").exists()


def test_audit_all(
    tmpdir,
):
    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    funky = TestFunc(a=2)
    message_path = tmpdir / funky._checksum / "messages"

    funky(
        cache_root=tmpdir,
        audit_flags=AuditFlag.ALL,
        messengers=FileMessenger(),
        messenger_args=dict(message_dir=message_path),
    )
    from glob import glob

    assert len(glob(str(tmpdir / funky._checksum / "proc*.log"))) == 1
    assert len(glob(str(message_path / "*.jsonld"))) == 7

    # commented out to speed up testing
    collect_messages(tmpdir / funky._checksum, message_path, ld_op="compact")
    assert (tmpdir / funky._checksum / "messages.jsonld").exists()
