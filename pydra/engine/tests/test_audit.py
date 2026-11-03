import json
from pathlib import Path
import glob as glob
from pydra.compose import python, shell, workflow
from pydra.utils.messenger import FileMessenger, PrintMessenger, collect_messages
from pydra.engine.audit import AuditFlag
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job
from fileformats.generic import File
from pydra.utils.hash import hash_function


def _slot(cache_root, task):
    """The one context slot holding the task's results"""
    (slot,) = (Path(cache_root) / task._checksum).iterdir()
    return slot


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
    message_path = _slot(tmpdir, funky) / "messages"
    assert (_slot(tmpdir, funky) / "messages").exists()

    collect_messages(_slot(tmpdir, funky), message_path, ld_op="compact")
    assert (_slot(tmpdir, funky) / "messages.jsonld").exists()


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
    message_path = _slot(tmpdir, funky) / "messages"

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
    message_path = _slot(tmpdir, shelly) / "messages"
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

    # Python is used as the executable as it is available on all platforms
    version_cmd = sp.run("python --version", shell=True, stdout=sp.PIPE).stdout.decode(
        "utf-8"
    )
    version_cmd = version_cmd.splitlines()[0]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = ["python", "-c", "pass"]

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly()

    import glob

    shelly(
        cache_root=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = _slot(tmpdir, shelly) / "messages"
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
    # providing messenger_dir for audit
    funky_task = Job(
        task=funky,
        submitter=Submitter(
            cache_root=tmpdir, audit_flags=AuditFlag.PROV, messengers=FileMessenger()
        ),
        name="funky",
    )
    message_path = funky_task.cache_dir / "my_messages"
    funky_task.audit.messenger_args = dict(message_dir=message_path)
    funky_task.run()
    assert (_slot(tmpdir, funky) / "my_messages").exists()

    collect_messages(_slot(tmpdir, funky), message_path, ld_op="compact")
    assert (_slot(tmpdir, funky) / "messages.jsonld").exists()


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
    message_path = _slot(tmpdir, wf) / "messages"
    assert message_path.exists()

    collect_messages(_slot(tmpdir, wf), message_path, ld_op="compact")
    assert (_slot(tmpdir, wf) / "messages.jsonld").exists()


def test_audit_all(
    tmpdir,
):
    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    funky = TestFunc(a=2)
    message_path = Path(tmpdir) / "messages"

    funky(
        cache_root=tmpdir,
        audit_flags=AuditFlag.ALL,
        messengers=FileMessenger(),
        messenger_args=dict(message_dir=message_path),
    )
    from glob import glob

    assert len(glob(str(_slot(tmpdir, funky) / "proc*.log"))) == 1
    assert len(glob(str(message_path / "*.jsonld"))) == 7

    # commented out to speed up testing
    collect_messages(_slot(tmpdir, funky), message_path, ld_op="compact")
    assert (_slot(tmpdir, funky) / "messages.jsonld").exists()
