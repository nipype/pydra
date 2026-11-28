"""The identity record is the preimage of the job checksum (RFC 0004 section 3.1, I1)."""

import hashlib

import pytest

from fileformats.generic import File

from pydra.compose import python, shell, workflow
from pydra.engine.job import Job
from pydra.engine.portable.identity import (
    PydraIdentityV1,
    PydraIdentityV2,
    StoreIdentity,
)
from pydra.engine.submitter import Submitter
from pydra.utils.digest import jcs
from pydra.utils.identity import (
    IDENTITY_SCHEME,
    job_record,
    record_checksum,
    task_record,
)


@python.define
def Add(a: int, b: int = 2) -> int:
    return a + b


def _key(task, inputs):
    return f"key-{inputs['a']}"


@python.define(checksum=_key)
def Keyed(a: int, b: int = 2) -> int:
    return a + b


@python.define(checksum=lambda task, inputs: f"k-{inputs['a']:03d}")
def Padded(a: int) -> int:
    return a


@python.define
class Hooked(python.Task["Hooked.Outputs"]):
    a: int

    @staticmethod
    def function(a: int) -> int:
        return a

    def __checksum__(self, inputs):
        return f"hooked-{inputs['a']}"

    class Outputs(python.Outputs):
        out: int


def _job(task):
    return Job(name="t", task=task, submitter=Submitter())


def test_scheme_name():
    assert IDENTITY_SCHEME == "pydra-identity@2"


def test_checksum_is_sha256_of_jcs_record():
    task = Add(a=1)
    record = task_record(task)
    assert list(sorted(record)) == [
        "inputs",
        "kind",
        "outputs",
        "scheme",
        "task",
        "walk",
    ]
    assert record["scheme"] == IDENTITY_SCHEME
    assert record["kind"] == "python"
    assert set(record["inputs"]) == {"a", "b"}
    assert record["task"]["code"].startswith("code@1:")
    assert record["outputs"].startswith("type@1:")
    expected = "python-" + hashlib.sha256(jcs(record)).hexdigest()
    assert len(expected) == len("python-") + 64
    assert record_checksum(record) == expected
    assert task._checksum == expected
    assert _job(task).checksum == expected
    assert task._hash == expected.removeprefix("python-")


def test_record_changes_with_inputs_and_is_stable():
    assert task_record(Add(a=1)) == task_record(Add(a=1))
    assert task_record(Add(a=1))["inputs"] != task_record(Add(a=2))["inputs"]
    assert Add(a=1)._checksum != Add(a=2)._checksum


def test_supplied_checksum_unchanged():
    # literals recorded at 3f734f9a, the base of the RFC 0004 work: the K2 checksums
    # of whole jobs and of the elements of split tasks are the supplied strings
    assert Keyed(a=1)._checksum == "key-1"
    assert Padded(a=7)._checksum == "k-007"
    assert Hooked(a=3)._checksum == "hooked-3"
    assert _job(Keyed(a=1)).checksum == "key-1"
    assert Keyed(b=5).split(a=[1, 2, 3])._element_checksums() == [
        "key-1",
        "key-2",
        "key-3",
    ]
    assert Padded().split(a=[4, 5])._element_checksums() == ["k-004", "k-005"]


def test_supplied_record_is_the_supplied_string():
    record = task_record(Keyed(a=1))
    assert record == {"scheme": IDENTITY_SCHEME, "supplied": "key-1"}
    assert record_checksum(record) == "key-1"


def test_check_key_accepts_supplied_and_digest():
    ident = PydraIdentityV2()
    task = Add(a=1)
    record = ident.key(_job(task))
    assert record == task_record(task)
    assert ident.check_key(task._checksum, record)
    supplied = ident.key(_job(Keyed(a=1)))
    assert ident.check_key("key-1", supplied)
    assert not ident.check_key("key-2", supplied)
    # a record that digests to another checksum is refused
    assert not ident.check_key(Add(a=2)._checksum, record)
    assert not ident.check_key(task._checksum, None)
    assert not ident.check_key(task._checksum, {"scheme": IDENTITY_SCHEME})


def test_check_key_rejects_other_scheme():
    ident = PydraIdentityV2()
    record = dict(task_record(Add(a=1)), scheme="pydra-identity@1")
    assert not ident.check_key(record_checksum(record), record)
    supplied = {"scheme": "pydra-identity@1", "supplied": "key-1"}
    assert not ident.check_key("key-1", supplied)


def test_identity_v2_is_a_store_identity():
    ident = PydraIdentityV2(recorded_only=["x"])
    assert isinstance(ident, StoreIdentity)
    assert ident.name == IDENTITY_SCHEME == "pydra-identity@2"
    assert ident.fingerprint_name == "pydra-context@1"
    assert ident.recorded_only == frozenset(
        {"x", "environment_facts", "environment_repository"}
    )
    ctx = {"a": 1, "x": 2}
    assert ident.fingerprint(ctx) == PydraIdentityV1(recorded_only=["x"]).fingerprint(
        ctx
    )
    job = _job(Add(a=1))
    assert ident.context(job, "lookup") == PydraIdentityV1().context(job, "lookup")


def test_walk_field_reflects_env(monkeypatch):
    monkeypatch.delenv("PYDRA_CODE_HASH", raising=False)
    monkeypatch.delenv("PYDRA_USER_PACKAGES", raising=False)
    default = task_record(Add(a=1))
    assert default["walk"] == {
        "code_hash": "user",
        "user_packages": [],
        "library_packages": [],
    }
    monkeypatch.setenv("PYDRA_CODE_HASH", "module")
    monkeypatch.setenv("PYDRA_USER_PACKAGES", "zeta, alpha,-numpy")
    record = task_record(Add(a=1))
    assert record["walk"] == {
        "code_hash": "module",
        "user_packages": ["alpha", "zeta"],
        "library_packages": ["numpy"],
    }
    assert record_checksum(record) != record_checksum(default)


def test_walk_field_reflects_task_code_hash(monkeypatch):
    monkeypatch.delenv("PYDRA_CODE_HASH", raising=False)

    @python.define(code_hash="function")
    def Inner(a: int) -> int:
        return a

    assert task_record(Inner(a=1))["walk"]["code_hash"] == "function"


def test_task_checksum_excludes_job_fields():
    task = Add(a=1)
    record = task_record(task)
    assert "job" not in record and "nodes" not in record
    # the job's record is the task's plus the job's own fields (none at this stage)
    assert job_record(_job(task)) == record
    with_job = dict(record, job={"environment": "docker@1:sha256:" + "0" * 64})
    assert record_checksum(with_job) != task._checksum
    assert task._checksum == record_checksum(record)


def test_job_keeps_record_and_pickles_it():
    import cloudpickle as cp

    job = _job(Add(a=1))
    record = job.record
    assert record == task_record(job.task)
    clone = cp.loads(cp.dumps(job))
    assert clone.__dict__["_record"] == record
    assert clone.checksum == job.checksum


def test_result_keeps_record(tmp_path):
    job = Job(
        name="t",
        task=Add(a=1),
        submitter=Submitter(store="pickle", cache_root=tmp_path / "cache"),
    )
    result = job.run()
    assert result.record == task_record(job.task)
    assert record_checksum(result.record) == job.checksum


def test_run_through_worker(any_worker, tmp_path):
    outputs = Add(a=1)(cache_root=tmp_path, worker=any_worker)
    assert outputs.out == 3


def test_key_is_the_carried_record_in_a_worker(monkeypatch):
    """A worker that cannot read the function's source (#936) still publishes a key that
    digests to the job's checksum: ``key`` returns the record carried with the job."""
    import cloudpickle as cp

    import pydra.utils.identity as identity

    job = _job(Add(a=1))
    job.checksum
    clone = cp.loads(cp.dumps(job))

    def unreadable(*args, **kwargs):
        raise OSError("could not get source code")

    monkeypatch.setattr(identity, "hash_function", unreadable)
    ident = PydraIdentityV2()
    assert ident.key(clone) == job.record
    assert ident.check_key(clone.checksum, ident.key(clone))
    assert ident.check_key(job.checksum, ident.key(clone))


def test_key_does_not_rebuild_a_changed_record(monkeypatch):
    import pydra.utils.identity as identity

    job = _job(Add(a=1))
    job.checksum
    monkeypatch.setattr(identity, "hash_function", lambda *a, **k: "0" * 32)
    ident = PydraIdentityV2()
    assert ident.check_key(job.checksum, ident.key(job))


def test_check_key_refuses_misfiled_and_malformed_keys():
    ident = PydraIdentityV2()
    task = Add(a=1)
    record = task_record(task)
    # a suffixed checksum under a record without "absent" is misfiled
    assert not ident.check_key(task._checksum + "-absent-" + "0" * 64, record)
    assert not ident.check_key(task._checksum[:-1], record)
    # a record carrying "absent" digests to neither (RFC 0004 key record: the absence suffix)
    with_absent = dict(record, absent=[["out", {"$int": "0"}, "dropped"]])
    assert not ident.check_key(task._checksum, with_absent)
    assert not ident.check_key(record_checksum(with_absent), record)
    for bad in (
        dict(record, inputs={"a": float("nan")}),
        dict(record, inputs={1: "x"}),
        dict(record, outputs="\ud800"),
        dict(record, inputs={"a": 2**60}),
        dict(record, task=object()),
        {"scheme": IDENTITY_SCHEME, "supplied": 5},
        [record],
        "record",
    ):
        assert ident.check_key(task._checksum, bad) is False


@python.define
def Fields(a: int, b: int = 2) -> int:
    return a + b


def test_python_record_shape():
    record = task_record(Fields(a=1))
    assert record["kind"] == "python"
    assert list(record["task"]) == ["refs", "values", "code"]
    assert set(record["inputs"]) == {"a", "b"}
    assert record["inputs"]["a"].startswith("py@1:")
    # the outputs digest follows the Outputs class
    other = python.define(lambda a: a, outputs={"out": str}, inputs={"a": int})
    assert task_record(other(a=1))["outputs"] != record["outputs"]


def test_shell_record_shape():
    record = task_record(shell.define("echo <text:str>")(text="hi"))
    assert record["kind"] == "shell"
    assert list(record["task"]) == ["refs", "values", "template", "version"]
    assert record["task"]["template"]["executable"] == "echo"
    assert list(record["task"]["template"]["fields"]) == ["append_args", "text"]
    assert record["task"]["version"] is None
    assert "executable" not in record["inputs"]
    assert record["inputs"]["text"].startswith("py@1:")
    assert record["outputs"].startswith("type@1:")
    assert record_checksum(record).startswith("shell-")


def test_workflow_record_shape():
    @workflow.define
    def Wf(a: int) -> int:
        return workflow.add(Add(a=a)).out

    record = task_record(Wf(a=1))
    assert record["kind"] == "workflow"
    assert list(record["task"]) == ["refs", "values", "code"]
    assert "constructor" not in record["inputs"]
    assert set(record["inputs"]) == {"a"}
    assert record_checksum(record).startswith("workflow-")


def test_outputs_and_kind_enter_the_checksum():
    a = task_record(Add(a=1))
    assert record_checksum(dict(a, outputs="type@1:" + "0" * 32)) != record_checksum(a)
    assert record_checksum(dict(a, kind="shell")).startswith("shell-")


def test_record_digests_are_ascii_independent_jcs():
    # a non-ASCII field name is written as UTF-8 by JCS, not escaped
    record = dict(task_record(Add(a=1)), inputs={"\u00e9": "py@1:" + "0" * 32})
    assert record_checksum(record) == (
        "python-" + hashlib.sha256(jcs(record)).hexdigest()
    )
    assert b"\\u00e9" not in jcs(record)


def test_walk_packages_are_sorted(monkeypatch):
    monkeypatch.setenv("PYDRA_USER_PACKAGES", "b,a,-d,-c")
    walk = task_record(Add(a=1))["walk"]
    assert walk["user_packages"] == ["a", "b"]
    assert walk["library_packages"] == ["c", "d"]


def test_hash_changes_with_no_executor_name():
    import typing as ty

    @python.define
    def Sum(a: ty.List[int]) -> int:
        return sum(a)

    for name in ("function", None):
        task = Sum(a=[1])
        monkeypatch_name = type(task)._executor_name
        type(task)._executor_name = name
        try:
            task._checksum
            task.a.append(2)
            assert task._hash_changes() == ["a"], name
            assert task._hashes["a"] != task._compute_hashes(exclude="x")[1]["a"]
        finally:
            type(task)._executor_name = monkeypatch_name


def test_result_record_through_worker_and_async(any_worker, tmp_path):
    from pydra.engine.submitter import Submitter

    with Submitter(worker=any_worker, store="pickle", cache_root=tmp_path) as sub:
        result = sub(Add(a=3))
    assert result.record is not None
    assert record_checksum(result.record) == Add(a=3)._checksum
    assert result.record == task_record(Add(a=3))

    @workflow.define
    def Wf(a: int) -> int:
        return workflow.add(Add(a=a)).out

    with Submitter(worker=any_worker, store="pickle", cache_root=tmp_path) as sub:
        wf_result = sub(Wf(a=3))
    assert wf_result.record == task_record(Wf(a=3))
    assert record_checksum(wf_result.record) == Wf(a=3)._checksum


@python.define
def WithFile(f: File) -> int:
    return 1


def test_changed_outputs_digest_is_reported(tmp_path, monkeypatch):
    """A changed key that is not an input field (the ``Outputs`` digest) is reported as
    a change, not raised as an ``AttributeError``"""
    path = tmp_path / "x.txt"
    path.write_text("x")
    job = Job(name="t", task=WithFile(f=path), submitter=Submitter())
    job.checksum
    monkeypatch.setattr(type(job.task), "_hash_changes", lambda self: ["Outputs"])
    with pytest.raises(RuntimeError, match="Input field hashes have changed"):
        job._check_for_hash_changes()
