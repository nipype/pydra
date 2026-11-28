"""Work directories, ``attempt``, ``_committed.json``, ``_error.json`` and the de-duplication
lease of a `PortableStore` (RFC 0003 sections 5.3, 5.4 and 11)."""

import errno
import functools
import getpass
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import textwrap
import threading
import time
import typing as ty
import uuid
from pathlib import Path

import pytest

from pydra.compose import python, workflow
from pydra.engine.runner import GraphRunner
from pydra.engine.job import Job
from pydra.engine.lease import LeaseSettings, _describe
from pydra.engine.portable.entry import CorruptEntry, EntryNotTrusted
from pydra.engine.portable.layout import cdir_for
from pydra.engine.portable.store import PortableStore
from pydra.engine.store import DeclaredVersionWarning, check_declared_code
from pydra.engine.submitter import Submitter
from pydra.scripts.cli import crash
from pydra.utils.storagedir import IS_WINDOWS, StoreAges, StorePolicy

from .test_portable_quarantine import SKEW, age, aged_store, skew  # noqa: F401
from .test_portable_store import (
    CTX,
    CTX2,
    IDENT,
    TRUST,
    Refusing,
    _Lost,
    _result,
    _work_under,
    entry_path,
    make,
    posix_only,
    publish,
    staging_names,
)


@python.define
def _Inc(a: int) -> int:
    return a + 1


@python.define
def _Boom(a: int) -> int:
    raise ValueError("boom from the job")


def _touch(path: str) -> int:
    Path(path).write_text("unpickled")
    return 7


class _Marker:
    """Unpickling it runs ``_touch``, which shows that a pickle was opened."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self) -> tuple:
        return (_touch, (self.path,))


def job_in(sub, task, attempt=None, resumable=False, name="j"):
    job = Job(task=task, submitter=sub, name=name)
    job.attempt = uuid.uuid4().hex if attempt is None else attempt
    if resumable:
        job._can_resume = True
    return job


def portable_submitter(tmp_path, **kw):
    store = kw.pop("store", None) or make(tmp_path)
    return store, Submitter(
        worker="debug", store=store, cache_root=tmp_path / "pc", **kw
    )


def results(root):
    return list((Path(root) / "results@1").iterdir())


# -- the work directory is a pure function of values the job carries ---------------------------------


def test_shared_root_requires_work_dir(tmp_path):
    with pytest.raises(ValueError, match="work_dir"):
        PortableStore(tmp_path / "r", trust="shared")
    store = PortableStore(tmp_path / "p", trust=TRUST)  # private: <root>/work/<attempt>
    job = _stub(attempt="abc123")
    assert store.work_dir(job) == tmp_path / "p" / "work" / "abc123"
    assert store.dedupe is True
    if IS_WINDOWS:
        return
    shared = PortableStore(
        tmp_path / "s",
        trust="shared",
        work_dir=functools.partial(_work_under, tmp_path),
    )
    assert shared.dedupe is False
    assert shared.work_dir(_stub(attempt="x")) == tmp_path / "work" / "x"


def _stub(attempt, checksum="python-abc", context=CTX, can_resume=False):
    from types import SimpleNamespace

    return SimpleNamespace(
        name="j",
        checksum=checksum,
        context=dict(context),
        attempt=attempt,
        can_resume=can_resume,
        committed=None,
        task=None,
    )


SCRIPT = textwrap.dedent("""
    import json, sys
    from types import SimpleNamespace
    from pydra.engine.portable.store import PortableStore

    root, trust, attempt, checksum, resume = sys.argv[1:6]
    store = PortableStore(root, trust=trust)
    job = SimpleNamespace(name="j", checksum=checksum, context=json.loads(sys.argv[6]),
                          attempt=attempt or None, can_resume=resume == "1")
    print(store.work_dir(job))
    """)


def _work_dir_elsewhere(tmp_path, attempt, checksum="python-abc", resume="0"):
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            SCRIPT,
            str(tmp_path / "root"),
            TRUST,
            attempt,
            checksum,
            resume,
            json.dumps(CTX),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return Path(proc.stdout.strip())


def test_work_dir_pure_across_processes(tmp_path):
    store = make(tmp_path)
    here = store.work_dir(_stub("a1"))
    assert here == _work_dir_elsewhere(tmp_path, "a1")
    assert here.is_absolute() and here.parent == tmp_path / "root" / "work"
    assert store.work_dir(_stub("a2")) != here
    # a job that can resume has one directory per address, whatever its attempt
    r1 = store.work_dir(_stub(None, can_resume=True))
    assert r1 == store.work_dir(_stub("zzz", can_resume=True))
    assert r1 == _work_dir_elsewhere(tmp_path, "", resume="1")
    assert r1 != store.work_dir(_stub(None, checksum="python-other", can_resume=True))
    with pytest.raises(ValueError, match="attempt"):
        store.work_dir(_stub(None))
    with pytest.raises(ValueError, match="attempt"):
        store.work_dir(_stub("../escape"))


# -- _committed.json ---------------------------------------------------------------------------------


def refusing_submitter(tmp_path):
    return portable_submitter(tmp_path, store=make(tmp_path, reuse=Refusing()))


def _occupy_and_commit(store, sub, out=2, occupant=3):
    """Commit ``out`` for a job whose address holds an entry with other outputs: the result is
    not stored, and is kept in its work directory (the store's hook refuses the occupant).
    """
    first = job_in(sub, _Inc(a=1), name="first")
    store.commit(_Lost(), _result(first, _Inc.Outputs(out=occupant)), first)
    job = job_in(sub, _Inc(a=1))
    result = _result(job, _Inc.Outputs(out=out))
    store.commit(_Lost(), result, job)
    return job, result


def test_stored_false_read_by_other_process_via_committed_json(tmp_path):
    store, sub = refusing_submitter(tmp_path)
    with sub:
        job, result = _occupy_and_commit(store, sub)
    assert result.stored is False and result.committed.kept is not None
    wd = store.work_dir(job)
    assert Path(result.committed.kept) == wd / "_entry"
    assert (wd / "_committed.json").is_file()
    script = textwrap.dedent("""
        import json, sys
        from types import SimpleNamespace
        from pydra.engine.portable.store import PortableStore

        root, trust, attempt, checksum, context = sys.argv[1:6]
        store = PortableStore(root, trust=trust)
        job = SimpleNamespace(name="j", checksum=checksum, context=json.loads(context),
                              attempt=attempt, can_resume=False, committed=None, task=None)
        r = store.read_result(job, settled=True)
        print(json.dumps([r.outputs, r.stored, r.errored]))
        """)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "root"),
            TRUST,
            job.attempt,
            job.checksum,
            json.dumps(job.context),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [{"out": 2}, False, False]


def test_committed_json_written_by_rename(tmp_path, monkeypatch):
    store, sub = portable_submitter(tmp_path)
    renames = []
    real = os.rename

    def rename(src, dst, *a, **kw):
        if os.path.basename(str(dst)) == "_committed.json":
            name = os.path.basename(str(src))
            renames.append((name, json.loads((wd / name).read_text())))
        return real(src, dst, *a, **kw)

    with sub:
        job = job_in(sub, _Inc(a=1))
        wd = store.work_dir(job)
        monkeypatch.setattr(os, "rename", rename)
        store.commit(_Lost(), _result(job, _Inc.Outputs(out=2)), job)
    ((src, doc),) = renames
    assert src.startswith("._committed.json.") and src.endswith(".tmp")
    assert doc["format"] == "pydra-committed@1" and doc["stored"] is True
    assert doc["kind"] == "entry" and doc["checksum"] == job.checksum
    assert sorted(p.name for p in wd.iterdir()) == ["_committed.json"]


def test_committed_json_names_the_entry_and_is_read_without_the_job_committed(tmp_path):
    store, sub = portable_submitter(tmp_path)
    with sub:
        job = job_in(sub, _Inc(a=1))
        result = _result(job, _Inc.Outputs(out=2))
        store.commit(_Lost(), result, job)
        assert job.committed is None  # this process never ran it: only the file says
        read = store.read_result(job)
    assert read.outputs.out == 2 and read.stored is True
    assert read.committed == result.committed
    assert Path(read.cache_dir) == store.entry_path(job.checksum, job.context)


def test_a_committed_json_of_another_job_is_not_followed(tmp_path):
    store, sub = portable_submitter(tmp_path)
    with sub:
        job = job_in(sub, _Inc(a=1))
        store.commit(_Lost(), _result(job, _Inc.Outputs(out=2)), job)
        other = job_in(sub, _Inc(a=5), attempt=job.attempt)  # reads job's record
        failed = store.read_result(other, settled=False)
    assert failed.errored and "work directory" in failed.failure


# -- kept outputs ------------------------------------------------------------------------------------


def test_kept_entry_rehashed_on_read(tmp_path):
    store, sub = refusing_submitter(tmp_path)
    with sub:
        job, result = _occupy_and_commit(store, sub)
        wd = store.work_dir(job)
        assert store.read_result(job).outputs.out == 2  # intact: read from _entry/
        assert staging_names(tmp_path / "root") == []  # moved out of staging
        member = wd / "_entry" / "context.json"
        os.chmod(member, 0o644)
        data = member.read_bytes()
        member.write_bytes(bytes([data[0] ^ 1]) + data[1:])  # same size, other content
        with pytest.raises(CorruptEntry):
            store.read_committed(job, result.committed)
        failed = store.read_result(job)
    assert failed.errored and failed.outputs is None
    assert str(wd) in failed.failure


def test_kept_entry_copied_across_filesystems_is_rehashed(tmp_path, monkeypatch):
    real = os.rename

    def rename(src, dst, *a, **kw):
        if os.path.basename(str(dst)) == "_entry":
            raise OSError(errno.EXDEV, "cross-device link")
        return real(src, dst, *a, **kw)

    store, sub = refusing_submitter(tmp_path)
    with sub:
        monkeypatch.setattr(os, "rename", rename)
        job, result = _occupy_and_commit(store, sub)
        monkeypatch.undo()
        assert store.read_result(job).outputs.out == 2
    assert staging_names(tmp_path / "root") == []
    assert (store.work_dir(job) / "_entry" / "entry.json").is_file()


def test_a_kept_entry_with_many_members_is_copied_across_filesystems(
    tmp_path, monkeypatch
):
    """The copy reads the members the way a read does: an ``entry.json`` over a mebibyte
    (thousands of file members) is copied, not refused as malformed"""
    import pydra.engine.portable.codecs as codecs

    real_init = codecs.Encoder.__init__
    many = {
        f"files/{'a' * 64}/m{i}": {"sha256": "b" * 64, "size": 1} for i in range(8000)
    }

    encoders = []

    def init(self, *a, **kw):  # the second entry staged (the kept one) is the large one
        real_init(self, *a, **kw)
        encoders.append(self)
        if len(encoders) == 2:
            self.members.update(many)

    real = os.rename

    def rename(src, dst, *a, **kw):
        if os.path.basename(str(dst)) == "_entry":
            raise OSError(errno.EXDEV, "cross-device link")
        return real(src, dst, *a, **kw)

    store, sub = refusing_submitter(tmp_path)
    with sub:
        monkeypatch.setattr(codecs.Encoder, "__init__", init)
        monkeypatch.setattr(os, "rename", rename)
        job, result = _occupy_and_commit(store, sub)
        monkeypatch.undo()
    kept = store.work_dir(job) / "_entry" / "entry.json"
    assert kept.stat().st_size > 1 << 20


def test_kept_entry_copy_that_changed_is_refused(tmp_path, monkeypatch):
    import pydra.engine.portable.store as mod

    real_copy = mod.copy_hashed

    def wrong(src, dst, name, mode):
        digest, size = real_copy(src, dst, name, mode)
        return ("0" * 64, size) if name == "context.json" else (digest, size)

    def rename(src, dst, *a, **kw):
        if os.path.basename(str(dst)) == "_entry":
            raise OSError(errno.EXDEV, "cross-device link")
        return real_rename(src, dst, *a, **kw)

    real_rename = os.rename
    store, sub = refusing_submitter(tmp_path)
    with sub:
        first = job_in(sub, _Inc(a=1), name="first")
        store.commit(_Lost(), _result(first, _Inc.Outputs(out=3)), first)
        job = job_in(sub, _Inc(a=1))
        result = _result(job, _Inc.Outputs(out=2))
        monkeypatch.setattr(os, "rename", rename)
        monkeypatch.setattr(mod, "copy_hashed", wrong)
        with pytest.raises(CorruptEntry):
            store.commit(_Lost(), result, job)
    assert not (store.work_dir(job) / "_entry").exists()


class _Opaque:
    pass


def _unencodable_commit(store, sub, marker):
    job = job_in(sub, _Inc(a=1))
    result = _result(job, {"out": _Marker(str(marker))})
    store.commit(_Lost(), result, job)  # no codec for it, and this root may not pickle
    return job, result


def test_unencodable_outputs_are_kept_as_outputs_pklz(tmp_path):
    marker = tmp_path / "marker"
    store, sub = portable_submitter(tmp_path)
    with sub:
        job, result = _unencodable_commit(store, sub, marker)
        wd = store.work_dir(job)
        assert result.stored is False and (wd / "_outputs.pklz").is_file()
        assert not results(tmp_path / "root") or not any(
            p.iterdir() for p in results(tmp_path / "root")
        )
        read = store.read_result(job)
    assert read.outputs.out == 7 and marker.read_text() == "unpickled"
    assert read.stored is False


@posix_only
def test_outputs_pklz_ownership_checked(tmp_path):
    marker = tmp_path / "marker"
    store, sub = portable_submitter(tmp_path)
    with sub:
        job, result = _unencodable_commit(store, sub, marker)
        wd = store.work_dir(job)
        os.chmod(wd, 0o770)  # a group can write the directory: its pickle is not opened
        with pytest.raises(EntryNotTrusted):
            store.read_committed(job, result.committed)
        refused = store.read_result(job)
        assert refused.errored and str(wd) in refused.failure
        assert not marker.exists()
        os.chmod(wd, 0o700)
        assert store.read_result(job).outputs.out == 7
    assert marker.exists()


# -- a resumable job's directory is stable across runs -------------------------------------------------


@python.define
def _Flaky(a: int, flag: str) -> ty.Any:
    if Path(flag).exists():
        raise RuntimeError("second run fails")
    return _Opaque()


def test_resume_stale_committed_json_not_read_after_failure(tmp_path):
    flag = tmp_path / "fail"
    store, sub = portable_submitter(tmp_path)
    store.visibility_timeout = 0.1
    with sub:
        first = job_in(sub, _Flaky(a=1, flag=str(flag)), resumable=True)
        wd = store.work_dir(first)
        assert wd.name.startswith("resume-") and first.attempt != "resume"
        first.run()
        committed = json.loads((wd / "_committed.json").read_text())
        assert (
            committed["stored"] is False
        )  # run 1's outputs were kept here, not stored
        flag.write_text("x")
        second = job_in(sub, _Flaky(a=1, flag=str(flag)), resumable=True)
        assert store.work_dir(second) == wd
        with pytest.raises(RuntimeError, match="second run fails"):
            second.run()
        assert not (
            wd / "_committed.json"
        ).exists()  # a run's start removes the last one's
        assert (wd / "_error.json").is_file()
        seen = job_in(sub, _Flaky(a=1, flag=str(flag)), resumable=True)
        got = seen.result(
            settled=True
        )  # another process's view: no job.committed, not errored
    assert got.errored is True and got.outputs is None


# -- failures -----------------------------------------------------------------------------------------


def test_errored_job_writes_error_json_not_entry(tmp_path):
    store, sub = portable_submitter(tmp_path)
    with sub:
        job = job_in(sub, _Boom(a=1))
        job.persist_pickle = False
        with pytest.raises(ValueError, match="boom"):
            job.run()
        wd = store.work_dir(job)
        assert job.error_file == wd / "_error.json"
        doc = json.loads((wd / "_error.json").read_text())
        assert "boom from the job" in "".join(doc["error message"])
        assert {"time of crash", "login name", "name with checksum"} <= set(doc)
        assert not (wd / "_error.pklz").exists()
        assert (wd / "_job.pklz").is_file()  # a failed job keeps its pickle
        failed = job.result()
    assert failed.errored and failed.errors["error message"] == doc["error message"]
    assert not any(p.iterdir() for p in results(tmp_path / "root"))
    assert staging_names(tmp_path / "root") == []


# -- the default work directories of the attempts a submitter hands out -------------------------------


@workflow.define
def _TwoIncs(a: int) -> int:
    first = workflow.add(_Inc(a=a), name="A")
    return workflow.add(_Inc(a=first.out), name="B").out


def test_the_emptied_work_directories_of_a_run_are_removed(tmp_path):
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        assert sub(_TwoIncs(a=1)).outputs.out == 3
    work = tmp_path / "root" / "work"
    assert not work.exists() or list(work.iterdir()) == []


def test_forgetting_an_outcome_keeps_the_work_directory(tmp_path):
    """Jobs on threads change the one process working directory, so one may restore it to
    another job's work directory; forgetting that job's outcome must not remove the
    directory under the process"""
    store = make(tmp_path)
    here = os.getcwd()
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(
            _TwoIncs(a=1), store, on_error="raise_at_end", submitter=sub
        )
        (a,) = runner.ready().dispatch
        result = a.run()
        wd = store.work_dir(a)
        try:
            os.chdir(wd)
            runner.complete(a, result)
            assert os.getcwd() and wd.is_dir()
            assert list(wd.iterdir()) == []
            sub.store.release_attempts()  # as on close: not the directory the process is in
            assert wd.is_dir()
        finally:
            os.chdir(here)
    assert not wd.exists()  # removed by the submitter's close, once nothing is in it


def test_forgetting_spares_a_directory_a_running_job_returns_to(tmp_path):
    from pydra.utils.general import hold_dir, release_dir

    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(
            _TwoIncs(a=1), store, on_error="raise_at_end", submitter=sub
        )
        (a,) = runner.ready().dispatch
        result = a.run()
        wd = store.work_dir(a)
        hold_dir(wd)
        try:
            runner.complete(a, result)
            assert wd.is_dir() and list(wd.iterdir()) == []
        finally:
            release_dir(wd)
        store.forget_committed(a)
        assert not wd.exists()


@python.define
def _BigBoom(n: int) -> int:
    raise RuntimeError("x" * n)


class _NoWait:
    """A retry clock that records each wait and moves on by it, never sleeping"""

    def __init__(self):
        self.waits = []

    def monotonic(self):
        return sum(self.waits)

    def sleep(self, seconds):
        self.waits.append(seconds)


def test_a_large_error_record_is_a_failure_read_at_once(tmp_path, monkeypatch):
    """However long the error message, a settled read names ``_error.json`` without waiting"""
    store, sub = portable_submitter(tmp_path)
    with sub:
        job = job_in(sub, _BigBoom(n=2 << 20))
        job.persist_pickle = False
        with pytest.raises(RuntimeError):
            job.run()
        assert (store.work_dir(job) / "_error.json").stat().st_size > 2 << 20
        clock = _NoWait()
        monkeypatch.setattr("pydra.utils.retry.time", clock)
        got = job.result(settled=True)
    assert clock.waits == []
    assert got.errored and "_error.json" in got.failure


def test_crash_cli_reads_error_json(tmp_path, cli_runner):
    store, sub = portable_submitter(tmp_path)
    with sub:
        job = job_in(sub, _Boom(a=1))
        with pytest.raises(ValueError):
            job.run()
        error_file = store.work_dir(job) / "_error.json"
    result = cli_runner(crash, [str(error_file)])
    assert result.exit_code == 0, result.output
    assert "boom from the job" in result.output


def test_a_never_handed_out_job_has_no_work_dir_and_fails_cleanly(tmp_path):
    store, sub = portable_submitter(tmp_path)
    with sub:
        job = Job(task=_Inc(a=1), submitter=sub, name="j")
        assert job.attempt is None
        with pytest.raises(ValueError, match="attempt"):
            job.cache_dir
        assert job.result() is None  # not run: nothing to find
        failed = job.result(settled=True)
    assert failed.errored and "no result" in failed.failure


# -- check_declared_code -------------------------------------------------------------------------------


def test_check_declared_code_reads_entry_path(tmp_path):
    store = make(tmp_path)
    context = {**CTX, "code": {"mod.f==1": "d1"}}
    c = publish(store, context=context)
    assert store.entry_path("python-abc", context) == entry_path(store.path, c)
    # an entry in the context's own slot: nothing to check, and no slot directory is asked for
    assert check_declared_code(store, "python-abc", context, None) is None
    # none stored: the store has no stored contexts to compare with
    assert not check_declared_code(store, "python-other", context, None)


class ServesAny:
    """A reuse hook with no ``accept``: it ranks every stored entry and serves it"""

    def bind(self, store):
        self._store = store

    def rank(self, current, stored, task):
        from pydra.engine.portable.reuse import Candidate

        return [Candidate(e, c, d, 1, "any") for e, c, d in stored]

    def verdict(self, candidate, current, task):
        from pydra.engine.portable.reuse import Verdict

        return Verdict.SERVE

    def served(self, candidate, current, task, entry):
        pass

    def compare(self, *args, **kwargs):
        return None


def test_check_declared_code_asks_the_reuse_hook(tmp_path):
    """The warning says what ``find`` would do: a hook that serves an entry of moved code
    makes it "reused", whether or not it has a ``ReusePolicy``'s ``accept``"""
    store = make(tmp_path, reuse=ServesAny())
    publish(store, context={**CTX, "code": {"mod.f==1": "d1"}})
    with pytest.warns(DeclaredVersionWarning, match="That result is reused"):
        check_declared_code(
            store, "python-abc", {**CTX2, "code": {"mod.f==1": "d2"}}, None
        )


# -- the de-duplication lease ----------------------------------------------------------------------------


@python.define
def _Count(a: int, log: str, pause: float) -> int:
    with open(log, "a") as f:
        f.write("ran\n")
    time.sleep(pause)
    return a + 1


def test_dedupe_lease_second_submitter_serves(tmp_path):
    log = tmp_path / "log"
    store, sub = portable_submitter(tmp_path, chdir=False)
    with sub:
        first = job_in(sub, _Count(a=1, log=str(log), pause=1.0))
        second = job_in(sub, _Count(a=1, log=str(log), pause=1.0))
        thread = threading.Thread(target=first.run)
        thread.start()
        deadline = time.monotonic() + 30
        while not log.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert log.exists()  # the first holds the lease and is computing
        served = second.run()  # waits for the lease, finds the entry, serves it
        thread.join()
        assert first.result().outputs.out == 2
    assert log.read_text() == "ran\n"
    assert served.outputs.out == 2 and second.committed.served is True
    assert store.dedupe is True


def test_no_dedupe_lease_is_always_held(tmp_path):
    store = make(tmp_path, dedupe=False)
    one = store.acquire("python-abc", CTX)
    two = store.acquire("python-abc", CTX)
    with one, two:  # both submitters compute, and one wins the rename
        assert one.prove() and two.prove()
    assert not (tmp_path / "root" / "leases").exists()
    # a resumable job is guarded whatever the store's default
    with store.acquire("python-abc", CTX, resumable=True) as lease:
        assert (tmp_path / "root" / "leases").is_dir() and lease.prove()


def test_anonymous_lease_holder_names_no_user(tmp_path):
    store = make(tmp_path)
    lease = store.acquire("python-abc", CTX)
    name = hashlib.sha256(
        f"{cdir_for('python-abc')}/{IDENT.fingerprint(CTX)}".encode()
    ).hexdigest()
    assert lease.lockfile == tmp_path / "root" / "leases" / f"{name}.lock"
    with lease:
        files = [p for p in lease.lockfile.rglob("holder.json")]
        assert len(files) == 1
        text = files[0].read_text()
        record = json.loads(text)
        assert set(record) == {"started", "token"}
        secrets = [getpass.getuser(), socket.gethostname(), str(os.getpid())]
        if hasattr(os, "getuid"):
            secrets.append(str(os.getuid()))
        for key, value in record.items():
            for secret in secrets:
                assert str(value) != secret, key
                if isinstance(value, str) and key != "token":
                    assert secret not in value, key
        assert isinstance(record["started"], float)
        assert isinstance(record["token"], str)
        assert "python-abc" not in "".join(
            p.name for p in lease.lockfile.parent.iterdir()
        )
    assert _describe({"pid": 1, "host": "h", "owner": "u"}, 2.0, True).startswith(
        "another process"
    )
    assert "pid 1" in _describe({"pid": 1, "host": "h"}, 2.0)
    with pytest.raises(ValueError, match="holder"):
        LeaseSettings(holder="nobody")


def test_a_symlinked_leases_directory_is_refused(tmp_path):
    if IS_WINDOWS:
        pytest.skip("needs symlinks")
    store = make(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "root" / "leases").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(OSError):
        store.acquire("python-abc", CTX)
    assert list((tmp_path / "elsewhere").iterdir()) == []


# -- through a worker ------------------------------------------------------------------------------------


def test_debug_cf_parity(any_worker, tmp_path):
    store = make(tmp_path)
    with Submitter(worker=any_worker, store=store, cache_root=tmp_path / "pc") as sub:
        job = job_in(sub, _Inc(a=1))
        sub.submit(job)
        got = job.result(settled=True)
        wd = store.work_dir(job)
    assert got.errored is False and got.outputs.out == 2 and got.stored is True
    # published under the execute context, which a worker process can make differ from the lookup
    # context, so the result is read by the EntryRef that publish returned, not by recomputing it
    ref = got.committed.entry
    assert Path(got.cache_dir).name == ref.fingerprint and Path(got.cache_dir).is_dir()
    assert Path(got.cache_dir).parent.name == ref.cdir == cdir_for(job.checksum)
    # emptied after the publish, except the record the runner removes once it holds the outcome
    assert sorted(p.name for p in wd.iterdir()) == ["_committed.json"]
    assert staging_names(tmp_path / "root") == []
    store.forget_committed(job)
    assert not wd.exists()


# -- collect ---------------------------------------------------------------------------------------------


def test_collect_removes_old_default_work_dirs_only(tmp_path, skew):  # noqa: F811
    store = aged_store(tmp_path, work=100)
    work = Path(store.path) / "work"

    def stub_dir(attempt, **kw):
        job = _stub(attempt, **kw)
        path = store.begin_run(job)
        (path / "_job.pklz").write_bytes(b"x")
        return path

    old = stub_dir("a" * 32)
    committed = stub_dir("b" * 32)
    (committed / "_committed.json").write_text("{}")
    young = stub_dir("c" * 32)
    resume = stub_dir(None, can_resume=True)
    foreign = work / "not a name"
    foreign.mkdir()
    for path in (old, resume):
        for p in path.iterdir():
            age(p, 200)
        age(path, 200)
    for p in committed.iterdir():  # old, but its record is young
        age(p, 10 if p.name == "_committed.json" else 200)
    age(committed, 200)
    for p in young.iterdir():
        age(p, 10)
    age(young, 10)
    report = store.collect()
    assert not old.exists() and not resume.exists()
    assert committed.exists() and young.exists() and foreign.exists()
    assert any(str(old) == r for r in report.removed)
    assert any(str(committed) == w and "younger" in why for w, why in report.kept)


def test_collect_spares_a_resume_dir_whose_lease_is_live(tmp_path, skew):  # noqa: F811
    """A resumable job's directory is old while its job runs (nothing is written at its top);
    the lease of the same hash, renewed by its holder, keeps it"""
    store = aged_store(tmp_path, work=100)
    # the skewed clock makes everything at least SKEW old; a lease is stale after more
    store._lease_settings = LeaseSettings(
        holder="anonymous", stale_after=10 * SKEW, timeout=20 * SKEW
    )
    job = _stub(None, can_resume=True)
    path = store.begin_run(job)
    (path / "_job.pklz").write_bytes(b"x")
    for p in (*path.iterdir(), path):
        age(p, 200)
    lease = store.acquire(job.checksum, job.context, resumable=True)
    assert lease.lockfile.name == path.name[len("resume-") :] + ".lock"
    with lease:
        report = store.collect()
        assert path.exists()
        assert any(str(path) == w and "lease" in why for w, why in report.kept)
    store.collect()
    assert not path.exists()


def test_collect_never_touches_a_caller_named_work_dir(tmp_path, skew):  # noqa: F811
    custom = tmp_path / "scratch"
    store = make(
        tmp_path,
        policy=StorePolicy(ages=StoreAges(work=100)),
        work_dir=lambda c, f, a: custom / a,
    )
    path = store.begin_run(_stub("d" * 32))
    assert path == custom / ("d" * 32) and stat.S_IMODE(path.stat().st_mode) == (
        0o700 if not IS_WINDOWS else stat.S_IMODE(path.stat().st_mode)
    )
    age(path, 200)
    store.collect()
    assert path.exists()


# -- the execute phase carries the lookup context --------------------------------------------------------


def test_second_run_serves_from_the_store(any_worker, tmp_path, monkeypatch):
    """A job run in a worker is published where the next submitter looks (RFC 0003 section 4),
    and the hit served to the second is recorded, so a settled read of it never waits.
    """
    log = tmp_path / "log"
    store = make(tmp_path)
    task = _Count(a=1, log=str(log), pause=0.0)
    with Submitter(worker=any_worker, store=store, cache_root=tmp_path / "pc") as sub:
        first = job_in(sub, task)
        sub.submit(first)
    with Submitter(worker=any_worker, store=store, cache_root=tmp_path / "pc2") as sub:
        second = job_in(sub, _Count(a=1, log=str(log), pause=0.0))
        sub.submit(second)
        waits = []
        clock = [0.0]

        class RetryTime:
            def __getattr__(self, name):
                return getattr(time, name)

            def monotonic(self):
                return clock[0]

            def sleep(self, seconds):
                waits.append(seconds)
                clock[0] += seconds

        monkeypatch.setattr("pydra.utils.retry.time", RetryTime())
        got = second.result(settled=True)
    assert waits == []
    assert log.read_text() == "ran\n"  # computed once
    assert got.outputs.out == 2 and not got.errored
    fingerprint = IDENT.fingerprint(first.context)
    assert [
        p.name for p in store.entry_path(first.checksum, first.context).parent.iterdir()
    ] == [fingerprint]


def test_execute_phase_returns_the_carried_context(tmp_path, monkeypatch):
    """A context that a worker process recomputes (an ``unknown`` nonce for a module it cannot
    attribute) never splits the slot: ``@1`` returns the context the job carries."""
    import pydra.engine.portable.identity as mod

    store, sub = portable_submitter(tmp_path)
    with sub:
        job = job_in(sub, _Inc(a=1))
        carried = job.context
        monkeypatch.setattr(
            mod, "job_context", lambda *a, **k: {"unknown": uuid.uuid4().hex}
        )
        assert IDENT.context(job, "execute") == carried
        assert IDENT.context(job, "lookup") != carried  # lookup still builds it


# -- provenance survives the emptying of the work directory ----------------------------------------------


def test_audit_messages_and_runtime_survive_a_stored_publish(tmp_path):
    from pydra.utils.messenger import AuditFlag, FileMessenger

    def audited(store, root):
        kw = dict(
            audit_flags=AuditFlag.ALL, messengers=FileMessenger(), cache_root=root
        )
        with Submitter(worker="debug", store=store, **kw) as sub:
            job = Job(task=_Inc(a=1), submitter=sub, name="j")
            job.attempt = uuid.uuid4().hex
            return job, job.run()

    from pydra.engine.store import PickleStore

    pjob, presult = audited(PickleStore(tmp_path / "pickle"), tmp_path / "pickle")
    expected = len(list(pjob.cache_dir.rglob("*.jsonld")))
    assert expected > 0 and presult.runtime is not None
    store = make(tmp_path)
    job, result = audited(store, tmp_path / "pc")
    assert result.runtime is not None and result.stored is True
    assert sorted(p.name for p in store.work_dir(job).iterdir()) == ["_committed.json"]
    # found from the entry's address and the attempt, under the root, and private
    ref = result.committed.entry
    record = Path(store.path) / "audit" / ref.cdir / ref.fingerprint / job.attempt
    assert record == job.record_dir and record.is_dir()
    assert (
        len(list(record.rglob("*.jsonld"))) == expected + 1
    )  # the publication message
    assert list(record.glob("proc-*.log"))  # the resource monitor's log
    if not IS_WINDOWS:
        for d in (record, record.parent, record.parent.parent, record / "messages"):
            assert stat.S_IMODE(d.stat().st_mode) & 0o077 == 0, d
    published = [
        json.loads(p.read_text())
        for p in record.rglob("*.jsonld")
        if "publication" in p.read_text()
    ]
    (message,) = published
    assert message["checksum"] == job.checksum and message["attempt"] == job.attempt
    assert message["entry"]["fingerprint"] == ref.fingerprint
    # evicting the entry removes its provenance with it
    store.evict(ref, result.committed.own_digest)
    assert not record.exists() and not record.parent.exists()
    other = Path(store.path) / "audit" / ref.cdir
    assert not other.exists() or not any(other.iterdir())
