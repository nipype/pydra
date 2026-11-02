import multiprocessing
import os
import platform
import sys
import attrs
import pytest
from pydra.compose import python
from pydra.engine.job import Job
from pydra.engine.result import Result, load_result
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.environments import native
from pydra.utils.hash import hash_function


@attrs.define
class Tagged(native.Native):
    """A native environment whose identity is set by ``tag``."""

    tag: str = ""


class PerProcess(native.Native):
    """A native environment whose identity differs in every process."""

    def _identity(self):
        return os.getpid()


class InWorker(native.Native):
    """A native environment whose identity tells a worker process from the main one."""

    def _identity(self):
        return multiprocessing.parent_process() is not None


@python.define
def AddOne(a: int) -> int:
    return a + 1


def inc_one(a: int) -> int:
    return a + 1


def inc_two(a: int) -> int:
    return a + 2


def _job(cache_root, env, reuse=None):
    return Job(
        task=AddOne(a=1),
        name="j",
        submitter=Submitter(cache_root=cache_root, environment=env, reuse=reuse),
    )


def test_version_replaces_code_hash(monkeypatch):
    # Unversioned, the two functions' code makes their checksums differ
    assert (
        python.define(inc_one, name="Inc")(a=1)._checksum
        != python.define(inc_two, name="Inc")(a=1)._checksum
    )
    # inc_two stands for an edit of inc_one: same module and name, different code
    monkeypatch.setattr(inc_two, "__qualname__", inc_one.__qualname__)
    v1 = python.define(inc_one, version="1", name="Inc")
    v1_edited = python.define(inc_two, version="1", name="Inc")
    v2 = python.define(inc_two, version="2", name="Inc")
    assert v1(a=1)._checksum == v1_edited(a=1)._checksum
    assert v1(a=1)._checksum != v2(a=1)._checksum
    assert v1(a=1)._checksum != v1(a=2)._checksum
    # A versioned task passed as a value hashes by its declared version too
    assert hash_function(v1(a=1)) == hash_function(v1_edited(a=1))
    assert hash_function(v1(a=1)) != hash_function(v2(a=1))


def test_adding_version_changes_checksum():
    assert python.define(inc_one)._version is None
    assert (
        python.define(inc_one)(a=1)._checksum
        != python.define(inc_one, version="1")(a=1)._checksum
    )


def test_context_recorded(tmp_path):
    job = _job(tmp_path, Tagged(tag="a"))
    result = job.run()
    assert job.context == {
        "environment_checksum": Tagged(tag="a").checksum(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "python_version": ".".join(platform.python_version_tuple()[:2]),
    }
    assert result.context == job.context
    assert load_result(job.checksum, [tmp_path]).context == job.context


def test_exact_reuses_same_context_only(tmp_path):
    _job(tmp_path, Tagged(tag="a")).run()
    same = _job(tmp_path, Tagged(tag="a")).result()
    assert same.environment_checksum == Tagged(tag="a").checksum()
    assert not same.cross_context

    other = _job(tmp_path, Tagged(tag="b"))
    assert other.checksum == _job(tmp_path, Tagged(tag="a")).checksum
    assert other.result() is None
    result = other.run()
    assert result.environment_checksum == Tagged(tag="b").checksum()
    assert not result.cross_context


def test_any_reuses_and_marks_other_context(tmp_path):
    _job(tmp_path, Tagged(tag="a")).run()
    assert not _job(tmp_path, Tagged(tag="a"), reuse="any").result().cross_context

    other = _job(tmp_path, Tagged(tag="b"), reuse="any")
    result = other.run()
    assert result.cross_context
    assert result.environment_checksum == Tagged(tag="a").checksum()
    assert result.outputs.out == 2
    stored = load_result(other.checksum, [tmp_path])
    assert stored.environment_checksum == Tagged(tag="a").checksum()
    assert not stored.cross_context


def test_lookup_entry_without_context(tmp_path):
    with PickleStore(tmp_path).acquire("abc") as lease:
        PickleStore(tmp_path).commit(lease, Result(cache_dir=lease.cache_dir))
    context = {"environment_checksum": "e", "platform": "p", "python_version": "v"}
    assert PickleStore(tmp_path).lookup("abc") is not None
    assert PickleStore(tmp_path).lookup("abc", context) is None
    assert PickleStore(tmp_path, reuse="any").lookup("abc", context).cross_context


def test_lookup_skips_rejected_location(tmp_path):
    primary, ro = tmp_path / "primary", tmp_path / "ro"
    context = {"environment_checksum": "e", "platform": "p", "python_version": "v"}
    for root, recorded in (
        (primary, {**context, "environment_checksum": "old"}),
        (ro, context),
    ):
        with PickleStore(root).acquire("abc") as lease:
            PickleStore(root).commit(
                lease, Result(cache_dir=lease.cache_dir, **recorded)
            )
    store = PickleStore(primary, [ro])
    assert store.lookup("abc", context).cache_dir == ro / "abc"
    assert store.lookup("abc").cache_dir == primary / "abc"


def test_lookup_accepts_own_result(tmp_path):
    with PickleStore(tmp_path).acquire("abc") as lease:
        PickleStore(tmp_path).commit(
            lease, Result(cache_dir=lease.cache_dir, job_uid="u")
        )
    context = {"environment_checksum": "e", "platform": "p", "python_version": "v"}
    result = PickleStore(tmp_path).lookup("abc", context, produced_by="u")
    assert result.job_uid == "u"
    assert not result.cross_context
    assert PickleStore(tmp_path).lookup("abc", context, produced_by="v") is None


def test_reuse_argument(tmp_path):
    assert Submitter(cache_root=tmp_path).store.reuse == "exact"
    assert Submitter(cache_root=tmp_path, reuse="any").store.reuse == "any"
    with pytest.raises(ValueError, match="reuse must be one of"):
        Submitter(cache_root=tmp_path, reuse="nearest")
    with pytest.raises(ValueError, match="'reuse' applies to the default store"):
        Submitter(cache_root=tmp_path, store=PickleStore(tmp_path), reuse="any")


def test_exact_reruns_in_other_context(worker, tmp_path):
    for tag in ("a", "b"):
        with Submitter(
            worker=worker, cache_root=tmp_path, environment=Tagged(tag=tag)
        ) as sub:
            result = sub(AddOne(a=1))
        assert result.outputs.out == 2
        assert result.environment_checksum == Tagged(tag=tag).checksum()


def test_own_result_accepted_across_contexts(tmp_path):
    """The worker process records a different context from the one the submitting
    process computes, yet the job's own result is accepted, and a later
    submission still judges it by context"""
    with Submitter(worker="cf", cache_root=tmp_path, environment=PerProcess()) as sub:
        first = sub(AddOne(a=1))
    assert first.outputs.out == 2
    assert first.environment_checksum != PerProcess().checksum()
    with Submitter(worker="cf", cache_root=tmp_path, environment=PerProcess()) as sub:
        second = sub(AddOne(a=1))
    assert second.outputs.out == 2
    assert second.job_uid != first.job_uid


def test_worker_judges_reuse_with_submitting_context(tmp_path):
    """An entry written by a worker matches the next worker's context but not the
    submitting process's, so the job reruns rather than never reading as done"""
    results = []
    for _ in range(2):
        with Submitter(worker="cf", cache_root=tmp_path, environment=InWorker()) as sub:
            results.append(sub(AddOne(a=1)))
    first, second = results
    assert first.environment_checksum != InWorker().checksum()
    assert second.environment_checksum == first.environment_checksum
    assert second.outputs.out == 2
    assert second.job_uid != first.job_uid


def test_worker_reuses_entry_accepted_by_submitting_context(tmp_path):
    """An entry matching the submitting process's context is reused in the worker,
    although the worker's own context differs, and the parent accepts it"""
    with Submitter(worker="debug", cache_root=tmp_path, environment=InWorker()) as sub:
        first = sub(AddOne(a=1))
    assert first.environment_checksum == InWorker().checksum()
    with Submitter(worker="cf", cache_root=tmp_path, environment=InWorker()) as sub:
        second = sub(AddOne(a=1))
    assert second.outputs.out == 2
    assert second.job_uid == first.job_uid
