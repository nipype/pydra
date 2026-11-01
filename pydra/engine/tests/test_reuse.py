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


@attrs.define
class Tagged(native.Native):
    """A native environment whose identity is set by ``tag``."""

    tag: str = ""


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
    # inc_two stands for an edit of inc_one: same module and name, different code
    monkeypatch.setattr(inc_two, "__qualname__", inc_one.__qualname__)
    v1 = python.define(inc_one, version="1", name="Inc")
    v1_edited = python.define(inc_two, version="1", name="Inc")
    v2 = python.define(inc_two, version="2", name="Inc")
    assert v1(a=1)._checksum == v1_edited(a=1)._checksum
    assert v1(a=1)._checksum != v2(a=1)._checksum
    assert v1(a=1)._checksum != v1(a=2)._checksum


def test_unversioned_checksum_unchanged():
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
        "python_version": platform.python_version(),
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
