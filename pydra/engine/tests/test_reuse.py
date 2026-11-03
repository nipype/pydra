import multiprocessing
import os
import threading
import platform
import sys
import typing as ty
import uuid
from pathlib import Path
import attrs
import cloudpickle
import pytest
from pydra.compose import python, shell, workflow
from pydra.engine.job import Job
from pydra.engine.result import Result, load_result
from pydra.engine.store import AnyReuse, ExactReuse, PickleStore, context_key
from pydra.engine.submitter import Submitter
from pydra.environments import native
from pydra.utils.hash import hash_function

# Worker processes cannot import the test modules, so they receive their classes by value
cloudpickle.register_pickle_by_value(sys.modules[__name__])


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
    assert job.store.lookup(job.checksum, job.context, job.task).context == job.context


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
    first = _job(tmp_path, Tagged(tag="a"))
    first.run()
    assert not _job(tmp_path, Tagged(tag="a"), reuse="any").result().cross_context

    other = _job(tmp_path, Tagged(tag="b"), reuse="any")
    result = other.run()
    assert result.cross_context
    assert result.environment_checksum == Tagged(tag="a").checksum()
    assert result.outputs.out == 2
    stored = load_result(f"{first.checksum}/{context_key(first.context)}", [tmp_path])
    assert stored.environment_checksum == Tagged(tag="a").checksum()
    assert not stored.cross_context


def _store_result(store, checksum, context):
    with store.acquire(checksum, context) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir, **context))


def test_contexts_keep_separate_slots(tmp_path):
    store = PickleStore(tmp_path)
    a = {"environment_checksum": "a", "platform": "p", "python_version": "v"}
    b = {**a, "environment_checksum": "b"}
    _store_result(store, "abc", a)
    _store_result(store, "abc", b)
    assert store.cache_dir("abc", a) != store.cache_dir("abc", b)
    assert store.lookup("abc", a, None).context == a
    assert store.lookup("abc", b, None).context == b
    assert store.lookup("abc", {**a, "platform": "q"}, None) is None


def test_any_prefers_exact_match_in_later_location(tmp_path):
    primary, ro = tmp_path / "primary", tmp_path / "ro"
    a = {"environment_checksum": "a", "platform": "p", "python_version": "v"}
    b = {**a, "environment_checksum": "b"}
    _store_result(PickleStore(primary), "abc", b)
    _store_result(PickleStore(ro), "abc", a)
    store = PickleStore(primary, [ro], reuse="any")
    exact = store.lookup("abc", a, None)
    assert exact.cache_dir == ro / "abc" / context_key(a)
    assert not exact.cross_context
    other = store.lookup("abc", {**a, "environment_checksum": "c"}, None)
    assert other.cross_context


def test_reuse_argument(tmp_path):
    assert isinstance(Submitter(cache_root=tmp_path).store.reuse, ExactReuse)
    assert isinstance(Submitter(cache_root=tmp_path, reuse="any").store.reuse, AnyReuse)
    policy = Equivalent([])
    assert Submitter(cache_root=tmp_path, reuse=policy).store.reuse is policy
    for bad in ("nearest", object()):
        with pytest.raises(ValueError, match="reuse must be one of"):
            Submitter(cache_root=tmp_path, reuse=bad)
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


def _record_run(log):
    """One file per run, since appends from concurrent processes to one file can be
    lost on Windows"""
    Path(log).mkdir(exist_ok=True)
    (Path(log) / uuid.uuid4().hex).touch()


@python.define
def Logged(a: int, log: str) -> int:
    _record_run(log)
    return a + 1


def _runs(log):
    return len(list(Path(log).iterdir()))


def test_worker_reuse_is_judged_by_the_recorded_context(tmp_path):
    """A job run in a worker records the context it carried from the submitting
    process, which the next submission is compared with, so it is reused"""
    log = tmp_path / "log"
    for _ in range(2):
        with Submitter(
            worker="cf", cache_root=tmp_path / "cache", environment=InWorker()
        ) as sub:
            result = sub(Logged(a=1, log=str(log)))
        assert result.outputs.out == 2
        assert result.environment_checksum == InWorker().checksum()
    assert _runs(log) == 1


def test_other_context_keeps_the_first_result(worker, tmp_path):
    log = tmp_path / "log"
    for tag in ("a", "b", "a", "b"):
        with Submitter(
            worker=worker, cache_root=tmp_path / "cache", environment=Tagged(tag=tag)
        ) as sub:
            result = sub(Logged(a=1, log=str(log)))
        assert result.environment_checksum == Tagged(tag=tag).checksum()
    assert _runs(log) == 2


def test_any_reuses_the_other_context_under_each_worker(worker, tmp_path):
    log = tmp_path / "log"
    results = []
    for tag in ("a", "b"):
        with Submitter(
            worker=worker,
            cache_root=tmp_path / "cache",
            environment=Tagged(tag=tag),
            reuse="any",
        ) as sub:
            results.append(sub(Logged(a=1, log=str(log))))
    assert [r.cross_context for r in results] == [False, True]
    assert results[1].environment_checksum == Tagged(tag="a").checksum()
    assert _runs(log) == 1


@pytest.mark.timeout(120)
def test_concurrent_submissions_run_a_job_once(tmp_path):
    """Two submitters, each with its own cf process, start the same job at once on one
    cache: the lock makes one of them compute it and the other reuse the result"""
    log = tmp_path / "log"
    start = threading.Barrier(2)
    results, errors = [], []

    def submit():
        try:
            with Submitter(
                worker="cf", n_procs=1, cache_root=tmp_path / "cache"
            ) as sub:
                start.wait(30)
                results.append(sub(Logged(a=1, log=str(log)), raise_errors=True))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(100)
    assert not errors
    assert [r.outputs.out for r in results] == [2, 2]
    assert _runs(log) == 1


def test_combined_lookup_carries_the_job_context(tmp_path):
    seen = []

    class Spy(PickleStore):
        def lookup(self, checksum, context, task):
            seen.append(context)
            return super().lookup(checksum, context, task)

    task = AddOne().split(a=[1, 2]).combine("a")
    with Submitter(cache_root=tmp_path, store=Spy(tmp_path)) as sub:
        assert sub(task).outputs.out == [2, 3]
    with Submitter(cache_root=tmp_path, store=Spy(tmp_path)) as sub:
        assert sub(task).outputs.out == [2, 3]
    assert seen and None not in seen


@pytest.mark.parametrize("local", [False, True])
def test_version_of_ambiguous_function_warns(local):
    def inner(a: int) -> int:
        return a

    inner.__module__ = "pydra.engine.tests.test_reuse"
    if not local:
        inner.__qualname__ = "inner"
        inner.__module__ = "__main__"
    with pytest.warns(UserWarning, match="version='1'"):
        python.define(inner, version="1")


def _key(task, inputs):
    return f"key-{inputs['a']}"


@python.define(checksum=_key)
def KeyedAddOne(a: int) -> int:
    return a + 1


@python.define
class HookedAddOne(python.Task["HookedAddOne.Outputs"]):
    a: int

    @staticmethod
    def function(a: int) -> int:
        return a + 1

    def __checksum__(self, inputs):
        return f"hooked-{inputs['a']}"

    class Outputs(python.Outputs):
        out: int


class Unhashable:
    def __bytes_repr__(self, cache):
        raise TypeError("Unhashable cannot be hashed")


@python.define(checksum=_key)
def KeyedLogged(a: int, log: str, m: ty.Any = None) -> int:
    _record_run(log)
    return a + 1


@pytest.mark.parametrize(
    "task, checksum",
    [(KeyedAddOne(a=1), "key-1"), (HookedAddOne(a=1), "hooked-1")],
)
def test_supplied_checksum_is_job_checksum(worker, tmp_path, task, checksum):
    job = Job(task=task, name="j", submitter=Submitter(cache_root=tmp_path))
    assert job.checksum == checksum
    assert task._hashes is None
    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        result = sub(task)
    assert result.outputs.out == 2
    assert result.cache_dir.parent == tmp_path / checksum


def test_checksum_hook_receives_the_inputs():
    seen = []

    def key(task, inputs):
        seen.append((task, inputs))
        return "k"

    task = python.define(inc_one, checksum=key)(a=3)
    assert task._checksum == "k"
    assert seen[0][0] is task
    assert seen[0][1]["a"] == 3


def test_checksum_hook_must_return_str():
    task = python.define(inc_one, checksum=lambda task, inputs: 1)(a=1)
    with pytest.raises(TypeError, match="must return a str"):
        task._checksum


def test_supplied_checksum_per_split_element(worker, tmp_path):
    log = tmp_path / "log"
    for _ in range(2):
        with Submitter(worker=worker, cache_root=tmp_path / "cache") as sub:
            result = sub(KeyedLogged(log=str(log), m=Unhashable()).split(a=[1, 2]))
        assert result.outputs.out == [2, 3]
    assert {p.name for p in (tmp_path / "cache").iterdir() if p.is_dir()} >= {
        "key-1",
        "key-2",
    }
    assert _runs(log) == 2


@workflow.define
def GoldenWf(a: int) -> int:
    return workflow.add(AddOne(a=a)).out


def test_tasks_without_hook_keep_checksums(monkeypatch):
    """Checksums of tasks without the hook, as computed at 277bee99"""
    # the module name depends on pytest's import mode, and is part of a declared version
    monkeypatch.setattr(inc_one, "__module__", "pydra.engine.tests.test_reuse")
    for task, checksum in [
        (
            python.define(inc_one, version="1")(a=1),
            "python-e87e7040c1d7894a31506139707e9425",
        ),
        (
            shell.define("echo <text:str>")(text="hi"),
            "shell-e04edfe0da3d1efc0d8eb9466535b1fe",
        ),
    ]:
        assert not hasattr(task, "__checksum__")
        assert task._checksum == checksum


def test_code_hashed_tasks_keep_checksums():
    """Checksums of tasks whose function's code is hashed, the same on every Python
    version (measured on 3.11 to 3.15)"""
    assert (AddOne(a=1)._checksum, GoldenWf(a=1)._checksum) == (
        "python-463063b196ab62ce6c8c1a59e2bdd535",
        "workflow-699b3cc655c8fb48e7dd8a66863c28e8",
    )


def _add_k(k):
    def add_k(a: int) -> int:
        return a + k

    return python.define(add_k)


def test_closure_variants_keep_their_own_results(worker, tmp_path):
    """Tasks made by one factory with different captured values do not share a result"""
    for k in (1, 2):
        with Submitter(worker=worker, cache_root=tmp_path) as sub:
            result = sub(_add_k(k)(a=1))
        assert result.outputs.out == 1 + k


@attrs.define
class Equivalent:
    """Accepts results whose environment is in the same class as the current one"""

    classes: list[set[str]]
    calls: list = attrs.field(factory=list)

    def accept(self, stored, current, task):
        self.calls.append((dict(stored), dict(current), task))
        return any(
            {stored["environment_checksum"], current["environment_checksum"]} <= c
            for c in self.classes
        )


def test_reuse_hook_marks_stored_context(worker, tmp_path):
    log = tmp_path / "log"
    a, b, c = (Tagged(tag=t).checksum() for t in "abc")
    policy = Equivalent([{a, b}])
    results = []
    for tag in "abc":
        with Submitter(
            worker=worker,
            cache_root=tmp_path / "cache",
            environment=Tagged(tag=tag),
            reuse=policy,
        ) as sub:
            results.append(sub(Logged(a=1, log=str(log))))
    assert [r.cross_context for r in results] == [False, True, False]
    assert results[1].context == results[0].context
    assert [r.environment_checksum for r in results] == [a, a, c]
    assert _runs(log) == 2


def test_reuse_policy_receives_contexts_and_task(tmp_path):
    a = {"environment_checksum": "a", "platform": "p", "python_version": "v"}
    b = {**a, "environment_checksum": "b"}
    _store_result(PickleStore(tmp_path), "abc", a)
    policy = Equivalent([])
    store = PickleStore(tmp_path, reuse=policy)
    task = AddOne(a=1)
    assert store.lookup("abc", b, task) is None
    assert policy.calls == [(a, b, task)]
    policy.classes.append({"a", "b"})
    result = store.lookup("abc", b, task)
    assert result.cross_context and result.context == a


@pytest.mark.parametrize("checksum", ["", ".", "..", "a/b", "/abs", "a\\b", "a\0b"])
def test_supplied_checksum_must_be_one_path_component(checksum):
    task = python.define(inc_one, checksum=lambda task, inputs: checksum)(a=1)
    with pytest.raises(ValueError, match="inc_one.*single path component"):
        task._checksum


def test_supplied_checksum_may_name_a_directory(tmp_path):
    task = python.define(inc_one, checksum=lambda task, inputs: "key_1.v-2")(a=1)
    with Submitter(cache_root=tmp_path) as sub:
        assert sub(task).cache_dir.parent == tmp_path / "key_1.v-2"


def _file_key(task, inputs):
    with open(inputs["path"]) as f:
        return f"file-{len(f.read())}"


@python.define(checksum=_file_key)
def Appends(path: str) -> int:
    with open(path, "a") as f:
        f.write("x")
    return 0


def test_supplied_checksum_change_during_run_raises(tmp_path):
    path = tmp_path / "data"
    path.write_text("x")
    job = Job(
        task=Appends(path=str(path)),
        name="j",
        submitter=Submitter(cache_root=tmp_path / "cache"),
    )
    with pytest.raises(RuntimeError, match="changed during its execution"):
        job.run()


class Lenient(ExactReuse):
    def accept(self, stored, current, task):
        return True


def test_policy_subclass_is_consulted(tmp_path):
    a = {"environment_checksum": "a", "platform": "p", "python_version": "v"}
    _store_result(PickleStore(tmp_path), "abc", a)
    store = PickleStore(tmp_path, reuse=Lenient())
    assert store.lookup("abc", {**a, "environment_checksum": "b"}, None).cross_context


@python.define(checksum=lambda task, inputs: f"k-{inputs['a']:03d}")
def IntKeyed(a: int) -> int:
    return a + 1


def test_split_hook_sees_only_elements(worker, tmp_path):
    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        result = sub(IntKeyed().split(a=[1, 2]))
    assert result.outputs.out == [2, 3]
    assert {"k-001", "k-002"} <= {p.name for p in tmp_path.iterdir()}


@python.define(checksum=lambda task, inputs: f"k-{inputs['a']}-{inputs['b']}")
def PairKeyed(a: int, b: int) -> int:
    return a + b


@pytest.mark.parametrize(
    "split",
    [
        lambda t: t.split(a=[1, 2], b=[3, 4]),
        lambda t: t.split(("a", "b"), a=[1, 2], b=[3, 4]),
        lambda t: t.split(a=[1, 2], b=[3, 4]).combine("a"),
        lambda t: t.split(["a", "b"], a=[1, 2], b=[3, 4]).combine("b"),
    ],
)
def test_element_checksums_match_the_jobs_run(worker, tmp_path, split):
    task = split(PairKeyed())
    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        sub(task)
    run = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("k-"))
    assert sorted(task._element_checksums()) == run
