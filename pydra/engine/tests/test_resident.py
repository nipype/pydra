"""Resident keys, providers and the ResidentSet (RFC 0005 section 3.1)"""

import hashlib
import json
import os
import sys
import typing as ty
import cloudpickle as cp
import pytest
from pydra.compose import python
from pydra.engine.submitter import Submitter
from pydra.engine.resident import (
    Device,
    Footprint,
    Provided,
    provided_fields,
    ResidentKey,
    ResidentSet,
    UnregisteredProvider,
    residency_digest,
)
from pydra.engine.job import Job
from pydra.utils.digest import digest

# the test module is not importable by name in a worker process: send it by value
cp.register_pickle_by_value(sys.modules[__name__])

LOADS: list[str] = []
CLOSES: list[str] = []


class ToyModel:
    def __init__(self, spec):
        self.spec = spec
        self.pid = os.getpid()
        self.token = os.urandom(8).hex()


class ToyProvider:
    name = "toy"

    def load(self, spec, device):
        LOADS.append(spec["id"])
        return ToyModel(spec)

    def close(self, obj):
        CLOSES.append(obj.spec["id"])

    def footprint(self, spec):
        return Footprint(host=spec.get("bytes", 1))

    def measure(self, obj):
        return None

    def version(self):
        return "1"


@python.define
def Embed(x: int, model: Provided[dict, "toy"]) -> tuple[int, str, int]:
    return x, model.token, model.pid


@python.define
def Plain(x: int) -> int:
    return x + 1


@pytest.fixture(autouse=True)
def _reset():
    LOADS.clear()
    CLOSES.clear()


def test_provided_spec_keyed_object_not(tmp_path):
    spec = {"id": "m1"}
    task = Embed(x=1, model=spec)
    before = task._checksum
    tokens = set()
    for n in range(2):
        with Submitter(cache_root=tmp_path / str(n), providers=[ToyProvider()]) as sub:
            res = sub(Embed(x=1, model=spec))
        tokens.add(res.outputs.out[1])
    assert len(tokens) == 2  # two different loaded objects...
    assert Embed(x=1, model=spec)._checksum == before  # ...one identity
    other = Embed(x=1, model={"id": "m2"})
    assert other._checksum != before  # the spec is what is hashed
    assert provided_fields(task) == [("model", "toy")]


def test_residency_and_identity_share_one_digest():
    spec = {"id": "m1", "n": 3}
    key = ResidentKey("provider", "toy", digest(spec))
    assert key.spec == digest(spec)
    canonical = json.dumps(
        {"kind": "provider", "name": "toy", "spec": digest(spec)},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert key.digest() == hashlib.sha256(canonical.encode()).hexdigest()
    task = Embed(x=1, model=spec)
    job = Job(task, submitter=Submitter(providers=[ToyProvider()]), name="e")
    expected = hashlib.sha256(
        json.dumps([key.digest()], separators=(",", ":")).encode()
    ).hexdigest()
    assert residency_digest(job) == expected


def test_no_residency_without_provided_fields():
    job = Job(Plain(x=1), submitter=Submitter(), name="p")
    assert residency_digest(job) is None


def test_plain_task_creates_no_resident_set(tmp_path):
    with Submitter(cache_root=tmp_path) as sub:
        sub(Plain(x=1))
        assert sub._resident_set is None


def _key(n):
    return ResidentKey("provider", "toy", f"json@1:{n}")


def test_eviction_order_lru():
    closed = []
    rs = ResidentSet(host_budget=2, device_budget=None)

    def load(n):
        return lambda: n

    kw = dict(footprint=Footprint(host=1))
    for n in "abc":
        rs.get(_key(n), load(n), close=closed.append, **kw)
        rs.release(_key(n))
        if n == "b":
            rs.get(_key("a"), load("a"), close=closed.append, **kw)  # touch a
            rs.release(_key("a"))
    assert closed == ["b"]  # a was touched after b, so b is least recent
    assert rs.keys() == [_key("a"), _key("c")]


def test_busy_resident_survives_tighter_budget():
    closed = []
    rs = ResidentSet(host_budget=1, device_budget=None)
    kw = dict(footprint=Footprint(host=1), close=closed.append)
    rs.get(_key("a"), lambda: "a", **kw)  # busy: not released
    rs.get(_key("b"), lambda: "b", **kw)
    assert closed == []  # over budget, nothing evictable
    assert set(rs.keys()) == {_key("a"), _key("b")}
    rs.release(_key("a"))
    rs.release(_key("b"))
    rs.get(_key("c"), lambda: "c", **kw)
    assert sorted(closed) == ["a", "b"]  # idle ones go once room is needed


def test_device_budget_evicts():
    closed = []
    rs = ResidentSet(host_budget=None, device_budget=10)
    for n, size in (("a", 6), ("b", 6)):
        rs.get(
            _key(n),
            lambda n=n: n,
            footprint=Footprint(device=size),
            close=closed.append,
        )
        rs.release(_key(n))
    assert closed == ["a"]


def test_debug_loads_once_across_two_calls(tmp_path):
    spec = {"id": "m1"}
    with Submitter(cache_root=tmp_path, providers=[ToyProvider()]) as sub:
        r1 = sub(Embed(x=1, model=spec))
        r2 = sub(Embed(x=2, model=spec))
        assert sub._resident_set is not None
    assert LOADS == ["m1"]
    assert r1.outputs.out[1] == r2.outputs.out[1]


def test_unregistered_provider_is_wiring_error(tmp_path):
    with Submitter(cache_root=tmp_path) as sub:
        with pytest.raises(Exception) as ei:
            sub(Embed(x=1, model={"id": "m1"}))
    chain = []
    e: BaseException | None = ei.value
    while e is not None:
        chain.append(e)
        e = e.__cause__ or e.__context__
    assert any(isinstance(e, UnregisteredProvider) for e in chain), chain


def test_provided_on_any_worker(any_worker, tmp_path):
    with Submitter(
        cache_root=tmp_path, worker=any_worker, providers=[ToyProvider()]
    ) as sub:
        res = sub(Embed(x=5, model={"id": "m9"}))
    x, token, pid = res.outputs.out
    assert x == 5 and token
    if any_worker == "debug":
        assert pid == os.getpid()
    else:
        assert pid != os.getpid()


def test_field_provided_spelling():
    f = python.arg(provided="toy")
    assert f.provided == "toy"
    assert python.arg().provided is None
