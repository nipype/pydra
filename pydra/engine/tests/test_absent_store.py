"""``Absent`` under a `PortableStore` (RFC 0002 section 2.3(a), R10a; RFC 0003 section 6.4): the
``$absent`` codec, the suffixed checksum a work directory is named by, and the jobs that complete
absent or gated off, which the store never sees."""

import json
from pathlib import Path

import pytest

from pydra.compose import workflow
from pydra.engine.absent import Absent
from pydra.engine.portable.store import PortableStore
from pydra.engine.submitter import Submitter

from .test_absent import Chain, Keyed, Maybe
from .test_gates import Fanout
from .test_portable_submitter import nonpickling
from .test_portable_store import make_v2 as make


def entry_docs(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in root.glob("results@1/*/*/*/entry.json")]


class Spy:
    """Records the job, or the checksum, of every `find`, `read_succeeded` and `publish` a
    `PortableStore` is asked for, and every checksum a work directory is named by."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, str]] = []
        self.found: list[str] = []  # the name of every job `find` is asked for
        real_find = PortableStore.find
        real_read = PortableStore.read_succeeded
        real_publish = PortableStore.publish
        real_work = PortableStore._work_path
        spy = self

        def find(self, job, **kw):
            spy.calls.append(("find", job.checksum))
            spy.found.append(job.name)
            return real_find(self, job, **kw)

        def read_succeeded(self, checksum, *args, **kw):
            spy.calls.append(("read_succeeded", checksum))
            return real_read(self, checksum, *args, **kw)

        def publish(self, checksum, *args, **kw):
            spy.calls.append(("publish", checksum))
            return real_publish(self, checksum, *args, **kw)

        def work_path(self, checksum, fingerprint, attempt):
            spy.calls.append(("work_dir", checksum))
            return real_work(self, checksum, fingerprint, attempt)

        monkeypatch.setattr(PortableStore, "find", find)
        monkeypatch.setattr(PortableStore, "read_succeeded", read_succeeded)
        monkeypatch.setattr(PortableStore, "publish", publish)
        monkeypatch.setattr(PortableStore, "_work_path", work_path)

    def of(self, checksum: str) -> list[str]:
        return [what for what, c in self.calls if c == checksum]

    def of_kind(self, kind: str) -> list[str]:
        return [c for what, c in self.calls if what == kind]


def test_absent_codec_round_trip_shared_root(tmp_path):
    # a shared root never pickles, so an Absent output is stored only through its codec
    store = nonpickling(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        result = sub(Maybe(x=-1))
    assert not result.errored and result.stored is True
    assert result.outputs.out == Absent(cause="negative", source="Maybe")
    (doc,) = entry_docs(tmp_path / "nopickle")
    assert doc["outputs"]["out"] == {
        "$absent": {"cause": "negative", "source": "Maybe"}
    }
    with Submitter(worker="debug", store=nonpickling(tmp_path)) as sub:
        again = sub(Maybe(x=-1))
    assert again.committed.served is True
    out = again.outputs.out
    assert isinstance(out, Absent) and (out.cause, out.source) == ("negative", "Maybe")


def test_work_dir_uses_suffixed_checksum(tmp_path, monkeypatch):
    spy = Spy(monkeypatch)
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        result = sub(Keyed(xs=[1, -1, 3]))
    assert result.outputs.out == ["1", "absent:negative", "3"]
    named = {c for what, c in spy.calls if what == "work_dir"}
    (suffixed,) = [c for c in named if c.startswith("gather-n3")]
    assert suffixed.startswith("gather-n3-absent-")
    assert "gather-n3" not in {c for _, c in spy.calls}  # never the unsuffixed key
    assert "publish" in spy.of(suffixed)


@workflow.define
def ChainFromAbsent(x: int) -> int:
    a = workflow.add(Maybe(x=x), name="A")
    b = workflow.add(Chain(x=a.out), name="B")  # a workflow node reads it too
    return b.out


@pytest.mark.parametrize("wf", [Chain, ChainFromAbsent])
def test_propagated_absent_never_calls_find(tmp_path, monkeypatch, wf):
    spy = Spy(monkeypatch)
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        result = sub(wf(x=-1))
    assert result.outputs.out == Absent(cause="negative", source="Maybe")
    assert "A" in spy.found and "B" not in spy.found and spy.of_kind("publish")
    # a propagated job's checksum is its absence identity, ``<type>-absent-<digest>``
    assert [c for _, c in spy.calls if "-absent-" in c] == []


def test_gated_off_never_calls_find(tmp_path, monkeypatch):
    spy = Spy(monkeypatch)
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        result = sub(Fanout(xs=[1, -1, -2]))
    assert result.outputs.out[0] == "X"
    assert [type(v) for v in result.outputs.out[1:]] == [Absent, Absent]
    assert spy.found.count("C") == 3 and spy.found.count("D") == 1  # D(0) only
    assert spy.of_kind("publish")
    # a gated-off job's checksum is its absence identity, ``<type>-absent-<digest>``
    assert [c for _, c in spy.calls if "-absent-" in c] == []
