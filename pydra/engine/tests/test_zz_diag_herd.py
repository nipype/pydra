"""TEMPORARY diagnostic for the Windows herd wait (RFC 0003 P4a review round 4); removed next commit."""

import collections
import functools
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

import pydra.engine.portable.store as mod
from pydra.engine.portable.store import PortableStore
from pydra.utils.storagedir import IS_WINDOWS, StorageDir, StorePolicy

CTX = {"python": "3.12", "platform": "test", "n": 1}


@pytest.mark.skipif(not IS_WINDOWS, reason="diagnostic for Windows")
def test_zz_diag_herd(tmp_path, monkeypatch):
    trace = collections.defaultdict(list)
    local = threading.local()

    def wrap(owner, name, fn, is_static=False):
        @functools.wraps(fn)
        def w(*a, **k):
            idx = getattr(local, "i", -1)
            t = time.monotonic()
            exc = None
            try:
                return fn(*a, **k)
            except BaseException as e:
                exc = f"{type(e).__name__}:{getattr(e, 'errno', '')}:{getattr(e, 'winerror', '')}"
                raise
            finally:
                arg = ""
                if a and isinstance(a[0], StorageDir):
                    arg = (
                        os.path.basename(a[0].path)
                        + "/"
                        + (
                            str(a[1])[:12]
                            if len(a) > 1 and isinstance(a[1], str)
                            else ""
                        )
                    )
                trace[idx].append(
                    (
                        round(t - getattr(local, "t0", t), 3),
                        round(time.monotonic() - t, 3),
                        name,
                        arg,
                        exc,
                    )
                )

        return staticmethod(w) if is_static else w

    for n in (
        "child",
        "rename",
        "lstat",
        "fstat",
        "rmdir",
        "unlink",
        "scandir",
        "open_read",
        "open_new",
        "set_mode",
    ):
        monkeypatch.setattr(
            StorageDir, n, wrap(StorageDir, "SD." + n, getattr(StorageDir, n))
        )
    for n in (
        "_container",
        "_set_aside",
        "_lost",
        "_attempt",
        "_cdir_dir",
        "_quarantine",
        "_settled",
        "_seal",
        "_publish",
        "_place",
    ):
        monkeypatch.setattr(
            PortableStore, n, wrap(PortableStore, n, getattr(PortableStore, n))
        )
    monkeypatch.setattr(
        PortableStore,
        "_attached",
        wrap(
            PortableStore,
            "_attached",
            PortableStore.__dict__["_attached"].__func__,
            True,
        ),
    )
    monkeypatch.setattr(
        PortableStore,
        "_occupied",
        wrap(
            PortableStore,
            "_occupied",
            PortableStore.__dict__["_occupied"].__func__,
            True,
        ),
    )
    monkeypatch.setattr(mod, "_backoff", wrap(mod, "_backoff", mod._backoff))
    monkeypatch.setattr(mod, "_discard", wrap(mod, "_discard", mod._discard))

    store = PortableStore(
        tmp_path / "root", trust="trusted", policy=StorePolicy(durability="none")
    )
    local.t0 = time.monotonic()
    store.publish("python-seed", None, CTX, {"a": 1}, None, None)
    report = []
    for rnd in range(4):
        name = f"python-herd{rnd}"
        trace.clear()
        local.t0 = time.monotonic()
        c = store.publish(name, None, CTX, {"a": 1}, None, None)
        victim = [
            d / name
            for d in (tmp_path / "root" / "results@1").iterdir()
            if (d / name).exists()
        ][0]
        shutil.rmtree(victim)
        victim.write_text("x")
        barrier = threading.Barrier(16)
        out = [None] * 16
        t00 = [0.0]

        def go(i):
            barrier.wait()
            local.i = i
            local.t0 = t00[0] or time.monotonic()
            t00[0] = local.t0
            t = time.monotonic()
            try:
                got = store.publish(name, None, CTX, {"a": 1}, None, None)
                out[i] = (got.won, got.stored, round(time.monotonic() - t, 3))
            except BaseException as e:
                out[i] = (repr(e), round(time.monotonic() - t, 3))

        ts = [threading.Thread(target=go, args=(i,)) for i in range(16)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        slow = max(range(16), key=lambda i: out[i][-1])
        report.append(f"ROUND {rnd} out={out}")
        if out[slow][-1] > 1:
            counts = collections.Counter((e[2], e[4]) for e in trace[slow])
            report.append(f" slow thread {slow} counts: {dict(counts)}")
            long = [e for e in trace[slow] if e[1] > 0.3]
            report.append(f" ops > 0.3s: {long[:40]}")
            report.append(f" first 60 ops: {trace[slow][:60]}")
            report.append(f" last 30 ops: {trace[slow][-30:]}")
            winner = [i for i in range(16) if out[i][0] is True]
            if winner:
                report.append(f" winner {winner[0]} ops: {trace[winner[0]][:60]}")
    pytest.fail("\n".join(report))
