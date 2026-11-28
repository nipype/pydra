"""``pydra-identity@2`` records the pool facts and never keys them (RFC 0005 sections 8 and 11, W11b)."""

import importlib
import importlib.util

import pytest

from pydra.engine.lease import LeaseSettings
from pydra.engine.portable.identity import PydraIdentityV2
from pydra.engine.portable.layout import LayoutHeader


def _importable(module: str, name: str) -> bool:
    try:
        if importlib.util.find_spec(module) is None:
            return False
        return hasattr(importlib.import_module(module), name)
    except ImportError:
        return False


HAS_WARM_POOL = _importable("pydra.workers.pool.pool", "WarmPool") and _importable(
    "pydra.engine.portable.store", "PortableStore"
)


def test_default_v2_root_accepts_warm_pool(tmp_path):
    """The header of a root made with the default ``pydra-identity@2`` lists ``pool`` in
    ``recorded_only``, so the facts a pool worker adds are recorded and split no slot, and a
    `WarmPool` opens on a store with that identity without configuration."""
    ident = PydraIdentityV2()
    assert "pool" in ident.recorded_only
    header = LayoutHeader.for_store(ident)
    assert "pool" in header.identity["recorded_only"]
    ctx = {"a": 1}
    assert ident.fingerprint({**ctx, "pool": {"worker": "w1"}}) == ident.fingerprint(
        {**ctx, "pool": {"worker": "w2"}}
    )
    assert ident.fingerprint({**ctx, "pool": {}}) == ident.fingerprint(ctx)

    if not HAS_WARM_POOL:
        pytest.skip("pydra.workers.pool is not part of this build")

    from pydra.engine.portable.store import PortableStore
    from pydra.workers.pool.classes import WorkerClass
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.settings import PoolSettings
    from pydra.workers.pool.substrate import ExternalSubstrate

    root = tmp_path / "store"
    root.mkdir(mode=0o700)
    store = PortableStore(root, trust="private", identity=PydraIdentityV2())
    lease = LeaseSettings(
        stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
    )
    pool = WarmPool(
        tmp_path / "pool",
        [WorkerClass(name="cpu", device="cpu", quantum_jobs=8, quantum_s=600.0)],
        ExternalSubstrate(),
        store=store,
        settings=PoolSettings(
            lease=lease, poll_s=0.05, idle_polls=2, timeout_grace=60.0
        ),
    )
    pool.close()


def test_v2_recorded_only_is_a_union():
    """``pool`` (W11b) and the environment facts (I6a, I6b) stay recorded-only whatever
    else is given."""
    ident = PydraIdentityV2(recorded_only={"mine"})
    assert {
        "mine",
        "pool",
        "environment_repository",
        "environment_facts",
    } <= ident.recorded_only
    assert {"pool", "environment_repository", "environment_facts"} <= (
        PydraIdentityV2().recorded_only
    )
