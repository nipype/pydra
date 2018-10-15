"""Tests for `pydra.engine.workers`."""
import pytest

from pydra.engine import workers


def test__get_worker():
    assert isinstance(workers._get_worker("mp")(), workers.MpWorker)
    assert isinstance(workers._get_worker("serial")(), workers.SerialWorker)
    assert isinstance(workers._get_worker("cf")(), workers.ConcurrentFuturesWorker)
    assert isinstance(workers._get_worker("dask")(), workers.DaskWorker)

    w = workers.SerialWorker
    assert workers._get_worker(w) is w

    w = workers.SerialWorker()
    assert workers._get_worker(w) is w

    with pytest.raises(KeyError):
        workers._get_worker("not_a_plugin")
