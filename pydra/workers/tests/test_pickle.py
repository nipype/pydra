import pickle as pkl

from pydra.workers import debug, cf, sge, slurm, dask, base


def test_pickle_debug_worker():
    """
    Test pickling of DebugWorker
    """
    worker = debug.Worker()
    worker2 = pkl.loads(pkl.dumps(worker))
    assert worker2.loop is None
    assert worker == worker2
