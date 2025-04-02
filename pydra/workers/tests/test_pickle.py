import pickle as pkl

from pydra.workers import debug, cf, sge, slurm


def test_pickle_debug_worker():
    """
    Test pickling of debug.Worker
    """
    worker = debug.Worker()
    worker2 = pkl.loads(pkl.dumps(worker))
    assert worker2.loop is None
    assert worker == worker2


def test_pickle_cf_worker():
    """
    Test pickling of cf.Worker
    """
    worker = cf.Worker()
    worker2 = pkl.loads(pkl.dumps(worker))
    assert worker2.loop is None
    assert worker == worker2


def test_pickle_sge_worker():
    """
    Test pickling of sge.Worker
    """
    worker = sge.Worker()
    worker2 = pkl.loads(pkl.dumps(worker))
    assert worker2.loop is None
    assert worker == worker2


def test_pickle_slurm_worker():
    """
    Test pickling of DebugWorker
    """
    worker = slurm.Worker()
    worker2 = pkl.loads(pkl.dumps(worker))
    assert worker2.loop is None
    assert worker == worker2
