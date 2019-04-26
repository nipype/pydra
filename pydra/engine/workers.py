import time
import multiprocessing as mp
import asyncio

# from pycon_utils import make_cluster
from dask.distributed import Client
import concurrent.futures as cf

import logging

logger = logging.getLogger("nipype.workflow")


class Worker(object):
    def __init__(self):
        self._pending = set()
        logger.debug("Initialize Worker")

    def run_el(self, interface, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def _remove_pending(self, task):
        self._pending.remove(task)


class MpWorker(Worker):
    def __init__(self, nr_proc=4):  # should be none
        self.nr_proc = nr_proc
        self.pool = mp.Pool(processes=self.nr_proc)
        logger.debug("Initialize MpWorker")

    def run_el(self, interface, inp):
        x = self.pool.apply_async(interface, inp)
        # returning dir_nm_el and Result object for the specific element
        return x.get()

    def close(self):
        # added this method since I was having somtetimes problem with reading results from (existing) files
        # i thought that pool.close() should work, but still was getting some errors, so testing terminate
        self.pool.terminate()


class SerialPool:
    """ a simply class to imitate a pool like in cf"""

    def submit(self, interface, **kwargs):
        self.res = interface(**kwargs)

    def result(self):
        return self.res

    def done(self):
        return True


class SerialWorker(Worker):
    def __init__(self):
        logger.debug("Initialize SerialWorker")
        self.pool = SerialPool()

    def run_el(self, interface, **kwargs):
        self.pool.submit(interface=interface, **kwargs)
        return self.pool

    def close(self):
        pass


class ConcurrentFuturesWorker(Worker):
    def __init__(self, nr_proc=2):
        super(ConcurrentFuturesWorker, self).__init__()
        self.nr_proc = nr_proc or mp.cpu_count()
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.nr_proc)
        logger.debug("Initialize ConcurrentFuture")
        self.loop = asyncio.get_event_loop()  # TODO: consider windows

    def run_el(self, interface, **kwargs):
        # wrap as asyncio task
        task = asyncio.create_task(exec_as_coro(self.loop, self.pool, interface))
        self._pending.add(task)
        logger.debug("Pending tasks: %s", self._pending)
        return task
        # return self.pool.submit(interface, **kwargs)

    def close(self):
        self.pool.shutdown()

    async def fetch_finished(self):
        done, pending = await asyncio.wait(
            self._pending, return_when=asyncio.FIRST_COMPLETED
        )
        # preserve pending tasks
        self._pending.union(pending)
        return done


class DaskWorker(Worker):
    def __init__(self):
        from distributed.deploy.local import LocalCluster

        logger.debug("Initialize Dask Worker")
        # self.cluster = LocalCluster()
        self.client = Client()  # self.cluster)
        # print("BOKEH", self.client.scheduler_info()["address"] + ":" + str(self.client.scheduler_info()["services"]["bokeh"]))

    def run_el(self, interface, **kwargs):
        print("DASK, run_el: ", interface, kwargs, time.time())
        # dask  doesn't copy the node second time, so it doesn't see that I change input in the meantime (??)
        x = self.client.submit(interface, **kwargs)
        print("DASK, status: ", x.status)
        # this important, otherwise dask will not finish the job
        x.add_done_callback(lambda x: print("DONE ", interface, kwargs))
        print("res", x.result())
        # returning dir_nm_el and Result object for the specific element
        # return x.result()
        return x

    def close(self):
        # self.cluster.close()
        self.client.close()


async def exec_as_coro(loop, pool, interface):
    res = await loop.run_in_executor(pool, interface)
    return interface, res
