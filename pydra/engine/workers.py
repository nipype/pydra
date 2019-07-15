import time
import asyncio

import concurrent.futures as cf

import logging

logger = logging.getLogger("pydra.worker")


class Worker(object):
    def __init__(self, loop=None):
        logger.debug("Initialize Worker")
        self.loop = loop

    def run_el(self, interface, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


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
    def __init__(self, nr_proc=None):
        super(ConcurrentFuturesWorker, self).__init__()
        self.nr_proc = nr_proc
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.nr_proc)
        # self.loop = asyncio.get_event_loop()
        logger.debug("Initialize ConcurrentFuture")

    def run_el(self, interface, **kwargs):
        # wrap as asyncio task
        if not self.loop:
            raise Exception("No event loop available to submit tasks")
        task = asyncio.create_task(exec_as_coro(self.loop, self.pool, interface._run))
        return task

    async def exec_as_coro(self, interface):  # sidx=None):
        res = await self.loop.run_in_executor(self.pool, interface._run)
        return res

    def close(self):
        self.pool.shutdown()

    async def fetch_finished(self, futures):
        """Awaits asyncio ``Tasks`` until one is finished

        Parameters
        ----------
        futures : set of ``Futures``
            Pending tasks

        Returns
        -------
        done : set
            Finished or cancelled tasks
        """
        done = set()
        try:
            done, pending = await asyncio.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )
        except ValueError:
            # nothing pending!
            pending = set()

        assert (
            done.union(pending) == futures
        ), "all tasks from futures should be either in done or pending"
        logger.debug(f"Tasks finished: {len(done)}")
        return done, pending


async def exec_as_coro(loop, pool, interface):
    res = await loop.run_in_executor(pool, interface)
    return res
