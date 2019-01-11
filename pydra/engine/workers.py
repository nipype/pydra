import os, time, pdb
import multiprocessing as mp
#from pycon_utils import make_cluster
from dask.distributed import Client
import concurrent.futures as cf

import logging
logger = logging.getLogger('nipype.workflow')


class Worker(object):
    def __init__(self):
        logger.debug("Initialize Worker")
        pass

    def run_el(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class MpWorker(Worker):
    def __init__(self, nr_proc=4):  #should be none
        self.nr_proc = nr_proc
        self.pool = mp.Pool(processes=self.nr_proc)
        logger.debug('Initialize MpWorker')

    def run_el(self, interface, inp):
        x = self.pool.apply_async(interface, (inp[0], inp[1]))
        # returning dir_nm_el and Result object for the specific element
        return x.get()

    def close(self):
        # added this method since I was having somtetimes problem with reading results from (existing) files
        # i thought that pool.close() should work, but still was getting some errors, so testing terminate
        self.pool.terminate()


class SerialWorker(Worker):
    def __init__(self):
        logger.debug("Initialize SerialWorker")
        pass

    def run_el(self, interface, inp):
        res = interface(inp[0], inp[1])
        # returning dir_nm_el and Result object for the specific element
        return res

    def close(self):
        pass


class ConcurrentFuturesWorker(Worker):
    def __init__(self, nr_proc=4):
        self.nr_proc = nr_proc
        self.pool = cf.ProcessPoolExecutor(self.nr_proc)
        logger.debug('Initialize ConcurrentFuture')

    def run_el(self, interface, inp):
        x = self.pool.submit(interface, inp[0], inp[1])
        #print("X, DONE", x.done())
        x.add_done_callback(lambda x: print("DONE ", interface, inp, x.done))
        #print("DIR", x.result())
        # returning dir_nm_el and Result object for the specific element
        return x.result()

    def close(self):
        self.pool.shutdown()


class DaskWorker(Worker):
    def __init__(self):
        from distributed.deploy.local import LocalCluster
        logger.debug("Initialize Dask Worker")
        #self.cluster = LocalCluster()
        self.client = Client()  #self.cluster)
        #print("BOKEH", self.client.scheduler_info()["address"] + ":" + str(self.client.scheduler_info()["services"]["bokeh"]))

    def run_el(self, interface, inp):
        print("DASK, run_el: ", interface, inp, time.time())
        # dask  doesn't copy the node second time, so it doesn't see that I change input in the meantime (??)
        x = self.client.submit(interface, inp[0], inp[1])
        print("DASK, status: ", x.status)
        # this important, otherwise dask will not finish the job
        x.add_done_callback(lambda x: print("DONE ", interface, inp))
        print("res", x.result())
        # returning dir_nm_el and Result object for the specific element
        return x.result()

    def close(self):
        #self.cluster.close()
        self.client.close()
