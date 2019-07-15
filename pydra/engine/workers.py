import time
import multiprocessing as mp
import asyncio
import sys
import re
from tempfile import gettempdir

# from pycon_utils import make_cluster
import concurrent.futures as cf

from .helpers import create_pyscript, read_and_display, save

import logging

logger = logging.getLogger("pydra.worker")


class Worker:
    def __init__(self, loop=None):
        logger.debug(f"Initializing {self.__class__.__name__}")
        self.loop = loop
        self._distributed = False

    def run_el(self, interface, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    async def fetch_finished(self, futures):
        raise NotImplementedError


class DistributedWorker(Worker):
    """Base Worker for distributed execution"""

    def __init__(self, loop=None):
        super(DistributedWorker, self).__init__(loop)
        self._distributed = True

    @staticmethod
    def _prepare_runscripts(task, interpreter="/bin/sh"):
        pyscript = create_pyscript(task.output_dir, task.checksum)
        batchscript = (task.output_dir / f"batchscript_{task.checksum}.sh")
        shebang = f"#!{interpreter}"
        bcmd = "\n".join((shebang, f"{sys.executable} {str(pyscript)}"))
        with batchscript.open('wt') as fp:
            fp.writelines(bcmd)
        return batchscript

    @staticmethod
    async def _awaitable(jobid):
        """Readily available coroutine"""
        return jobid


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
    def __init__(self, nr_proc=None):
        super(ConcurrentFuturesWorker, self).__init__()
        self.nr_proc = nr_proc or mp.cpu_count()
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.nr_proc)
        # self.loop = asyncio.get_event_loop()
        logger.debug("Initialize ConcurrentFuture")

    def run_el(self, runnable, **kwargs):
        assert self.loop, "No event loop available to submit tasks"
        task = asyncio.create_task(self.exec_as_coro(runnable))
        return task

    async def exec_as_coro(self, runnable):
        res = await self.loop.run_in_executor(self.pool, runnable._run)
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
        for fut in done:
            await fut
        return pending


class DaskWorker(Worker):
    def __init__(self):
        logger.debug("Initialize Dask Worker")
        from dask.distributed import Client
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


class SlurmWorker(DistributedWorker):
    _cmd = "sbatch"

    def __init__(self, poll_delay=1, sbatch_args=None, **kwargs):
        """Initialize Slurm Worker

        Parameters
        ----------
        poll_delay : seconds
            Delay between polls to slurmd
        sbatch_args : str
            Additional sbatch arguments
        """
        super(SlurmWorker, self).__init__()
        if not poll_delay or poll_delay < 0:
            poll_delay = 0
        self.poll_delay = poll_delay
        self.sbatch_args = sbatch_args or ""
        self.sacct_re = re.compile(
            '(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +'
            '(?P<exit_code>\\d+):\\d+'
        )
        self.pending = set()
        self.failed = set()

    def run_el(self, task):
        """
        Worker submission API
        """
        odir = task.output_dir
        if (odir.anchor + odir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        save(odir, task=task)
        runscript = self._prepare_runscripts(task, interpreter="/bin/bash")
        task = asyncio.create_task(self._submit_job(task, runscript))
        return task

    async def _submit_job(self, task, batchscript):
        """Wraps Slurm batch submission (sbatch)"""
        sargs = self.sbatch_args.split()
        jobname = re.search(r"(?<=-J )\S+|(?<=--job-name=)\S+", self.sbatch_args)
        if not jobname:
            jobname = ".".join((task.name, task.checksum))
            sargs.append(f"--job-name={jobname}")
        output = re.search(r"(?<=-o )\S+|(?<=--output=)\S+", self.sbatch_args)
        if not output:
            output = str(batchscript.parent / 'slurm-%j.out')
            sargs.append(f"--output={output}")
        sargs.append(str(batchscript))
        # TO CONSIDER: add random sleep to avoid overloading calls
        _, stdout, _ = await read_and_display('sbatch', *sargs, hide_display=True)
        jobid = re.search(r"\d+", stdout)
        if not jobid:
            raise RuntimeError("Could not extract job ID")
        return jobid.group()

    def close(self):
        pass

    async def _poll_jobs(self, raise_exception):
        sargs = ("-h", "-j", ",".join(self.pending))
        _, stdout, _ = await read_and_display('squeue', *sargs, hide_display=True)
        if not stdout:
            if raise_exception:
                raise RuntimeError("Nothing to run")
            return True
        return False

    async def _poll_job(self, jobid):
        sargs = ("-h", "-j", jobid)
        rc, stdout, stderr = await read_and_display('squeue', *sargs, hide_display=True)
        if not stdout or "slurm_load_jobs" in stderr:
            # job is no longer running - check exit code
            await self._verify_exit_code(jobid)
            return jobid
        elif rc != 0:
            raise RuntimeError("Job query failed")
        return None

    async def _verify_exit_code(self, jobid):
        sargs = ("-n", "-X", "-j", jobid, "-o", "JobID,State,ExitCode")
        _, stdout, _ = await read_and_display('sacct', *sargs, hide_display=True)
        if not stdout:
            raise RuntimeError("Job information not found")
        m = self.sacct_re.search(stdout)
        if int(m.group('exit_code')) != 0 or m.group('status') != "COMPLETED":
            # TODO: potential for requeuing
            self.pending.remove(jobid)
            self.failed.add(jobid)

    async def fetch_finished(self, futures):
        """
        Waits until at least one job finishes
        Basic implementation of FIRST_COMPLETED
        """
        done = set()
        raise_exception = False
        # wait for all futures to complete (submit to Slurm)
        submitted = await asyncio.gather(*futures)
        for job in submitted:
            # tracker for submitted job IDs
            self.pending.add(job)

        while True:
            # poll all submitted jobs until single job has completed
            for job in self.pending:
                done = await self._poll_job(job)
                if done is not None:
                    self.pending.remove(job)
                    # break polling and check if pending jobs
                    return self.pending
                await asyncio.sleep(self.poll_delay)
            raise_exception = await self._poll_jobs(raise_exception)
