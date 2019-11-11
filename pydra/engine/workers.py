import asyncio
import sys
import re
from tempfile import gettempdir
from pathlib import Path

import concurrent.futures as cf

from .helpers import create_pyscript, read_and_display_async, save

import logging

logger = logging.getLogger("pydra.worker")


class Worker:
    def __init__(self, loop=None):
        logger.debug(f"Initializing {self.__class__.__name__}")
        self.loop = loop

    def run_el(self, interface, **kwargs):
        """Returns coroutine for task execution"""
        raise NotImplementedError

    def close(self):
        pass

    async def fetch_finished(self, futures):
        """
        Awaits asyncio `Tasks` until one is finished.

        Parameters
        ----------
        futures : set of asyncio awaitables
            Task execution coroutines or asyncio `Tasks`

        Returns
        -------
        pending : set
            Pending asyncio `Task`s
        """
        done = set()
        try:
            done, pending = await asyncio.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )
        except ValueError:
            # nothing pending!
            pending = set()
        logger.debug(f"Tasks finished: {len(done)}")
        return pending


class DistributedWorker(Worker):
    """Base Worker for distributed execution"""

    def __init__(self, loop=None, max_jobs=None):
        super().__init__(loop=loop)
        self.max_jobs = max_jobs
        self._jobs = 0

    def _prepare_runscripts(self, task, interpreter="/bin/sh"):
        script_dir = (
            task.cache_dir / f"{self.__class__.__name__}_scripts" / task.checksum
        )
        script_dir.mkdir(parents=True, exist_ok=True)
        if not (script_dir / "_task.pkl").exists():
            save(script_dir, task=task)
        pyscript = create_pyscript(script_dir, task.checksum)
        batchscript = script_dir / f"batchscript_{task.checksum}.sh"
        bcmd = "\n".join(
            (
                f"#!{interpreter}",
                f"#SBATCH --output={str(script_dir / 'slurm-%j.out')}",
                f"{sys.executable} {str(pyscript)}",
            )
        )
        with batchscript.open("wt") as fp:
            fp.writelines(bcmd)
        return script_dir, pyscript, batchscript

    async def fetch_finished(self, futures):
        """
        Awaits asyncio `Task`s until one is finished.
        Limits number of submissions based on max_jobs attr.

        Parameters
        ----------
        futures : set of asyncio awaitables
            Task execution coroutines or asyncio `Task`s

        Returns
        -------
        pending : set
            Pending asyncio `Task`s
        """
        done, unqueued = set(), set()
        job_slots = self.max_jobs - self._jobs if self.max_jobs else float("inf")
        if len(futures) > job_slots:
            # convert to list to simplify indexing
            logger.warning(f"Reducing queued jobs due to max jobs ({self.max_jobs})")
            futures = list(futures)
            futures, unqueued = set(futures[:job_slots]), set(futures[job_slots:])
        try:
            self._jobs += len(futures)
            done, pending = await asyncio.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )
        except ValueError:
            # nothing pending!
            pending = set()
        self._jobs -= len(done)
        logger.debug(f"Tasks finished: {len(done)}")
        # ensure pending + unqueued tasks persist
        return pending.union(unqueued)


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
    def __init__(self, n_procs=None):
        super(ConcurrentFuturesWorker, self).__init__()
        self.n_procs = n_procs
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.n_procs)
        # self.loop = asyncio.get_event_loop()
        logger.debug("Initialize ConcurrentFuture")

    def run_el(self, runnable, **kwargs):
        assert self.loop, "No event loop available to submit tasks"
        return self.exec_as_coro(runnable)

    async def exec_as_coro(self, runnable):
        res = await self.loop.run_in_executor(self.pool, runnable._run)
        return res

    def close(self):
        self.pool.shutdown()


class SlurmWorker(DistributedWorker):
    _cmd = "sbatch"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
    )

    def __init__(
        self, loop=None, max_jobs=None, poll_delay=1, sbatch_args=None, **kwargs
    ):
        """Initialize Slurm Worker

        Parameters
        ----------
        poll_delay : seconds
            Delay between polls to slurmd
        sbatch_args : str
            Additional sbatch arguments
        max_jobs : int
            Maximum number of submitted jobs
        """
        super().__init__(loop=loop, max_jobs=max_jobs)
        if not poll_delay or poll_delay < 0:
            poll_delay = 0
        self.poll_delay = poll_delay
        self.sbatch_args = sbatch_args or ""

    def run_el(self, runnable):
        """
        Worker submission API
        """
        script_dir, _, batch_script = self._prepare_runscripts(runnable)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        return self._submit_job(runnable, batch_script)

    async def _submit_job(self, task, batchscript):
        """Coroutine that submits task runscript and polls job until completion or error."""
        sargs = self.sbatch_args.split()
        jobname = re.search(r"(?<=-J )\S+|(?<=--job-name=)\S+", self.sbatch_args)
        if not jobname:
            jobname = ".".join((task.name, task.checksum))
            sargs.append(f"--job-name={jobname}")
        output = re.search(r"(?<=-o )\S+|(?<=--output=)\S+", self.sbatch_args)
        if not output:
            self.output = str(batchscript.parent / "slurm-%j.out")
            sargs.append(f"--output={self.output}")
        error = re.search(r"(?<=-e )\S+|(?<=--error=)\S+", self.sbatch_args)
        if not error:
            self.error = str(batchscript.parent / "slurm-%j.err")
            sargs.append(f"--error={self.error}")
        sargs.append(str(batchscript))
        # TO CONSIDER: add random sleep to avoid overloading calls
        _, stdout, _ = await read_and_display_async("sbatch", *sargs, hide_display=True)
        jobid = re.search(r"\d+", stdout)
        if not jobid:
            raise RuntimeError("Could not extract job ID")
        jobid = jobid.group()
        self.output = self.output.replace("%j", jobid)
        self.error = self.error.replace("%j", jobid)
        # intermittent polling
        while True:
            # 3 possibilities
            # False: job is still pending/working
            # True: job is complete
            # Exception: Polling / job failure
            done = await self._poll_job(jobid)
            if done:
                return task
            await asyncio.sleep(self.poll_delay)

    async def _poll_job(self, jobid):
        cmd = ("squeue", "-h", "-j", jobid)
        logger.debug(f"Polling job {jobid}")
        rc, stdout, stderr = await read_and_display_async(*cmd, hide_display=True)
        if not stdout or "slurm_load_jobs error" in stderr:
            # job is no longer running - check exit code
            status = await self._verify_exit_code(jobid)
            return status
        return False

    async def _verify_exit_code(self, jobid):
        cmd = ("sacct", "-n", "-X", "-j", jobid, "-o", "JobID,State,ExitCode")
        _, stdout, _ = await read_and_display_async(*cmd, hide_display=True)
        if not stdout:
            raise RuntimeError("Job information not found")
        m = self._sacct_re.search(stdout)
        if int(m.group("exit_code")) != 0 or m.group("status") != "COMPLETED":
            if m.group("status") in ["RUNNING", "PENDING"]:
                return False
            # TODO: potential for requeuing
            # parsing the error message
            error_line = Path(self.error).read_text().split("\n")[-2]
            if "Exception" in error_line:
                error_message = error_line.replace("Exception: ", "")
            elif "Error" in error_line:
                error_message = error_line.replace("Exception: ", "")
            else:
                error_message = "Job failed (unknown reason - TODO)"
            raise Exception(error_message)
        return True
