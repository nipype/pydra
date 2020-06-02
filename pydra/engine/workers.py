"""Execution workers."""
import asyncio
import sys
import re
from tempfile import gettempdir
from pathlib import Path
from shutil import copyfile

import concurrent.futures as cf

from .core import TaskBase
from .helpers import get_available_cpus, read_and_display_async, save, load_and_run

import logging

logger = logging.getLogger("pydra.worker")


class Worker:
    """A base class for execution of tasks."""

    def __init__(self, loop=None):
        """Initialize the worker."""
        logger.debug(f"Initializing {self.__class__.__name__}")
        self.loop = loop

    def run_el(self, interface, **kwargs):
        """Return coroutine for task execution."""
        raise NotImplementedError

    def close(self):
        """Close this worker."""

    async def fetch_finished(self, futures):
        """
        Awaits asyncio's :class:`asyncio.Task` until one is finished.

        Parameters
        ----------
        futures : set of asyncio awaitables
            Task execution coroutines or asyncio :class:`asyncio.Task`

        Returns
        -------
        pending : set
            Pending asyncio :class:`asyncio.Task`.

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
    """Base Worker for distributed execution."""

    def __init__(self, loop=None, max_jobs=None):
        """Initialize the worker."""
        super().__init__(loop=loop)
        self.max_jobs = max_jobs
        """Maximum number of concurrently running jobs."""
        self._jobs = 0

    def _prepare_runscripts(self, task, interpreter="/bin/sh", rerun=False):

        if isinstance(task, TaskBase):
            checksum = task.checksum
            cache_dir = task.cache_dir
            ind = None
        else:
            ind = task[0]
            checksum = task[-1].checksum_states()[ind]
            cache_dir = task[-1].cache_dir

        script_dir = cache_dir / f"{self.__class__.__name__}_scripts" / checksum
        script_dir.mkdir(parents=True, exist_ok=True)
        if ind is None:
            if not (script_dir / "_task.pkl").exists():
                save(script_dir, task=task)
        else:
            copyfile(task[1], script_dir / "_task.pklz")

        task_pkl = script_dir / "_task.pklz"
        if not task_pkl.exists() or not task_pkl.stat().st_size:
            raise Exception("Missing or empty task!")

        batchscript = script_dir / f"batchscript_{checksum}.sh"
        python_string = f"""'from pydra.engine.helpers import load_and_run; load_and_run(task_pkl="{str(task_pkl)}", ind={ind}, rerun={rerun}) '
        """
        bcmd = "\n".join(
            (
                f"#!{interpreter}",
                f"#SBATCH --output={str(script_dir / 'slurm-%j.out')}",
                f"{sys.executable} -c " + python_string,
            )
        )
        with batchscript.open("wt") as fp:
            fp.writelines(bcmd)
        return script_dir, batchscript

    async def fetch_finished(self, futures):
        """
        Awaits asyncio's :class:`asyncio.Task` until one is finished.

        Limits number of submissions based on
        py:attr:`DistributedWorker.max_jobs`.

        Parameters
        ----------
        futures : set of asyncio awaitables
            Task execution coroutines or asyncio :class:`asyncio.Task`

        Returns
        -------
        pending : set
            Pending asyncio :class:`asyncio.Task`.

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
    """A simple class to imitate a pool executor of concurrent futures."""

    def submit(self, interface, **kwargs):
        """Send new task."""
        self.res = interface(**kwargs)

    def result(self):
        """Get the result of a task."""
        return self.res

    def done(self):
        """Return whether the task is finished."""
        return True


class SerialWorker(Worker):
    """A worker to execute linearly."""

    def __init__(self):
        """Initialize worker."""
        logger.debug("Initialize SerialWorker")
        self.pool = SerialPool()

    def run_el(self, interface, rerun=False, **kwargs):
        """Run a task."""
        self.pool.submit(interface=interface, rerun=rerun, **kwargs)
        return self.pool

    def close(self):
        """Return whether the task is finished."""


class ConcurrentFuturesWorker(Worker):
    """A worker to execute in parallel using Python's concurrent futures."""

    def __init__(self, n_procs=None):
        """Initialize Worker."""
        super(ConcurrentFuturesWorker, self).__init__()
        self.n_procs = get_available_cpus() if n_procs is None else n_procs
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.n_procs)
        # self.loop = asyncio.get_event_loop()
        logger.debug("Initialize ConcurrentFuture")

    def run_el(self, runnable, rerun=False, **kwargs):
        """Run a task."""
        assert self.loop, "No event loop available to submit tasks"
        return self.exec_as_coro(runnable, rerun=rerun)

    async def exec_as_coro(self, runnable, rerun=False):
        """Run a task (coroutine wrapper)."""
        if isinstance(runnable, TaskBase):
            res = await self.loop.run_in_executor(self.pool, runnable._run, rerun)
        else:  # it could be tuple that includes pickle files with tasks and inputs
            ind, task_main_pkl, task_orig = runnable
            res = await self.loop.run_in_executor(
                self.pool, load_and_run, task_main_pkl, ind, rerun
            )
        return res

    def close(self):
        """Finalize the internal pool of tasks."""
        self.pool.shutdown()


class SlurmWorker(DistributedWorker):
    """A worker to execute tasks on SLURM systems."""

    _cmd = "sbatch"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
    )

    def __init__(
        self, loop=None, max_jobs=None, poll_delay=1, sbatch_args=None, **kwargs
    ):
        """
        Initialize SLURM Worker.

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
        self.error = {}

    def run_el(self, runnable, rerun=False):
        """Worker submission API."""
        script_dir, batch_script = self._prepare_runscripts(runnable, rerun=rerun)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")

        if isinstance(runnable, TaskBase):
            checksum = runnable.checksum
            cache_dir = runnable.cache_dir
            name = runnable.name
        else:
            checksum = runnable[-1].checksum_states()[runnable[0]]
            cache_dir = runnable[-1].cache_dir
            name = runnable[-1].name

        return self._submit_job(
            batch_script, name=name, checksum=checksum, cache_dir=cache_dir
        )

    async def _submit_job(self, batchscript, name, checksum, cache_dir):
        """Coroutine that submits task runscript and polls job until completion or error."""
        script_dir = cache_dir / f"{self.__class__.__name__}_scripts" / checksum
        sargs = self.sbatch_args.split()
        jobname = re.search(r"(?<=-J )\S+|(?<=--job-name=)\S+", self.sbatch_args)
        if not jobname:
            jobname = ".".join((name, checksum))
            sargs.append(f"--job-name={jobname}")
        output = re.search(r"(?<=-o )\S+|(?<=--output=)\S+", self.sbatch_args)
        if not output:
            output_file = str(script_dir / "slurm-%j.out")
            sargs.append(f"--output={output_file}")
        error = re.search(r"(?<=-e )\S+|(?<=--error=)\S+", self.sbatch_args)
        if not error:
            error_file = str(script_dir / "slurm-%j.err")
            sargs.append(f"--error={error_file}")
        else:
            error_file = None
        sargs.append(str(batchscript))
        # TO CONSIDER: add random sleep to avoid overloading calls
        _, stdout, _ = await read_and_display_async("sbatch", *sargs, hide_display=True)
        jobid = re.search(r"\d+", stdout)
        if not jobid:
            raise RuntimeError("Could not extract job ID")
        jobid = jobid.group()
        if error_file:
            error_file = error_file.replace("%j", jobid)
        self.error[jobid] = error_file.replace("%j", jobid)
        # intermittent polling
        while True:
            # 3 possibilities
            # False: job is still pending/working
            # True: job is complete
            # Exception: Polling / job failure
            done = await self._poll_job(jobid)
            if done:
                return True
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
        error_file = self.error[jobid]
        if int(m.group("exit_code")) != 0 or m.group("status") != "COMPLETED":
            if m.group("status") in ["RUNNING", "PENDING"]:
                return False
            # TODO: potential for requeuing
            # parsing the error message
            error_line = Path(error_file).read_text().split("\n")[-2]
            if "Exception" in error_line:
                error_message = error_line.replace("Exception: ", "")
            elif "Error" in error_line:
                error_message = error_line.replace("Exception: ", "")
            else:
                error_message = "Job failed (unknown reason - TODO)"
            raise Exception(error_message)
        return True


class DaskWorker(Worker):
    """ A worker to execute in parallel using Dask.distributed.
        This is an experimental implementation with limited testing.
    """

    def __init__(self, **kwargs):
        """Initialize Worker."""
        super(DaskWorker, self).__init__()
        try:
            from dask.distributed import Client
        except ImportError:
            logger.critical("Please instiall Dask distributed.")
            raise
        self.client = None
        self.client_args = kwargs
        logger.debug("Initialize Dask")

    def run_el(self, runnable, rerun=False, **kwargs):
        """Run a task."""
        return self.exec_dask(runnable, rerun=rerun)

    async def exec_dask(self, runnable, rerun=False):
        """Run a task (coroutine wrapper)."""
        if self.client is None:
            from dask.distributed import Client

            self.client = await Client(**self.client_args, asynchronous=True)
        future = self.client.submit(runnable._run, rerun)
        result = await future
        return result

    def close(self):
        """Finalize the internal pool of tasks."""
        pass
