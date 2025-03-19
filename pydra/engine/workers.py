"""Execution workers."""

import asyncio
import sys
import json
import abc
import re
import inspect
import typing as ty
from tempfile import gettempdir
from pathlib import Path
from shutil import copyfile, which
import cloudpickle as cp
import concurrent.futures as cf
from pydra.engine.core import Task
from pydra.engine.specs import TaskDef
from pydra.engine.helpers import (
    get_available_cpus,
    read_and_display_async,
    save,
    load_task,
)

import logging
import random

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.specs import Result

DefType = ty.TypeVar("DefType", bound="TaskDef")


class Worker(metaclass=abc.ABCMeta):
    """A base class for execution of tasks."""

    plugin_name: str
    loop: asyncio.AbstractEventLoop

    def __init__(self, loop=None):
        """Initialize the worker."""
        logger.debug(f"Initializing {self.__class__.__name__}")
        self.loop = loop

    @abc.abstractmethod
    def run(self, task: "Task[DefType]", rerun: bool = False) -> "Result":
        """Return coroutine for task execution."""
        pass

    async def run_async(self, task: "Task[DefType]", rerun: bool = False) -> "Result":
        assert self.is_async, "Worker is not asynchronous"
        if task.is_async:  # only for workflows at this stage and the foreseeable
            # These jobs are run in the primary process but potentially farm out
            # workflow jobs to other processes/job-schedulers
            return await task.run_async(rerun=rerun)
        else:
            return await self.run(task=task, rerun=rerun)

    def close(self):
        """Close this worker."""

    @property
    def is_async(self) -> bool:
        """Return whether the worker is asynchronous."""
        return inspect.iscoroutinefunction(self.run)

    async def fetch_finished(
        self, futures
    ) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
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
        done: set
            Completed asyncio :class:`asyncio.Task`

        """
        done = set()
        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(f) if not isinstance(f, asyncio.Task) else f
                    for f in futures
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except ValueError:
            # nothing pending!
            pending = set()
        logger.debug(f"Tasks finished: {len(done)}")
        return pending, done


class DistributedWorker(Worker):
    """Base Worker for distributed execution."""

    def __init__(self, loop=None, max_jobs=None):
        """Initialize the worker."""
        super().__init__(loop=loop)
        self.max_jobs = max_jobs
        """Maximum number of concurrently running jobs."""
        self._jobs = 0

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
                [
                    asyncio.create_task(f) if not isinstance(f, asyncio.Task) else f
                    for f in futures
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except ValueError:
            # nothing pending!
            pending = set()
        self._jobs -= len(done)
        logger.debug(f"Tasks finished: {len(done)}")
        # ensure pending + unqueued tasks persist
        return pending.union(unqueued), done


class DebugWorker(Worker):
    """A worker to execute linearly."""

    plugin_name: str = "debug"

    def __init__(self, **kwargs):
        """Initialize worker."""
        logger.debug("Initialize SerialWorker")

    def run(
        self,
        task: "Task[DefType]",
        rerun: bool = False,
    ) -> "Result":
        """Run a task."""
        return task.run(rerun=rerun)

    def close(self):
        """Return whether the task is finished."""

    async def fetch_finished(self, futures):
        raise NotImplementedError("DebugWorker does not support async execution")


class ConcurrentFuturesWorker(Worker):
    """A worker to execute in parallel using Python's concurrent futures."""

    plugin_name = "cf"

    n_procs: int
    loop: asyncio.AbstractEventLoop
    pool: cf.ProcessPoolExecutor

    def __init__(self, n_procs: int | None = None):
        """Initialize Worker."""
        super().__init__()
        self.n_procs = get_available_cpus() if n_procs is None else n_procs
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.n_procs)
        # self.loop = asyncio.get_event_loop()
        logger.debug("Initialize ConcurrentFuture")

    async def run(
        self,
        task: "Task[DefType]",
        rerun: bool = False,
    ) -> "Result":
        """Run a task."""
        assert self.loop, "No event loop available to submit tasks"
        return await self.loop.run_in_executor(
            self.pool, self.uncloudpickle_and_run, cp.dumps(task), rerun
        )

    @classmethod
    def uncloudpickle_and_run(cls, task_pkl: bytes, rerun: bool) -> "Result":
        """Unpickle and run a task."""
        task: Task[DefType] = cp.loads(task_pkl)
        return task.run(rerun=rerun)

    def close(self):
        """Finalize the internal pool of tasks."""
        self.pool.shutdown()


class SlurmWorker(DistributedWorker):
    """A worker to execute tasks on SLURM systems."""

    plugin_name = "slurm"
    _cmd = "sbatch"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
    )

    def __init__(self, loop=None, max_jobs=None, poll_delay=1, sbatch_args=None):
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

    def _prepare_runscripts(self, task, interpreter="/bin/sh", rerun=False):
        if isinstance(task, Task):
            cache_dir = task.cache_dir
            ind = None
            uid = task.uid
        else:
            assert isinstance(task, tuple), f"Expecting a task or a tuple, not {task!r}"
            assert len(task) == 2, f"Expecting a tuple of length 2, not {task!r}"
            ind = task[0]
            cache_dir = task[-1].cache_dir
            uid = f"{task[-1].uid}_{ind}"

        script_dir = cache_dir / f"{self.__class__.__name__}_scripts" / uid
        script_dir.mkdir(parents=True, exist_ok=True)
        if ind is None:
            if not (script_dir / "_task.pkl").exists():
                save(script_dir, task=task)
        else:
            copyfile(task[1], script_dir / "_task.pklz")

        task_pkl = script_dir / "_task.pklz"
        if not task_pkl.exists() or not task_pkl.stat().st_size:
            raise Exception("Missing or empty task!")

        batchscript = script_dir / f"batchscript_{uid}.sh"
        python_string = (
            f"""'from pydra.engine.helpers import load_and_run; """
            f"""load_and_run("{task_pkl}", rerun={rerun}) '"""
        )
        bcmd = "\n".join(
            (
                f"#!{interpreter}",
                f"#SBATCH --output={script_dir / 'slurm-%j.out'}",
                f"{sys.executable} -c " + python_string,
            )
        )
        with batchscript.open("wt") as fp:
            fp.writelines(bcmd)
        return script_dir, batchscript

    async def run(self, task: "Task[DefType]", rerun: bool = False) -> "Result":
        """Worker submission API."""
        script_dir, batch_script = self._prepare_runscripts(task, rerun=rerun)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        script_dir = task.cache_dir / f"{self.__class__.__name__}_scripts" / task.uid
        sargs = self.sbatch_args.split()
        jobname = re.search(r"(?<=-J )\S+|(?<=--job-name=)\S+", self.sbatch_args)
        if not jobname:
            jobname = ".".join((task.name, task.uid))
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
        sargs.append(str(batch_script))
        # TO CONSIDER: add random sleep to avoid overloading calls
        rc, stdout, stderr = await read_and_display_async(
            "sbatch", *sargs, hide_display=True
        )
        jobid = re.search(r"\d+", stdout)
        if rc:
            raise RuntimeError(f"Error returned from sbatch: {stderr}")
        elif not jobid:
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
                if (
                    done in ["CANCELLED", "TIMEOUT", "PREEMPTED"]
                    and "--no-requeue" not in self.sbatch_args
                ):
                    # loading info about task with a specific uid
                    info_file = task.cache_dir / f"{task.uid}_info.json"
                    if info_file.exists():
                        checksum = json.loads(info_file.read_text())["checksum"]
                        if (task.cache_dir / f"{checksum}.lock").exists():
                            # for pyt3.8 we could you missing_ok=True
                            (task.cache_dir / f"{checksum}.lock").unlink()
                    cmd_re = ("scontrol", "requeue", jobid)
                    await read_and_display_async(*cmd_re, hide_display=True)
                else:
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
            if m.group("status") in ["CANCELLED", "TIMEOUT", "PREEMPTED"]:
                return m.group("status")
            elif m.group("status") in ["RUNNING", "PENDING"]:
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


class SGEWorker(DistributedWorker):
    """A worker to execute tasks on SLURM systems."""

    plugin_name = "sge"

    _cmd = "qsub"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
    )

    def __init__(
        self,
        loop=None,
        max_jobs=None,
        poll_delay=1,
        qsub_args=None,
        write_output_files=True,
        max_job_array_length=50,
        indirect_submit_host=None,
        max_threads=None,
        poll_for_result_file=True,
        default_threads_per_task=1,
        polls_before_checking_evicted=60,
        collect_jobs_delay=30,
        default_qsub_args="",
        max_mem_free=None,
    ):
        """
        Initialize SGE Worker.

        Parameters
        ----------
        poll_delay : seconds
            Delay between polls to slurmd
        qsub_args : str
            Additional qsub arguments
        max_jobs : int
            Maximum number of submitted jobs
        write_output_files : bool
            Turns on/off writing to output files for individual tasks
        max_job_array_length : int
            Number of jobs an SGE job array can hold
        indirect_submit_host : str
            Name of a submit node in the SGE cluster through which to run SGE qsub commands
        max_threads : int
            Maximum number of threads that will be scheduled for SGE submission at once
        poll_for_result_file : bool
            If true, a task is complete when its _result.pklz file exists
            If false, a task is complete when its job array is indicated complete by qstat/qacct polling
        default_threads_per_task : int
            Sets the number of slots SGE should request for a task if sgeThreads
            is not a field in the task input_spec
        polls_before_checking_evicted : int
            Number of poll_delays before running qacct to check if a task has been evicted by SGE
        collect_jobs_delay : int
            Number of seconds to wait for the list of jobs for a job array to fill

        """
        super().__init__(loop=loop, max_jobs=max_jobs)
        if not poll_delay or poll_delay < 0:
            poll_delay = 0
        self.poll_delay = poll_delay
        self.qsub_args = qsub_args or ""
        self.error = {}
        self.write_output_files = (
            write_output_files  # set to False to avoid OSError: Too many open files
        )
        self.tasks_to_run_by_threads_requested = {}
        self.output_by_jobid = {}
        self.jobid_by_task_uid = {}
        self.max_job_array_length = max_job_array_length
        self.threads_used = 0
        self.job_completed_by_jobid = {}
        self.indirect_submit_host = indirect_submit_host
        self.max_threads = max_threads
        self.default_threads_per_task = default_threads_per_task
        self.poll_for_result_file = poll_for_result_file
        self.polls_before_checking_evicted = polls_before_checking_evicted
        self.result_files_by_jobid = {}
        self.collect_jobs_delay = collect_jobs_delay
        self.task_pkls_rerun = {}
        self.default_qsub_args = default_qsub_args
        self.max_mem_free = max_mem_free

    def _prepare_runscripts(self, task, interpreter="/bin/sh", rerun=False):
        if isinstance(task, Task):
            cache_dir = task.cache_dir
            ind = None
            uid = task.uid
            try:
                task_qsub_args = task.qsub_args
            except Exception:
                task_qsub_args = self.default_qsub_args
        else:
            ind = task[0]
            cache_dir = task[-1].cache_dir
            uid = f"{task[-1].uid}_{ind}"
            try:
                task_qsub_args = task[-1].qsub_args
            except Exception:
                task_qsub_args = self.default_qsub_args

        script_dir = cache_dir / f"{self.__class__.__name__}_scripts" / uid
        script_dir.mkdir(parents=True, exist_ok=True)
        if ind is None:
            if not (script_dir / "_task.pkl").exists():
                save(script_dir, task=task)
        else:
            copyfile(task[1], script_dir / "_task.pklz")

        task_pkl = script_dir / "_task.pklz"
        if not task_pkl.exists() or not task_pkl.stat().st_size:
            raise Exception("Missing or empty task!")

        batchscript = script_dir / f"batchscript_{uid}.job"

        if task_qsub_args not in self.tasks_to_run_by_threads_requested:
            self.tasks_to_run_by_threads_requested[task_qsub_args] = []
        self.tasks_to_run_by_threads_requested[task_qsub_args].append(
            (str(task_pkl), ind, rerun)
        )

        return (
            script_dir,
            batchscript,
            task_pkl,
            ind,
            task.output_dir,
            task_qsub_args,
        )

    async def get_tasks_to_run(self, task_qsub_args, mem_free):
        # Extract the first N tasks to run
        if mem_free is not None and self.max_mem_free is not None:
            max_job_array_length = min(
                self.max_job_array_length, int(self.max_mem_free / mem_free)
            )
        else:
            max_job_array_length = self.max_job_array_length
        tasks_to_run_copy, self.tasks_to_run_by_threads_requested[task_qsub_args] = (
            self.tasks_to_run_by_threads_requested[task_qsub_args][
                :max_job_array_length
            ],
            self.tasks_to_run_by_threads_requested[task_qsub_args][
                max_job_array_length:
            ],
        )
        return tasks_to_run_copy

    async def check_for_results_files(self, jobid, threads_requested):
        for task in list(self.result_files_by_jobid[jobid]):
            if self.result_files_by_jobid[jobid][task].exists():
                del self.result_files_by_jobid[jobid][task]
                self.threads_used -= threads_requested

    async def run(self, task: "Task[DefType]", rerun: bool = False) -> "Result":
        """Worker submission API."""
        (
            script_dir,
            batch_script,
            task_pkl,
            ind,
            output_dir,
            task_qsub_args,
        ) = self._prepare_runscripts(task, rerun=rerun)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        interpreter = "/bin/sh"
        threads_requested = self.default_threads_per_task
        if "smp" in task_qsub_args:
            smp_index = task_qsub_args.split().index("smp")
            if (
                smp_index + 1 < len(task_qsub_args.split())
                and task_qsub_args.split()[smp_index + 1].isdigit()
            ):
                threads_requested = int(task_qsub_args.split()[smp_index + 1])
        # Get the amount of mem_free requested for the job
        mem_free = None
        if "mem_free" in task_qsub_args:
            mem_free_cmd = [
                word for word in task_qsub_args.split() if word.startswith("mem_free")
            ][0]
            if len(re.findall(r"\d+", mem_free_cmd)) > 0:
                mem_free = int(re.findall(r"\d+", mem_free_cmd)[0])

        if (
            len(self.tasks_to_run_by_threads_requested.get(task_qsub_args))
            <= self.max_job_array_length
        ):
            await asyncio.sleep(self.collect_jobs_delay)
        tasks_to_run = await self.get_tasks_to_run(task_qsub_args, mem_free)

        if mem_free is not None:
            summed_mem_free_cmd = re.sub(
                str(mem_free), str(len(tasks_to_run) * mem_free), mem_free_cmd
            )
            task_qsub_args = re.sub(mem_free_cmd, summed_mem_free_cmd, task_qsub_args)

        if len(tasks_to_run) > 0:
            if self.max_threads is not None:
                while self.threads_used > self.max_threads - threads_requested * len(
                    tasks_to_run
                ):
                    await asyncio.sleep(self.poll_delay)
            self.threads_used += threads_requested * len(tasks_to_run)

            python_string = f"""import sys; from pydra.engine.helpers import load_and_run; \
                task_pkls={[task_tuple for task_tuple in tasks_to_run]}; \
                task_index=int(sys.argv[1])-1; \
                load_and_run(task_pkls[task_index][0], rerun=task_pkls[task_index][1])"""
            bcmd_job = "\n".join(
                (
                    f"#!{interpreter}",
                    f"{sys.executable} {Path(batch_script).with_suffix('.py')}"
                    + " $SGE_TASK_ID",
                )
            )

            bcmd_py = python_string

            # Better runtime when the python contents are written to file
            # rather than given by cmdline arg -c
            with Path(batch_script).with_suffix(".py").open("wt") as fp:
                fp.write(bcmd_py)

            with batch_script.open("wt") as fp:
                fp.writelines(bcmd_job)

            script_dir = (
                task.cache_dir / f"{self.__class__.__task.name__}_scripts" / task.uid
            )
            script_dir.mkdir(parents=True, exist_ok=True)
            sargs = ["-t"]
            sargs.append(f"1-{len(tasks_to_run)}")
            sargs = sargs + task_qsub_args.split()

            jobname = re.search(r"(?<=-N )\S+", task_qsub_args)

            if not jobname:
                jobname = ".".join((task.name, task.uid))
                sargs.append("-N")
                sargs.append(jobname)
            output = re.search(r"(?<=-o )\S+", self.qsub_args)

            if not output:
                output_file = str(script_dir / "sge-%j.out")
                if self.write_output_files:
                    sargs.append("-o")
                    sargs.append(output_file)
            error = re.search(r"(?<=-e )\S+", self.qsub_args)
            if not error:
                error_file = str(script_dir / "sge-%j.out")
                if self.write_output_files:
                    sargs.append("-e")
                    sargs.append(error_file)
            else:
                error_file = None
            sargs.append(str(batch_script))

            await asyncio.sleep(random.uniform(0, 5))

            jobid = await self.submit_array_job(sargs, tasks_to_run, error_file)

            if self.poll_for_result_file:
                self.result_files_by_jobid[jobid] = {}
                for task_pkl, ind, rerun in tasks_to_run:
                    task = load_task(task_pkl=task_pkl, ind=ind)
                    self.result_files_by_jobid[jobid][task] = (
                        task.output_dir / "_result.pklz"
                    )

            poll_counter = 0
            while True:
                # 3 possibilities
                # False: job is still pending/working
                # True: job is complete
                # Exception: Polling / job failure
                # done = await self._poll_job(jobid)
                if self.poll_for_result_file:
                    if len(self.result_files_by_jobid[jobid]) > 0:
                        for task in list(self.result_files_by_jobid[jobid]):
                            if self.result_files_by_jobid[jobid][task].exists():
                                del self.result_files_by_jobid[jobid][task]
                                self.threads_used -= threads_requested

                    else:
                        exit_status = await self._verify_exit_code(jobid)
                        if exit_status == "ERRORED":
                            jobid = await self._rerun_job_array(
                                task.cache_dir,
                                task.uid,
                                sargs,
                                tasks_to_run,
                                error_file,
                                jobid,
                            )
                        else:
                            for task_pkl, ind, rerun in tasks_to_run:
                                if task_pkl in self.task_pkls_rerun:
                                    del self.task_pkls_rerun[task_pkl]
                            return True

                    if poll_counter >= self.polls_before_checking_evicted:
                        # Checking for evicted for jobid
                        exit_status = await self._verify_exit_code(jobid)
                        if exit_status == "ERRORED":
                            jobid = await self._rerun_job_array(
                                task.cache_dir,
                                task.uid,
                                sargs,
                                tasks_to_run,
                                error_file,
                                jobid,
                            )
                        poll_counter = 0
                    poll_counter += 1
                    await asyncio.sleep(self.poll_delay)
                else:
                    done = await self._poll_job(jobid, task.cache_dir)
                    if done:
                        if done == "ERRORED":  # If the SGE job was evicted, rerun it
                            jobid = await self._rerun_job_array(
                                task.cache_dir,
                                task.uid,
                                sargs,
                                tasks_to_run,
                                error_file,
                                jobid,
                            )
                        else:
                            self.job_completed_by_jobid[jobid] = True
                            self.threads_used -= threads_requested * len(tasks_to_run)
                            return True
                    # Don't poll exactly on the same interval to avoid overloading SGE
                    await asyncio.sleep(
                        random.uniform(max(0, self.poll_delay - 2), self.poll_delay + 2)
                    )

    async def _rerun_job_array(
        self, cache_dir, uid, sargs, tasks_to_run, error_file, evicted_jobid
    ):
        for task_pkl, ind, rerun in tasks_to_run:
            sge_task = load_task(task_pkl=task_pkl, ind=ind)
            application_task_pkl = sge_task.output_dir / "_task.pklz"
            if (
                not application_task_pkl.exists()
                or load_task(task_pkl=application_task_pkl).result() is None
                or load_task(task_pkl=application_task_pkl).result().errored
            ):
                self.task_pkls_rerun[task_pkl] = None
                info_file = cache_dir / f"{sge_task.uid}_info.json"
                if info_file.exists():
                    checksum = json.loads(info_file.read_text())["checksum"]
                    if (cache_dir / f"{checksum}.lock").exists():
                        # for pyt3.8 we could use missing_ok=True
                        (cache_dir / f"{checksum}.lock").unlink()
                    # Maybe wait a little to check if _error.pklz exists - not getting found immediately

        # If the previous job array failed, run the array's script again and get the new jobid
        jobid = await self.submit_array_job(sargs, tasks_to_run, error_file)
        self.result_files_by_jobid[jobid] = self.result_files_by_jobid[evicted_jobid]
        return jobid

    async def submit_array_job(self, sargs, tasks_to_run, error_file):
        if self.indirect_submit_host is not None:
            indirect_submit_host_prefix = []
            indirect_submit_host_prefix.append("ssh")
            indirect_submit_host_prefix.append(self.indirect_submit_host)
            indirect_submit_host_prefix.append('""export SGE_ROOT=/opt/sge;')
            rc, stdout, stderr = await read_and_display_async(
                *indirect_submit_host_prefix,
                str(Path(which("qsub")).parent / "qsub"),
                *sargs,
                '""',
                hide_display=True,
            )
        else:
            rc, stdout, stderr = await read_and_display_async(
                "qsub", *sargs, hide_display=True
            )
        jobid = re.search(r"\d+", stdout)
        if rc:
            raise RuntimeError(f"Error returned from qsub: {stderr}")
        elif not jobid:
            raise RuntimeError("Could not extract job ID")
        jobid = jobid.group()
        self.output_by_jobid[jobid] = (rc, stdout, stderr)

        for task_pkl, ind, rerun in tasks_to_run:
            self.jobid_by_task_uid[Path(task_pkl).parent.name] = jobid

        if error_file:
            error_file = str(error_file).replace("%j", jobid)
        self.error[jobid] = str(error_file).replace("%j", jobid)
        return jobid

    async def get_output_by_task_pkl(self, task_pkl):
        jobid = self.jobid_by_task_uid.get(task_pkl.parent.name)
        while jobid is None:
            jobid = self.jobid_by_task_uid.get(task_pkl.parent.name)
            await asyncio.sleep(1)
        job_output = self.output_by_jobid.get(jobid)
        while job_output is None:
            job_output = self.output_by_jobid.get(jobid)
            await asyncio.sleep(1)
        return job_output

    async def _submit_job(
        self,
        batchscript,
        name,
        uid,
        cache_dir,
        task_pkl,
        ind,
        output_dir,
        task_qsub_args,
    ):
        """Coroutine that submits task runscript and polls job until completion or error."""
        await self._submit_jobs(
            batchscript,
            name,
            uid,
            cache_dir,
            output_dir,
            task_qsub_args,
        )
        if self.poll_for_result_file:
            while True:
                result_file = output_dir / "_result.pklz"
                if result_file.exists() and str(task_pkl) not in self.task_pkls_rerun:
                    return True
                await asyncio.sleep(self.poll_delay)
        else:
            rc, stdout, stderr = await self.get_output_by_task_pkl(task_pkl)
            while True:
                jobid = self.jobid_by_task_uid.get(task_pkl.parent.name)
                if self.job_completed_by_jobid.get(jobid):
                    return True
                else:
                    await asyncio.sleep(self.poll_delay)

    async def _poll_job(self, jobid, cache_dir):
        cmd = ("qstat", "-j", jobid)
        logger.debug(f"Polling job {jobid}")
        rc, stdout, stderr = await read_and_display_async(*cmd, hide_display=True)

        if not stdout:
            # job is no longer running - check exit code
            status = await self._verify_exit_code(jobid)
            return status
        return False

    async def _verify_exit_code(self, jobid):
        cmd = ("qacct", "-j", jobid)
        rc, stdout, stderr = await read_and_display_async(*cmd, hide_display=True)
        if not stdout:
            await asyncio.sleep(10)
            rc, stdout, stderr = await read_and_display_async(*cmd, hide_display=True)

        # job is still pending/working
        if re.match(r"error: job id .* not found", stderr):
            return False

        if not stdout:
            return "ERRORED"

        # Read the qacct stdout into dictionary stdout_dict
        for line in stdout.splitlines():
            line_split = line.split()
            if len(line_split) > 1:
                if line_split[0] == "failed":
                    if not line_split[1].isdigit():
                        return "ERRORED"
                    elif not int(line_split[1]) == 0:
                        return "ERRORED"
        return True


class DaskWorker(Worker):
    """A worker to execute in parallel using Dask.distributed.
    This is an experimental implementation with limited testing.
    """

    plugin_name = "dask"

    def __init__(self, **kwargs):
        """Initialize Worker."""
        super().__init__()
        try:
            from dask.distributed import Client  # noqa: F401
        except ImportError:
            logger.critical("Please instiall Dask distributed.")
            raise
        self.client = None
        self.client_args = kwargs
        logger.debug("Initialize Dask")

    async def run(
        self,
        task: "Task[DefType]",
        rerun: bool = False,
    ) -> "Result":
        from dask.distributed import Client

        async with Client(**self.client_args, asynchronous=True) as client:
            return await client.submit(task.run, rerun)

    def close(self):
        """Finalize the internal pool of tasks."""
        pass


class PsijWorker(Worker):
    """A worker to execute tasks using PSI/J."""

    def __init__(self, **kwargs):
        """
        Initialize PsijWorker.

        Parameters
        ----------
        subtype : str
            Scheduler for PSI/J.
        """
        try:
            import psij
        except ImportError:
            logger.critical("Please install psij.")
            raise
        logger.debug("Initialize PsijWorker")
        self.psij = psij

    def make_spec(self, cmd=None, arg=None):
        """
        Create a PSI/J job specification.

        Parameters
        ----------
        cmd : str, optional
            Executable command. Defaults to None.
        arg : list, optional
            List of arguments. Defaults to None.

        Returns
        -------
        psij.JobDef
            PSI/J job specification.
        """
        spec = self.psij.JobSpec()
        spec.executable = cmd
        spec.arguments = arg

        return spec

    def make_job(self, spec, attributes):
        """
        Create a PSI/J job.

        Parameters
        ----------
        definition : psij.JobDef
            PSI/J job specification.
        attributes : any
            Job attributes.

        Returns
        -------
        psij.Job
            PSI/J job.
        """
        job = self.psij.Job()
        job.spec = spec
        return job

    async def run(
        self,
        task: "Task[DefType]",
        rerun: bool = False,
    ) -> "Result":
        """
        Run a task (coroutine wrapper).

        Raises
        ------
        Exception
            If stderr is not empty.

        Returns
        -------
        None
        """
        jex = self.psij.JobExecutor.get_instance(self.subtype)
        absolute_path = Path(__file__).parent

        cache_dir = task.cache_dir
        file_path = cache_dir / "runnable_function.pkl"
        with open(file_path, "wb") as file:
            cp.dump(task.run, file)
        func_path = absolute_path / "run_pickled.py"
        spec = self.make_spec("python", [func_path, file_path])

        if rerun:
            spec.arguments.append("--rerun")

        spec.stdout_path = cache_dir / "demo.stdout"
        spec.stderr_path = cache_dir / "demo.stderr"

        job = self.make_job(spec, None)
        jex.submit(job)
        job.wait()

        if spec.stderr_path.stat().st_size > 0:
            with open(spec.stderr_path, "r") as stderr_file:
                stderr_contents = stderr_file.read()
            raise Exception(
                f"stderr_path '{spec.stderr_path}' is not empty. Contents:\n{stderr_contents}"
            )

        return task.result()

    def close(self):
        """Finalize the internal pool of tasks."""
        pass


class PsijLocalWorker(PsijWorker):
    """A worker to execute tasks using PSI/J on the local machine."""

    subtype = "local"
    plugin_name = f"psij-{subtype}"


class PsijSlurmWorker(PsijWorker):
    """A worker to execute tasks using PSI/J using SLURM."""

    subtype = "slurm"
    plugin_name = f"psij-{subtype}"


WORKERS = {
    w.plugin_name: w
    for w in (
        DebugWorker,
        ConcurrentFuturesWorker,
        SlurmWorker,
        DaskWorker,
        SGEWorker,
        PsijLocalWorker,
        PsijSlurmWorker,
    )
}
