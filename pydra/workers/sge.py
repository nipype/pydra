import asyncio
import sys
import json
import re
import attrs
import typing as ty
from tempfile import gettempdir
from pathlib import Path
from shutil import copyfile, which
import random
import logging
from pydra.engine.job import Job, save, load_job
from pydra.workers import base

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


@attrs.define
class SgeWorker(base.Worker):
    """A worker to execute tasks on SLURM systems. Initialize SGE Worker.

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
        If true, a job is complete when its _result.pklz file exists
        If false, a job is complete when its job array is indicated complete by qstat/qacct polling
    default_threads_per_task : int
        Sets the number of slots SGE should request for a job if sgeThreads
        is not a field in the job input_spec
    polls_before_checking_evicted : int
        Number of poll_delays before running qacct to check if a job has been evicted by SGE
    collect_jobs_delay : int
        Number of seconds to wait for the list of jobs for a job array to fill
    """

    _cmd = "qsub"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
    )

    poll_delay: int = attrs.field(default=1, converter=base.ensure_non_negative)
    qsub_args: str = ""
    write_output_files: bool = True
    max_job_array_length: int = 50
    indirect_submit_host: str | None = None
    max_threads: int | None = None
    poll_for_result_file: bool = True
    default_threads_per_task: int = 1
    polls_before_checking_evicted: int = 60
    collect_jobs_delay: int = 30
    default_qsub_args: str = ""
    max_mem_free: int | None = None
    error: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    tasks_to_run_by_threads_requested: dict[str, ty.Any] = attrs.field(
        factory=dict, init=False
    )
    output_by_jobid: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    jobid_by_task_uid: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    threads_used: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    job_completed_by_jobid: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    result_files_by_jobid: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    job_pkls_rerun: dict[str, ty.Any] = attrs.field(factory=dict, init=False)

    _INTERNAL_DICT_ATTRS = [
        "error",
        "tasks_to_run_by_threads_requested",
        "output_by_jobid",
        "jobid_by_task_uid",
        "threads_used",
        "job_completed_by_jobid",
        "result_files_by_jobid",
        "job_pkls_rerun",
    ]

    def __getstate__(self) -> dict[str, ty.Any]:
        """Return state for pickling."""
        state = super().__getstate__()
        for atr in self._INTERNAL_DICT_ATTRS:
            del state[atr]
        return state

    def __setstate__(self, state: dict[str, ty.Any]):
        """Set state for unpickling."""
        super().__setstate__(state)
        for atr in self._INTERNAL_DICT_ATTRS:
            setattr(self, atr, {})

    def _prepare_runscripts(self, job, interpreter="/bin/sh", rerun=False):
        if isinstance(job, Job):
            cache_root = job.cache_root
            ind = None
            uid = job.uid
            try:
                task_qsub_args = job.qsub_args
            except Exception:
                task_qsub_args = self.default_qsub_args
        else:
            ind = job[0]
            cache_root = job[-1].cache_root
            uid = f"{job[-1].uid}_{ind}"
            try:
                task_qsub_args = job[-1].qsub_args
            except Exception:
                task_qsub_args = self.default_qsub_args

        script_dir = cache_root / f"{self.plugin_name()}_scripts" / uid
        script_dir.mkdir(parents=True, exist_ok=True)
        if ind is None:
            if not (script_dir / "_job.pklz").exists():
                save(script_dir, job=job)
        else:
            copyfile(job[1], script_dir / "_job.pklz")

        job_pkl = script_dir / "_job.pklz"
        if not job_pkl.exists() or not job_pkl.stat().st_size:
            raise Exception("Missing or empty job!")

        batchscript = script_dir / f"batchscript_{uid}.job"

        if task_qsub_args not in self.tasks_to_run_by_threads_requested:
            self.tasks_to_run_by_threads_requested[task_qsub_args] = []
        self.tasks_to_run_by_threads_requested[task_qsub_args].append(
            (str(job_pkl), ind, rerun)
        )

        return (
            script_dir,
            batchscript,
            job_pkl,
            ind,
            job.cache_dir,
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
        for job in list(self.result_files_by_jobid[jobid]):
            if self.result_files_by_jobid[jobid][job].exists():
                del self.result_files_by_jobid[jobid][job]
                self.threads_used -= threads_requested

    async def run(self, job: Job[base.TaskType], rerun: bool = False) -> "Result":
        """Worker submission API."""
        (
            script_dir,
            batch_script,
            job_pkl,
            ind,
            cache_dir,
            task_qsub_args,
        ) = self._prepare_runscripts(job, rerun=rerun)
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

            python_string = f"""import sys; from pydra.engine.job import load_and_run; \
                job_pkls={[task_tuple for task_tuple in tasks_to_run]}; \
                task_index=int(sys.argv[1])-1; \
                load_and_run(job_pkls[task_index][0], rerun=job_pkls[task_index][1])"""
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

            script_dir = job.cache_root / f"{self.plugin_name()}_scripts" / job.uid
            script_dir.mkdir(parents=True, exist_ok=True)
            sargs = ["-t"]
            sargs.append(f"1-{len(tasks_to_run)}")
            sargs = sargs + task_qsub_args.split()

            jobname = re.search(r"(?<=-N )\S+", task_qsub_args)

            if not jobname:
                jobname = ".".join((job.name, job.uid))
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
                for job_pkl, ind, rerun in tasks_to_run:
                    job = load_job(job_pkl=job_pkl, ind=ind)
                    self.result_files_by_jobid[jobid][job] = (
                        job.cache_dir / "_result.pklz"
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
                        for job in list(self.result_files_by_jobid[jobid]):
                            if self.result_files_by_jobid[jobid][job].exists():
                                del self.result_files_by_jobid[jobid][job]
                                self.threads_used -= threads_requested

                    else:
                        exit_status = await self._verify_exit_code(jobid)
                        if exit_status == "ERRORED":
                            jobid = await self._rerun_job_array(
                                job.cache_root,
                                job.uid,
                                sargs,
                                tasks_to_run,
                                error_file,
                                jobid,
                            )
                        else:
                            for job_pkl, ind, rerun in tasks_to_run:
                                if job_pkl in self.job_pkls_rerun:
                                    del self.job_pkls_rerun[job_pkl]
                            return True

                    if poll_counter >= self.polls_before_checking_evicted:
                        # Checking for evicted for jobid
                        exit_status = await self._verify_exit_code(jobid)
                        if exit_status == "ERRORED":
                            jobid = await self._rerun_job_array(
                                job.cache_root,
                                job.uid,
                                sargs,
                                tasks_to_run,
                                error_file,
                                jobid,
                            )
                        poll_counter = 0
                    poll_counter += 1
                    await asyncio.sleep(self.poll_delay)
                else:
                    done = await self._poll_job(jobid, job.cache_root)
                    if done:
                        if done == "ERRORED":  # If the SGE job was evicted, rerun it
                            jobid = await self._rerun_job_array(
                                job.cache_root,
                                job.uid,
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
        self, cache_root, uid, sargs, tasks_to_run, error_file, evicted_jobid
    ):
        for job_pkl, ind, rerun in tasks_to_run:
            sge_task = load_job(job_pkl=job_pkl, ind=ind)
            application_job_pkl = sge_task.cache_dir / "_job.pklz"
            if (
                not application_job_pkl.exists()
                or load_job(job_pkl=application_job_pkl).result() is None
                or load_job(job_pkl=application_job_pkl).result().errored
            ):
                self.job_pkls_rerun[job_pkl] = None
                info_file = cache_root / f"{sge_task.uid}_info.json"
                if info_file.exists():
                    checksum = json.loads(info_file.read_text())["checksum"]
                    if (cache_root / f"{checksum}.lock").exists():
                        # for pyt3.8 we could use missing_ok=True
                        (cache_root / f"{checksum}.lock").unlink()
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
            rc, stdout, stderr = await base.read_and_display_async(
                *indirect_submit_host_prefix,
                str(Path(which("qsub")).parent / "qsub"),
                *sargs,
                '""',
                hide_display=True,
            )
        else:
            rc, stdout, stderr = await base.read_and_display_async(
                "qsub", *sargs, hide_display=True
            )
        jobid = re.search(r"\d+", stdout)
        if rc:
            raise RuntimeError(f"Error returned from qsub: {stderr}")
        elif not jobid:
            raise RuntimeError("Could not extract job ID")
        jobid = jobid.group()
        self.output_by_jobid[jobid] = (rc, stdout, stderr)

        for job_pkl, ind, rerun in tasks_to_run:
            self.jobid_by_task_uid[Path(job_pkl).parent.name] = jobid

        if error_file:
            error_file = str(error_file).replace("%j", jobid)
        self.error[jobid] = str(error_file).replace("%j", jobid)
        return jobid

    async def get_output_by_job_pkl(self, job_pkl):
        jobid = self.jobid_by_task_uid.get(job_pkl.parent.name)
        while jobid is None:
            jobid = self.jobid_by_task_uid.get(job_pkl.parent.name)
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
        cache_root,
        job_pkl,
        ind,
        cache_dir,
        task_qsub_args,
    ):
        """Coroutine that submits job runscript and polls job until completion or error."""
        await self._submit_jobs(
            batchscript,
            name,
            uid,
            cache_root,
            cache_dir,
            task_qsub_args,
        )
        if self.poll_for_result_file:
            while True:
                result_file = cache_dir / "_result.pklz"
                if result_file.exists() and str(job_pkl) not in self.job_pkls_rerun:
                    return True
                await asyncio.sleep(self.poll_delay)
        else:
            rc, stdout, stderr = await self.get_output_by_job_pkl(job_pkl)
            while True:
                jobid = self.jobid_by_task_uid.get(job_pkl.parent.name)
                if self.job_completed_by_jobid.get(jobid):
                    return True
                else:
                    await asyncio.sleep(self.poll_delay)

    async def _poll_job(self, jobid, cache_root):
        cmd = ("qstat", "-j", jobid)
        logger.debug(f"Polling job {jobid}")
        rc, stdout, stderr = await base.read_and_display_async(*cmd, hide_display=True)

        if not stdout:
            # job is no longer running - check exit code
            status = await self._verify_exit_code(jobid)
            return status
        return False

    async def _verify_exit_code(self, jobid):
        cmd = ("qacct", "-j", jobid)
        rc, stdout, stderr = await base.read_and_display_async(*cmd, hide_display=True)
        if not stdout:
            await asyncio.sleep(10)
            rc, stdout, stderr = await base.read_and_display_async(
                *cmd, hide_display=True
            )

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


# Alias so it can be referred to as sge.Worker
Worker = SgeWorker
