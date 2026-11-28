import asyncio
import sys
import re
import attrs
import typing as ty
from tempfile import gettempdir
from pathlib import Path
from shutil import copyfile, which
import random
import logging
import warnings
from pydra.engine.job import Job, save, load_job
from pydra.engine.store import remove_from_cache
from pydra.workers import base

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


@attrs.define
class SgeWorker(base.Worker):
    """A worker to execute tasks on SGE systems.

    Parameters
    ----------
    poll_delay : seconds
        Delay between polls to the scheduler
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
    collect_jobs_delay : int, optional
        Deprecated and unused: a job array is made from the jobs of one ``submit_many`` call
    pe_name : str
        Parallel environment that ``ComputeResources.cpus`` is requested from
    mem_resource : str
        Resource that ``ComputeResources.mem`` is requested as
    gpu_resource : str
        Resource that ``ComputeResources.gpus`` is requested as
    gpu_type_resource : str, optional
        Resource that ``ComputeResources.gpu_type`` is requested as; ``None`` leaves
        it unmapped
    """

    batches = True
    reports_done = True

    _cmd = "qsub"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +(?P<exit_code>\\d+):\\d+"
    )

    poll_delay: int = attrs.field(default=1, converter=base.ensure_non_negative)
    qsub_args: str = ""
    write_output_files: bool = True
    max_job_array_length: int = 50
    indirect_submit_host: str | None = None
    max_threads: int | None = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(1))
    )
    poll_for_result_file: bool = True
    default_threads_per_task: int = attrs.field(
        default=1, validator=attrs.validators.ge(1)
    )
    polls_before_checking_evicted: int = 60
    collect_jobs_delay: int | None = None
    default_qsub_args: str = ""
    pe_name: str = "smp"
    mem_resource: str = "mem_free"
    gpu_resource: str = "gpu"
    gpu_type_resource: str | None = "gpu_type"
    max_mem_free: int | None = None  # gigabytes across one job array
    error: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    output_by_jobid: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    jobid_by_task_uid: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    threads_used: int = attrs.field(default=0, init=False)
    job_completed_by_jobid: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    result_files_by_jobid: dict[str, ty.Any] = attrs.field(factory=dict, init=False)
    job_pkls_rerun: dict[str, ty.Any] = attrs.field(factory=dict, init=False)

    _INTERNAL_DICT_ATTRS = [
        "error",
        "output_by_jobid",
        "jobid_by_task_uid",
        "job_completed_by_jobid",
        "result_files_by_jobid",
        "job_pkls_rerun",
    ]

    def __attrs_post_init__(self):
        if self.collect_jobs_delay is not None:
            warnings.warn(
                "SgeWorker(collect_jobs_delay=) is unused and will be removed: "
                "jobs are grouped into arrays by submit_many",
                DeprecationWarning,
                stacklevel=3,
            )

    def __getstate__(self) -> dict[str, ty.Any]:
        """Return state for pickling."""
        state = super().__getstate__()
        for atr in self._INTERNAL_DICT_ATTRS:
            del state[atr]
        del state["threads_used"]
        return state

    def __setstate__(self, state: dict[str, ty.Any]):
        """Set state for unpickling."""
        super().__setstate__(state)
        for atr in self._INTERNAL_DICT_ATTRS:
            setattr(self, atr, {})
        self.threads_used = 0

    def _prepare_runscripts(self, job, interpreter="/bin/sh", rerun=False):
        if isinstance(job, Job):
            cache_root = job.cache_root
            ind = None
            uid = job.uid
            resources = job.resources
            try:
                task_qsub_args = job.qsub_args
            except Exception:
                task_qsub_args = self.default_qsub_args
        else:
            ind = job[0]
            cache_root = job[-1].cache_root
            uid = f"{job[-1].uid}_{ind}"
            resources = job[-1].resources
            try:
                task_qsub_args = job[-1].qsub_args
            except Exception:
                task_qsub_args = self.default_qsub_args
        if resources is not None:
            task_qsub_args = self._apply_resources(task_qsub_args, resources)

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

        return (
            script_dir,
            batchscript,
            job_pkl,
            ind,
            job.cache_dir,
            task_qsub_args,
        )

    def _apply_resources(self, qsub_args, resources):
        """Replace the arguments the node's resources set, so the node wins."""
        overridden = set()
        if resources.mem is not None:
            overridden.add(self.mem_resource)
        if resources.time is not None:
            overridden.add("h_rt")
        if resources.gpus is not None:
            overridden.add(self.gpu_resource)
        if resources.gpu_type is not None and self.gpu_type_resource is not None:
            overridden.add(self.gpu_type_resource)
        tokens = qsub_args.split()
        kept = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "-pe" and resources.cpus is not None:
                i += 3
            elif tokens[i] == "-l" and i + 1 < len(tokens):
                items = [
                    item
                    for item in tokens[i + 1].split(",")
                    if item.split("=")[0] not in overridden
                ]
                if items:
                    kept += ["-l", ",".join(items)]
                i += 2
            else:
                kept.append(tokens[i])
                i += 1
        new = resources.sge_args(
            pe=self.pe_name,
            mem=self.mem_resource,
            gpu=self.gpu_resource,
            gpu_type=self.gpu_type_resource,
        )
        return " ".join(kept + new)

    def _requested(self, qsub_args):
        """The threads and memory (``mem_free_cmd``) a job's qsub arguments ask for."""
        tokens = qsub_args.split()
        threads = self.default_threads_per_task
        if "-pe" in tokens:
            i = tokens.index("-pe")
            if i + 2 < len(tokens) and tokens[i + 2].isdigit():
                threads = int(tokens[i + 2])
        if threads < 1:
            raise ValueError(f"an SGE job needs at least one thread ({qsub_args!r})")
        if self.max_threads is not None and threads > self.max_threads:
            raise ValueError(
                f"an SGE job asking for {threads} threads never starts under "
                f"max_threads={self.max_threads} ({qsub_args!r})"
            )
        mem_free = mem_free_cmd = None
        for word in tokens:
            if word.startswith(self.mem_resource):
                mem_free_cmd = word
                digits = re.findall(r"\d+", word)
                if digits:
                    mem_free = int(digits[0])
                break
        return threads, mem_free, mem_free_cmd

    @staticmethod
    def _gigabytes(mem_free_cmd):
        """The memory in a ``name=<digits><K|M|G|T>`` argument, in gigabytes.

        A bare number is bytes, as SGE reads it.
        """
        match = re.search(r"=(\d+)([KMGT]?)", mem_free_cmd.upper())
        scale = {"": 1 / 1024**3, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}
        return int(match.group(1)) * scale[match.group(2)]

    def _array_length(self, mem_free_cmd):
        """How many jobs one array may hold, given the memory each asks for"""
        if mem_free_cmd is not None and self.max_mem_free is not None:
            return min(
                self.max_job_array_length,
                max(1, int(self.max_mem_free / self._gigabytes(mem_free_cmd))),
            )
        return self.max_job_array_length

    async def check_for_results_files(self, jobid, threads_requested):
        for job in list(self.result_files_by_jobid[jobid]):
            if self.result_files_by_jobid[jobid][job].exists():
                del self.result_files_by_jobid[jobid][job]
                self.threads_used -= threads_requested

    async def run(self, job: Job[base.TaskType], rerun: bool = False) -> "Result":
        """Worker submission API: a round of one job."""
        (outcome,) = await self._submit_round([job], [rerun])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def submit_many(self, jobs, done=None):
        """Submit the jobs of one round together and wait for all of them.

        Jobs with the same ``qsub`` arguments go to one ``qsub -t`` array, in chunks of
        at most ``max_job_array_length`` (fewer when ``max_mem_free`` allows less).
        Returns, in order, ``True`` for a job that completed and the exception that
        ended it otherwise.

        Parameters
        ----------
        jobs : Sequence[Job]
            The jobs of the round
        done : Callable[[Job, bool | Exception], None], optional
            Called as each job ends, before the others have
        """
        jobs = list(jobs)
        return await self._submit_round(jobs, [job.rerun for job in jobs], done)

    async def _submit_round(self, jobs, reruns, done=None):
        outcomes: list = [None] * len(jobs)
        groups: dict = {}
        for i, (job, rerun) in enumerate(zip(jobs, reruns)):
            try:
                script_dir, batch_script, job_pkl, ind, _, qsub_args = (
                    self._prepare_runscripts(job, rerun=rerun)
                )
            except Exception as exc:
                outcomes[i] = exc
                if done is not None:
                    done(job, exc)
                continue
            groups.setdefault(qsub_args, []).append(
                (i, script_dir, batch_script, (str(job_pkl), ind, rerun))
            )

        async def array(qsub_args, chunk):
            first = jobs[chunk[0][0]]
            try:
                outcome = await self._run_array(
                    first, chunk[0][1], chunk[0][2], qsub_args, [c[3] for c in chunk]
                )
            except Exception as exc:
                outcome = exc
            for i, *_ in chunk:
                outcomes[i] = outcome
                if done is not None:
                    done(jobs[i], outcome)

        runs = []
        for qsub_args, members in groups.items():
            try:
                threads, _, mem_cmd = self._requested(qsub_args)
            except ValueError as exc:  # refused: this group's jobs fail, the rest run
                for i, *_ in members:
                    outcomes[i] = exc
                    if done is not None:
                        done(jobs[i], exc)
                continue
            size = self._array_length(mem_cmd)
            if self.max_threads is not None:
                size = min(size, max(1, self.max_threads // threads))
            for start in range(0, len(members), size):
                runs.append(array(qsub_args, members[start : start + size]))
        await asyncio.gather(*runs)
        return outcomes

    async def _run_array(
        self, job, script_dir, batch_script, task_qsub_args, tasks_to_run
    ):
        """Submit ``tasks_to_run`` as one ``qsub -t`` array and wait for it to end"""
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        interpreter = "/bin/sh"
        cache_root, uid = job.cache_root, job.uid
        threads_requested, mem_free, mem_free_cmd = self._requested(task_qsub_args)

        if mem_free is not None:
            summed_mem_free_cmd = re.sub(
                str(mem_free), str(len(tasks_to_run) * mem_free), mem_free_cmd
            )
            task_qsub_args = re.sub(mem_free_cmd, summed_mem_free_cmd, task_qsub_args)

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

        jobid = await self.submit_array_job(sargs, tasks_to_run, error_file)

        if self.poll_for_result_file:
            self.result_files_by_jobid[jobid] = {}
            for job_pkl, ind, rerun in tasks_to_run:
                element = load_job(job_pkl=job_pkl, ind=ind)
                self.result_files_by_jobid[jobid][element] = (
                    element.cache_dir / "_result.pklz"
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
                    for element in list(self.result_files_by_jobid[jobid]):
                        if self.result_files_by_jobid[jobid][element].exists():
                            del self.result_files_by_jobid[jobid][element]
                            self.threads_used -= threads_requested

                else:
                    exit_status = await self._verify_exit_code(jobid)
                    if exit_status == "ERRORED":
                        jobid = await self._rerun_job_array(
                            cache_root,
                            uid,
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
                            cache_root,
                            uid,
                            sargs,
                            tasks_to_run,
                            error_file,
                            jobid,
                        )
                    poll_counter = 0
                poll_counter += 1
                await asyncio.sleep(self.poll_delay)
            else:
                done = await self._poll_job(jobid, cache_root)
                if done:
                    if done == "ERRORED":  # If the SGE job was evicted, rerun it
                        jobid = await self._rerun_job_array(
                            cache_root,
                            uid,
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
                remove_from_cache(sge_task.lockfile)

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
