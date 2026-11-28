import asyncio
import sys
import re
import typing as ty
from tempfile import gettempdir
from pathlib import Path
import getpass
import os
import logging
import attrs
from pydra.engine.job import Job, save
from pydra.engine.store import remove_from_cache
from pydra.workers import base

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


class SlurmJobCancelled(RuntimeError):
    """A job that was cancelled; it is never requeued."""


@attrs.define
class SlurmWorker(base.Worker):
    """A worker to execute tasks on SLURM systems.

    Parameters
    ----------
    poll_delay
        Seconds between poll cycles. Each cycle makes one ``squeue --user=<your uid>`` call
        for every job awaiting a poll.
    sbatch_args
        Extra ``sbatch`` options for every job.
    requeue_budget
        How many times one job is requeued after ``TIMEOUT``, ``PREEMPTED``,
        ``NODE_FAIL`` or ``BOOT_FAIL``.
    max_array_size
        The elements of a split node that become runnable together are submitted as
        ``sbatch --array`` of at most this many tasks (task ids ``0`` to
        ``max_array_size - 1``, so keep it below the site's ``MaxArraySize``). ``0``
        turns arrays off, and each job is then its own ``sbatch``. Arrays are also not
        used when ``sbatch_args`` set ``--output``, ``--error`` or ``--array``, and only
        the elements of one ``submit_many`` call (one round of the runner) are grouped.
    squeue_failures
        Consecutive failed ``squeue`` polls (non-zero exit, controller error or
        timeout) tolerated before the waiting jobs are failed. A failed poll is
        skipped, not read as an empty queue.
    squeue_timeout
        Seconds after which one ``squeue`` call counts as failed.

    Notes
    -----
    An array is accepted or refused as one request: on a partition whose QOS caps a user's
    submitted jobs (``MaxSubmitPU``, counting every array task) a larger array is rejected
    whole. Set ``max_array_size`` at or below that limit, or to ``0``.
    """

    batches = True
    reports_done = True

    _cmd = "sbatch"
    _requeue_states = ("TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL")

    _sacct_re = re.compile(
        r"(?P<jobid>[^|\s]*)\|(?P<status>\w*)(?: by \d+)?\+?\|"
        r"(?P<exit_code>\d+):\d+"
    )

    poll_delay: int = attrs.field(default=1, converter=base.ensure_non_negative)
    sbatch_args: str = ""
    requeue_budget: int = attrs.field(default=3, validator=attrs.validators.ge(0))
    max_array_size: int = attrs.field(default=1000, validator=attrs.validators.ge(0))
    squeue_failures: int = attrs.field(default=3, validator=attrs.validators.ge(1))
    squeue_timeout: float = attrs.field(default=60.0, validator=attrs.validators.gt(0))
    error: dict[str, ty.Any] = attrs.field(factory=dict)
    _poller: ty.Any = attrs.field(default=None, init=False, repr=False)
    _poll_loop: ty.Any = attrs.field(default=None, init=False, repr=False)
    _waiters: list = attrs.field(factory=list, init=False, repr=False)
    _active: int = attrs.field(default=0, init=False, repr=False)
    _retired: list = attrs.field(factory=list, init=False, repr=False)

    _runtime_state = (
        "error",
        "_poller",
        "_poll_loop",
        "_waiters",
        "_active",
        "_retired",
    )

    def __getstate__(self) -> dict[str, ty.Any]:
        """Return state for pickling."""
        state = super().__getstate__()
        for name in self._runtime_state:
            del state[name]
        return state

    def __setstate__(self, state: dict[str, ty.Any]):
        """Set state for unpickling."""
        state["error"] = {}
        state["_poller"] = state["_poll_loop"] = None
        state["_waiters"] = []
        state["_active"] = 0
        state["_retired"] = []
        super().__setstate__(state)

    def _prepare_runscripts(self, job, interpreter="/bin/sh", rerun=False):
        cache_root = job.cache_root
        uid = job.uid
        script_dir = cache_root / f"{self.plugin_name()}_scripts" / uid
        script_dir.mkdir(parents=True, exist_ok=True)
        if not (script_dir / "_job.pklz").exists():
            save(script_dir, job=job)

        job_pkl = script_dir / "_job.pklz"
        if not job_pkl.exists() or not job_pkl.stat().st_size:
            raise Exception("Missing or empty job!")

        batchscript = script_dir / f"batchscript_{uid}.sh"
        python_string = (
            f"""'from pydra.engine.job import load_and_run; """
            f"""load_and_run("{job_pkl}", rerun={rerun}) '"""
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

    @staticmethod
    def _apply_resources(sargs, resources):
        """Replace the worker-wide options of every family the node's resources set.

        Options of one family conflict (``--mem`` with ``--mem-per-cpu``, ``--gres=gpu``
        with ``--gpus``), so the node's resources replace the whole family.
        """
        families = []
        if resources.cpus is not None:
            families.append(("--cpus-per-task", "-c"))
        if resources.mem is not None:
            families.append(("--mem", "--mem-per-cpu", "--mem-per-gpu"))
        if resources.time is not None:
            families.append(("--time", "-t"))
        if resources.gpus is not None:
            families.append(
                (
                    "--gpus",
                    "-G",
                    "--gpus-per-node",
                    "--gpus-per-task",
                    "--gpus-per-socket",
                    "--cpus-per-gpu",
                    "--mem-per-gpu",
                )
            )
        names = {name for family in families for name in family}
        kept = []
        i = 0
        while i < len(sargs):
            token = sargs[i]
            if token.startswith("--"):
                name, has_value, value = token.partition("=")
            elif token.startswith("-") and len(token) > 1:
                name, value = token[:2], token[2:]
                has_value = bool(value)
            else:
                name, has_value, value = None, True, ""
            if name is not None and not has_value and i + 1 < len(sargs):
                value = sargs[i + 1]
            consumed = 1 if has_value else 2
            if name in names:
                i += consumed
            elif name == "--gres" and resources.gpus is not None:
                items = value.split(",")
                rest = [x for x in items if not x.startswith("gpu")]
                if rest:
                    kept.append(f"--gres={','.join(rest)}")
                i += consumed
            else:
                kept.append(token)
                i += 1
        return kept + resources.slurm_args()

    async def run(self, job: "Job[base.TaskType]", rerun: bool = False) -> "Result":
        """Worker submission API."""
        self._active += 1
        try:
            jobid, error_file = await self._submit_single(job, rerun)
            return await self._wait(job, jobid, error_file)
        finally:
            self._release(1)

    def _release(self, count):
        self._active -= count
        if not self._active:  # the last job is gone: stop polling the queue
            self._stop_poller()

    async def submit_many(self, jobs, done=None):
        """Submit the jobs of one round together and wait for all of them.

        The elements of a split node that share a name, resources, cache root and
        rerun flag go to ``sbatch --array`` in chunks of at most ``max_array_size``; any
        other job is its own ``sbatch``. Returns, in order, ``True`` for a job that
        completed and the exception that ended it otherwise.

        Parameters
        ----------
        jobs : Sequence[Job]
            The jobs of the round
        done : Callable[[Job, bool | Exception], None], optional
            Called as each job ends, before the others have
        """
        jobs = list(jobs)
        self._active += len(jobs)
        try:
            placed = await self._place(jobs)

            async def finish(job, place):
                try:
                    if isinstance(place, Exception):
                        raise place
                    outcome = await self._wait(job, *place)
                except Exception as exc:
                    outcome = exc
                if done is not None:
                    done(job, outcome)
                return outcome

            return list(
                await asyncio.gather(*(finish(j, p) for j, p in zip(jobs, placed)))
            )
        finally:
            self._release(len(jobs))

    async def _place(self, jobs):
        """Submit the jobs; the (id, error file) of each, or the exception it met"""
        groups: dict = {}
        chunks = []
        for i, job in enumerate(jobs):
            if self._array_eligible(job):
                key = (
                    job.name,
                    repr(job.resources),
                    str(job.cache_root),
                    bool(job.rerun),
                )
                groups.setdefault(key, []).append(i)
            else:
                chunks.append([i])
        for indices in groups.values():
            size = self.max_array_size
            chunks += [indices[k : k + size] for k in range(0, len(indices), size)]

        placed: list = [None] * len(jobs)

        async def submit(indices):
            chunk = [jobs[i] for i in indices]
            try:
                if len(chunk) == 1:
                    results = [await self._submit_single(chunk[0], chunk[0].rerun)]
                else:
                    results = await self._submit_array(chunk, chunk[0].rerun)
            except Exception as exc:
                results = [exc] * len(chunk)
            for i, result in zip(indices, results):
                placed[i] = result

        await asyncio.gather(*(submit(c) for c in chunks))
        return placed

    def _stop_poller(self):
        poller, self._poller = self._poller, None
        if poller is not None and not poller.done():
            poller.cancel()
            self._retired.append(poller)

    def close(self):
        """Stop the queue poller, leaving no pending task behind."""
        self._stop_poller()
        retired = [p for p in self._retired if not p.done()]
        self._retired = []
        loop = self._poll_loop
        if retired and loop is not None and not loop.is_running():
            if not loop.is_closed():
                loop.run_until_complete(
                    asyncio.gather(*retired, return_exceptions=True)
                )

    async def _wait(self, job, jobid, error_file):
        """Poll a submitted job to its end, requeueing it within the budget"""
        self.error[jobid] = error_file
        requeues = 0
        # intermittent polling
        while True:
            # 3 possibilities
            # False: job is still pending/working
            # True: job is complete
            # Exception: Polling / job failure
            done = await self._poll_job(jobid)
            if done:
                if (
                    done in self._requeue_states
                    and "--no-requeue" not in self.sbatch_args
                ):
                    if requeues >= self.requeue_budget:
                        raise RuntimeError(
                            f"Slurm job {jobid} ended {done} and the requeue budget "
                            f"({self.requeue_budget}) is spent"
                        )
                    requeues += 1
                    remove_from_cache(job.lockfile)
                    cmd_re = ("scontrol", "requeue", jobid)
                    rc, _, stderr = await base.read_and_display_async(
                        *cmd_re, hide_display=True
                    )
                    if rc:
                        raise RuntimeError(
                            f"Slurm job {jobid} ended {done} but `scontrol requeue` "
                            f"failed: {stderr.strip()}"
                        )
                else:
                    return True

    def _worker_sargs(self, job):
        sargs = self.sbatch_args.split()
        if job.resources is not None:
            sargs = self._apply_resources(sargs, job.resources)
        return sargs

    async def _sbatch(self, sargs):
        """Run ``sbatch`` and return the job id it reports."""
        # TO CONSIDER: add random sleep to avoid overloading calls
        rc, stdout, stderr = await base.read_and_display_async(
            "sbatch", *sargs, hide_display=True
        )
        jobid = re.search(r"\d+", stdout)
        if rc:
            raise RuntimeError(f"Error returned from sbatch: {stderr}")
        elif not jobid:
            raise RuntimeError("Could not extract job ID")
        return jobid.group()

    async def _submit_single(self, job, rerun):
        """Submit one job with its own ``sbatch``; return its id and error file."""
        script_dir, batch_script = self._prepare_runscripts(job, rerun=rerun)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        script_dir = job.cache_root / f"{self.plugin_name()}_scripts" / job.uid
        sargs = self._worker_sargs(job)
        jobname = re.search(r"(?<=-J )\S+|(?<=--job-name=)\S+", self.sbatch_args)
        if not jobname:
            jobname = ".".join((job.name, job.uid))
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
            error_file = error.group()
        sargs.append(str(batch_script))
        jobid = await self._sbatch(sargs)
        return jobid, error_file.replace("%j", jobid)

    # ---- array submission -------------------------------------------------------

    def _array_eligible(self, job) -> bool:
        """Whether ``job`` may be submitted as one task of an ``sbatch --array``.

        Only elements of a split node are batched, and only when the worker-wide
        ``sbatch_args`` do not set the output, error or array options, because those
        are per-array in an array submission.
        """
        if getattr(job, "state_index", None) is None or self.max_array_size < 2:
            return False
        for token in self.sbatch_args.split():
            if token.startswith(("--output", "--error", "--array")) or token[:2] in (
                "-o",
                "-e",
                "-a",
            ):
                return False
        return True

    async def _submit_array(self, jobs, rerun):
        """Submit ``jobs`` as one ``sbatch --array``; return (id, error file) per job.

        Task ``i`` runs ``jobs[i]`` from its own pickle. Standard output and error go
        to ``slurm-<array id>_<task>.out/.err`` in one directory beside the first
        job's scripts.
        """
        first = jobs[0]
        pickles = [
            self._prepare_runscripts(job, rerun=rerun)[0] / "_job.pklz" for job in jobs
        ]
        array_dir = (
            first.cache_root
            / f"{self.plugin_name()}_scripts"
            / f"array_{first.uid}_{len(jobs)}"
        )
        array_dir.mkdir(parents=True, exist_ok=True)
        manifest = array_dir / "jobs.txt"
        manifest.write_text("\n".join(str(p) for p in pickles) + "\n")
        script = array_dir / "batchscript_array.sh"
        script.write_text(
            "\n".join(
                (
                    "#!/bin/sh",
                    f"{sys.executable} -c "
                    + repr(
                        "import sys; from pathlib import Path; "
                        "from pydra.engine.job import load_and_run; "
                        "pkls = Path(sys.argv[1]).read_text().splitlines(); "
                        f"load_and_run(Path(pkls[int(sys.argv[2])]), rerun={bool(rerun)})"
                    )
                    + f' "{manifest}" "$SLURM_ARRAY_TASK_ID"',
                )
            )
            + "\n"
        )
        sargs = self._worker_sargs(first)
        if not re.search(r"(?<=-J )\S+|(?<=--job-name=)\S+", self.sbatch_args):
            sargs.append(f"--job-name={first.name}.{first.uid}")
        out = str(array_dir / "slurm-%A_%a.out")
        err = str(array_dir / "slurm-%A_%a.err")
        sargs += [f"--array=0-{len(jobs) - 1}", f"--output={out}", f"--error={err}"]
        sargs.append(str(script))
        array_id = await self._sbatch(sargs)
        return [
            (
                f"{array_id}_{i}",
                err.replace("%A", array_id).replace("%a", str(i)),
            )
            for i in range(len(jobs))
        ]

    # ---- polling ----------------------------------------------------------------

    async def _poll_job(self, jobid):
        logger.debug(f"Polling job {jobid}")
        if jobid in await self._queue_snapshot():
            return False
        # job is no longer queued or running - check exit code
        return await self._verify_exit_code(jobid)

    async def _queue_snapshot(self) -> frozenset[str]:
        """The ids this user has in the queue, from one ``squeue`` per poll cycle.

        Every job awaiting a poll shares the call of the next cycle; cycles are
        ``poll_delay`` seconds apart.
        """
        loop = asyncio.get_running_loop()
        if self._poll_loop is not loop:
            self._waiters, self._poll_loop = [], loop
        if self._poller is None or self._poller.done():
            self._poller = loop.create_task(self._poll_cycles())
        future = loop.create_future()
        self._waiters.append(future)
        return await future

    async def _poll_cycles(self):
        failures = 0
        waiters: list = []
        try:
            while self._waiters:
                waiters, self._waiters = self._waiters, []
                try:
                    queued = await asyncio.wait_for(
                        self._squeue_all(), self.squeue_timeout
                    )
                except Exception as exc:
                    if isinstance(exc, asyncio.TimeoutError):
                        exc = RuntimeError(
                            f"`{' '.join(self._squeue_cmd())}` did not finish within "
                            f"squeue_timeout={self.squeue_timeout}s"
                        )
                    failures += 1
                    if failures < self.squeue_failures:
                        logger.warning(f"squeue failed, skipping this cycle: {exc}")
                        self._waiters[:0] = waiters  # asked again next cycle
                    else:
                        failure = RuntimeError(
                            f"squeue failed {failures} times in a row: {exc}"
                        )
                        for future in waiters:
                            if not future.done():
                                future.set_exception(failure)
                        failures = 0
                else:
                    failures = 0
                    for future in waiters:
                        if not future.done():
                            future.set_result(queued)
                waiters = []
                await asyncio.sleep(self.poll_delay)
        finally:
            # a poller that dies (e.g. is cancelled) must not leave jobs waiting
            stranded = [*waiters, *self._waiters]
            self._waiters = []
            for future in stranded:
                if not future.done():
                    future.set_exception(RuntimeError("Slurm queue poller stopped"))

    @staticmethod
    def _squeue_cmd() -> tuple[str, ...]:
        # a numeric uid is accepted by --user; Slurm runs only on POSIX, but the
        # worker is importable and unit-testable elsewhere
        user = os.getuid() if hasattr(os, "getuid") else getpass.getuser()
        return ("squeue", "-h", f"--user={user}", "-r", "-o", "%i")

    async def _squeue_all(self) -> frozenset[str]:
        rc, stdout, stderr = await base.read_and_display_async(
            *self._squeue_cmd(), hide_display=True
        )
        if rc or "slurm_load_jobs error" in stderr:
            raise RuntimeError(f"squeue exited {rc}: {stderr.strip()}")
        return frozenset(stdout.split())

    async def _verify_exit_code(self, jobid):
        cmd = ("sacct", "-n", "-X", "-P", "-j", jobid, "-o", "JobID,State,ExitCode")
        _, stdout, _ = await base.read_and_display_async(*cmd, hide_display=True)
        if not stdout:
            raise RuntimeError("Job information not found")
        m = self._sacct_re.search(stdout)
        if m is None:
            raise RuntimeError(f"Could not parse sacct output: {stdout!r}")
        error_file = self.error[jobid]
        if m.group("status") == "CANCELLED":
            raise SlurmJobCancelled(
                f"Slurm job {jobid} was cancelled and is not requeued"
            )
        if int(m.group("exit_code")) != 0 or m.group("status") != "COMPLETED":
            if m.group("status") in self._requeue_states:
                return m.group("status")
            elif m.group("status") in ["RUNNING", "PENDING"]:
                return False
            # TODO: potential for requeuing
            # parsing the error message
            try:
                error_text = Path(error_file).read_text()
            except OSError:
                error_text = ""
            error_lines = [line for line in error_text.splitlines() if line]
            error_line = error_lines[-1] if error_lines else ""
            if "Exception" in error_line:
                error_message = error_line.replace("Exception: ", "")
            elif "Error" in error_line:
                error_message = error_line.replace("Exception: ", "")
            else:
                error_message = "Job failed (unknown reason - TODO)"
            raise Exception(error_message)
        return True


# Alias so it can be referred to as slurm.Worker
Worker = SlurmWorker
