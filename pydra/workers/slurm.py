import asyncio
import sys
import json
import re
import typing as ty
from tempfile import gettempdir
from pathlib import Path
from shutil import copyfile
import logging
import attrs
from pydra.engine.job import Job, save
from pydra.workers import base


logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


@attrs.define
class SlurmWorker(base.Worker):
    """A worker to execute tasks on SLURM systems."""

    _cmd = "sbatch"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
    )

    poll_delay: int = attrs.field(default=1, converter=base.ensure_non_negative)
    sbatch_args: str = ""
    error: dict[str, ty.Any] = attrs.field(factory=dict)

    def __getstate__(self) -> dict[str, ty.Any]:
        """Return state for pickling."""
        state = super().__getstate__()
        del state["error"]
        return state

    def __setstate__(self, state: dict[str, ty.Any]):
        """Set state for unpickling."""
        state["error"] = {}
        super().__setstate__(state)

    def _prepare_runscripts(self, job, interpreter="/bin/sh", rerun=False):
        if isinstance(job, Job):
            cache_root = job.cache_root
            ind = None
            uid = job.uid
        else:
            assert isinstance(job, tuple), f"Expecting a job or a tuple, not {job!r}"
            assert len(job) == 2, f"Expecting a tuple of length 2, not {job!r}"
            ind = job[0]
            cache_root = job[-1].cache_root
            uid = f"{job[-1].uid}_{ind}"

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

    async def run(self, job: "Job[base.TaskType]", rerun: bool = False) -> "Result":
        """Worker submission API."""
        script_dir, batch_script = self._prepare_runscripts(job, rerun=rerun)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        script_dir = job.cache_root / f"{self.plugin_name()}_scripts" / job.uid
        sargs = self.sbatch_args.split()
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
            error_file = None
        sargs.append(str(batch_script))
        # TO CONSIDER: add random sleep to avoid overloading calls
        rc, stdout, stderr = await base.read_and_display_async(
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
                    # loading info about job with a specific uid
                    info_file = job.cache_root / f"{job.uid}_info.json"
                    if info_file.exists():
                        checksum = json.loads(info_file.read_text())["checksum"]
                        if (job.cache_root / f"{checksum}.lock").exists():
                            # for pyt3.8 we could you missing_ok=True
                            (job.cache_root / f"{checksum}.lock").unlink()
                    cmd_re = ("scontrol", "requeue", jobid)
                    await base.read_and_display_async(*cmd_re, hide_display=True)
                else:
                    return True
            await asyncio.sleep(self.poll_delay)

    async def _poll_job(self, jobid):
        cmd = ("squeue", "-h", "-j", jobid)
        logger.debug(f"Polling job {jobid}")
        rc, stdout, stderr = await base.read_and_display_async(*cmd, hide_display=True)
        if not stdout or "slurm_load_jobs error" in stderr:
            # job is no longer running - check exit code
            status = await self._verify_exit_code(jobid)
            return status
        return False

    async def _verify_exit_code(self, jobid):
        cmd = ("sacct", "-n", "-X", "-j", jobid, "-o", "JobID,State,ExitCode")
        _, stdout, _ = await base.read_and_display_async(*cmd, hide_display=True)
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


# Alias so it can be referred to as slurm.Worker
Worker = SlurmWorker
