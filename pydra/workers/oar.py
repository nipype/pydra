import asyncio
import os
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
class OarWorker(base.Worker):
    """A worker to execute tasks on OAR systems."""

    _cmd = "oarsub"

    poll_delay: int = attrs.field(default=1, converter=base.ensure_non_negative)
    oarsub_args: str = ""
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
                f"{sys.executable} -c " + python_string,
            )
        )
        with batchscript.open("wt") as fp:
            fp.writelines(bcmd)
        os.chmod(batchscript, 0o544)
        return script_dir, batchscript

    async def run(self, job: "Job[base.TaskType]", rerun: bool = False) -> "Result":
        """Worker submission API."""
        script_dir, batch_script = self._prepare_runscripts(job, rerun=rerun)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        script_dir = job.cache_root / f"{self.plugin_name()}_scripts" / job.uid
        sargs = self.oarsub_args.split()
        jobname = re.search(r"(?<=-n )\S+|(?<=--name=)\S+", self.oarsub_args)
        if not jobname:
            jobname = ".".join((job.name, job.uid))
            sargs.append(f"--name={jobname}")
        output = re.search(r"(?<=-O )\S+|(?<=--stdout=)\S+", self.oarsub_args)
        if not output:
            output_file = str(script_dir / "oar-%jobid%.out")
            sargs.append(f"--stdout={output_file}")
        error = re.search(r"(?<=-E )\S+|(?<=--stderr=)\S+", self.oarsub_args)
        if not error:
            error_file = str(script_dir / "oar-%jobid%.err")
            sargs.append(f"--stderr={error_file}")
        else:
            error_file = None
        sargs.append(str(batch_script))
        # TO CONSIDER: add random sleep to avoid overloading calls
        rc, stdout, stderr = await base.read_and_display_async(
            self._cmd, *sargs, hide_display=True
        )
        jobid = re.search(r"OAR_JOB_ID=(\d+)", stdout)
        if rc:
            raise RuntimeError(f"Error returned from oarsub: {stderr}")
        elif not jobid:
            raise RuntimeError("Could not extract job ID")
        jobid = jobid.group(1)
        if error_file:
            error_file = error_file.replace("%jobid%", jobid)
        self.error[jobid] = error_file.replace("%jobid%", jobid)
        # intermittent polling
        while True:
            # 4 possibilities
            # False: job is still pending/working
            # Terminated: job is complete
            # Error + idempotent: job has been stopped and resubmited with another jobid
            # Error: Job failure
            done = await self._poll_job(jobid)
            if not done:
                await asyncio.sleep(self.poll_delay)
            elif done == "Terminated":
                return True
            elif done == "Error" and "idempotent" in self.oarsub_args:
                jobid = await self._handle_resubmission(jobid, job)
                continue
            else:
                error_file = self.error[jobid]
                if not Path(error_file).exists():
                    logger.debug(
                        f"No error file for job {jobid}. Checking if job was resubmitted by OAR..."
                    )
                    jobid = await self._handle_resubmission(jobid, job)
                    if jobid:
                        continue
                    for _ in range(5):
                        if Path(error_file).exists():
                            break
                        await asyncio.sleep(1)
                    else:
                        raise RuntimeError(
                            f"OAR error file not found: {error_file}, and no resubmission detected."
                        )
                error_line = Path(error_file).read_text().split("\n")[-2]
                if "Exception" in error_line:
                    error_message = error_line.replace("Exception: ", "")
                elif "Error" in error_line:
                    error_message = error_line.replace("Error: ", "")
                else:
                    error_message = "Job failed (unknown reason - TODO)"
                raise Exception(error_message)
                return True

    async def _poll_job(self, jobid):
        cmd = ("oarstat", "-J", "-s", "-j", jobid)
        logger.debug(f"Polling job {jobid}")
        _, stdout, _ = await base.read_and_display_async(*cmd, hide_display=True)
        if not stdout:
            raise RuntimeError("Job information not found")
        status = json.loads(stdout)[jobid]
        if status in ["Waiting", "Launching", "Running", "Finishing"]:
            return False
        return status

    async def _handle_resubmission(self, jobid, job):
        logger.debug(f"Job {jobid} has been stopped. Looking for its resubmission...")
        # loading info about task with a specific uid
        info_file = job.cache_root / f"{job.uid}_info.json"
        if info_file.exists():
            checksum = json.loads(info_file.read_text())["checksum"]
            lock_file = job.cache_root / f"{checksum}.lock"
            if lock_file.exists():
                lock_file.unlink()
        cmd_re = ("oarstat", "-J", "--sql", f"resubmit_job_id='{jobid}'")
        _, stdout, _ = await base.read_and_display_async(*cmd_re, hide_display=True)
        if stdout:
            return next(iter(json.loads(stdout).keys()), None)
        else:
            return None


# Alias so it can be referred to as oar.Worker
Worker = OarWorker
