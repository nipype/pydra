import typing as ty
from pathlib import Path
import cloudpickle as cp
import logging
from pydra.engine.job import Job
from . import base

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


class PsijWorker(base.Worker):
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
        task : psij.JobDef
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
        job: Job[base.TaskType],
        rerun: bool = False,
    ) -> "Result":
        """
        Run a job (coroutine wrapper).

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

        cache_dir = job.cache_dir
        file_path = cache_dir / "runnable_function.pkl"
        with open(file_path, "wb") as file:
            cp.dump(job.run, file)
        func_path = absolute_path / "run_pickled.py"
        spec = self.make_spec("python", [func_path, file_path])

        if rerun:
            spec.arguments.append("--rerun")

        spec.stdout_path = cache_dir / "demo.stdout"
        spec.stderr_path = cache_dir / "demo.stderr"

        psij_job = self.make_job(spec, None)
        jex.submit(psij_job)
        psij_job.wait()

        if spec.stderr_path.stat().st_size > 0:
            with open(spec.stderr_path, "r") as stderr_file:
                stderr_contents = stderr_file.read()
            raise Exception(
                f"stderr_path '{spec.stderr_path}' is not empty. Contents:\n{stderr_contents}"
            )

        return job.result()

    def close(self):
        """Finalize the internal pool of tasks."""
        pass
