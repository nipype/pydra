"""Job I/O definitions."""

from pathlib import Path
import typing as ty
import attrs
import pickle
import time
import os
import cloudpickle as cp
import getpass
from time import strftime
from filelock import SoftFileLock
from fileformats.generic import FileSet
from pydra.utils.general import (
    attrs_values,
    attrs_fields,
    is_workflow,
)
from pydra.utils.typing import copy_nested_files
from pydra.compose import workflow, base

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


TaskType = ty.TypeVar("TaskType", bound=base.Task)
OutputsType = ty.TypeVar("OutputsType", bound=base.Outputs)


@attrs.define(kw_only=True)
class Runtime:
    """Represent run time metadata."""

    rss_peak_gb: ty.Optional[float] = None
    """Peak in consumption of physical RAM."""
    vms_peak_gb: ty.Optional[float] = None
    """Peak in consumption of virtual memory."""
    cpu_peak_percent: ty.Optional[float] = None
    """Peak in cpu consumption."""


@attrs.define(kw_only=True)
class Result(ty.Generic[OutputsType]):
    """Metadata regarding the outputs of processing."""

    cache_dir: Path
    outputs: OutputsType | None = None
    runtime: Runtime | None = None
    errored: bool = False
    task: base.Task[OutputsType] | None = None

    CLOUD_PICKLE_ATTRS = ("outputs", "task")

    def __getstate__(self):
        state = attrs_values(self)
        for attr in self.CLOUD_PICKLE_ATTRS:
            if state[attr] is not None:
                state[attr] = cp.dumps(state[attr])
        return state

    def __setstate__(self, state):
        for attr in self.CLOUD_PICKLE_ATTRS:
            if state[attr] is not None:
                state[attr] = cp.loads(state[attr])
        for name, val in state.items():
            setattr(self, name, val)

    def get_output_field(self, field_name):
        """Used in get_values in Workflow

        Parameters
        ----------
        field_name : `str`
            Name of field in LazyField object
        """
        if field_name == "all_":
            return attrs_values(self.outputs)
        else:
            return getattr(self.outputs, field_name)

    @property
    def errors(self):
        if self.errored:
            error_file = self.cache_dir / "_error.pklz"
            if error_file.exists():
                with open(error_file, "rb") as f:
                    return cp.load(f)
        return None

    @property
    def job(self):
        job_pkl = self.cache_dir / "_job.pklz"
        if not job_pkl.exists():
            return None
        with open(job_pkl, "rb") as f:
            return cp.load(f)

    @property
    def return_values(self):
        return_values_pkl = self.cache_dir / "_return_values.pklz"
        if not return_values_pkl.exists():
            return None
        with open(return_values_pkl, "rb") as f:
            return cp.load(f)


@attrs.define(kw_only=True)
class RuntimeSpec:
    """
    Specification for a job.

    From CWL::

        InlineJavascriptRequirement
        SchemaDefRequirement
        DockerRequirement
        SoftwareRequirement
        InitialWorkDirRequirement
        EnvVarRequirement
        ShellCommandRequirement
        ResourceRequirement

        InlineScriptRequirement

    """

    outdir: ty.Optional[str] = None
    container: ty.Optional[str] = "shell"
    network: bool = False


def load_result(
    checksum: str,
    readonly_caches: list[Path],
    retries: int = 10,
    polling_interval: float = 0.1,
) -> Result | None:
    """
    Restore a result from the cache.

    Parameters
    ----------
    checksum : :obj:`str`
        Unique identifier of the job to be loaded.
    readonly_caches : :obj:`list` of :obj:`os.pathlike`
        List of cache directories, in order of priority, where
        the checksum will be looked for.
    retries : :obj:`int`
        Number of times to retry loading the result if the file is not
        completely written.
    polling_interval : :obj:`float`
        Time to wait between retries.

    Returns
    -------
    result : :obj:`Result` | None
        The result object if found, otherwise None.

    """
    if not readonly_caches:
        return None
    # TODO: if there are issues with loading, we might need to
    # TODO: sleep and repeat loads (after checking that there are no lock files!)
    for location in readonly_caches:
        if (location / checksum).exists():
            result_file = location / checksum / "_result.pklz"
            if result_file.exists() and result_file.stat().st_size > 0:
                # Load the result file, retrying if necessary while waiting for the file
                # to be written completely.
                for _ in range(retries):
                    try:
                        with open(result_file, "rb") as fp:
                            return cp.load(fp)
                    except (pickle.UnpicklingError, EOFError):
                        # if the file is not finished writing
                        # wait and retry
                        time.sleep(polling_interval)
            return None
    return None


def save(
    task_path: Path,
    result: "Result | None" = None,
    job: "Job[TaskType] | None" = None,
    return_values: dict[str, ty.Any] | None = None,
    name_prefix: str = None,
) -> None:
    """
    Save a :class:`~pydra.compose.base.Task` object and/or results.

    Parameters
    ----------
    task_path : :obj:`Path`
        Write directory
    result : :obj:`Result`
        Result to pickle and write
    job : :class:`~pydra.compose.base.Task`
        Job to pickle and write
    return_values : :obj:`dict`
        Return values to pickle and write
    """

    if job is None and result is None:
        raise ValueError("Nothing to be saved")

    if not isinstance(task_path, Path):
        task_path = Path(task_path)
    task_path.mkdir(parents=True, exist_ok=True)
    if name_prefix is None:
        name_prefix = ""

    lockfile = task_path.parent / (task_path.name + "_save.lock")
    with SoftFileLock(lockfile):
        if result:
            if result.task and is_workflow(result.task) and result.outputs is not None:
                # copy files to the workflow directory
                result.outputs = copyfile_workflow(
                    wf_path=task_path, outputs=result.outputs
                )
            with (task_path / f"{name_prefix}_result.pklz").open("wb") as fp:
                cp.dump(result, fp)
        if job:
            with (task_path / f"{name_prefix}_job.pklz").open("wb") as fp:
                cp.dump(job, fp)
        if return_values:
            with (task_path / f"{name_prefix}_return_values.pklz").open("wb") as fp:
                cp.dump(job, fp)


def copyfile_workflow(
    wf_path: os.PathLike, outputs: workflow.Outputs
) -> workflow.Outputs:
    """if file in the wf results, the file will be copied to the workflow directory"""

    for field in attrs_fields(outputs):
        value = getattr(outputs, field.name)
        # if the field is a path or it can contain a path _copyfile_single_value is run
        # to move all files and directories to the workflow directory
        new_value = copy_nested_files(
            value, wf_path, mode=FileSet.CopyMode.hardlink_or_copy
        )
        setattr(outputs, field.name, new_value)
    return outputs


def gather_runtime_info(fname):
    """
    Extract runtime information from a file.

    Parameters
    ----------
    fname : :obj:`os.pathlike`
        The file containing runtime information

    Returns
    -------
    runtime : :obj:`Runtime`
        A runtime object containing the collected information.

    """

    runtime = Runtime(rss_peak_gb=None, vms_peak_gb=None, cpu_peak_percent=None)

    # Read .prof file in and set runtime values
    data = [
        [float(el) for el in line.strip().split(",")]
        for line in Path(fname).read_text().splitlines()
    ]
    if data:
        runtime.rss_peak_gb = max([val[2] for val in data]) / 1024
        runtime.vms_peak_gb = max([val[3] for val in data]) / 1024
        runtime.cpu_peak_percent = max([val[1] for val in data])

    """
    runtime.prof_dict = {
        'time': vals[:, 0].tolist(),
        'cpus': vals[:, 1].tolist(),
        'rss_GiB': (vals[:, 2] / 1024).tolist(),
        'vms_GiB': (vals[:, 3] / 1024).tolist(),
    }
    """
    return runtime


def create_checksum(name, inputs):
    """
    Generate a checksum name for a given combination of job name and inputs.

    Parameters
    ----------
    name : :obj:`str`
        Job name.
    inputs : :obj:`str`
        String of inputs.

    """
    return "-".join((name, inputs))


def record_error(error_path, error):
    """Write an error file."""

    error_message = str(error)

    resultfile = error_path / "_result.pklz"
    if not resultfile.exists():
        error_message += """\n
    When creating this error file, the results file corresponding
    to the job could not be found."""

    name_checksum = str(error_path.name)
    timeofcrash = strftime("%Y%m%d-%H%M%S")
    try:
        login_name = getpass.getuser()
    except KeyError:
        login_name = f"UID{os.getuid():d}"

    full_error = {
        "time of crash": timeofcrash,
        "login name": login_name,
        "name with checksum": name_checksum,
        "error message": error,
    }

    with (error_path / "_error.pklz").open("wb") as fp:
        cp.dump(full_error, fp)

    return error_path / "_error.pklz"
