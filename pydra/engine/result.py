"""Job I/O definitions."""

from pathlib import Path
import typing as ty
import attrs
import functools
import inspect
import io
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


_SKELETONS = ("_make_skeleton_class", "_make_skeleton_enum")
_SETSTATE = "_class_setstate"
_TRACKER = ("_DYNAMIC_CLASS_TRACKER_BY_ID", "_DYNAMIC_CLASS_TRACKER_LOCK")


class UnsupportedCloudpickleError(RuntimeError):
    """The installed cloudpickle lacks the helpers pydra intercepts to load results
    without changing the classes of the loading process"""


@functools.cache
def _tracker_args() -> dict[str, int]:
    """The position of the class tracker id in the arguments of each cloudpickle
    skeleton function, checking that every helper pydra intercepts exists"""
    missing = [
        n for n in (*_SKELETONS, _SETSTATE, *_TRACKER) if not hasattr(cp.cloudpickle, n)
    ]
    positions = {}
    for name in _SKELETONS:
        if name in missing:
            continue
        parameters = list(inspect.signature(getattr(cp.cloudpickle, name)).parameters)
        if "class_tracker_id" in parameters:
            positions[name] = parameters.index("class_tracker_id")
        else:
            missing.append(f"{name}(class_tracker_id=...)")
    if missing:
        raise UnsupportedCloudpickleError(
            f"cloudpickle {cp.__version__} has no {', '.join(missing)}, which pydra "
            "uses to load results without replacing the task classes of this process. "
            "Install a cloudpickle release that provides them (pydra is tested with "
            "cloudpickle 3.x), and report the incompatibility to pydra"
        )
    return positions


class _ClassKeepingUnpickler(pickle.Unpickler):
    """Unpickles objects whose classes (or enums) cloudpickle pickled by value without
    changing those classes when this process already holds them (when it pickled
    them): such a class is restored as it is, rather than its attributes, such as the
    function of a task, being replaced by the pickled copies"""

    def __init__(self, file: ty.BinaryIO):
        super().__init__(file)
        self._existing: set[int] = set()

    def find_class(self, module: str, name: str) -> ty.Any:
        if module == "cloudpickle.cloudpickle":
            if name in _SKELETONS:
                return functools.partial(self._make_skeleton, name)
            if name == _SETSTATE:
                return self._class_setstate
        return super().find_class(module, name)

    def _make_skeleton(self, name: str, *args: ty.Any) -> type:
        tracker_id = args[_tracker_args()[name]]
        with cp.cloudpickle._DYNAMIC_CLASS_TRACKER_LOCK:
            existing = tracker_id in cp.cloudpickle._DYNAMIC_CLASS_TRACKER_BY_ID
        klass = getattr(cp.cloudpickle, name)(*args)
        if existing:
            self._existing.add(id(klass))
        return klass

    def _class_setstate(self, obj: type, state: ty.Any) -> type:
        if id(obj) in self._existing:
            return obj
        return getattr(cp.cloudpickle, _SETSTATE)(obj, state)


def _loads(data: bytes) -> ty.Any:
    _tracker_args()
    return _ClassKeepingUnpickler(io.BytesIO(data)).load()


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
    environment_checksum: str | None = None
    platform: str | None = None
    python_version: str | None = None
    cross_context: bool = False
    """Whether the result was reused from an entry recorded in another context."""

    CLOUD_PICKLE_ATTRS = ("outputs", "task")
    CONTEXT_ATTRS = ("environment_checksum", "platform", "python_version")

    @property
    def context(self) -> dict[str, str | None]:
        """The execution context that produced the result, recorded beside it."""
        return {name: getattr(self, name) for name in self.CONTEXT_ATTRS}

    def __getstate__(self):
        state = attrs_values(self)
        for attr in self.CLOUD_PICKLE_ATTRS:
            if state[attr] is not None:
                state[attr] = cp.dumps(state[attr])
        return state

    def __setstate__(self, state):
        for attr in self.CLOUD_PICKLE_ATTRS:
            if state[attr] is not None:
                state[attr] = _loads(state[attr])
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

    clashes_to_avoid: set[Path] = set()
    for field in attrs_fields(outputs):
        value = getattr(outputs, field.name)
        # if the field is a path or it can contain a path _copyfile_single_value is run
        # to move all files and directories to the workflow directory
        new_value = copy_nested_files(
            value,
            wf_path,
            mode=FileSet.CopyMode.hardlink_or_copy,
            clashes_to_avoid=clashes_to_avoid,
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
