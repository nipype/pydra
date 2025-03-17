"""Administrative support for the engine framework."""

import asyncio
import asyncio.subprocess as asp
from pathlib import Path
import os
import inspect
import sys
import getpass
import typing as ty
import subprocess as sp
import re
from time import strftime
from traceback import format_exception
import attrs
from filelock import SoftFileLock, Timeout
import cloudpickle as cp
from fileformats.core import FileSet
from pydra.utils.typing import StateArray


if ty.TYPE_CHECKING:
    from .specs import TaskDef, Result, WorkflowOutputs, WorkflowDef
    from .core import Task
    from pydra.design.base import Field
    from pydra.engine.lazy import LazyField


PYDRA_ATTR_METADATA = "__PYDRA_METADATA__"

DefType = ty.TypeVar("DefType", bound="TaskDef")


def plot_workflow(
    workflow_task: "WorkflowDef",
    out_dir: Path,
    plot_type: str = "simple",
    export: ty.Sequence[str] | None = None,
    name: str | None = None,
    output_dir: Path | None = None,
    lazy: ty.Sequence[str] | ty.Set[str] = (),
):
    """creating a graph - dotfile and optionally exporting to other formats"""
    from .core import Workflow

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construct the workflow object with all of the fields lazy
    wf = Workflow.construct(workflow_task, lazy=lazy)

    if not name:
        name = f"graph_{type(workflow_task).__name__}"
    if plot_type == "simple":
        graph = wf.graph()
        dotfile = graph.create_dotfile_simple(outdir=out_dir, name=name)
    elif plot_type == "nested":
        graph = wf.graph()
        dotfile = graph.create_dotfile_nested(outdir=out_dir, name=name)
    elif plot_type == "detailed":
        graph = wf.graph(detailed=True)
        dotfile = graph.create_dotfile_detailed(outdir=out_dir, name=name)
    else:
        raise Exception(
            f"type of the graph can be simple, detailed or nested, "
            f"but {plot_type} provided"
        )
    if not export:
        return dotfile
    else:
        if export is True:
            export = ["png"]
        elif isinstance(export, str):
            export = [export]
        formatted_dot = []
        for ext in export:
            formatted_dot.append(graph.export_graph(dotfile=dotfile, ext=ext))
        return dotfile, formatted_dot


def attrs_fields(definition, exclude_names=()) -> list[attrs.Attribute]:
    """Get the fields of a definition, excluding some names."""
    return [
        field for field in definition.__attrs_attrs__ if field.name not in exclude_names
    ]


def attrs_values(obj, **kwargs) -> dict[str, ty.Any]:
    """Get the values of an attrs object."""
    return {
        n: v
        for n, v in attrs.asdict(obj, recurse=False, **kwargs).items()
        if not n.startswith("_")
    }


def list_fields(definition: "type[TaskDef] | TaskDef") -> list["Field"]:
    """List the fields of a task definition"""
    if not inspect.isclass(definition):
        definition = type(definition)
    if not attrs.has(definition):
        return []
    return [
        f.metadata[PYDRA_ATTR_METADATA]
        for f in attrs.fields(definition)
        if PYDRA_ATTR_METADATA in f.metadata
    ]


def fields_dict(definition: "type[TaskDef] | TaskDef") -> dict[str, "Field"]:
    """Returns the fields of a definition in a dictionary"""
    return {f.name: f for f in list_fields(definition)}


# from .specs import MultiInputFile, MultiInputObj, MultiOutputObj, MultiOutputFile


def from_list_if_single(obj: ty.Any) -> ty.Any:
    """Converts a list to a single item if it is of length == 1"""

    if obj is attrs.NOTHING:
        return obj
    if is_lazy(obj):
        return obj
    obj = list(obj)
    if len(obj) == 1:
        return obj[0]
    return obj


def print_help(defn: "TaskDef[DefType]") -> list[str]:
    """Visit a task object and print its input/output interface."""
    from pydra.design.base import NO_DEFAULT

    lines = [f"Help for {defn.__class__.__name__}"]
    if list_fields(defn):
        lines += ["Input Parameters:"]
    for f in list_fields(defn):
        if (defn._task_type == "python" and f.name == "function") or (
            defn._task_type == "workflow" and f.name == "constructor"
        ):
            continue
        default = ""
        if f.default is not NO_DEFAULT and not f.name.startswith("_"):
            default = f" (default: {f.default})"
        try:
            name = f.type.__name__
        except AttributeError:
            name = str(f.type)
        lines += [f"- {f.name}: {name}{default}"]
    output_klass = defn.Outputs
    if list_fields(output_klass):
        lines += ["Output Parameters:"]
    for f in list_fields(output_klass):
        try:
            name = f.type.__name__
        except AttributeError:
            name = str(f.type)
        lines += [f"- {f.name}: {name}"]
    print("\n".join(lines))
    return lines


def load_result(checksum, cache_locations):
    """
    Restore a result from the cache.

    Parameters
    ----------
    checksum : :obj:`str`
        Unique identifier of the task to be loaded.
    cache_locations : :obj:`list` of :obj:`os.pathlike`
        List of cache directories, in order of priority, where
        the checksum will be looked for.

    """
    if not cache_locations:
        return None
    # TODO: if there are issues with loading, we might need to
    # TODO: sleep and repeat loads (after checking that there are no lock files!)
    for location in cache_locations:
        if (location / checksum).exists():
            result_file = location / checksum / "_result.pklz"
            if result_file.exists() and result_file.stat().st_size > 0:
                return cp.loads(result_file.read_bytes())
            return None
    return None


def save(
    task_path: Path,
    result: "Result | None" = None,
    task: "Task[DefType] | None" = None,
    name_prefix: str = None,
) -> None:
    """
    Save a :class:`~pydra.engine.core.TaskBase` object and/or results.

    Parameters
    ----------
    task_path : :obj:`Path`
        Write directory
    result : :obj:`Result`
        Result to pickle and write
    task : :class:`~pydra.engine.core.TaskBase`
        Task to pickle and write
    """
    from pydra.engine.core import is_workflow

    if task is None and result is None:
        raise ValueError("Nothing to be saved")

    if not isinstance(task_path, Path):
        task_path = Path(task_path)
    task_path.mkdir(parents=True, exist_ok=True)
    if name_prefix is None:
        name_prefix = ""

    lockfile = task_path.parent / (task_path.name + "_save.lock")
    with SoftFileLock(lockfile):
        if result:
            if (
                result.definition
                and is_workflow(result.definition)
                and result.outputs is not None
            ):
                # copy files to the workflow directory
                result.outputs = copyfile_workflow(
                    wf_path=task_path, outputs=result.outputs
                )
            with (task_path / f"{name_prefix}_result.pklz").open("wb") as fp:
                cp.dump(result, fp)
        if task:
            with (task_path / f"{name_prefix}_task.pklz").open("wb") as fp:
                cp.dump(task, fp)


def copyfile_workflow(
    wf_path: os.PathLike, outputs: "WorkflowOutputs"
) -> "WorkflowOutputs":
    """if file in the wf results, the file will be copied to the workflow directory"""
    from .helpers_file import copy_nested_files

    for field in attrs_fields(outputs):
        value = getattr(outputs, field.name)
        # if the field is a path or it can contain a path _copyfile_single_value is run
        # to move all files and directories to the workflow directory
        new_value = copy_nested_files(value, wf_path, mode=FileSet.CopyMode.hardlink)
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
    from .specs import Runtime

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


async def read_stream_and_display(stream, display):
    """
    Read from stream line by line until EOF, display, and capture the lines.

    See Also
    --------
    This `discussion on StackOverflow
    <https://stackoverflow.com/questions/17190221>`__.

    """
    output = []
    while True:
        line = await stream.readline()
        if not line:
            break
        output.append(line)
        if display is not None:
            display(line)  # assume it doesn't block
    return b"".join(output).decode()


async def read_and_display_async(*cmd, hide_display=False, strip=False):
    """
    Capture standard input and output of a process, displaying them as they arrive.

    Works line-by-line.

    """
    # start process
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asp.PIPE, stderr=asp.PIPE
    )

    stdout_display = sys.stdout.buffer.write if not hide_display else None
    stderr_display = sys.stderr.buffer.write if not hide_display else None
    # read child's stdout/stderr concurrently (capture and display)
    try:
        stdout, stderr = await asyncio.gather(
            read_stream_and_display(process.stdout, stdout_display),
            read_stream_and_display(process.stderr, stderr_display),
        )
    except Exception:
        process.kill()
        raise
    finally:
        # wait for the process to exit
        rc = await process.wait()
    if strip:
        return rc, stdout.strip(), stderr
    else:
        return rc, stdout, stderr


def read_and_display(*cmd, strip=False, hide_display=False):
    """Capture a process' standard output."""
    try:
        process = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    except Exception:
        # TODO editing some tracing?
        raise

    if strip:
        return (
            process.returncode,
            process.stdout.decode("utf-8").strip(),
            process.stderr.decode("utf-8"),
        )
    else:
        return (
            process.returncode,
            process.stdout.decode("utf-8"),
            process.stderr.decode("utf-8"),
        )


def execute(cmd, strip=False):
    """
    Run the event loop with coroutine.

    Uses :func:`read_and_display_async` unless a loop is
    already running, in which case :func:`read_and_display`
    is used.

    Parameters
    ----------
    cmd : :obj:`list` or :obj:`tuple`
        The command line to be executed.
    strip : :obj:`bool`
        TODO

    """
    rc, stdout, stderr = read_and_display(*cmd, strip=strip)
    """
    loop = get_open_loop()
    if loop.is_running():
        rc, stdout, stderr = read_and_display(*cmd, strip=strip)
    else:
        rc, stdout, stderr = loop.run_until_complete(
            read_and_display_async(*cmd, strip=strip)
        )
    """
    return rc, stdout, stderr


def create_checksum(name, inputs):
    """
    Generate a checksum name for a given combination of task name and inputs.

    Parameters
    ----------
    name : :obj:`str`
        Task name.
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
    to the task could not be found."""

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


def get_open_loop():
    """
    Get current event loop.

    If the loop is closed, a new
    loop is created and set as the current event loop.

    Returns
    -------
    loop : :obj:`asyncio.EventLoop`
        The current event loop

    """
    if os.name == "nt":
        loop = asyncio.ProactorEventLoop()  # for subprocess' pipes on Windows
    else:
        try:
            loop = asyncio.get_event_loop()
        # in case RuntimeError: There is no current event loop in thread 'MainThread'
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
    return loop


def get_available_cpus():
    """
    Return the number of CPUs available to the current process or, if that is not
    available, the total number of CPUs on the system.

    Returns
    -------
    n_proc : :obj:`int`
        The number of available CPUs.
    """
    # Will not work on some systems or if psutil is not installed.
    # See https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_affinity
    try:
        import psutil

        return len(psutil.Process().cpu_affinity())
    except (AttributeError, ImportError, NotImplementedError):
        pass

    # Not available on all systems, including macOS.
    # See https://docs.python.org/3/library/os.html#os.sched_getaffinity
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    # Last resort
    return os.cpu_count()


def load_and_run(task_pkl: Path, rerun: bool = False) -> Path:
    """
    loading a task from a pickle file, settings proper input
    and running the task

    Parameters
    ----------
    task_pkl : :obj:`Path`
        The path to pickled task file

    Returns
    -------
    resultfile : :obj:`Path`
        The path to the pickled result file
    """

    from .specs import Result

    try:
        task: Task[DefType] = load_task(task_pkl=task_pkl)
    except Exception:
        if task_pkl.parent.exists():
            etype, eval, etr = sys.exc_info()
            traceback = format_exception(etype, eval, etr)
            errorfile = record_error(task_pkl.parent, error=traceback)
            result = Result(output=None, runtime=None, errored=True, definition=None)
            save(task_pkl.parent, result=result)
        raise

    resultfile = task.output_dir / "_result.pklz"
    try:
        if task.submitter.worker.is_async:
            task.submitter.loop.run_until_complete(
                task.submitter.worker.run_async(task, rerun=rerun)
            )
        else:
            task.submitter.worker.run(task, rerun=rerun)
    except Exception as e:
        # creating result and error files if missing
        errorfile = task.output_dir / "_error.pklz"
        if not errorfile.exists():  # not sure if this is needed
            etype, eval, etr = sys.exc_info()
            traceback = format_exception(etype, eval, etr)
            errorfile = record_error(task.output_dir, error=traceback)
        if not resultfile.exists():  # not sure if this is needed
            result = Result(output=None, runtime=None, errored=True, definition=None)
            save(task.output_dir, result=result)
        e.add_note(f" full crash report is here: {errorfile}")
        raise
    return resultfile


# async def load_and_run_async(task_pkl):
#     """
#     loading a task from a pickle file, settings proper input
#     and running the workflow
#     """
#     task = load_task(task_pkl=task_pkl)
#     await task()


def load_task(task_pkl: Path | str) -> "Task[DefType]":
    """loading a task from a pickle file, settings proper input for the specific ind"""
    if isinstance(task_pkl, str):
        task_pkl = Path(task_pkl)
    task = cp.loads(task_pkl.read_bytes())
    return task


def position_sort(args):
    """
    Sort objects by position, following Python indexing conventions.

    Ordering is positive positions, lowest to highest, followed by unspecified
    positions (``None``) and negative positions, lowest to highest.

    >>> position_sort([(None, "d"), (-3, "e"), (2, "b"), (-2, "f"), (5, "c"), (1, "a")])
    ['a', 'b', 'c', 'd', 'e', 'f']

    Parameters
    ----------
    args : list of (int/None, object) tuples

    Returns
    -------
    list of objects
    """
    import bisect

    pos, none, neg = [], [], []
    for entry in args:
        position = entry[0]
        if position is None:
            # Take existing order
            none.append(entry[1])
        elif position < 0:
            # Sort negatives while collecting
            bisect.insort(neg, entry)
        else:
            # Sort positives while collecting
            bisect.insort(pos, entry)

    return [arg for _, arg in pos] + none + [arg for _, arg in neg]


class PydraFileLock:
    """Wrapper for filelock's SoftFileLock that makes it work with asyncio."""

    def __init__(self, lockfile):
        self.lockfile = lockfile
        self.timeout = 0.1

    async def __aenter__(self):
        lock = SoftFileLock(self.lockfile)
        acquired_lock = False
        while not acquired_lock:
            try:
                lock.acquire(timeout=0)
                acquired_lock = True
            except Timeout:
                await asyncio.sleep(self.timeout)
                if self.timeout <= 2:
                    self.timeout = self.timeout * 2
        self.lock = lock
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.lock.release()
        return None


def parse_format_string(fmtstr: str) -> set[str]:
    """Parse a argstr format string and return all keywords used in it."""
    identifier = r"[a-zA-Z_]\w*"
    attribute = rf"\.{identifier}"
    item = r"\[\w+\]"
    # Example: var.attrs[key][0].attr2 (capture "var")
    field_with_lookups = (
        f"({identifier})(?:{attribute}|{item})*"  # Capture only the keyword
    )
    conversion = "(?:!r|!s)"
    nobrace = "[^{}]*"
    # Example: 0{pads[hex]}x (capture "pads")
    fmtspec = f"{nobrace}(?:{{({identifier}){nobrace}}}{nobrace})?"  # Capture keywords in definition
    full_field = f"{{{field_with_lookups}{conversion}?(?::{fmtspec})?}}"

    all_keywords = re.findall(full_field, fmtstr)
    return set().union(*all_keywords) - {""}


def fields_in_formatter(formatter: str | ty.Callable[..., str]) -> set[str]:
    """Extract all field names from a formatter string or function."""
    if isinstance(formatter, str):
        return parse_format_string(formatter)
    else:
        return set(inspect.signature(formatter).parameters.keys())


def ensure_list(obj, tuple2list=False):
    """
    Return a list whatever the input object is.

    Examples
    --------
    >>> ensure_list(list("abc"))
    ['a', 'b', 'c']
    >>> ensure_list("abc")
    ['abc']
    >>> ensure_list(tuple("abc"))
    [('a', 'b', 'c')]
    >>> ensure_list(tuple("abc"), tuple2list=True)
    ['a', 'b', 'c']
    >>> ensure_list(None)
    []
    >>> ensure_list(5.0)
    [5.0]

    """

    if obj is attrs.NOTHING:
        return attrs.NOTHING
    if obj is None:
        return []
    # list or numpy.array (this might need some extra flag in case an array has to be converted)
    elif isinstance(obj, list) or hasattr(obj, "__array__"):
        return obj
    elif tuple2list and isinstance(obj, tuple):
        return list(obj)
    elif is_lazy(obj):
        return obj
    return [obj]


def is_lazy(obj):
    """Check whether an object is a lazy field or has any attribute that is a Lazy Field"""
    from pydra.engine.lazy import LazyField

    return isinstance(obj, LazyField)


T = ty.TypeVar("T")
U = ty.TypeVar("U")


def state_array_support(
    function: ty.Callable[T, U],
) -> ty.Callable[T | StateArray[T], U | StateArray[U]]:
    """
    Decorator to convert a allow a function to accept and return StateArray objects,
    where the function is applied to each element of the StateArray.
    """

    def state_array_wrapper(
        value: "T | StateArray[T] | LazyField[T]",
    ) -> "U | StateArray[U] | LazyField[U]":
        if is_lazy(value):
            return value
        if isinstance(value, StateArray):
            return StateArray(function(v) for v in value)
        return function(value)

    return state_array_wrapper
