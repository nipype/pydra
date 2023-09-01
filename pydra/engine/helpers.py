"""Administrative support for the engine framework."""
import asyncio
import asyncio.subprocess as asp
from pathlib import Path
import os
import sys
from uuid import uuid4
import getpass
import typing as ty
import subprocess as sp
import re
from time import strftime
from traceback import format_exception
import attr
import attrs  # New defaults
from filelock import SoftFileLock, Timeout
import cloudpickle as cp
from .specs import (
    Runtime,
    attr_fields,
    Result,
    LazyField,
    File,
)
from .helpers_file import copy_nested_files
from ..utils.typing import TypeParser
from fileformats.core import FileSet
from .specs import MultiInputFile, MultiInputObj, MultiOutputObj, MultiOutputFile


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
    if obj is attr.NOTHING:
        return attr.NOTHING
    if obj is None:
        return []
    # list or numpy.array (this might need some extra flag in case an array has to be converted)
    elif isinstance(obj, list) or hasattr(obj, "__array__"):
        return obj
    elif tuple2list and isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, LazyField):
        return obj
    return [obj]


def from_list_if_single(obj):
    """Converts a list to a single item if it is of length == 1"""
    if obj is attr.NOTHING:
        return obj
    if isinstance(obj, LazyField):
        return obj
    obj = list(obj)
    if len(obj) == 1:
        return obj[0]
    return obj


def print_help(obj):
    """Visit a task object and print its input/output interface."""
    lines = [f"Help for {obj.__class__.__name__}"]
    input_klass = make_klass(obj.input_spec)
    if attr.fields(input_klass):
        lines += ["Input Parameters:"]
    for f in attr.fields(input_klass):
        default = ""
        if f.default != attr.NOTHING and not f.name.startswith("_"):
            default = f" (default: {f.default})"
        try:
            name = f.type.__name__
        except AttributeError:
            name = str(f.type)
        lines += [f"- {f.name}: {name}{default}"]
    output_klass = make_klass(obj.output_spec)
    if attr.fields(output_klass):
        lines += ["Output Parameters:"]
    for f in attr.fields(output_klass):
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


def save(task_path: Path, result=None, task=None, name_prefix=None):
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
            if task_path.name.startswith("Workflow") and result.output is not None:
                # copy files to the workflow directory
                result = copyfile_workflow(wf_path=task_path, result=result)
            with (task_path / f"{name_prefix}_result.pklz").open("wb") as fp:
                cp.dump(result, fp)
        if task:
            with (task_path / f"{name_prefix}_task.pklz").open("wb") as fp:
                cp.dump(task, fp)


def copyfile_workflow(wf_path: os.PathLike, result):
    """if file in the wf results, the file will be copied to the workflow directory"""
    for field in attr_fields(result.output):
        value = getattr(result.output, field.name)
        # if the field is a path or it can contain a path _copyfile_single_value is run
        # to move all files and directories to the workflow directory
        new_value = copy_nested_files(value, wf_path, mode=FileSet.CopyMode.hardlink)
        setattr(result.output, field.name, new_value)
    return result


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


def make_klass(spec):
    """
    Create a data class given a spec.

    Parameters
    ----------
    spec :
        TODO

    """
    if spec is None:
        return None
    fields = spec.fields
    if fields:
        newfields = {}
        for item in fields:
            if len(item) == 2:
                name = item[0]
                if isinstance(item[1], attr._make._CountingAttr):
                    newfield = item[1]
                else:
                    newfield = attr.ib(type=item[1])
            else:
                if (
                    any([isinstance(ii, attr._make._CountingAttr) for ii in item])
                    or len(item) > 4
                ):
                    raise ValueError(
                        "syntax not valid, you can use (name, attr), "
                        "(name, type, default), (name, type, default, metadata)"
                        "or (name, type, metadata)"
                    )
                kwargs = {}
                if len(item) == 3:
                    name, tp = item[:2]
                    if isinstance(item[-1], dict) and "help_string" in item[-1]:
                        mdata = item[-1]
                        kwargs["metadata"] = mdata
                    else:
                        kwargs["default"] = item[-1]
                elif len(item) == 4:
                    name, tp, dflt, mdata = item
                    kwargs["default"] = dflt
                    kwargs["metadata"] = mdata
                newfield = attr.ib(
                    type=tp,
                    **kwargs,
                )
            checker_label = f"'{name}' field of {spec.name}"
            type_checker = TypeParser[newfield.type](newfield.type, label=checker_label)
            if newfield.type in (MultiInputObj, MultiInputFile):
                converter = attr.converters.pipe(ensure_list, type_checker)
            elif newfield.type in (MultiOutputObj, MultiOutputFile):
                converter = attr.converters.pipe(from_list_if_single, type_checker)
            else:
                converter = type_checker
            newfield.converter = converter
            newfield.on_setattr = attr.setters.convert
            if "allowed_values" in newfield.metadata:
                if newfield._validator is None:
                    newfield._validator = allowed_values_validator
                elif isinstance(newfield._validator, ty.Iterable):
                    if allowed_values_validator not in newfield._validator:
                        newfield._validator.append(allowed_values_validator)
                elif newfield._validator is not allowed_values_validator:
                    newfield._validator = [
                        newfield._validator,
                        allowed_values_validator,
                    ]
            newfields[name] = newfield
        fields = newfields
    return attrs.make_class(
        spec.name, fields, bases=spec.bases, kw_only=True, on_setattr=None
    )


def allowed_values_validator(_, attribute, value):
    """checking if the values is in allowed_values"""
    allowed = attribute.metadata["allowed_values"]
    if value is attr.NOTHING or isinstance(value, LazyField):
        pass
    elif value not in allowed:
        raise ValueError(
            f"value of {attribute.name} has to be from {allowed}, but {value} provided"
        )


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
    return "_".join((name, inputs))


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


def output_from_inputfields(output_spec, input_spec):
    """
    Collect values from output from input fields.
    If names_only is False, the output_spec is updated,
    if names_only is True only the names are returned

    Parameters
    ----------
    output_spec :
        TODO
    input_spec :
        TODO

    """
    current_output_spec_names = [f.name for f in attr.fields(make_klass(output_spec))]
    new_fields = []
    for fld in attr.fields(make_klass(input_spec)):
        if "output_file_template" in fld.metadata:
            if "output_field_name" in fld.metadata:
                field_name = fld.metadata["output_field_name"]
            else:
                field_name = fld.name
            # not adding if the field already in the output_spec
            if field_name not in current_output_spec_names:
                # TODO: should probably remove some of the keys
                new_fields.append(
                    (field_name, attr.ib(type=File, metadata=fld.metadata))
                )
    output_spec.fields += new_fields
    return output_spec


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


def load_and_run(
    task_pkl, ind=None, rerun=False, submitter=None, plugin=None, **kwargs
):
    """
    loading a task from a pickle file, settings proper input
    and running the task
    """
    try:
        task = load_task(task_pkl=task_pkl, ind=ind)
    except Exception:
        if task_pkl.parent.exists():
            etype, eval, etr = sys.exc_info()
            traceback = format_exception(etype, eval, etr)
            errorfile = record_error(task_pkl.parent, error=traceback)
            result = Result(output=None, runtime=None, errored=True)
            save(task_pkl.parent, result=result)
        raise

    resultfile = task.output_dir / "_result.pklz"
    try:
        task(rerun=rerun, plugin=plugin, submitter=submitter, **kwargs)
    except Exception as excinfo:
        # creating result and error files if missing
        errorfile = task.output_dir / "_error.pklz"
        if not errorfile.exists():  # not sure if this is needed
            etype, eval, etr = sys.exc_info()
            traceback = format_exception(etype, eval, etr)
            errorfile = record_error(task.output_dir, error=traceback)
        if not resultfile.exists():  # not sure if this is needed
            result = Result(output=None, runtime=None, errored=True)
            save(task.output_dir, result=result)
        raise type(excinfo)(
            str(excinfo.with_traceback(None)),
            f" full crash report is here: {errorfile}",
        )
    return resultfile


async def load_and_run_async(task_pkl, ind=None, submitter=None, rerun=False, **kwargs):
    """
    loading a task from a pickle file, settings proper input
    and running the workflow
    """
    task = load_task(task_pkl=task_pkl, ind=ind)
    await task._run(submitter=submitter, rerun=rerun, **kwargs)


def load_task(task_pkl, ind=None):
    """loading a task from a pickle file, settings proper input for the specific ind"""
    if isinstance(task_pkl, str):
        task_pkl = Path(task_pkl)
    task = cp.loads(task_pkl.read_bytes())
    if ind is not None:
        ind_inputs = task.get_input_el(ind)
        task.inputs = attr.evolve(task.inputs, **ind_inputs)
        task._pre_split = True
        task.state = None
        # resetting uid for task
        task._uid = uuid4().hex
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


def argstr_formatting(argstr, inputs, value_updates=None):
    """formatting argstr that have form {field_name},
    using values from inputs and updating with value_update if provided
    """
    inputs_dict = attr.asdict(inputs, recurse=False)
    # if there is a value that has to be updated (e.g. single value from a list)
    if value_updates:
        inputs_dict.update(value_updates)
    # getting all fields that should be formatted, i.e. {field_name}, ...
    inp_fields = re.findall(r"{\w+}", argstr)
    inp_fields_float = re.findall(r"{\w+:[0-9.]+f}", argstr)
    inp_fields += [re.sub(":[0-9.]+f", "", el) for el in inp_fields_float]
    val_dict = {}
    for fld in inp_fields:
        fld_name = fld[1:-1]  # extracting the name form {field_name}
        fld_value = inputs_dict[fld_name]
        fld_attr = getattr(attrs.fields(type(inputs)), fld_name)
        if fld_value is attr.NOTHING or (
            fld_value is False
            and TypeParser.matches_type(fld_attr.type, ty.Union[Path, bool])
        ):
            # if value is NOTHING, nothing should be added to the command
            val_dict[fld_name] = ""
        else:
            val_dict[fld_name] = fld_value

    # formatting string based on the val_dict
    argstr_formatted = argstr.format(**val_dict)
    # removing extra commas and spaces after removing the field that have NOTHING
    argstr_formatted = (
        argstr_formatted.replace("[ ", "[")
        .replace(" ]", "]")
        .replace("[,", "[")
        .replace(",]", "]")
        .strip()
    )
    return argstr_formatted


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


def parse_copyfile(fld: attr.Attribute, default_collation=FileSet.CopyCollation.any):
    """Gets the copy mode from the 'copyfile' value from a field attribute"""
    copyfile = fld.metadata.get("copyfile", FileSet.CopyMode.any)
    if isinstance(copyfile, tuple):
        mode, collation = copyfile
    elif isinstance(copyfile, str):
        try:
            mode, collation = copyfile.split(",")
        except ValueError:
            mode = copyfile
            collation = default_collation
        else:
            collation = FileSet.CopyCollation[collation]
        mode = FileSet.CopyMode[mode]
    else:
        if copyfile is True:
            mode = FileSet.CopyMode.copy
        elif copyfile is False:
            mode = FileSet.CopyMode.link
        elif copyfile is None:
            mode = FileSet.CopyMode.any
        else:
            mode = copyfile
        collation = default_collation
    if not isinstance(mode, FileSet.CopyMode):
        raise TypeError(
            f"Unrecognised type for mode copyfile metadata of {fld}, {mode}"
        )
    if not isinstance(collation, FileSet.CopyCollation):
        raise TypeError(
            f"Unrecognised type for collation copyfile metadata of {fld}, {collation}"
        )
    return mode, collation
