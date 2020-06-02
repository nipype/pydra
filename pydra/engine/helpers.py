"""Administrative support for the engine framework."""
import asyncio
import asyncio.subprocess as asp
import attr
import cloudpickle as cp
from pathlib import Path
from filelock import SoftFileLock
import os
import sys
from hashlib import sha256
import subprocess as sp
import getpass
import uuid
from time import strftime
from traceback import format_exception


from .specs import Runtime, File, Directory, attr_fields, Result
from .helpers_file import hash_file, hash_dir, copyfile, is_existing_file


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
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    elif tuple2list and isinstance(obj, tuple):
        return list(obj)
    return [obj]


def print_help(obj):
    """Visit a task object and print its input/output interface."""
    lines = ["Help for {}".format(obj.__class__.__name__)]
    input_klass = make_klass(obj.input_spec)
    if attr.fields(input_klass):
        lines += ["Input Parameters:"]
    for f in attr.fields(input_klass):
        default = ""
        if f.default != attr.NOTHING and not f.name.startswith("_"):
            default = " (default: {})".format(f.default)
        try:
            name = f.type.__name__
        except AttributeError:
            name = str(f.type)
        lines += ["- {}: {}{}".format(f.name, name, default)]
    output_klass = make_klass(obj.output_spec)
    if attr.fields(output_klass):
        lines += ["Output Parameters:"]
    for f in attr.fields(output_klass):
        try:
            name = f.type.__name__
        except AttributeError:
            name = str(f.type)
        lines += ["- {}: {}".format(f.name, name)]
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
            if task_path.name.startswith("Workflow"):
                # copy files to the workflow directory
                result = copyfile_workflow(wf_path=task_path, result=result)
            with (task_path / f"{name_prefix}_result.pklz").open("wb") as fp:
                cp.dump(result, fp)
        if task:
            with (task_path / f"{name_prefix}_task.pklz").open("wb") as fp:
                cp.dump(task, fp)


def copyfile_workflow(wf_path, result):
    """ if file in the wf results, the file will be copied to the workflow directory"""
    for field in attr_fields(result.output):
        value = getattr(result.output, field.name)
        new_value = _copyfile_single_value(wf_path=wf_path, value=value)
        if new_value != value:
            setattr(result.output, field.name, new_value)
    return result


def _copyfile_single_value(wf_path, value):
    """ checking a single value for files that need to be copied to the wf dir"""
    if isinstance(value, (tuple, list)):
        return [_copyfile_single_value(wf_path, val) for val in value]
    elif isinstance(value, dict):
        return {
            key: _copyfile_single_value(wf_path, val) for (key, val) in value.items()
        }
    elif is_existing_file(value):
        new_path = wf_path / Path(value).name
        copyfile(originalfile=value, newfile=new_path, copy=True, use_hardlink=True)
        return new_path
    else:
        return value


def task_hash(task):
    """
    Calculate the checksum of a task.

    input hash, output hash, environment hash

    Parameters
    ----------
    task : :class:`~pydra.engine.core.TaskBase`
        The input task.

    """
    return NotImplementedError


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
        newfields = dict()
        for item in fields:
            if len(item) == 2:
                if isinstance(item[1], attr._make._CountingAttr):
                    newfields[item[0]] = item[1]
                else:
                    newfields[item[0]] = attr.ib(type=item[1])
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
                else:
                    if len(item) == 3:
                        name, tp = item[:2]
                        if isinstance(item[-1], dict) and "help_string" in item[-1]:
                            mdata = item[-1]
                            newfields[name] = attr.ib(type=tp, metadata=mdata)
                        else:
                            dflt = item[-1]
                            newfields[name] = attr.ib(type=tp, default=dflt)
                    elif len(item) == 4:
                        name, tp, dflt, mdata = item
                        newfields[name] = attr.ib(type=tp, default=dflt, metadata=mdata)
        fields = newfields
    return attr.make_class(spec.name, fields, bases=spec.bases, kw_only=True)


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
        login_name = "UID{:d}".format(os.getuid())

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
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    return loop


def hash_function(obj):
    """Generate hash of object."""
    return sha256(str(obj).encode()).hexdigest()


def hash_value(value, tp=None, metadata=None):
    """calculating hash or returning values recursively"""
    if metadata is None:
        metadata = {}
    if isinstance(value, (tuple, list)):
        return [hash_value(el, tp, metadata) for el in value]
    elif isinstance(value, dict):
        dict_hash = {k: hash_value(v, tp, metadata) for (k, v) in value.items()}
        # returning a sorted object
        return [list(el) for el in sorted(dict_hash.items(), key=lambda x: x[0])]
    else:  # not a container
        if (
            (tp is File or "pydra.engine.specs.File" in str(tp))
            and is_existing_file(value)
            and "container_path" not in metadata
        ):
            return hash_file(value)
        elif (
            (tp is File or "pydra.engine.specs.Directory" in str(tp))
            and is_existing_file(value)
            and "container_path" not in metadata
        ):
            return hash_dir(value)
        else:
            return value


def output_names_from_inputfields(inputs):
    """
    Collect outputs from input fields with output_file_template.

    Parameters
    ----------
    inputs :
        TODO

    """
    output_names = []
    for fld in attr_fields(inputs):
        if "output_file_template" in fld.metadata:
            if "output_field_name" in fld.metadata:
                field_name = fld.metadata["output_field_name"]
            else:
                field_name = fld.name
            output_names.append(field_name)
    return output_names


def output_from_inputfields(output_spec, inputs):
    """
    Collect values from output from input fields.

    Parameters
    ----------
    output_spec :
        TODO
    inputs :
        TODO

    """
    for fld in attr_fields(inputs):
        if "output_file_template" in fld.metadata:
            value = getattr(inputs, fld.name)
            if "output_field_name" in fld.metadata:
                field_name = fld.metadata["output_field_name"]
            else:
                field_name = fld.name
            output_spec.fields.append(
                (field_name, attr.ib(type=File, metadata={"value": value}))
            )
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
    except Exception as excinfo:
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
        if not resultfile.exists():
            etype, eval, etr = sys.exc_info()
            traceback = format_exception(etype, eval, etr)
            errorfile = record_error(task.output_dir, error=traceback)
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
    """ loading a task from a pickle file, settings proper input for the specific ind"""
    if isinstance(task_pkl, str):
        task_pkl = Path(task_pkl)
    task = cp.loads(task_pkl.read_bytes())
    if ind is not None:
        _, inputs_dict = task.get_input_el(ind)
        task.inputs = attr.evolve(task.inputs, **inputs_dict)
        task.state = None
    return task
