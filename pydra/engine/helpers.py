import asyncio
import asyncio.subprocess as asp
import dataclasses as dc
import cloudpickle as cp
from pathlib import Path
import os
import sys
from hashlib import sha256
import subprocess as sp

from .specs import Runtime, File, LazyField


def ensure_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def print_help(obj):
    help = ["Help for {}".format(obj.__class__.__name__)]
    input_klass = make_klass(obj.input_spec)
    if dc.fields(input_klass):
        help += ["Input Parameters:"]
    for f in dc.fields(input_klass):
        default = ""
        if f.default is not dc.MISSING and not f.name.startswith("_"):
            default = " (default: {})".format(f.default)
        try:
            name = f.type.__name__
        except AttributeError:
            name = str(f.type)
        help += ["- {}: {}{}".format(f.name, name, default)]
    output_klass = make_klass(obj.output_spec)
    if dc.fields(output_klass):
        help += ["Output Parameters:"]
    for f in dc.fields(output_klass):
        try:
            name = f.type.__name__
        except AttributeError:
            name = str(f.type)
        help += ["- {}: {}".format(f.name, name)]
    print("\n".join(help))
    return help


def load_result(checksum, cache_locations):
    if not cache_locations:
        return None
    for location in cache_locations:
        if (location / checksum).exists():
            result_file = location / checksum / "_result.pklz"
            if result_file.exists() and result_file.stat().st_size > 0:
                return cp.loads(result_file.read_bytes())
            return None
    return None


def save(task_path: Path, result=None, task=None):
    """
    Save ``Task`` object and/or results.

    Parameters
    ----------
    task_path : Path
        Write directory
    result : Result
        Result to pickle and write
    task : Task
        Task to pickle and write
    """
    if task is None and result is None:
        raise ValueError("Nothing to be saved")
    task_path.mkdir(parents=True, exist_ok=True)
    if result:
        with (task_path / "_result.pklz").open("wb") as fp:
            cp.dump(result, fp)
    if task:
        with (task_path / "_task.pklz").open("wb") as fp:
            cp.dump(task, fp)


def task_hash(task_obj):
    """
    input hash, output hash, environment hash

    :param task_obj:
    :return:
    """
    return NotImplementedError


def gather_runtime_info(fname):
    runtime = Runtime(rss_peak_gb=None, vms_peak_gb=None, cpu_peak_percent=None)

    # Read .prof file in and set runtime values
    with open(fname, "rt") as fp:
        data = [[float(el) for el in val.strip().split(",")] for val in fp.readlines()]
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
    if spec is None:
        return None
    return dc.make_dataclass(spec.name, spec.fields, bases=spec.bases)


# https://stackoverflow.com/questions/17190221
async def read_stream_and_display(stream, display):
    """Read from stream line by line until EOF, display, and capture the lines.

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
    """Capture cmd's stdout, stderr while displaying them as they arrive
    (line by line).

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
    """Capture cmd's stdout

    """
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


# run the event loop with coroutine read_and_display_async
# or run read_and_display if a loop is already running
def execute(cmd, strip=False):
    loop = get_open_loop()
    if loop.is_running():
        rc, stdout, stderr = read_and_display(*cmd, strip=strip)
    else:
        rc, stdout, stderr = loop.run_until_complete(
            read_and_display_async(*cmd, strip=strip)
        )
    return rc, stdout, stderr


def create_checksum(name, inputs):
    return "_".join((name, inputs))


def record_error(error_path, error):
    with (error_path / "_error.pklz").open("wb") as fp:
        cp.dump(error, fp)


def get_open_loop():
    """
    Gets current event loop. If the loop is closed, a new
    loop is created and set as the current event loop.

    Returns
    -------
    loop : EventLoop
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


def create_pyscript(script_path, checksum):
    """
    Create standalone script for task execution in
    a different environment.

    Parameters
    ----------
    script_path : Path
    checksum : str
        ``Task``'s checksum

    Returns
    -------
    pyscript : File
        Execution script
    """
    task_pkl = script_path / "_task.pklz"
    if not task_pkl.exists() or not task_pkl.stat().st_size:
        raise Exception("Missing or empty task!")

    content = f"""import cloudpickle as cp
from pathlib import Path


cache_path = Path("{str(script_path)}")
task_pkl = (cache_path / "_task.pklz")
task = cp.loads(task_pkl.read_bytes())

# submit task
task()

if not task.result():
    raise Exception("Something went wrong")
print("Completed", task.checksum, task)
task_pkl.unlink()
"""
    pyscript = script_path / f"pyscript_{checksum}.py"
    with pyscript.open("wt") as fp:
        fp.writelines(content)
    return pyscript


def hash_function(obj):
    return sha256(str(obj).encode()).hexdigest()


def hash_file(afile, chunk_len=8192, crypto=sha256, raise_notfound=False):
    """
    Computes hash of a file using 'crypto' module
    """
    if afile is None or isinstance(afile, LazyField):
        return None
    if not os.path.isfile(afile):
        if raise_notfound:  # WHY??
            raise RuntimeError('File "%s" not found.' % afile)
        return None

    crypto_obj = crypto()
    with open(afile, "rb") as fp:
        while True:
            data = fp.read(chunk_len)
            if not data:
                break
            crypto_obj.update(data)
    return crypto_obj.hexdigest()


def output_names_from_inputfields(input_spec):
    """ collecting outputs from input fields with output_file_template"""
    output_names = []
    for fld in dc.fields(make_klass(input_spec)):
        if "output_file_template" in fld.metadata:
            if "output_field_name" in fld.metadata:
                field_name = fld.metadata["output_field_name"]
            else:
                field_name = fld.name
            output_names.append(field_name)
    return output_names


def output_from_inputfields(output_spec, input_spec, inputs):
    """ collecting values from output from input fields"""
    for fld in dc.fields(make_klass(input_spec)):
        if "output_file_template" in fld.metadata:
            value = getattr(inputs, fld.name)
            if "output_field_name" in fld.metadata:
                field_name = fld.metadata["output_field_name"]
            else:
                field_name = fld.name
            output_spec.fields.append(
                (field_name, File, dc.field(metadata={"value": value}))
            )
    return output_spec
