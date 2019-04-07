import asyncio
import asyncio.subprocess as asp
import dataclasses as dc
import cloudpickle as cp
from pathlib import Path
import os
import sys

from .specs import Runtime


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
            if result_file.exists():
                try:
                    return cp.loads(result_file.read_bytes())
                except EOFError:
                    return None
            return None
    return None


def save_result(result_path: Path, result):
    with (result_path / "_result.pklz").open("wb") as fp:
        cp.dump(result, fp)


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
@asyncio.coroutine
def read_stream_and_display(stream, display):
    """Read from stream line by line until EOF, display, and capture the lines.

    """
    output = []
    while True:
        line = yield from stream.readline()
        if not line:
            break
        output.append(line)
        display(line)  # assume it doesn't block
    return b"".join(output).decode()


@asyncio.coroutine
def read_and_display(*cmd):
    """Capture cmd's stdout, stderr while displaying them as they arrive
    (line by line).

    """
    # start process
    process = yield from asyncio.create_subprocess_exec(
        *cmd, stdout=asp.PIPE, stderr=asp.PIPE
    )

    # read child's stdout/stderr concurrently (capture and display)
    try:
        stdout, stderr = yield from asyncio.gather(
            read_stream_and_display(process.stdout, sys.stdout.buffer.write),
            read_stream_and_display(process.stderr, sys.stderr.buffer.write),
        )
    except Exception:
        process.kill()
        raise
    finally:
        # wait for the process to exit
        rc = yield from process.wait()
    return rc, stdout, stderr


# run the event loop
def execute(cmd):
    if os.name == "nt":
        loop = asyncio.ProactorEventLoop()  # for subprocess' pipes on Windows
        asyncio.set_event_loop(loop)
    else:
        if asyncio.get_event_loop().is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    rc, stdout, stderr = loop.run_until_complete(read_and_display(*cmd))
    loop.close()
    return rc, stdout, stderr


def create_checksum(name, inputs):
    return "_".join((name, inputs.hash))


def get_inputs(needed_outputs):
    in_dict = {}
    for outlink in needed_outputs:
        result = load_result(outlink.cache_location)
        if result:
            in_dict[outlink.input] = getattr(result.output, outlink.output)
    return in_dict


def record_error(error_path, error):
    with (error_path / "_error.pklz").open("wb") as fp:
        cp.dump(error, fp)
