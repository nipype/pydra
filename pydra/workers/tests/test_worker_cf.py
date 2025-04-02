import os
import shutil
from pathlib import Path
import random
import platform
import typing as ty
import pytest
import cloudpickle as cp
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job
from pydra.compose import workflow
from fileformats.generic import Directory, File
from pydra.engine.tests.utils import Multiply, RaiseXeq1
from pydra.utils.general import position_sort
from pydra.compose.shell.templating import parse_format_string
from pydra.engine.job import save, load_and_run
from pydra.workers.cf import get_available_cpus
from pydra.utils.hash import hash_function


def test_get_available_cpus():
    assert get_available_cpus() > 0
    try:
        import psutil

        has_psutil = True
    except ImportError:
        has_psutil = False

    if hasattr(os, "sched_getaffinity"):
        assert get_available_cpus() == len(os.sched_getaffinity(0))

    if has_psutil and platform.system().lower() != "darwin":
        assert get_available_cpus() == len(psutil.Process().cpu_affinity())

    if platform.system().lower() == "darwin":
        assert get_available_cpus() == os.cpu_count()
