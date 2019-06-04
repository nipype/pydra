""" This file contains defines parameters for nipy that we use to fill
settings in setup.py, the nipy top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nipy
"""

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
]

description = "Pydra dataflow engine"

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """========================================================
Pydra: Dataflow Engine
========================================================

Pydra is a rewrite of the Nipype engine with mapping and joining as
first-class operations. It forms the core of the Nipype 2.0 ecosystem.
"""

# versions
NETWORKX_MIN_VERSION = "1.9"
PYTEST_MIN_VERSION = "4.4.0"

__packagename__ = "pydra"
__author__ = __maintainer__ = "nipype developers"
__email__ = "neuroimaging@python.org"
__license__ = "Apache License, 2.0"
__status__ = "Pre-Alpha"
__description__ = description
__longdesc__ = long_description
__url__ = "https://github.com/nipype/pydra"

DOWNLOAD_URL = "http://github.com/nipype/{name}/archives/{ver}.tar.gz".format(
    name=__packagename__, ver=__version__
)
PLATFORMS = "OS Independent"
MAJOR = __version__.split(".")[0]
MINOR = __version__.split(".")[1]
MICRO = __version__.replace("-", ".").split(".")[2]
ISRELEASE = (
    len(__version__.replace("-", ".").split(".")) == 3
    or "post" in __version__.replace("-", ".").split(".")[-1]
)
VERSION = __version__
PROVIDES = ["pydra"]
REQUIRES = [
    "networkx>=%s" % NETWORKX_MIN_VERSION,
    "pytest>=%s" % PYTEST_MIN_VERSION,
    'dataclasses; python_version < "3.7"',
    "cloudpickle",
    "filelock",
]

SETUP_REQUIRES = ["setuptools>=27.0"]
TESTS_REQUIRES = ["pytest-cov", "codecov", "pytest-env", "pytest-xdist", "pyld"]
LINKS_REQUIRES = []

EXTRA_REQUIRES = {
    "tests": TESTS_REQUIRES,
    "dev": TESTS_REQUIRES + ["yapf>=0.22"],
    "plugins": ["dask", "distributed"],
}


def _list_union(iterable):
    return list(set(sum(iterable, [])))


# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES["all"] = _list_union(EXTRA_REQUIRES.values())
