# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path
from packaging.version import Version

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
sys.path.insert(1, str(Path(__file__).parent / "sphinxext"))
from pydra import __version__
from github_link import make_linkcode_resolve


# -- Project information -----------------------------------------------------

project = "Pydra: A simple dataflow engine with scalable semantics"
copyright = "2019 - 2020, The Nipype Developers team"
author = "The Nipype Developers team"

# The full version, including alpha/beta/rc tags
release = __version__
version = Version(release).public


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.linkcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinxcontrib.apidoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "api/pydra.rst"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for extensions ---------------------------------------------------

# Autodoc
autodoc_mock_imports = ["cloudpickle", "matplotlib", "numpy", "psutil"]
apidoc_module_dir = "../pydra"
apidoc_output_dir = "api"
apidoc_excluded_paths = ["conftest.py", "*/tests/*", "tests/*", "data/*"]
apidoc_separate_modules = True
apidoc_extra_args = ["--module-first", "-d 1", "-T"]

# Napoleon
# Accept custom section names to be parsed for numpy-style docstrings
# of parameters.
# Requires pinning sphinxcontrib-napoleon to a specific commit while
# https://github.com/sphinx-contrib/napoleon/pull/10 is merged.
napoleon_use_param = False
napoleon_custom_sections = [("Inputs", "Parameters"), ("Outputs", "Parameters")]

# Intersphinx
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Linkcode
# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "pydra",
    "https://github.com/nipype/pydra/blob/{revision}/" "{package}/{path}#L{lineno}",
)

# Sphinx-versioning
scv_show_banner = True
