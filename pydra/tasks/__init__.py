""" Pydra tasks

The ``pydra.tasks`` namespace is reserved for collections of Tasks, to be managed and
packaged separately.
To create a task package, please fork the `pydra-tasks-template
<https://github.com/nipype/pydra-tasks-template>`__.
"""
try:
    __import__("pkg_resources").declare_namespace(__name__)
except ImportError:
    pass  # must not have setuptools
