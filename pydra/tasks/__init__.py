""" Pydra tasks

The ``pydra.tasks`` namespace is reserved for collections of Tasks, to be managed and
packaged separately.
To create a task package, please fork the `pydra-tasks-template
<https://github.com/nipype/pydra-tasks-template>`__.
"""
# This call enables pydra.tasks to be used as a namespace package when installed
# in editable mode. In normal installations it has no effect.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
