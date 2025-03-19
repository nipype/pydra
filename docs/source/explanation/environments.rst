Software environments
=====================

Pydra supports running tasks within encapsulated software environments, such as Docker_
and Singularity_ containers. This can be specified at runtime or during workflow
construction, and allows tasks to be run in environments that are isolated from the
host system, and that have specific software dependencies.

The environment a task runs within is specified by the ``environment`` argument passed
to the execution call (e.g. ``my_task(plugin="cf", environment="docker")``) or in the
``workflow.add()`` call in workflow constructors.

Specifying at execution
-----------------------

Work in progress...


Specifying at workflow construction
-----------------------------------

Work in progress...



Implementing new environment types
----------------------------------

Work in progress...


.. _Docker: https://www.docker.com/
.. _Singularity: https://sylabs.io/singularity/
