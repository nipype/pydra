.. _home:

Pydra
=====

Pydra is a lightweight, Python 3.11+ dataflow engine for computational graph construction,
manipulation, and distributed execution. Designed as a successor to Nipype_,
Pydra is a general-purpose engine that supports analytics in any scientific domain.
Pydra helps build reproducible, scalable, reusable scientific workflows
that combine Python functions and shell commands.

The power of Pydra lies in ease of constructing workflows, containing complex
multiparameter map-reduce operations, in Python code and the use of a global cache (see
:ref:`Design philosophy` for the rationale behind its design).

Pydra's key features are:

* Combine diverse tasks, implemented in `Python functions <./tutorial/python.html>`__ or stand-alone `shell commands <./tutorial/shell.html>`__, into coherent `workflows <./tutorial/workflow.html>`__
* Map-reduce like semantics (see :ref:`Splitting and combining`)
* Dynamic workflow construction using Python code (see :ref:`Dynamic construction`)
* Modular execution systems for varied deployment on cloud, HPC, etc... (see `Execution options <./tutorial/advanced-execution.html>`__)
* Support for the execution of tasks in containerized environments (see :ref:`Containers and environments`)
* Global caching to reduce recomputation (see :ref:`Hashing and caching`)
* Support for strong type-checking, including file types, at workflow construction time (see :ref:`Typing and file-formats`)


Installation
------------

Pydra itself is a pure-Python package, which has only a handful of dependencies,
therefore, it is straightforward to install via pip for Python >= 3.11

.. code-block:: bash

    $ pip install pydra

Pre-designed tasks are available under the `pydra.tasks.*` package namespace. These tasks
are implemented within separate packages that are typically specific to a given shell-command toolkit such as FSL_, AFNI_ or ANTs_,
or a collection of related tasks/workflows (e.g. `niworkflows`_). Pip can be used to
install these packages as well:


.. code-block:: bash

    $ pip install pydra-fsl pydra-ants

Of course, if you use Pydra to execute commands within toolkits, you will need to
either have those commands installed on the execution machine, or use containers
environments (see `Environments <../explanation/environments.html>`__) to run them.


Tutorials and notebooks
-----------------------

The following tutorials provide a step-by-step guide to using Pydra.
They can be read in any order, but it is recommended to start with :ref:`Getting started`.
The tutorials are implemented as Jupyter notebooks, which can be downloaded and run locally
or run online using the |Binder| within each tutorial.

If you decide to download the notebooks and run locally, be sure to install the necessary
dependencies with ``pip install -e /path/to/your/pydra[tutorial]``.


Execution
~~~~~~~~~

Learn how to execute existing tasks (including workflows) on different systems

* :ref:`Getting started`
* :ref:`Execution options`

Design
~~~~~~

Learn how to design your own tasks

* :ref:`Python-tasks`
* :ref:`Shell-tasks`
* :ref:`Workflows`
* :ref:`Canonical task form`

Examples
~~~~~~~~

The following comprehensive examples demonstrate how to use Pydra to build and execute
complex workflows

* :ref:`T1w MRI preprocessing`
* :ref:`One-level GLM`
* :ref:`Two-Level GLM`

How-to Guides
-------------

The following guides provide step-by-step instructions on how to

* :ref:`Create a task package`
* :ref:`Port interfaces from Nipype`

Reference
---------

See the full reference documentation for Pydra

* :ref:`API`
* :ref:`genindex`
* :ref:`modindex`


.. toctree::
    :maxdepth: 2
    :caption: Tutorials: Execution
    :hidden:

    tutorial/getting-started
    tutorial/advanced-execution

.. toctree::
    :maxdepth: 2
    :caption: Tutorials: Design
    :hidden:

    tutorial/python
    tutorial/shell
    tutorial/workflow
    tutorial/canonical-form


.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/t1w-preprocess
   examples/first-level-glm
   examples/two-level-glm

.. toctree::
   :maxdepth: 2
   :caption: How-to
   :hidden:

   howto/create-task-package
   howto/port-from-nipype

.. toctree::
   :maxdepth: 2
   :caption: Explanation
   :hidden:

   explanation/splitting-combining
   explanation/conditional-lazy
   explanation/environments
   explanation/hashing-caching
   explanation/typing
   explanation/design-approach

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/api
   genindex
   modindex

.. _FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL
.. _ANTs: http://stnava.github.io/ANTs/
.. _AFNI: https://afni.nimh.nih.gov/
.. _niworkflows: https://niworkflows.readthedocs.io/en/latest/
.. _Nipype: https://nipype.readthedocs.io/en/latest/
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/nipype/pydra/develop
