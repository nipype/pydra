.. _home:

Pydra
=====

Pydra is a lightweight, Python 3.11+ dataflow engine for computational graph construction,
manipulation, and distributed execution. Designed as a successor to created for Nipype_,
Pydra is a general-purpose engine that supports analytics in any scientific domain.
Pydra helps build reproducible, scalable, reusable, and fully automated, provenance
tracked scientific workflows that combine Python functions and shell commands.

The power of Pydra lies in ease of workflow creation and execution for complex
multiparameter map-reduce operations, and the use of global cache.

Pydra's key features are:
- Modular execution backends (see `Advanced execution <./tutorial/advanced-execution.html>`__)
- Map-reduce like semantics (see `Splitting and combining <./explanation/splitting-combining.html>`__)
- Global cache support to reduce recomputation (see `Hashing and caching <./explanation/hashing-caching.html>`__)
- Support for execution of Tasks in containerized environments (see `Environments <./explanation/environments.html>`__)
- Strong type-checking and type-hinting support (see `Typing <./explanation/typing.html>`__)

See :ref:`Design philosophy` for more an explanation of the design of Pydra.

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
environments (see [Environments](../explanation/environments.html)) to run them.


Tutorials
---------

* :ref:`Getting started`
* :ref:`Advanced execution`
* :ref:`Python-tasks`
* :ref:`Shell-tasks`
* :ref:`Workflows`
* :ref:`Canonical (dataclass) task form`

Examples
--------

* :ref:`T1w MRI preprocessing`
* :ref:`One-level GLM`
* :ref:`Two-Level GLM`

How-to Guides
-------------

* :ref:`Create a task package`
* :ref:`Port interfaces from Nipype`

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
    :maxdepth: 2
    :caption: Task execution
    :hidden:

    tutorial/getting-started
    tutorial/advanced-execution

.. toctree::
    :maxdepth: 2
    :caption: Design
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
   :caption: In-depth
   :hidden:

   explanation/design-approach
   explanation/splitting-combining
   explanation/typing
   explanation/conditional-lazy
   explanation/environments


.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/api

.. _FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL
.. _ANTs: http://stnava.github.io/ANTs/
.. _AFNI: https://afni.nimh.nih.gov/
.. _niworkflows: https://niworkflows.readthedocs.io/en/latest/
.. _Nipype: https://nipype.readthedocs.io/en/latest/
.. _
