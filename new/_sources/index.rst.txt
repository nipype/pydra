.. _home:

Pydra
=====

Pydra is a lightweight dataflow engine written in Python. Although designed to succeed
Nipype_ in order to address the needs of the neuroimaging community, Pydra can be used
for analytics in any scientific domain. Pydra facilitates the design of reproducible,
scalable and robust workflows that can link diverse processing tasks implemented as
shell commands or Python functions.

**Key features:**

* Combine diverse tasks (`Python functions <./tutorial/3-python.html>`__ or `shell commands <./tutorial/4-shell.html>`__) into coherent, robust `workflows <./tutorial/5-workflow.html>`__
* Dynamic workflow construction using Python code (see :ref:`Dynamic construction`)
* Concurrent execution on `choice of computing platform (e.g. workstation, SLURM, SGE, Dask, etc...) <./tutorial/3-advanced-execution.html#Workers>`__
* Map-reduce-like semantics (see :ref:`Splitting and combining`)
* Global caching to reduce recomputation (see :ref:`Caches and hashes`)
* Tasks can be executed in separate software environments, e.g. containers (see :ref:`Software environments`)
* Strong type-checking, including file types, before execution (see :ref:`Typing and file-formats`)

See :ref:`Design philosophy` for more details on the rationale behind Pydra's design.


Installation
------------

Pydra is implemented purely in Python and has a small number of dependencies
It is easy to install via pip for Python >= 3.11 (preferably within a
`virtual environment`_):

.. code-block:: bash

   $ pip install pydra

Pre-designed tasks are available under the `pydra.tasks.*` namespace. These tasks
are typically implemented within separate packages that are specific to a given
shell-command toolkit, such as FSL_ (*pydra-fsl*), AFNI_ (*pydra-afni*) or
ANTs_ (*pydra-ants*), or a collection of related tasks/workflows, such as Niworkflows
(*pydra-niworkflows*). Pip can be used to install these extension packages as well:

.. code-block:: bash

   $ pip install pydra-fsl pydra-ants

Of course, if you use Pydra to execute commands within non-Python toolkits, you will
need to either have those commands installed on the execution machine, or use containers
to run them (see :ref:`Software environments`).


Tutorials and notebooks
-----------------------

The following tutorials provide a step-by-step guide to using Pydra. They can be
studied in any order, but it is recommended to start with :ref:`Getting started` and
step through the list from there.

The tutorials are written in Jupyter notebooks, which can be downloaded and run locally
or run online using the |Binder| button within each tutorial.

If you decide to download the notebooks and run locally, be sure to install the necessary
dependencies (ideally within a  `virtual environment`_):

.. code-block:: bash

   $ pip install -e /path/to/your/pydra[tutorial]


Execution
~~~~~~~~~

Learn how to execute existing tasks (including workflows) on different systems

* :ref:`Getting started`
* :ref:`Advanced execution`
* :ref:`Troubleshooting`

Design
~~~~~~

Learn how to design your own tasks, wrapped shell commands or Python functions, or
workflows,

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

    tutorial/1-getting-started
    tutorial/2-advanced-execution
    tutorial/3-troubleshooting

.. toctree::
    :maxdepth: 2
    :caption: Tutorials: Design
    :hidden:

    tutorial/4-python
    tutorial/5-shell
    tutorial/6-workflow
    tutorial/7-canonical-form


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

   explanation/design-approach
   explanation/splitting-combining
   explanation/conditional-lazy
   explanation/environments
   explanation/hashing-caching
   explanation/typing


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
.. _virtual environment: https://docs.python.org/3/library/venv.html
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/nipype/pydra/develop
