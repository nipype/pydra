.. _home:

Pydra
=====

Pydra is a new lightweight dataflow engine written in Python, which provides a simple way to
implement scientific workflows that use a mix of shell commands and Python functions.

Pydra is developed as an open-source project in the neuroimaging community,
but it is designed as a general-purpose dataflow engine to support any scientific domain.

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

Examples
--------

* :ref:`T1w MRI preprocessing`

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
    :caption: Execution
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


.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/t1w-preprocess

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
   explanation/conditional-lazy
   explanation/typing
   explanation/hashing-caching
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
