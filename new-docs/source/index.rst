.. _home:

Pydra
=====

Pydra is a new lightweight dataflow engine written in Python, which provides a simple way to
implement scientific workflows that use a mix of shell commands and Python functions.

Pydra is developed as an open-source project in the neuroimaging community,
but it is designed as a general-purpose dataflow engine to support any scientific domain.

See the :ref:`Design philosophy` for more an explanation of the design
philosophy and goals of Pydra.

Installation
------------

Pydra itself is a pure-Python package, which has only a handful of dependencies,
therefore, it is straightforward to install via pip

.. code-block:: bash

    $ pip install pydra

Of course, if you use Pydra to execute shell-commands tools, you will need to either have
those commands installed on the execution machine, or use software containers
(e.g., Docker or Singularity) to run them.


Tutorials
---------

* :ref:`Getting started`
* :ref:`Execution options`
* :ref:`Python-tasks`
* :ref:`Shell-tasks`
* :ref:`Workflows`

Examples
--------

* :ref:`Real-world example`

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

   examples/real-example

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
   explanation/hashing-caching
   explanation/conditional-lazy
   explanation/provenance


.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/api
