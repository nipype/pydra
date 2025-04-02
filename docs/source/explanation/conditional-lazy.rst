Dynamic construction
====================

Pydra workflows are constructed dynamically by workflow "constructor" functions. These
functions can use any valid Python code, allowing rich and complex workflows to be
constructed based on the inputs to the workflow. For example, a workflow constructor
could include conditional branches, loops, or other control flow structures, to tailor
the workflow to the specific inputs provided.


Lazy fields
-----------

Pydra workflows are constructed by the assignment of "lazy field" placeholders from
the outputs of upstream nodes to the inputs of downstream nodes. These placeholders,
which are instances of the :class:`pydra.engine.specs.LazyField` class, are replaced
by the actual values they represent when the workflow is run.


Caching of workflow construction
--------------------------------

Workflows are constructed just before they are executed to produce a Directed Acyclic Graph
(DAG) of nodes. Tasks are generated from these nodes as upstream inputs become available
and added to the execution stack. If the workflow has been split, either at the top-level,
in an upstream node or at the current node, then a separate task will be generated for
split.


Nested workflows and lazy conditionals
--------------------------------------

Since lazy fields are only evaluated at runtime, they can't be used in conditional
statements that construct the workflow. However, if there is a section of a workflow
that needs to be conditionally included or excluded based on upstream outputs, that
section can be implemented in a nested workflow and that upstream be connected to the
nested workflow.
