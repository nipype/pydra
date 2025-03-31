Glossary
========

.. glossary::

    Task
      A unit of work within the system. It represents one complete processing step in a workflow.

    Worker
        Encapsulation of a task execution environment. It is responsible for executing
        tasks and managing their lifecycle. Workers can be local (e.g., a thread or
        process) or remote (e.g., high-performance cluster).

    Job
        A task that has been instantiated with resolved with concrete inputs.
        (i.e. not lazy-values or state-arrays) and assigned to a worker.

    Workflow
      A series of tasks executed in a specific order.

    Node
        A single task within the context of a workflow, which is assigned a name and
        references a state. Note this task can be nested workflow task.

    State
        The combination of all upstream splits and combines with any splitters and
        combiners for a given node, it is used to track how many jobs, and their
        parameterisations, need to be run for a given workflow node.
