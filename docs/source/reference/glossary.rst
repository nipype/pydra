Glossary
========

.. glossary::

    Environment
        An environment refers to a specific software encapsulation, such as a Docker
        or Singularity image, that is used to run a task.

    Field
        A field is a parameter of a task, or a task outputs object, that can be set to
        a specific value. Fields are specified to be of any types, including objects
        and file-system objects.

    Hook
        A hook is a user-defined function that is executed at a specific point in the task
        execution process. Hooks can be used to prepare/finalise the task cache directory
        or send notifications

    Job
        A task that has been instantiated with resolved with concrete inputs.
        (i.e. not lazy-values or state-arrays) and assigned to a worker. Whereas a
        a task describes "what" is to be done and a submitter object describes
        "how" it is to be done, a job combines them both to describe a concrete unit
        of processing.

    Lazy-fields
        A lazy-field is a field that is not immediately resolved to a value. Instead,
        it is a placeholder that will be resolved at runtime, allowing for dynamic
        parameterisation of tasks.

    Node
        A single task within the context of a workflow, which is assigned a name and
        references a state. Note this task can be nested workflow task.

    State
        The combination of all upstream splits and combines with any splitters and
        combiners for a given node, it is used to track how many jobs, and their
        parameterisations, need to be run for a given workflow node.

    State-array
        A state array is a collection of parameterised tasks or values that were generated
        by a split operation either at the current or upstream node of a workflow.

    Submitter
        A submitter object parameterises how a task is to be executed, by specifying
        an worker, environment, cache-root directory and other key execution parameters.
        The submitter

    Task
      A task describeas a unit of work within the system. It represents a unit of processing, either
      standalone or as one step in a larger workflow. Tasks can be of various types,
      including Python functions, shell commands, and nested workflows. Tasks are
      parameterised, meaning they can accept inputs and produce

    Worker
        Encapsulation of a task execution environment. It is responsible for executing
        tasks and managing their lifecycle. Workers can be local (e.g., a thread or
        process) or remote (e.g., high-performance cluster).

    Workflow
      A Directed-Acyclic-Graph (DAG) of parameterised tasks, to be executed in order.
      Note that a Workflow object is created by a WorkflowTask, by the WorkflowTask's
      `construct()` method at runtime.
