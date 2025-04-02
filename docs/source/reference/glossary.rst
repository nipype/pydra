Glossary
========

.. glossary::

    Cache-root
        The root directory in which separate cache directories for each job are created.
        Job cache directories are named within the cache-root directory using a unique
        checksum for the job based on the task's parameters and software environment,
        so that if the same job is run again the outputs from the previous run can be
        reuused.

    Combiner
        A combiner is used to combine :ref:`State-array` values created by a split operation
        defined by a :ref:`Splitter` on the current node, upstream workflow nodes or
        stand-alone tasks.

    Container-ndim
        The number of dimensions of the container object to be flattened into a single
        state array when splitting over nested containers/multi-dimension arrays.
        For example, a list-of-list-of-floats or a 2D numpy array with `container_ndim=1`,
        the outer list/2D would be split into a 1-D state array consisting of
        list-of-floats or 1D numpy arrays, respectively. Whereas with
        `container_ndim=2` they would be split into a state-array of floats consisiting
        of all the elements of the inner-lists/array.

    Environment
        An environment refers to a specific software encapsulation, such as a Docker
        or Singularity image, in which a shell tasks are run. They are specified in the
        Submitter object to be used when executing a task.

    Field
        A field is a parameter of a task, or an output in a task outputs class.
        Fields define the expected datatype of the parameter and other metadata
        parameters that control how the field is validated and passed through to the
        execution of the task.

    Hook
        A hook is a user-defined function that is executed at a specific point either before
        or after a task is run. Hooks can be used to prepare/finalise the task cache directory
        or send notifications

    Job
        A job consists of a :ref:`Task` with all inputs resolved
        (i.e. not lazy-values or state-arrays) and a Submitter object. It therefore
        represents a concrete unit of work to be executed, be combining "what" is to be
        done (Task) with "how" it is to be done (Submitter).

    Lazy-fields
        A lazy-field is a field that is not immediately resolved to a value. Instead,
        it is a placeholder that will be resolved at runtime when a workflow is executed,
        allowing for dynamic parameterisation of tasks.

    Node
        A single task within the context of a workflow. It is assigned a unique name
        within the workflow and references a state object that determines the
        state-array of jobs to be run if present (if the state is None then a single
        job will be run for each node).

    Read-only-caches
        A read-only cache is a cache root directory that was created by a previous
        pydra run. The read-only caches are checked for matching job checksums, which
        are reused if present. However, new job cache dirs are written to the cache root
        so the read-only caches are not modified during the execution.

    State
        The combination of all upstream splits and combines with any splitters and
        combiners for a given node. It is used to track how many jobs, and their
        parameterisations, that need to be run for a given workflow node.

    State-array
        A state array is a collection of parameterised tasks or values that were generated
        by a split operation either at the current or upstream node of a workflow. The
        size of the array is determined by the :ref:`State` of the workflow node.

    Splitter
        Defines how a task's inputs are to be split into multiple jobs. For example if
        a task's input takes an integer, a list of integers can be passed to it split
        over to create a ref:`State-array` of jobs. Different combinations of

    Submitter
        A submitter object parameterises how a task is to be executed, by defining the
        worker, environment, cache-root directory and other key execution parameters to
        be used when executing a task.

    Task
      A task describes a unit of work to be done (but not how it will be), either
      standalone or as one step in a larger workflow. Tasks can be of various types,
      including Python functions, shell commands, and nested workflows. Tasks are
      parameterised, meaning they can accept inputs and produce

    Worker
        Encapsulation of a task execution environment. It is responsible for executing
        tasks and managing their lifecycle. Workers can be local (e.g., debug and
        concurrent-futures multiprocess) or orchestrated through a remote scheduler
        (e.g., SLURM, SGE).

    Workflow
      A Directed-Acyclic-Graph (DAG) of parameterised tasks, to be executed in order.
      Note that a Workflow object is created by a :class:`WorkflowTask`'s
      `construct()` method at runtime and is not directly created by the end user.
