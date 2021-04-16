Dataflows Components: Task and Workflow
=======================================
A *Task* is the basic runnable component of *Pydra* and is described by the
class ``TaskBase``. A *Task* has named inputs and outputs, thus allowing
construction of dataflows. It can be hashed and executes in a specific working
directory. Any *Pydra*'s *Task* can be used as a function in a script, thus allowing
dual use in *Pydra*'s *Workflows* and in standalone scripts. There are several
classes that inherit from ``TaskBase`` and each has a different application:


Function Tasks
--------------

* ``FunctionTask`` is a *Task* that executes Python functions. Most Python functions
  declared in an existing library, package, or interactively in a terminal can
  be converted to a ``FunctionTask`` by using *Pydra*'s decorator - ``mark.task``.

  .. code-block:: python

     import numpy as np
     from pydra import mark
     fft = mark.annotate({'a': np.ndarray,
                      'return': float})(np.fft.fft)
     fft_task = mark.task(fft)()
     result = fft_task(a=np.random.rand(512))


  `fft_task` is now a *Pydra* *Task* and result will contain a *Pydra*'s ``Result`` object.
  In addition, the user can use Python's function annotation or another *Pydra*
  decorator --- ``mark.annotate`` in order to specify the output. In the
  following example, we decorate an arbitrary Python function to create named
  outputs:

  .. code-block:: python

     @mark.task
     @mark.annotate(
         {"return": {"mean": float, "std": float}}
     )
     def mean_dev(my_data):
         import statistics as st
         return st.mean(my_data), st.stdev(my_data)

     result = mean_dev(my_data=[...])()

  When the *Task* is executed `result.output` will contain two attributes: `mean`
  and `std`. Named attributes facilitate passing different outputs to
  different downstream nodes in a dataflow.


.. _shell_command_task:

Shell Command Tasks
-------------------

* ``ShellCommandTask`` is a *Task* used to run shell commands and executables.
  It can be used with a simple command without any arguments, or with specific
  set of arguments and flags, e.g.:

  .. code-block:: python

     ShellCommandTask(executable="pwd")

     ShellCommandTask(executable="ls", args="my_dir")

  The *Task* can accommodate more complex shell commands by allowing the user to
  customize inputs and outputs of the commands.
  One can generate an input
  specification to specify names of inputs, positions in the command, types of
  the inputs, and other metadata.
  As a specific example, FSL's BET command (Brain
  Extraction Tool) can be called on the command line as:

  .. code-block:: python

    bet input_file output_file -m

  Each of the command argument can be treated as a named input to the
  ``ShellCommandTask``, and can be included in the input specification.
  As shown next, even an output is specified by constructing
  the *out_file* field form a template:

  .. code-block:: python

    bet_input_spec = SpecInfo(
        name="Input",
        fields=[
        ( "in_file", File,
          { "help_string": "input file ...",
            "position": 1,
            "mandatory": True } ),
        ( "out_file", str,
          { "help_string": "name of output ...",
            "position": 2,
            "output_file_template":
                              "{in_file}_br" } ),
        ( "mask", bool,
          { "help_string": "create binary mask",
            "argstr": "-m", } ) ],
        bases=(ShellSpec,) )

    ShellCommandTask(executable="bet",
                     input_spec=bet_input_spec)

  More details are in the :ref:`Input Specification section`.

Container Tasks
---------------
* ``ContainerTask`` class is a child class of ``ShellCommandTask`` and serves as
  a parent class for ``DockerTask`` and ``SingularityTask``. Both *Container Tasks*
  run shell commands or executables within containers with specific user defined
  environments using Docker_ and Singularity_ software respectively.
  This might be extremely useful for users and projects that require environment
  encapsulation and sharing.
  Using container technologies helps improve scientific
  workflows reproducibility, one of the key concept behind *Pydra*.

  These *Container Tasks* can be defined by using
  ``DockerTask`` and ``SingularityTask`` classes directly, or can be created
  automatically from ``ShellCommandTask``, when an optional argument
  ``container_info`` is used when creating a *Shell Task*. The following two
  types of syntax are equivalent:

  .. code-block:: python

     DockerTask(executable="pwd", image="busybox")

     ShellCommandTask(executable="ls",
          container_info=("docker", "busybox"))

Workflows
---------
* ``Workflow`` - is a subclass of *Task* that provides support for creating *Pydra*
  dataflows. As a subclass, a *Workflow* acts like a *Task* and has inputs, outputs,
  is hashable, and is treated as a single unit. Unlike *Tasks*, workflows embed
  a directed acyclic graph. Each node of the graph contains a *Task* of any type,
  including another *Workflow*, and can be added to the *Workflow* simply by calling
  the ``add`` method. The connections between *Tasks* are defined by using so
  called *Lazy Inputs* or *Lazy Outputs*. These are special attributes that allow
  assignment of values when a *Workflow* is executed rather than at the point of
  assignment. The following example creates a *Workflow* from two *Pydra* *Tasks*.

  .. code-block:: python

    # creating workflow with two input fields
    wf = Workflow(input_spec=["x", "y"])
    # adding a task and connecting task's input
    # to the workflow input
    wf.add(mult(name="mlt",
                   x=wf.lzin.x, y=wf.lzin.y))
    # adding anoter task and connecting
    # task's input to the "mult" task's output
    wf.add(add2(name="add", x=wf.mlt.lzout.out))
    # setting worflow output
    wf.set_output([("out", wf.add.lzout.out)])


Task's State
------------
All Tasks, including Workflows, can have an optional attribute representing an instance of the State class.
This attribute controls the execution of a Task over different input parameter sets.
This class is at the heart of Pydra's powerful Map-Reduce over arbitrary inputs of nested dataflows feature.
The State class formalizes how users can specify arbitrary combinations.
Its functionality is used to create and track different combinations of input parameters,
and optionally allow limited or complete recombinations.
In order to specify how the inputs should be split into parameter sets, and optionally combined after
the Task execution, the user can set splitter and combiner attributes of the State class.

.. code-block:: python

  task_with_state =
        add2(x=[1, 5]).split("x").combine("x")

In this example, the ``State`` class is responsible for creating a list of two
separate inputs, *[{x: 1}, {x:5}]*, each run of the *Task* should get one
element from the list.
The results are grouped back when returning the result from the *Task*.
While this example
illustrates mapping and grouping of results over a single parameter, *Pydra*
extends this to arbitrary combinations of input fields and downstream grouping
over nested dataflows. Details of how splitters and combiners power *Pydra*'s
scalable dataflows are described in the next section.



.. _Docker: https://www.docker.com/
.. _Singularity: https://www.singularity.lbl.gov/
