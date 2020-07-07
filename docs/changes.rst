Release Notes
=============

0.6.2
-----

* Use pkgutil to declare ``pydra.tasks`` as a namespace package, ensuring better support for
  editable mode.

0.6.1
-----

* Add ``pydra.tasks`` namespace package to enable separate packages of ``Task``\s to be
  installed into ``pydra.tasks``.
* Raise error when task or workflow name conflicts with names of attributes, methods, or
  other tasks already added to workflow
* Mention ``requirements.txt`` in README

0.6
---

* removing the tutorial to a `separate repo <https://github.com/nipype/pydra-tutorial>`__
* adding windows tests to codecov
* accepting ``None`` as a valid output from a ``FunctionTask``, also for function that returns multiple values
* fixing slurm error files
* adding ``wf._connection`` to ``checksum``
* allowing for updates of ``wf._connections``
* editing output, so it works with ``numpy.arrays``
* removing ``to_job`` and pickling task instead (workers read the tasks and set the proper input, so the multile copies of the input are not kept in the memory)
* adding standalone function ``load_and_run`` that can load and run a task from a pickle file
* removing ``create_pyscript`` and simplifying the slurm worker
* improving error reports in errors flies
* fixing ``make_class`` so the ``Output`` is properly formatted

0.5
---

* fixing ``hash_dir`` function
* adding ``get_available_cpus`` to get the number of CPUs available to the current process or available on the system
* adding simple implementation for ``BoshTask`` that uses boutiques descriptor
* adding azure to CI
* fixing code for windows
* etelementry updates
* adding more verbose output for task ``result`` - returns values or indices for input fields
* adding an experimental implementation of Dask Worker (limited testing with ci)

0.4
---

* reorganization of the ``State`` class, fixing small issues with the class
* fixing some paths issues on windows os
* adding osx and window sto the travis runs (right now allowing for failures for windows)
* adding ``PydraStateError`` for exception in the ``State`` class
* small fixes to the hashing functions, adding more tests
* adding ``hash_dir`` to calculate hash for ``Directory`` type

0.3.1
-----

* passing ``wf.cache_locations`` to the task
* using ``rerun`` from submitter to all task
* adding ``test_rerun`` and ``propagate_rerun`` for workflows
* fixing task with a full combiner
* adding ``cont_dim`` to specify dimensionality of the input variables (how much the input is nested)

0.3
---

* adding sphinx documentation
* moving from ``dataclasses`` to ``attrs``
* adding ``container`` flag to the ``ShellCommandTask``
* fixing ``cmdline``, ``command_args`` and ``container_args`` for tasks with states
* adding ``CONTRIBUTING.md``
* fixing hash calculations for inputs with a list of files
* using ``attr.NOTHING`` for input that is not set

0.2.2
-----

* supporting tuple as a single element of an input

0.2.1
-----

* fixing: nodes with states and input fields (from splitter) that are empty were failing

0.2
---

* big changes in ``ShellTask``, ``DockerTask`` and ``SingularityTask``
    * customized input specification and output specification for ``Task``\s
    * adding singularity checks to Travis CI
    * binding all input files to the container
* changes in ``Workflow``
    * passing all outputs to the next node: ``lzout.all_``
    * fixing inner splitter
* allowing for ``splitter`` and ``combiner`` updates
* adding ``etelementry`` support

0.1
---

* Core dataflow creation and management API
* Distributed workers:
    * concurrent futures
    * SLURM
* Notebooks for Pydra concepts

0.0.1
-----

Initial Pydra Dataflow Engine release.
