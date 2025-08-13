Software environments
=====================

Pydra supports running tasks within encapsulated software environments, such as Docker_
and Singularity_ containers. This can be specified at runtime or during workflow
construction, and allows tasks to be run in environments that are isolated from the
host system, and that have specific software dependencies.

The environment a task runs within is specified by the ``environment`` argument passed
to the execution call (e.g. ``my_task(worker="cf", environment="docker")``) or in the
``workflow.add()`` call in workflow constructors.

Specifying at execution
-----------------------

The environment for a task can be specified at execution time by passing the ``environment`` argument to the task call.
This can be an instance of `pydra.environments.native.Environment` (for the host system),
`pydra.environments.docker.Environment` (for Docker containers), or
`pydra.environments.singularity.Environment` (for Singularity containers), or a custom environment.

Example:

.. code-block:: python

    from pydra.environments import native, docker, singularity
    from pydra.compose import shell
    # Define a simple shell task
    shelly = shell.fuse("echo hello")

    # Execute with a native environment
    outputs_native = shelly(environment=native.Environment())

    # Execute with a Docker environment (assuming busybox image is available)
    outputs_docker = shelly(environment=docker.Environment(image="busybox"))

    # Execute with a Singularity environment (assuming an image is available)
    outputs_singularity = shelly(environment=singularity.Environment(image="/path/to/image.sif"))

Alternatively, when using a `pydra.engine.submitter.Submitter`, the environment can be specified in the Submitter constructor:

.. code-block:: python

    from pydra.engine.submitter import Submitter
    from pydra.environments import native
    from pydra.compose import shell

    shelly = shell.fuse("echo hello")
    with Submitter(environment=native.Environment()) as sub:
        result = sub(shelly)


Specifying at workflow construction
-----------------------------------

When constructing a workflow, the environment can be specified in the ``workflow.add()`` call.
This ensures that all tasks within that workflow branch will execute in the specified environment.

Example:

.. code-block:: python

    from pydra.environments import singularity
    from pydra.compose import workflow, shell
    from fileformats.generic import File

    image = "/path/to/my_singularity_image.sif" # Replace with your Singularity image path

    Singu = shell.define("cat {file}")

    def MyWorkflow(file: File) -> str:
        singu_task = workflow.add(
            Singu(file=file),
            environment=singularity.Environment(image=image),
        )
        return singu_task.stdout

    # Now you can use MyWorkflow, and the 'cat' task will run in the Singularity environment


Implementing new environment types
----------------------------------

Custom environment types can be implemented by creating a new class that inherits from `pydra.environments.Environment`.
These custom environment classes are typically located in the `pydra/environments/` directory.

Example (simplified custom environment):

.. code-block:: python

    from pydra.environments import Environment as PydraEnvironment

    class MyCustomEnvironment(PydraEnvironment):
        def __init__(self, some_config: str):
            super().__init__()
            self.some_config = some_config

        def _setup(self):
            # Logic to set up the custom environment
            print(f"Setting up custom environment with config: {self.some_config}")

        def _execute(self, command: list):
            # Logic to execute a command within the custom environment
            # This is where you would integrate with a custom execution system
            print(f"Executing command: {' '.join(command)} in custom environment")
            # For demonstration, just return a dummy result
            return {"stdout": "Custom environment output", "return_code": 0}

        def _tear_down(self):
            # Logic to tear down the custom environment
            print("Tearing down custom environment")

Then, you can use your custom environment like any other built-in environment:

.. code-block:: python

    from pydra.compose import shell
    # Assume MyCustomEnvironment is defined as above
    my_task = shell.fuse("echo Hello from custom env")
    outputs = my_task(environment=MyCustomEnvironment(some_config="test"))
    print(outputs.stdout)


.. _Docker: https://www.docker.com/
.. _Singularity: https://sylabs.io/singularity/
