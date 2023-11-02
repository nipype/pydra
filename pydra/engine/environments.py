from .helpers import execute

from pathlib import Path


class Environment:
    """
    Base class for environments that are used to execute tasks.
    Right now it is asssumed that the environment, including container images,
    are available and are not removed at the end
    TODO: add setup and teardown methods
    """

    def setup(self):
        pass

    def execute(self, task):
        """
        Execute the task in the environment.

        Parameters
        ----------
        task : TaskBase
            the task to execute

        Returns
        -------
        output
            Output of the task.
        """
        raise NotImplementedError

    def teardown(self):
        pass


class Native(Environment):
    """
    Native environment, i.e. the tasks are executed in the current python environment.
    """

    def execute(self, task):
        keys = ["return_code", "stdout", "stderr"]
        values = execute(task.command_args(), strip=task.strip)
        output = dict(zip(keys, values))
        if output["return_code"]:
            msg = f"Error running '{task.name}' task with {task.command_args()}:"
            if output["stderr"]:
                msg += "\n\nstderr:\n" + output["stderr"]
            if output["stdout"]:
                msg += "\n\nstdout:\n" + output["stdout"]
            raise RuntimeError(msg)
        return output


class Container(Environment):
    """
    Base class for container environments used by Docker and Singularity.

    Parameters
    ----------
    image : str
        Name of the container image
    tag : str
        Tag of the container image
    output_cpath : str
        Path to the output directory in the container
    xargs : dict
        Extra arguments to be passed to the container
    """

    def __init__(self, image, tag="latest", output_cpath="/output_pydra", xargs=None):
        self.image = image
        self.tag = tag
        self.xargs = xargs
        self.output_cpath = output_cpath

    @staticmethod
    def bind(loc, mode="ro", root="/mnt/pydra"):  # TODO
        # XXX Failure mode: {loc} overwrites a critical directory in image
        # To fix, we'll need to update any args within loc to a new location
        # such as /mnt/pydra/loc
        loc_abs = Path(loc).absolute()
        return f"{loc_abs}:{root}{loc_abs}:{mode}"  # TODO: moving entire path?


class Docker(Container):
    """Docker environment."""

    def execute(self, task, root="/mnt/pydra"):
        docker_img = f"{self.image}:{self.tag}"
        # mounting all input locations
        mounts = task.get_bindings(root=root)

        # todo adding xargsy etc
        docker_args = ["docker", "run", "-v", self.bind(task.cache_dir, "rw")]
        docker_args.extend(
            " ".join(
                [f"-v {key}:{val[0]}:{val[1]}" for (key, val) in mounts.items()]
            ).split()
        )
        docker_args.extend(["-w", f"{root}{task.output_dir}"])
        keys = ["return_code", "stdout", "stderr"]
        # print("\n Docker args", docker_args)

        values = execute(
            docker_args + [docker_img] + task.command_args(root="/mnt/pydra"),
            strip=task.strip,
        )
        output = dict(zip(keys, values))
        if output["return_code"]:
            if output["stderr"]:
                raise RuntimeError(output["stderr"])
            else:
                raise RuntimeError(output["stdout"])
        return output


class Singularity(Container):
    """Singularity environment."""

    def execute(self, task, root="/mnt/pydra"):
        singularity_img = f"{self.image}:{self.tag}"
        # mounting all input locations
        mounts = task.get_bindings(root=root)

        # todo adding xargsy etc
        singularity_args = [
            "singularity",
            "exec",
            "-B",
            self.bind(task.cache_dir, "rw"),
        ]
        singularity_args.extend(
            " ".join(
                [f"-B {key}:{val[0]}:{val[1]}" for (key, val) in mounts.items()]
            ).split()
        )
        singularity_args.extend(["--pwd", f"{root}{task.output_dir}"])
        keys = ["return_code", "stdout", "stderr"]

        values = execute(
            singularity_args + [singularity_img] + task.command_args(root="/mnt/pydra"),
            strip=task.strip,
        )
        output = dict(zip(keys, values))
        if output["return_code"]:
            if output["stderr"]:
                raise RuntimeError(output["stderr"])
            else:
                raise RuntimeError(output["stdout"])
        return output
