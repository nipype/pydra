from .helpers import execute


class Environment:
    def setup(self):
        pass

    def execute(self, task):
        raise NotImplementedError

    def teardown(self):
        pass


class Native(Environment):
    def execute(self, task):
        args = task.render_arguments_in_root()
        keys = ["return_code", "stdout", "stderr"]
        values = execute(args, strip=task.strip)
        output = dict(zip(keys, values))
        if output["return_code"]:
            if output["stderr"]:
                raise RuntimeError(output["stderr"])
            else:
                raise RuntimeError(output["stdout"])
        return output


class Docker(Environment):
    def __init__(self, image, tag="latest"):
        self.image = image
        self.tag = tag

    @staticmethod
    def bind(loc, mode="ro"):
        # XXX Failure mode: {loc} overwrites a critical directory in image
        # To fix, we'll need to update any args within loc to a new location
        # such as /mnt/pydra/loc
        return f"{loc}:{loc}:{mode}"

    def execute(self, task):
        # XXX Need to mount all input locations
        docker_img = f"{self.image}:{self.tag}"
        # Renders arguments where `File`s have an additional prefix
        args = task.render_arguments_in_root("/mnt/pydra")
        # Skips over any inputs in task.cache_dir
        # Needs to include `out_file`s when not relative to working dir
        # Possibly a `TargetFile` type to distinguish between `File` and `str`?
        mounts = task.get_inputs_in_root(root="/mnt/pydra")

        docker_args = ["docker", "run", "-v", self.bind(task.cache_dir, "rw")]
        docker_args.extend(flatten(["-v", self.bind(mount)] for mount in mounts))
        keys = ["return_code", "stdout", "stderr"]
        values = execute(docker_args + [docker_img] + args, strip=task.strip)
        output = dict(zip(keys, values))
        if output["return_code"]:
            if output["stderr"]:
                raise RuntimeError(output["stderr"])
            else:
                raise RuntimeError(output["stdout"])
        # Any outputs that have been created with a re-rooted path need
        # to be de-rooted
        task.finalize_outputs("/mnt/pydra")
        return output
