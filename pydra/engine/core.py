"""Abstract base classes for compute objects"""


class Interface():

    inputs_ = None
    output_names_ = None
    version_ = None
    api_version_ = 0.0.1
    hints_ = None
    validate_ = None
    configs_ = None

    def __init__(self):
        self.result = None

    @property
    def inputs(self):
        pass

    @property
    def output_names(self):
        pass

    @property
    def version(self):
        """Semantic version of the tool"""
        return self.version_

    @property
    def api_version(self):
        """Semantic version of the Interface class"""
        return self.api_version_

    @property
    def validate(self):
        return self.validate_

    @property
    def configs(self):
        """Configuration objects for parameters of the interface"""
        return self.configs_

    def run_interface(self):
        output_dict = None
        return output_dict

    def __call__(self, *args, **kwargs):
        return self.run_interface(*args, **kwargs)

    def from_spec(self, path):
        """Create interface from a spec
        :param path: url or local path of a Boutiques/CWL spec
        """
        pass

    def to_spec(self, path, format):
        """Convert Interface to a Boutiques or CWL spec

        :param path: Write spec to path
        :param format: Use this format: 'Boutiques' or 'CWL'
        :return: path of converted file
        """
        pass


class Task(Interface):

    execution_environment_ = None

    @property
    def result(self):
        if self.result:
            return self.result
        else:
            load_result()

    def run(self, **kwargs):
        if not cache_available():
            env = create_environment(self.execution_environment_) # includes isolation of inputs
            id = record_provenance(self, env)
            resources = start_monitor()
            outputs = None
            try:
                outputs = self(**kwargs)
                resources = stop_monitor()
            except Exception as e:
                record_error(self, e)
            update_provenance(id, outputs, resources)
        return self.result


class FunctionTask(Task):
    def __init__(self, func=None, **kwargs):
        pass

def to_interface(func_to_decorate):
    def create_func(**original_kwargs):
        original_kwargs['function'] = func_to_decorate
        function_task = FunctionTask(func = func_to_decorate,
                                     **original_kwargs)
        return function_task
    return create_func
