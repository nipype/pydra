import typing as ty
import re
from pathlib import Path
from copy import copy
from typing import Self
import attrs.validators
from pydra.utils.typing import is_optional, is_fileset_or_union, is_truthy_falsy
from pydra.utils.general import get_fields
from pydra.utils.typing import StateArray, is_lazy
from pydra.utils.hash import hash_function
import os
import itertools
from collections import Counter
import attrs
import cloudpickle as cp
from pydra.utils.messenger import AuditFlag, Messenger
from pydra.utils.general import (
    attrs_fields,
    attrs_values,
)
from pydra.utils.hash import Cache, hash_single, register_serializer
from .field import Field, Arg, Out


if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.environments.base import Environment
    from pydra.workers.base import Worker
    from pydra.engine.result import Result
    from pydra.engine.hooks import TaskHooks

TaskType = ty.TypeVar("TaskType", bound="Task")


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class Outputs:
    """Base class for all output definitions"""

    RESERVED_FIELD_NAMES = ("inputs",)
    BASE_ATTRS = ()

    _cache_dir: Path = attrs.field(default=None, init=False, repr=False)
    _node = attrs.field(default=None, init=False, repr=False)

    @property
    def inputs(self):
        """The inputs object associated with a lazy-outputs object"""
        if self._node is None:
            raise AttributeError(
                f"{self} outputs object is not a lazy output of a workflow node"
            )
        return self._node.inputs

    @classmethod
    def _from_job(cls, job: "Job[TaskType]") -> Self:
        """Collect the outputs of a job. This is just an abstract base method that
        should be used by derived classes to set default values for the outputs.

        Parameters
        ----------
        job : Job[TaskType]
            The job whose outputs are being collected.

        Returns
        -------
        outputs : Outputs
            The outputs of the job
        """
        defaults = {}
        for output in get_fields(cls):
            if output.mandatory:
                default = attrs.NOTHING
            elif isinstance(output.default, attrs.Factory):
                default = output.default.factory()
            else:
                default = output.default
            defaults[output.name] = default
        outputs = cls(**defaults)
        outputs._cache_dir = job.cache_dir
        return outputs

    @property
    def _results(self) -> "Result[Self]":
        results_path = self._cache_dir / "_job.pklz"
        if not results_path.exists():
            raise FileNotFoundError(f"Job results file {results_path} not found")
        with open(results_path, "rb") as f:
            return cp.load(f)

    def __iter__(self) -> ty.Generator[str, None, None]:
        """The names of the fields in the output object"""
        return iter(sorted(f.name for f in attrs_fields(self)))

    def __getitem__(self, name_or_index: str | int) -> ty.Any:
        """Return the value for the given attribute

        Parameters
        ----------
        name : str
            the name of the attribute to return

        Returns
        -------
        Any
            the value of the attribute
        """
        if isinstance(name_or_index, int):
            return list(self)[name_or_index]
        try:
            return getattr(self, name_or_index)
        except AttributeError:
            raise KeyError(
                f"{self} doesn't have an attribute {name_or_index}"
            ) from None

    def __eq__(self, other: ty.Any) -> bool:
        """Check if two tasks are equal"""
        values = attrs.asdict(self)
        fields = get_fields(self)
        try:
            other_values = attrs.asdict(other)
        except AttributeError:
            return False
        try:
            other_fields = get_fields(other)
        except AttributeError:
            return False
        if fields != other_fields:
            return False
        for field in get_fields(self):
            if field.hash_eq:
                values[field.name] = hash_function(values[field.name])
                other_values[field.name] = hash_function(other_values[field.name])
        return values == other_values

    def __repr__(self) -> str:
        """A string representation of the task"""
        fields_str = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in get_fields(self)
            if getattr(self, f.name) != f.default
        )
        return f"{self.__class__.__name__}({fields_str})"


OutputsType = ty.TypeVar("OutputType", bound=Outputs)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class Task(ty.Generic[OutputsType]):
    """Base class for all tasks"""

    # Task type to be overridden in derived classes
    @classmethod
    def _task_type(cls) -> str:
        for base in cls.__mro__:
            parts = base.__module__.split(".")
            if parts[:2] == ["pydra", "compose"]:
                return parts[2]
        raise RuntimeError(
            f"Cannot determine task type for {cls.__name__} in module {cls.__module__} "
            "because none of its base classes are in the pydra.compose namespace:\n"
            + "\n".join(f"{b.__name__!r} in {b.__module__!r}" for b in cls.__mro__)
        )

    # The attribute containing the function/executable used to run the task
    _executor_name = None

    # Class attributes
    TASK_CLASS_ATTRS = ("xor",)
    _xor: frozenset[frozenset[str | None]] = (
        frozenset()
    )  # overwritten in derived classes

    # The following fields are used to store split/combine state information
    _splitter = attrs.field(default=None, init=False, repr=False)
    _combiner = attrs.field(default=None, init=False, repr=False)
    _container_ndim = attrs.field(default=None, init=False, repr=False)
    _hashes = attrs.field(default=None, init=False, eq=False, repr=False)

    RESERVED_FIELD_NAMES = ("split", "combine")
    BASE_ATTRS = ()

    def __call__(
        self,
        /,
        cache_root: os.PathLike | None = None,
        worker: "str | ty.Type[Worker] | Worker" = "debug",
        environment: "Environment | None" = None,
        rerun: bool = False,
        readonly_caches: ty.Iterable[os.PathLike] | None = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers: ty.Iterable[Messenger] | None = None,
        messenger_args: dict[str, ty.Any] | None = None,
        hooks: "TaskHooks | None" = None,
        **kwargs: ty.Any,
    ) -> OutputsType:
        """Create a job from this task and execute it to produce a result.

        Parameters
        ----------
        cache_root : os.PathLike, optional
            Cache directory where the working directory/results for the job will be
            stored, by default None
        worker : str or Worker, optional
            The worker to use, by default "cf"
        environment: Environment, optional
            The execution environment to use, by default None
        rerun : bool, optional
            Whether to force the re-computation of the job results even if existing
            results are found, by default False
        readonly_caches : list[os.PathLike], optional
            Alternate cache locations to check for pre-computed results, by default None
        audit_flags : AuditFlag, optional
            Configure provenance tracking. available flags: :class:`~pydra.utils.messenger.AuditFlag`
            Default is no provenance tracking.
        messenger : :class:`Messenger` or :obj:`list` of :class:`Messenger` or None
            Messenger(s) used by Audit. Saved in the `audit` attribute.
            See available flags at :class:`~pydra.utils.messenger.Messenger`.
        messengers_args : messengers_args : dict[str, Any], optional
            Argument(s) used by `messegner`. Saved in the `audit` attribu
        **kwargs : dict
            Keyword arguments to pass on to the worker initialisation

        Returns
        -------
        OutputsType or list[OutputsType]
            The output interface of the job, or in the case of split tasks, a list of
            output interfaces
        """
        from pydra.engine.submitter import (  # noqa: F811
            Submitter,
            WORKER_KWARG_FAIL_NOTE,
        )

        try:
            with Submitter(
                audit_flags=audit_flags,
                cache_root=cache_root,
                readonly_caches=readonly_caches,
                messenger_args=messenger_args,
                messengers=messengers,
                environment=environment,
                worker=worker,
                **kwargs,
            ) as sub:
                result = sub(
                    self,
                    hooks=hooks,
                    rerun=rerun,
                )
        except TypeError as e:
            # Catch any inadvertent passing of task parameters to the
            # execution call
            if hasattr(e, "__notes__") and WORKER_KWARG_FAIL_NOTE in e.__notes__:
                if match := re.match(
                    r".*got an unexpected keyword argument '(\w+)'", str(e)
                ):
                    if match.group(1) in self:
                        e.add_note(
                            f"Note that the unrecognised argument, {match.group(1)!r}, is "
                            f"an input of the task {self!r} that has already been "
                            f"parameterised (it is being called to execute it)"
                        )
            raise
        if result.errored:
            if result.errors:
                time_of_crash = result.errors["time of crash"]
                error_message = "\n".join(result.errors["error message"])
            else:
                time_of_crash = "UNKNOWN-TIME"
                error_message = "NOT RETRIEVED"
            raise RuntimeError(
                f"Job {self} failed @ {time_of_crash} with the "
                f"following errors:\n{error_message}\n"
                "To inspect, please load the pickled job object from here: "
                f"{result.cache_dir}/_job.pklz"
            )
        return result.outputs

    def split(
        self,
        splitter: ty.Union[str, ty.List[str], ty.Tuple[str, ...], None] = None,
        /,
        overwrite: bool = False,
        container_ndim: ty.Optional[dict] = None,
        **inputs,
    ) -> Self:
        """
        Run this job parametrically over lists of split inputs.

        Parameters
        ----------
        splitter : str or list[str] or tuple[str] or None
            the fields which to split over. If splitting over multiple fields, lists of
            fields are interpreted as outer-products and tuples inner-products. If None,
            then the fields to split are taken from the keyword-arg names.
        overwrite : bool, optional
            whether to overwrite an existing split on the node, by default False
        container_ndim : dict, optional
            Container dimensions for specific inputs, used in the splitter.
            If input name is not in container_ndim, it is assumed that the input values has
            a container dimension of 1, so only the most outer dim will be used for splitting.
        **inputs
            fields to split over, will be automatically wrapped in a StateArray object
            and passed to the node inputs

        Returns
        -------
        self : TaskBase
            a reference to the job
        """
        from pydra.engine.state import unwrap_splitter
        from pydra.engine import lazy

        if self._splitter and not overwrite:
            raise ValueError(
                f"Cannot overwrite existing splitter {self._splitter} on {self}, "
                "set 'overwrite=True' to do so"
            )
        if splitter:
            unwraped_split = list(unwrap_splitter(splitter))
            if duplicated := [f for f, c in Counter(unwraped_split).items() if c > 1]:
                raise ValueError(f"Splitter fields {duplicated} are duplicated")
            split_names = set(
                s for s in unwraped_split if not s.startswith("_") and "." not in s
            )
            input_names = set(inputs)
            if missing_inputs := list(split_names - input_names):
                raise ValueError(
                    f"Splitter fields {missing_inputs} need to be provided as a keyword "
                    f"arguments to the split method (provided {list(inputs)})"
                )
            if unrecognised_inputs := list(input_names - split_names):
                raise ValueError(
                    f"Provided inputs {unrecognised_inputs} are not present in the "
                    f"splitter {splitter}"
                )
        else:
            # If no splitter is provided, use the names of the inputs as combinatorial splitter
            split_names = splitter = list(inputs)
        for field_name in container_ndim or []:
            if field_name not in split_names:
                raise ValueError(
                    f"Container dimension for {field_name} is provided but the field "
                    f"is not present in the inputs"
                )
        split_inputs = {}
        for name, value in inputs.items():
            if isinstance(value, lazy.LazyField):
                split_val = value.split()
            elif isinstance(value, ty.Iterable) and not isinstance(
                value, (ty.Mapping, str)
            ):
                split_val = StateArray(value)
            else:
                raise TypeError(
                    f"Could not split {value!r} as it is not a sequence type"
                )
            split_inputs[name] = split_val
        split_def = attrs.evolve(self, **split_inputs)
        split_def._splitter = splitter
        split_def._container_ndim = container_ndim
        return split_def

    def combine(
        self,
        combiner: ty.Union[ty.List[str], str],
        overwrite: bool = False,
    ) -> Self:
        """
        Combine inputs parameterized by one or more previous tasks.

        Parameters
        ----------
        combiner : list[str] or str
            the field or list of inputs to be combined (i.e. not left split) after the
            job has been run
        overwrite : bool
            whether to overwrite an existing combiner on the node
        **kwargs : dict[str, Any]
            values for the job that will be "combined" before they are provided to the
            node

        Returns
        -------
        self : Self
            a reference to the outputs object
        """
        if self._combiner and not overwrite:
            raise ValueError(
                f"Attempting to overwrite existing combiner {self._combiner} on {self}, "
                "set 'overwrite=True' to do so"
            )
        if isinstance(combiner, str):
            combiner = [combiner]
        local_names = set(c for c in combiner if "." not in c and not c.startswith("_"))
        if unrecognised := local_names - set(self):
            raise ValueError(
                f"Combiner fields {unrecognised} are not present in the task"
            )
        combined_def = copy(self)
        combined_def._combiner = combiner
        return combined_def

    def __repr__(self) -> str:
        """A string representation of the task"""
        fields_str = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in get_fields(self)
            if getattr(self, f.name) != f.default
        )
        return f"{self.__class__.__name__}({fields_str})"

    def __iter__(self) -> ty.Generator[str, None, None]:
        """Iterate through all the names in the task"""
        return (
            f.name
            for f in get_fields(self)
            if not (f.name.startswith("_") or f.name in self.RESERVED_FIELD_NAMES)
        )

    def __eq__(self, other: ty.Any) -> bool:
        """Check if two tasks are equal"""
        values = attrs.asdict(self, recurse=False)
        if not isinstance(other, Task):
            return False
        try:
            other_values = attrs.asdict(other, recurse=False)
        except AttributeError:
            return False
        if set(values) != set(other_values):
            return False  # Return if attribute keys don't match
        for field in get_fields(self):
            if field.hash_eq:
                values[field.name] = hash_function(values[field.name])
                other_values[field.name] = hash_function(other_values[field.name])
        if values != other_values:
            return False
        hash_cache = Cache()
        if hash_function(type(self), cache=hash_cache) != hash_function(
            type(other), cache=hash_cache
        ):
            return False
        try:
            other_outputs = other.Outputs
        except AttributeError:
            return False
        return hash_function(self.Outputs, cache=hash_cache) == hash_function(
            other_outputs, cache=hash_cache
        )

    def __getitem__(self, name: str) -> ty.Any:
        """Return the value for the given attribute, resolving any templates

        Parameters
        ----------
        name : str
            the name of the attribute to return

        Returns
        -------
        Any
            the value of the attribute
        """
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(f"{self} doesn't have an attribute {name}") from None

    @property
    def _hash(self):
        hsh, self._hashes = self._compute_hashes()
        return hsh

    @property
    def _checksum(self):
        return f"{self._task_type()}-{self._hash}"

    def _hash_changes(self):
        """Detects any changes in the hashed values between the current inputs and the
        previously calculated values"""
        _, new_hashes = self._compute_hashes()
        return [k for k, v in new_hashes.items() if v != self._hashes[k]]

    def _compute_hashes(self) -> ty.Tuple[bytes, ty.Dict[str, bytes]]:
        """Compute a basic hash for any given set of fields."""
        inp_dict = {}
        for field in get_fields(self):
            if isinstance(field, Out):
                continue  # Skip output fields
            # removing values that are not set from hash calculation
            if getattr(self, field.name) is attrs.NOTHING:
                continue
            if getattr(field, "container_path", False):
                continue
            inp_dict[field.name] = getattr(self, field.name)
        # Include the outputs class, just in case any names or types have changed
        inp_dict["Outputs"] = self.Outputs
        hash_cache = Cache()
        field_hashes = {
            k: hash_function(v, cache=hash_cache) for k, v in inp_dict.items()
        }
        return hash_function(sorted(field_hashes.items())), field_hashes

    def _rule_violations(self) -> list[str]:
        """Check rules and returns a list of errors."""

        field: Arg
        errors = []
        for field in get_fields(self):
            value = self[field.name]

            if is_lazy(value):
                continue

            if (
                value is attrs.NOTHING
                and not getattr(field, "path_template", False)
                and not field.readonly
            ):
                errors.append(f"Mandatory field {field.name!r} is not set")

            # Raise error if any required field is unset.
            if (
                not (
                    value is None
                    or value is False
                    or (
                        is_optional(field.type)
                        and is_fileset_or_union(field.type)
                        and value is True
                    )
                )
                and field.requires
                and not any(rs.satisfied(self) for rs in field.requires)
            ):
                if len(field.requires) > 1:
                    qualification = (
                        " at least one of the following requirements to be satisfied: "
                    )
                else:
                    qualification = ""
                errors.append(
                    f"{field.name!r} requires{qualification} {[str(r) for r in field.requires]}"
                )
        # Collect alternative fields associated with this field.
        for xor_set in self._xor:
            mutually_exclusive = {name: self[name] for name in xor_set if name}
            are_set = [f"{n}={v!r}" for n, v in mutually_exclusive.items() if v]
            if len(are_set) > 1:
                errors.append(
                    f"Mutually exclusive fields ({', '.join(sorted(are_set))}) are set "
                    "together"
                )
            elif not are_set and None not in xor_set:
                errors.append(
                    "At least one of the mutually exclusive fields should be set: "
                    + ", ".join(f"{n}={v!r}" for n, v in mutually_exclusive.items())
                )
        return errors

    def _check_rules(self):
        """Check if all rules are satisfied."""

        attrs.validate(self)

        if errors := self._rule_violations():
            raise ValueError(
                f"Found the following errors in job {self} task:\n" + "\n".join(errors)
            )

    @classmethod
    def _check_arg_refs(
        cls,
        inputs: list[Arg],
        outputs: list[Out],
        xor: frozenset[frozenset[str | None]],
    ) -> None:
        """
        Checks if all fields referenced in requirements and xor are present in the inputs
        are valid field names
        """
        field: Field
        input_names = set(inputs)
        for field in itertools.chain(inputs.values(), outputs.values()):
            if unrecognised := (
                set([r.name for s in field.requires for r in s]) - input_names
            ):
                raise ValueError(
                    "'Unrecognised' field names in referenced in the requirements "
                    f"of {field} " + str(list(unrecognised))
                )

        for xor_set in xor:
            if unrecognised := xor_set - (input_names | {None}):
                raise ValueError(
                    f"Unrecognised field names in referenced in the xor {xor_set}: "
                    + str(list(unrecognised))
                )
            for field_name in xor_set:
                if field_name is None:  # i.e. none of the fields being set is valid
                    continue
                type_ = inputs[field_name].type
                if not is_truthy_falsy(type_):
                    raise ValueError(
                        f"Fields included in a 'xor' ({field_name!r}) must be an optional type or a "
                        f"truthy/falsy type, not type {type_}"
                    )

    def _check_resolved(self):
        """Checks that all the fields in the task have been resolved"""
        if lazy_values := [n for n, v in attrs_values(self).items() if is_lazy(v)]:
            raise ValueError(
                f"Cannot execute {self} because the following fields "
                f"still have lazy values {lazy_values}"
            )


# def set_none_default_if_optional(field: Field) -> None:
#     if is_optional(field.type) and field.mandatory:
#         field.default = None


@register_serializer
def bytes_repr_task(obj: Task, cache: Cache) -> ty.Iterator[bytes]:
    yield f"task[{obj._task_type()}]:(".encode()
    for field in get_fields(obj):
        yield f"{field.name}=".encode()
        yield hash_single(getattr(obj, field.name), cache)
        yield b","
    yield b"_splitter="
    yield hash_single(obj._splitter, cache)
    yield b",_combiner="
    yield hash_single(obj._combiner, cache)
    yield b",_container_ndim="
    yield hash_single(obj._container_ndim, cache)
    yield b",_xor="
    yield hash_single(obj._xor, cache)
    yield b")"
