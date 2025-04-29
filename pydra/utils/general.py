"""Administrative support for the engine framework."""

from pathlib import Path
import inspect
import sys
import typing as ty
from collections.abc import Mapping, Collection
from copy import copy
import re
import attrs
import ast
import tempfile
import importlib
import types
import sysconfig
import platformdirs
import builtins
import pkgutil
import logging
import pydra.scripts
from ._version import __version__


logger = logging.getLogger("pydra")
if ty.TYPE_CHECKING:
    from pydra.compose.base import Task, Field  # noqa
    from pydra.compose import workflow


SCRIPTS_DIR = Path(pydra.scripts.__file__).parent

PYDRA_ATTR_METADATA = "__PYDRA_METADATA__"

TaskType = ty.TypeVar("TaskType", bound="Task")


user_cache_root = Path(
    platformdirs.user_cache_dir(
        appname="pydra",
        appauthor="nipype",
        version=__version__,
    )
)

default_run_cache_root = user_cache_root / "run-cache"


def add_exc_note(e: Exception, note: str) -> Exception:
    """Adds a note to an exception in a Python <3.11 compatible way

    Parameters
    ----------
    e : Exception
        the exception to add the note to
    note : str
        the note to add

    Returns
    -------
    Exception
        returns the exception again
    """
    if hasattr(e, "add_note"):
        e.add_note(note)
    else:
        e.args = (e.args[0] + "\n" + note,)
    return e


def exc_info_matches(exc_info, match, regex=False):
    if exc_info.value.__cause__ is not None:
        msg = str(exc_info.value.__cause__)
    else:
        msg = str(exc_info.value)
    if regex:
        return re.match(".*" + match, msg)
    else:
        return match in msg


def get_undefined_symbols(
    func, exclude_signature_type_hints: bool = False, ignore_decorator: bool = False
):
    """
    Check the source code of a function and detect any symbols that aren't defined in its scope.

    Parameters
    ----------
    func : callable
        The function to analyze.

    Returns
    -------
    set
        A set of undefined symbols.
    """
    # Get the source code of the function
    source = inspect.getsource(func)

    # De-indent the source code if required
    indent = re.match(r"^\s*", source).group()
    source = ("\n" + source).replace("\n" + indent, "\n")

    if ignore_decorator:
        # Remove the decorator from the source code, i.e. everything before the first
        # unindented 'def ' keyword.
        source = re.match(
            r"(.*\n)(def .*)", "\n" + source, flags=re.MULTILINE | re.DOTALL
        ).group(2)

    # Parse the source code into an AST
    tree = ast.parse(source)

    # Define a visitor class to traverse the AST
    class SymbolVisitor(ast.NodeVisitor):

        def __init__(self):
            # Initialize sets to track defined and used symbols
            self.defined_symbols = set()
            self.used_symbols = set()

        def visit_FunctionDef(self, node):
            # Add function arguments to defined symbols
            for arg in node.args.args:
                self.defined_symbols.add(arg.arg)
            if exclude_signature_type_hints:
                # Exclude type hints from the defined symbols
                type_hints_visitor = SymbolVisitor()
                if node.returns:
                    type_hints_visitor.visit(node.returns)
                for arg in node.args.args:
                    if arg.annotation:
                        type_hints_visitor.visit(arg.annotation)
                type_hint_symbols = type_hints_visitor.used_symbols - self.used_symbols
            self.generic_visit(node)
            if exclude_signature_type_hints:
                # Remove type hints from the used symbols
                self.used_symbols -= type_hint_symbols

        def visit_Assign(self, node):
            # Add assigned variables to defined symbols
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.defined_symbols.add(target.id)
            self.generic_visit(node)

        def visit_Name(self, node):
            # Add all variable names to used symbols
            if isinstance(node.ctx, ast.Load):
                self.used_symbols.add(node.id)
            self.generic_visit(node)

        @property
        def undefined_symbols(self):
            return self.used_symbols - self.defined_symbols - get_builtin_type_names()

    # Create a visitor instance and visit the AST
    visitor = SymbolVisitor()
    visitor.visit(tree)

    return visitor.undefined_symbols


def get_builtin_type_names():
    """
    Get a list of built-in object type names in Python.

    Returns
    -------
    set
        A set of built-in object type names.
    """
    return set(name for name, obj in vars(builtins).items() if isinstance(obj, type))


def in_stdlib(obj: types.FunctionType | type) -> str | bool:
    """Check if a type is in the standard library and return the name of the module if
    so."""
    module = inspect.getmodule(obj)
    if module is None:
        return False
    if module.__name__.startswith("builtins"):
        return "builtins"
    if module.__name__ == "types" and obj.__name__ not in dir(types):
        return False
    toplevel = module.__name__.split(".")[0]
    if toplevel in STDLIB_MODULES:
        return toplevel
    return False


def _stdlib_modules() -> frozenset[str]:
    """List all standard library modules."""
    std_lib_modules = set(sys.builtin_module_names)
    std_lib_path = sysconfig.get_path("stdlib")
    std_lib_modules.update(m[1] for m in pkgutil.iter_modules([std_lib_path]))
    return frozenset(std_lib_modules)


STDLIB_MODULES: frozenset[str] = _stdlib_modules()

# Example usage:
# print(list_standard_library_modules())


def plot_workflow(
    workflow_task: "workflow.Task",
    out_dir: Path,
    plot_type: str = "simple",
    export: ty.Sequence[str] | None = None,
    name: str | None = None,
    lazy: ty.Sequence[str] | ty.Set[str] | None = None,
) -> Path | tuple[Path, list[Path]]:
    """creating a graph - dotfile and optionally exporting to other formats"""
    from pydra.engine.workflow import Workflow

    if inspect.isclass(workflow_task):
        workflow_task = workflow_task()

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    if lazy is None:
        lazy = [n for n, v in attrs_values(workflow_task).items() if v is attrs.NOTHING]

    # Construct the workflow object with all of the fields lazy
    wf = Workflow.construct(workflow_task, lazy=lazy)

    if not name:
        name = f"graph_{type(workflow_task).__name__}"
    if plot_type == "simple":
        graph = wf.graph()
        dotfile = graph.create_dotfile_simple(outdir=out_dir, name=name)
    elif plot_type == "nested":
        graph = wf.graph()
        dotfile = graph.create_dotfile_nested(outdir=out_dir, name=name)
    elif plot_type == "detailed":
        graph = wf.graph(detailed=True)
        dotfile = graph.create_dotfile_detailed(outdir=out_dir, name=name)
    else:
        raise Exception(
            f"type of the graph can be simple, detailed or nested, "
            f"but {plot_type} provided"
        )
    if not export:
        return dotfile
    else:
        if export is True:
            export = ["png"]
        elif isinstance(export, str):
            export = [export]
        formatted_dot = []
        for ext in export:
            formatted_dot.append(graph.export_graph(dotfile=dotfile, ext=ext))
        return dotfile, formatted_dot


def show_workflow(
    workflow_task: "workflow.Task",
    plot_type: str = "simple",
    lazy: ty.Sequence[str] | ty.Set[str] | None = None,
    use_lib: str | None = None,
    figsize: tuple[int, int] | None = None,
    **kwargs,
) -> None:
    """creating a graph and showing it"""
    out_dir = Path(tempfile.mkdtemp())
    png_graph = plot_workflow(
        workflow_task,
        out_dir=out_dir,
        plot_type=plot_type,
        export="png",
        lazy=lazy,
    )[1][0]

    if use_lib in ("matplotlib", None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.image import imread
        except ImportError:
            if use_lib == "matplotlib":
                raise ImportError(
                    "Please install either matplotlib to display the workflow image."
                )
        else:
            use_lib = "matplotlib"

            # Read the image
            img = imread(png_graph)

            if figsize is not None:
                plt.figure(figsize=figsize)
            # Display the image
            plt.imshow(img, **kwargs)
            plt.axis("off")
            plt.show()

    if use_lib in ("PIL", None):
        try:
            from PIL import Image
        except ImportError:
            msg = " or matplotlib" if use_lib is None else ""
            raise ImportError(
                f"Please install either Pillow{msg} to display the workflow image."
            )
        # Open the PNG image
        img = Image.open(png_graph)

        # Display the image
        img.show(**kwargs)


def attrs_fields(task, exclude_names=()) -> list[attrs.Attribute]:
    """Get the fields of a task, excluding some names."""
    return [field for field in task.__attrs_attrs__ if field.name not in exclude_names]


def attrs_values(obj, **kwargs) -> dict[str, ty.Any]:
    """Get the values of an attrs object."""
    return {
        n: v
        for n, v in attrs.asdict(obj, recurse=False, **kwargs).items()
        if not n.startswith("_")
    }


class _TaskFieldsList(dict[str, "Field"]):
    """A list of task fields. Acts like list in that you can iterate over the values
    but also access them like a dict or by attribute."""

    def __iter__(self):
        return iter(self.values())

    def __getattr__(self, name):
        return self[name]

    def __dir__(self):
        return sorted(self.keys())


def get_fields(task: "type[Task] | Task") -> _TaskFieldsList:
    """List the fields of a task"""
    if not inspect.isclass(task):
        task = type(task)
    if not attrs.has(task):
        return _TaskFieldsList()
    return _TaskFieldsList(
        **{
            f.name: f.metadata[PYDRA_ATTR_METADATA]
            for f in attrs.fields(task)
            if PYDRA_ATTR_METADATA in f.metadata
        }
    )


def asdict(obj, **kwargs) -> dict[str, ty.Any]:
    """Get the values of an attrs object."""
    return {f.name: getattr(obj, f.name) for f in get_fields(obj)}


def from_list_if_single(obj: ty.Any) -> ty.Any:
    """Converts a list to a single item if it is of length == 1"""
    from pydra.utils.typing import is_lazy

    if obj is attrs.NOTHING:
        return obj
    if is_lazy(obj):
        return obj
    if isinstance(obj, ty.Sequence) and not isinstance(obj, str):
        obj = list(obj)
        if len(obj) == 1:
            return obj[0]
    return obj


def wrap_text(text: str, width: int = 79, indent_size=4) -> str:
    """Wraps text to a given width, respecting word boundaries."""
    indent = " " * indent_size
    if len(text) <= width:
        return indent + text
    lines = []
    for line in text.splitlines():
        if len(line) > width:
            words = line.split()
            split_line = indent
            for word in words:
                if len(split_line) + len(word) + 1 > width:
                    lines.append(split_line.rstrip())
                    split_line = indent
                split_line += word + " "
            lines.append(split_line.rstrip())
        else:
            lines.append(indent + line)
    return "\n".join(lines).rstrip()


def task_help(
    task_type: "type[Task] | Task", line_width: int = 79, help_indent: int = 4
) -> list[str] | None:
    """Visit a job object and print its input/output interface."""
    import pydra.compose.base

    if isinstance(task_type, pydra.compose.base.Task):
        task_type = type(task_type)

    plugin_name = next(
        n
        for n, t in get_plugin_classes(pydra.compose, "Task").items()
        if issubclass(task_type, t)
    ).capitalize()

    header = f"Help for {plugin_name} task '{task_type.__name__}'"
    hyphen_line = "-" * len(header)
    lines = [hyphen_line, header, hyphen_line]
    inputs = get_fields(task_type)
    if inputs:
        lines.extend(["", "Inputs:"])
    if any(hasattr(i, "position") for i in inputs):
        inputs = sorted(inputs)
    for inpt in inputs:
        lines.extend(
            inpt.markdown_listing(
                line_width, help_indent=help_indent, as_input=True
            ).split("\n")
        )
    outputs = get_fields(task_type.Outputs)
    if outputs:
        lines.extend(["", "Outputs:"])
    for output in outputs:
        lines.extend(
            output.markdown_listing(line_width, help_indent=help_indent).split("\n")
        )
    lines.append("")
    return lines


def print_help(task: "Task[TaskType]") -> None:
    """Print help for a task."""
    lines = task_help(task)
    print("\n".join(lines))


def position_sort(args):
    """
    Sort objects by position, following Python indexing conventions.

    Ordering is positive positions, lowest to highest, followed by unspecified
    positions (``None``) and negative positions, lowest to highest.

    >>> position_sort([(None, "d"), (-3, "e"), (2, "b"), (-2, "f"), (5, "c"), (1, "a")])
    ['a', 'b', 'c', 'd', 'e', 'f']

    Parameters
    ----------
    args : list of (int/None, object) tuples

    Returns
    -------
    list of objects
    """
    import bisect

    pos, none, neg = [], [], []
    for entry in args:
        position = entry[0]
        if position is None:
            # Take existing order
            none.append(entry[1])
        elif position < 0:
            # Sort negatives while collecting
            bisect.insort(neg, entry)
        else:
            # Sort positives while collecting
            bisect.insort(pos, entry)

    return [arg for _, arg in pos] + none + [arg for _, arg in neg]


def ensure_list(obj, tuple2list=False):
    """
    Return a list whatever the input object is.

    Examples
    --------
    >>> ensure_list(list("abc"))
    ['a', 'b', 'c']
    >>> ensure_list("abc")
    ['abc']
    >>> ensure_list(tuple("abc"))
    [('a', 'b', 'c')]
    >>> ensure_list(tuple("abc"), tuple2list=True)
    ['a', 'b', 'c']
    >>> ensure_list(None)
    []
    >>> ensure_list(5.0)
    [5.0]

    """
    from pydra.utils.typing import is_lazy

    if obj is attrs.NOTHING:
        return attrs.NOTHING
    if obj is None:
        return []
    # list or numpy.array (this might need some extra flag in case an array has to be converted)
    elif isinstance(obj, list) or hasattr(obj, "__array__"):
        return obj
    elif tuple2list and isinstance(obj, tuple):
        return list(obj)
    elif is_lazy(obj):
        return obj
    # elif is_container(obj):
    #     raise NotImplementedError("just checking for now")
    return [obj]


def ensure_file_list(filename):
    """Return a list given either a string or a list."""
    if isinstance(filename, (str, bytes)):
        return [filename]
    elif isinstance(filename, list):
        return filename
    elif is_container(filename):
        return [x for x in filename]

    return None


# dj: copied from misc
def is_container(item):
    """
    Check if item is a container (list, tuple, dict, set).

    Parameters
    ----------
    item : :obj:`object`
        Input object to check.

    Returns
    -------
    output : :obj:`bool`
        ``True`` if container ``False`` otherwise.

    """
    if isinstance(item, str):
        return False
    elif hasattr(item, "__iter__"):
        return True

    return False


def is_workflow(obj):
    """Check whether an object is a :class:`Workflow` instance."""
    from pydra.compose.workflow import Task
    from pydra.engine.workflow import Workflow

    return isinstance(obj, (Task, Workflow))


def get_plugin_classes(namespace: types.ModuleType, class_name: str) -> dict[str, type]:
    """
    Get all classes within sub-packages of namespace package with a given name, e.g.
    "Worker" within "pydra.workers.*" sub-packages.

    Parameters
    ----------
    namespace : :obj:`str`
        The namespace to search for subclasses.
    base_class : :obj:`type`
        The base class to search for subclasses of.

    Returns
    -------
    :obj:`dict[str, type]`
        A dictionary mapping the sub-package name to classes of 'class_name' within
        the namespace package
    """
    sub_packages = [
        importlib.import_module(f"{namespace.__name__}.{m.name}")
        for m in pkgutil.iter_modules(namespace.__path__)
        if not m.name.startswith("base")
    ]
    return {
        pkg.__name__.split(".")[-1]: getattr(pkg, class_name)
        for pkg in sub_packages
        if hasattr(pkg, class_name)
    }


def unstructure(
    task_class: "type[Task]",
    filter: ty.Callable[[attrs.Attribute, ty.Any], bool] | None = None,
    value_serializer: (
        ty.Callable[[ty.Any, attrs.Attribute, ty.Any], ty.Any] | None
    ) = None,
    **kwargs: ty.Any,
) -> ty.Dict[str, ty.Any]:
    """Converts a Pydra task class into a dictionary representation that can be serialized
    (e.g. to a file), then reread and passed to an appropriate `pydra.compose.*.define`
    method to recreate the task.

    Parameters
    ----------
    task_class : type[pydra.compose.base.Task]
        The Pydra task class to convert.
    filter : callable, optional
        A function to filter out certain attributes from the task definition passed
        through to `attrs.asdict`. It should take an attribute and its value as
        arguments and return a boolean (True to keep the attribute, False to filter it out).
    value_serializer : callable, optional
        A function to serialize the value of an attribute. It should take the task
        definition, the attribute, and its value as arguments and return a serialized
        value.
    **kwargs : dict
        Additional keyword arguments to pass to `attrs.asdict` (i.e. all except `filter`,
        and `value_serializer`), e.g. `recurse`. See the `attrs` documentation for more
        details.

    Returns
    -------
    dict[str, ty.Any]
        A dictionary representation of the Pydra task.
    """
    from pydra.compose.base import Out

    if filter is None:
        filter = filter_out_defaults

    def full_val_serializer(
        obj: ty.Any, field: attrs.Attribute, value: ty.Any
    ) -> ty.Any:
        """A wrapper for the value serializer to handle the case where it is None."""
        if value_serializer is not None:
            value = value_serializer(obj, field, value)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and not isinstance(value, dict):
            # If the value is a mapping, convert it to a dict
            value = dict(value)
        elif isinstance(value, Collection) and not isinstance(value, list):
            # If the value is not a collection or string, convert it to a list
            value = list(value)
        return value

    input_fields = get_fields(task_class)
    executor = input_fields.pop(task_class._executor_name).default
    input_dicts = [
        attrs.asdict(i, filter=filter, value_serializer=full_val_serializer, **kwargs)
        for i in input_fields
        if (
            not isinstance(i, Out)  # filter out outarg fields
            and i.name not in task_class.BASE_ATTRS
        )
    ]
    output_dicts = [
        attrs.asdict(o, filter=filter, value_serializer=full_val_serializer, **kwargs)
        for o in get_fields(task_class.Outputs)
        if o.name not in task_class.Outputs.BASE_ATTRS
    ]
    dct = {
        "type": task_class._task_type(),
        task_class._executor_name: executor,
        "name": task_class.__name__,
        "inputs": {d.pop("name"): d for d in input_dicts},
        "outputs": {d.pop("name"): d for d in output_dicts},
    }
    class_attrs = {a: getattr(task_class, "_" + a) for a in task_class.TASK_CLASS_ATTRS}
    # Convert the frozensets to lists for serialization
    class_attrs["xor"] = [list(x) for x in class_attrs.pop("xor")]
    if value_serializer:
        # We need to create a mock attrs object for the class attrs to apply the
        # value_serializer to it
        mock_cls = _make_attrs_class(
            {a: task_class.__annotations__.get(a, ty.Any) for a in class_attrs}
        )
        attrs_fields = {f.name: f for f in attrs.fields(mock_cls)}
        mock = mock_cls(**class_attrs)
        class_attrs = {
            n: full_val_serializer(mock, attrs_fields[n], v)
            for n, v in class_attrs.items()
        }
    dct.update(class_attrs)

    return dct


def structure(task_class_dict: dict[str, ty.Any]) -> type["Task"]:
    """Converts a dictionary representation created by `unstructure` into a Pydra task
    class.

    Parameters
    ----------
    task_class_dict: dict[str, Any]
        the dictionary representation to unserialize

    Returns
    -------
    type[pydra.compose.base.Task]
        the unserialized task class
    """
    dct = copy(task_class_dict)
    task_type = dct.pop("type")
    mod = importlib.import_module(f"pydra.compose.{task_type}")
    return mod.define(dct.pop(mod.Task._executor_name), **dct)


def filter_out_defaults(atr: attrs.Attribute, value: ty.Any) -> bool:
    """Filter out values that match the attributes default value."""
    if isinstance(atr.default, attrs.Factory) and atr.default.factory() == value:
        return False
    if value == atr.default:
        return False
    return True


def _make_attrs_class(field_types: dict[str, type]) -> type:
    """Creates an attrs given the a dictionary of field names and their types.

    Parameters
    ----------
    field_types : dict[str, type]
        A dictionary mapping field names to their types.

    Returns
    -------
    type
        the attrs class.
    """
    return attrs.define(
        type(
            "MockAttrsClass",
            (),
            {n: attrs.field(type=t) for n, t in field_types.items()},
        )
    )
