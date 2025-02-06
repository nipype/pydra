from pathlib import Path
import re
import ast
import inspect
import types
import sys
import platformdirs
import builtins
import pkgutil
from pydra.engine._version import __version__

user_cache_dir = Path(
    platformdirs.user_cache_dir(
        appname="pydra",
        appauthor="nipype",
        version=__version__,
    )
)

default_run_cache_dir = user_cache_dir / "run-cache"


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


def in_stdlib(obj: types.FunctionType | type) -> bool:
    """Check if a type is in the standard library."""
    module = inspect.getmodule(obj)
    if module is None:
        return False
    if module.__name__.startswith("builtins"):
        return True
    if module.__name__ == "types" and obj.__name__ not in dir(types):
        return False
    return module.__name__.split(".")[-1] in STDLIB_MODULES


def _stdlib_modules() -> frozenset[str]:
    """List all standard library modules."""
    std_lib_modules = set(sys.builtin_module_names)
    for _, modname, ispkg in pkgutil.iter_modules():
        if not ispkg:
            std_lib_modules.add(modname)
    return frozenset(std_lib_modules)


STDLIB_MODULES: frozenset[str] = _stdlib_modules()

# Example usage:
# print(list_standard_library_modules())
