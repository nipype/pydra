"""Utilities for `pydra.engine`."""
import functools
import importlib


def _check_dependencies(*pkgs):
    """Decorator to check that `*pkgs` are available."""

    def f(function):
        @functools.wraps(function)
        def g(*args, **kwargs):
            for p in pkgs:
                try:
                    importlib.import_module(p)
                except ModuleNotFoundError:
                    msg = "The Python package '{}' is required for this functionality but is not installed."
                    raise ImportError(msg.format(p))

            return function(*args, **kwargs)

        return g

    return f
