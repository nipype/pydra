"""Warm worker pools: worker classes, placement and fit.

Nothing here is imported by ``import pydra``; import the submodules directly.
"""

import importlib
import typing as ty

_EXPORTS = {
    "WorkerClass": "classes",
    "parse_size": "classes",
    "CanStart": "placement",
    "ClassTried": "placement",
    "DeclaredFit": "placement",
    "Fit": "placement",
    "FitRequest": "placement",
    "Placement": "placement",
    "Unplaceable": "placement",
    "Verdict": "placement",
    "place": "placement",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> ty.Any:
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
