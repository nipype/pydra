"""Warm worker pools: worker classes, placement and fit, the pool executor and its substrates.

Nothing here is imported by ``import pydra``; import the submodules directly.
"""

import importlib
import typing as ty

_EXPORTS = {
    "WorkerClass": "classes",
    "parse_size": "classes",
    "Calibration": "calibration",
    "DirectoryCalibration": "calibration",
    "MemoryFit": "calibration",
    "ParityRecord": "calibration",
    "Sample": "calibration",
    "TimeRecord": "calibration",
    "UnfitRecord": "calibration",
    "MeasuredFit": "fit",
    "CanStart": "placement",
    "ClassTried": "placement",
    "DeclaredFit": "placement",
    "Fit": "placement",
    "FitRequest": "placement",
    "Placement": "placement",
    "Unplaceable": "placement",
    "Verdict": "placement",
    "fit_request": "placement",
    "place": "placement",
    "PoolBusy": "pool",
    "PoolFactsWouldSplitSlots": "pool",
    "PoolFenced": "pool",
    "PoolJobFailed": "pool",
    "PoolSettings": "settings",
    "UnplaceableJob": "pool",
    "WarmPool": "pool",
    "ExternalSubstrate": "substrate",
    "LocalSubstrate": "substrate",
    "NodeLocalStorage": "substrate",
    "Substrate": "substrate",
    "WorkerHandle": "substrate",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> ty.Any:
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
