"""Routers: choose which of a submitter's workers runs a job."""

import typing as ty

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

DEFAULT_WORKER = "default"
"""Name under which a submitter's ``worker`` is always available to a router."""

Router = ty.Callable[["Job"], "str | None"]
"""A router is called with each job at dispatch and returns the name of a worker, or
``None`` for the submitter's ``worker``."""


def default_router(job: "Job") -> str:
    """Send every job to the submitter's ``worker`` (a single-worker run)."""
    return DEFAULT_WORKER


def tag_router(routes: ty.Mapping[str, str]) -> Router:
    """Build a router from a mapping of resource tag to worker name.

    A job goes to the worker of the first of its ``resources.tags`` found in ``routes``,
    and to the submitter's ``worker`` if it has no resources or none of its tags match.

    Parameters
    ----------
    routes : Mapping[str, str]
        Tag to the name of a worker in ``Submitter(workers=...)``
    """
    routes = dict(routes)

    def route(job: "Job") -> "str | None":
        for tag in job.resources.tags if job.resources is not None else ():
            if tag in routes:
                return routes[tag]
        return None

    return route
