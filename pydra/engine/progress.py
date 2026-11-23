"""A count of the changes to a runner's state, so that `GraphRunner.ready
<pydra.engine.runner.GraphRunner.ready>` tells whether a pass advanced anything without
reading the state again. Changes are counted per thread: a runner is driven from one."""

import threading
from collections import defaultdict


class _Count(threading.local):
    changes = 0


_local = _Count()


def changed() -> None:
    """Count one change of the state a pass of `GraphRunner.ready` reads, in this thread"""
    _local.changes += 1


def changes() -> int:
    """The changes `changed` has counted in this thread"""
    return _local.changes


class _Tracking:
    """A mapping that counts each change of its items (`changed`)"""

    def __setitem__(self, key, value):
        if key not in self or self[key] is not value:
            changed()
        super().__setitem__(key, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        changed()

    def pop(self, key, *default):
        if key in self:
            changed()
        return super().pop(key, *default)

    def popitem(self):
        item = super().popitem()
        changed()
        return item

    def clear(self):
        if self:
            changed()
        super().clear()

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


class TrackedDict(_Tracking, dict):
    """A dict that counts its changes (`changed`)"""


class TrackedDefaultDict(_Tracking, defaultdict):
    """A defaultdict that counts its changes (`changed`)"""


class TrackedSet(set):
    """A set that counts its changes (`changed`)"""

    def add(self, item):
        if item not in self:
            changed()
            super().add(item)

    def discard(self, item):
        if item in self:
            changed()
            super().discard(item)

    def remove(self, item):
        super().remove(item)
        changed()

    def pop(self):
        item = super().pop()
        changed()
        return item

    def clear(self):
        if self:
            changed()
        super().clear()

    def update(self, *iterables):
        for items in iterables:
            for item in items:
                self.add(item)


def tracked(value):
    """``value`` as a container that counts its changes, if it is a plain dict"""
    if type(value) is dict:
        return TrackedDict(value)
    if type(value) is defaultdict:
        return TrackedDefaultDict(value.default_factory, value)
    return value


UNSET = object()
