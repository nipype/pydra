"""Bounded exponential backoff: the one retry primitive."""

import random
import time
import typing as ty

import attrs


@attrs.frozen
class Backoff:
    """Waits of ``first``, doubled after each try up to ``cap``, ended by a count of tries,
    a deadline, or both.

    Parameters
    ----------
    first : float
        Seconds waited before the second try.
    cap : float
        Most seconds waited between two tries.
    attempts : int | None
        Most tries; None for no count.
    deadline : float | None
        Most seconds from the first try to the start of the last; None for no deadline.
        `tries` may override it.
    jitter : bool
        Wait a uniformly random part of each wait, which spreads contending writers.
    """

    first: float
    cap: float
    attempts: int | None = None
    deadline: float | None = None
    jitter: bool = False

    def __attrs_post_init__(self) -> None:
        if self.first < 0 or self.cap < self.first:
            raise ValueError(f"need 0 <= first <= cap (got {self.first}, {self.cap})")
        if self.attempts is not None and self.attempts < 1:
            raise ValueError(f"attempts must be at least 1 (got {self.attempts})")

    def tries(
        self,
        deadline: float | None = None,
        *,
        clock: ty.Callable[[], float] | None = None,
        sleep: ty.Callable[[float], None] | None = None,
        rng: ty.Callable[[], float] | None = None,
    ) -> "Tries":
        """Iterate over the tries, waiting before each but the first.

        ``deadline`` (seconds from now) replaces the configured one. ``clock``, ``sleep``
        and ``rng`` default to `time.monotonic`, `time.sleep` and `random.random`, read
        when the tries begin.
        """
        clock = clock or time.monotonic
        sleep = sleep or time.sleep
        rng = rng or random.random
        seconds = self.deadline if deadline is None else deadline
        return Tries(
            self, None if seconds is None else clock() + seconds, clock, sleep, rng
        )


class Tries:
    """The tries of one `Backoff`: iterating yields the try number (0 first) after the wait
    before it, and stops once no try is left; `last` says whether the current try is the last.
    """

    def __init__(self, backoff: Backoff, end: float | None, clock, sleep, rng) -> None:
        self._backoff, self._end = backoff, end
        self._clock, self._sleep, self._rng = clock, sleep, rng
        self._n = -1

    def __iter__(self) -> "Tries":
        return self

    def __next__(self) -> int:
        if self._n >= 0:
            if self.last:
                raise StopIteration
            wait = min(self._backoff.first * 2**self._n, self._backoff.cap)
            if self._backoff.jitter:
                wait *= self._rng()
            if self._end is not None:
                wait = min(wait, max(self._end - self._clock(), 0.0))
            self._sleep(wait)
        self._n += 1
        return self._n

    @property
    def last(self) -> bool:
        """Whether no try may follow the current one."""
        b = self._backoff
        if b.attempts is not None and self._n + 1 >= b.attempts:
            return True
        return self._end is not None and self._clock() >= self._end

    def wait(self) -> bool:
        """Wait before the next try; False (without waiting) when none is left."""
        try:
            next(self)
        except StopIteration:
            return False
        return True
