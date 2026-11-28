import pytest

from pydra.utils.retry import Backoff


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def run(backoff, deadline=None, rng=lambda: 1.0, fail=lambda n: True):
    clock = FakeClock()
    tries = backoff.tries(deadline, clock=clock, sleep=clock.sleep, rng=rng)
    seen = []
    for n in tries:
        seen.append((n, tries.last))
        if not fail(n):
            break
    return seen, clock.waits


def test_the_clock_and_sleep_are_read_when_the_tries_begin(monkeypatch):
    """A clock and sleep patched into `time` after import are the ones the tries use"""
    clock = FakeClock()
    monkeypatch.setattr("time.monotonic", clock)
    monkeypatch.setattr("time.sleep", clock.sleep)
    tries = Backoff(first=1, cap=1, deadline=2.5).tries()
    assert list(tries) == [0, 1, 2, 3]
    assert clock.waits == [1, 1, 0.5]


def test_attempts_double_up_to_the_cap():
    seen, waits = run(Backoff(first=0.1, cap=0.3, attempts=5))
    assert [n for n, _ in seen] == [0, 1, 2, 3, 4]
    assert [last for _, last in seen] == [False] * 4 + [True]
    assert waits == pytest.approx([0.1, 0.2, 0.3, 0.3])


def test_a_success_ends_the_tries():
    seen, waits = run(Backoff(first=1, cap=1, attempts=5), fail=lambda n: n < 1)
    assert [n for n, _ in seen] == [0, 1]
    assert waits == [1]


def test_the_deadline_ends_the_tries_and_clips_the_last_wait():
    seen, waits = run(Backoff(first=1, cap=4, deadline=6))
    assert waits == [1, 2, 3]
    assert seen[-1] == (3, True)


def test_a_deadline_of_zero_gives_one_try():
    seen, waits = run(Backoff(first=1, cap=1, deadline=60), deadline=0)
    assert seen == [(0, True)]
    assert waits == []


def test_jitter_waits_a_random_part():
    _, waits = run(Backoff(first=1, cap=8, attempts=3, jitter=True), rng=lambda: 0.25)
    assert waits == [0.25, 0.5]


def test_wait_says_whether_a_try_is_left():
    clock = FakeClock()
    tries = Backoff(first=1, cap=1, attempts=2).tries(clock=clock, sleep=clock.sleep)
    next(tries)
    assert tries.wait() is True
    assert tries.wait() is False
    assert clock.waits == [1]


@pytest.mark.parametrize(
    "kwargs",
    [dict(first=-1, cap=1), dict(first=2, cap=1), dict(first=0, cap=0, attempts=0)],
)
def test_bad_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        Backoff(**kwargs)


def test_next_wait_leaves_the_sleep_to_the_caller():
    clock = FakeClock()
    tries = Backoff(first=1, cap=2, attempts=3).tries(clock=clock, sleep=clock.sleep)
    assert [tries.next_wait(), tries.next_wait(), tries.next_wait()] == [1, 2, None]
    assert clock.waits == []
