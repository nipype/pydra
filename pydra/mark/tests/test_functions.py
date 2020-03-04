# -*- coding: utf-8 -*-

import pytest
import random

from ..functions import task, annotate
from ...engine.task import FunctionTask


def test_task_equivalence():
    def add_two(a):
        return a + 2

    canonical = FunctionTask(add_two, a=3)

    decorated1 = task(add_two)(a=3)

    @task
    def addtwo(a):
        return a + 2

    decorated2 = addtwo(a=3)

    assert canonical.checksum == decorated1.checksum

    c_res = canonical._run()
    d1_res = decorated1._run()
    d2_res = decorated2._run()

    assert c_res.output.hash == d1_res.output.hash
    assert c_res.output.hash == d2_res.output.hash


def test_annotation_equivalence():
    def direct(a: int) -> int:
        return a + 2

    @annotate({"return": int})
    def partial(a: int):
        return a + 2

    @annotate({"a": int, "return": int})
    def indirect(a):
        return a + 2

    assert direct.__annotations__ == partial.__annotations__
    assert direct.__annotations__ == indirect.__annotations__

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert direct(a) == partial(a)
    assert direct(a) == indirect(a)


def test_annotation_override():
    @annotate({"a": float, "return": float})
    def annotated(a: int) -> int:
        return a + 2

    assert annotated.__annotations__ == {"a": float, "return": float}


def test_invalid_annotation():
    with pytest.raises(TypeError):

        @annotate({"b": int})
        def addtwo(a):
            return a + 2


def test_annotated_task():
    @task
    def square(in_val: float):
        return in_val ** 2

    res = square(in_val=2.0)()
    assert res.output.out == 4.0


def test_return_annotated_task():
    @task
    @annotate({"in_val": float, "return": {"squared": float}})
    def square(in_val):
        return in_val ** 2

    res = square(in_val=2.0)()
    assert res.output.squared == 4.0


def test_return_halfannotated_annotated_task():
    @task
    @annotate({"in_val": float, "return": float})
    def square(in_val):
        return in_val ** 2

    res = square(in_val=2.0)()
    assert res.output.out == 4.0


def test_return_annotated_task_multiple_output():
    @task
    @annotate({"in_val": float, "return": {"squared": float, "cubed": float}})
    def square(in_val):
        return in_val ** 2, in_val ** 3

    res = square(in_val=2.0)()
    assert res.output.squared == 4.0
    assert res.output.cubed == 8.0


def test_return_halfannotated_task_multiple_output():
    @task
    @annotate({"in_val": float, "return": (float, float)})
    def square(in_val):
        return in_val ** 2, in_val ** 3

    res = square(in_val=2.0)()
    assert res.output.out1 == 4.0
    assert res.output.out2 == 8.0
