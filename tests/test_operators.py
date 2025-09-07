from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    sum,
)

from .strategies import assert_close, small_floats


@pytest.mark.basic_math_ops
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    "Check that the main operators all return the same value of the python version"
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.basic_math_ops
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.basic_math_ops
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.basic_math_ops
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.basic_math_ops
@given(small_floats)
def test_lt(a: float) -> None:
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.basic_math_ops
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.basic_math_ops
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


@pytest.mark.basic_math_properties
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as sigmoid of the negative
    * It crosses 0 at 0.5
    * It is  strictly increasing.
    """
    assert 0. < sigmoid(a) <= 1.
    assert_close(1 - sigmoid(a), sigmoid(-a))
    assert_close(sigmoid(0), 0.5)
    assert sigmoid(a) <= sigmoid(a + 1)


@pytest.mark.basic_math_properties
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    if lt(a, b) and lt(b, c):
        assert lt(a, c)


@pytest.mark.basic_math_properties
@given(small_floats, small_floats)
def test_symmetric(a: float, b: float) -> None:
    assert_close(mul(a, b) , mul(b, a))


@pytest.mark.basic_math_properties
@given(small_floats, small_floats, small_floats)
def test_distribute(a: float, b: float, c: float) -> None:
    l = mul(c, add(a, b))
    r = add(mul(c, a), mul(c, b))
    assert_close(l, r)


@pytest.mark.basic_math_properties
@given(small_floats, small_floats, small_floats)
def test_other(a: float, b: float, c: float) -> None:
    l = add(add(a, b), c)
    r = add(a, add(b, c))
    assert_close(l, r)


@pytest.mark.high_order_fn
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.high_order_fn
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    a = sum(ls1) + sum(ls2)
    b = sum(addLists(ls1, ls2))
    assert_close(a, b)


@pytest.mark.high_order_fn
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), sum(ls))


@pytest.mark.high_order_fn
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.high_order_fn
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
