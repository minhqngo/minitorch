"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return float(-x)


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    return (x > 0) * x


def tanh(x: float) -> float:  # tanh(x) = 2*sigmoid(2x) - 1
    return 2.0 * (1.0 / (1.0 + math.exp(-2.0 * x))) - 1.0


def tanh_back(x: float, d: float) -> float:  # d/dx tanh(x) = 1 - tanh^2(x)
    t = 2.0 * (1.0 / (1.0 + math.exp(-2.0 * x))) - 1.0
    return d * (1.0 - t * t)


EPS = 1e-6


def log(x: float) -> float:
    return math.log(x + EPS)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / x


def inv(x: float) -> float:
    return 1 / x


def inv_back(x: float, d: float) -> float:
    return -d / x ** 2


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.


def sqrt(x: float) -> float:
    return math.sqrt(x)


def sqrt_back(x: float, d: float) -> float:
    return d / (2 * math.sqrt(x))


def pow(x: float, p: float) -> float:
    return x ** p


def pow_back(x: float, p: float, d: float) -> float:
    return d * p * (x ** (p - 1))


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    return lambda ls: [fn(x) for x in ls]


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    return lambda xls, yls: [fn(x, y) for x, y in zip(xls, yls)]


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    def f(ls: Iterable[float]) -> float:
        v = start
        for x in ls:
            v = fn(v, x)
        return v
    
    return f


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.)(ls)


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.)(ls)
