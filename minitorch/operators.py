"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable
# ## Task 0.1

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(a: float, b: float) -> float:
    """Returns product of two floats."""
    return a * b


# - id
def id(a: float) -> float:
    """Returns input."""
    return a


# - add
def add(a: float, b: float) -> float:
    """Returns sum of two floats."""
    return a + b


# - neg
def neg(a: float) -> float:
    """Returns negative of a float."""
    return -a


# - lt
def lt(a: float, b: float) -> bool:
    """Returns true if first arg is less than second."""
    return a < b


# - eq
def eq(a: float, b: float) -> bool:
    """Returns True if args are equal."""
    return a == b


# - max
def max(a: float, b: float) -> float:
    """Returns the maximum of two floats."""
    return a if a > b else b


# - is_close
def is_close(a: float, b: float) -> bool:
    """Returns true if absolute diff of args is < 1e-2."""
    result = max(a - b, b - a) < 1e-2
    return result


# - sigmoid
def sigmoid(a: float) -> float:
    """Returns sigmoid of input."""
    return 1 / (1 + math.exp(-a)) if a >= 0 else math.exp(a) / (1 + math.exp(a))


# - relu
def relu(a: float) -> float:
    """Returns relu of input, i.e. returns 0 if function is less than 0, else input."""
    return max(0.0, a)


# - log
def log(a: float) -> float:
    """Returns logarithm of a float."""
    return math.log(a)


# - exp
def exp(a: float) -> float:
    """Returns exponential of a float."""
    return math.exp(a)


# - log_back
def log_back(input_val: float, back_grad: float) -> float:
    """Backward pass calculation for logarithm."""
    return back_grad / input_val


# - inv
def inv(a: float) -> float:
    """Returns one over a float."""
    return 1.0 / a


# - inv_back
def inv_back(input_val: float, back_grad: float) -> float:
    """Backward pass calculation for one over a float."""
    return -back_grad / input_val**2


# - relu_back
def relu_back(input_val: float, back_grad: float) -> float:
    """Backward pass calculation for relu."""
    return back_grad if input_val >= 0 else 0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
# - zipWith
# - reduce
def map(fn: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    """Uses the provided function and iteratively applies to to the iterable.

    Args:
    ----
        fn (Callable[[float], float]): function to apply to each element
        iterable (Iterable): the input iterable to process

    Returns:
    -------
        Iterable: new iterable with the function applied to each element.

    """
    result = []
    for item in iterable:
        result.append(fn(item))
    return result


def zipWith(
    iter1: Iterable[float], iter2: Iterable[float], fn: Callable[[float, float], float]
) -> Iterable[float]:
    """Takes two iterables and applies a function to the zip of the two iterables.

    Args:
    ----
        iter1 (Iterable): first iterable to be zipped
        iter2 (Iterable): second iterable to be zipped
        fn (Callable[[float, float], float]): function to be applied to zipped iterables

    Returns:
    -------
        Iterable: result of function applied to zipped iterables

    """
    result = []
    for item1, item2 in zip(iter1, iter2):
        result.append(fn(item1, item2))
    return result


def reduce(
    iterable: Iterable[float], fn: Callable[[float, float], float], default: float
) -> float:
    """Iteratively applies a function to each element of an iterable, starting from an initial value.

    Args:
    ----
        iterable (Iterable): iterable we want to reduce
        fn (Callable[[float, float], float]): function of two arguments which we'll apply to each element
        default (float): initial value used to start the reduction process (e.g. 1 for prod)

    Returns:
    -------
        float: result of reduction

    """
    result = default
    if next(iter(iterable), None) is None:
        # not all iterables have a defined next() function, although python does automatically call iter() at runtime
        return result
    for item in iterable:
        result = fn(item, result)
    return result


#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def negList(l: list[float]) -> list[float]:
    """Negates every element of a list.

    Args:
    ----
        l (list[float]): List to be negated.

    Returns:
    -------
        list[float]: Negated list.

    """
    return list(map(neg, l))


def addLists(l1: list, l2: list) -> list:
    """Takes two lists and adds element-wise.

    Args:
    ----
        l1 (list): first list
        l2 (list): second list

    Returns:
    -------
        list: list where each element is element-wise sum of l1 and l2

    """
    return list(zipWith(l1, l2, add))


def sum(l: list[float]) -> float:
    """Sums all elements in a list.

    Args:
    ----
        l (list[float]): list to be summed

    Returns:
    -------
        float: number resulting from summing

    """
    return reduce(l, add, default=0.0)


def prod(l: list[float]) -> float:
    """Product of all elements in a list.

    Args:
    ----
        l (list[float]): list to be producted

    Returns:
    -------
        float: result of product of all elements in the list

    """
    return reduce(l, mul, default=1.0)
