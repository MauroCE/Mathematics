from utils import sign


def bisection_method(f, x_lower, x_upper, eps, max_iter=1000):
    """
    This function tries to implement the bisection method for finding the root
    of a real-valued function. Because I cannot evaluate derivatives with
    high accuracy without external libraries, for now I will make the function
    and its derivative parameters of my algorithm.

    :param f: Function for which we want to find the root.
    :type f: func
    :param x_lower: lower bound of the interval that we are considering.
    :type x_lower: float
    :param x_upper: upper bound of the interval that we are considering.
    :type x_upper: float
    :return:
    """
    if x_upper < x_lower:
        raise ValueError(
            "Search interval must not be empty."
        )
    if f(x_lower) * f(x_upper) > 0:
        raise ValueError(
            "Function values must be of opposite signs."
        )
    iteration = 0
    while iteration <= max_iter:
        # Find midpoint and check if we need to continue
        x_midpoint = (x_lower + x_upper) / 2
        if f(x_midpoint) == 0 or x_upper - x_lower < eps:
            return x_midpoint
        # Keep the point with opposite sign
        if sign(f(x_lower)) != sign(f(x_midpoint)):
            x_upper = x_midpoint
        else:
            x_lower = x_midpoint
    return x_midpoint
