from math import log, exp, cos


def func(x):
    """
    Function used in the first example of Golden Section Method in the
    lecture notes.

    :param x: Function input
    :type x: float
    :return: function value
    :rtype: float
    """
    x = float(x)
    # By default math.log uses math.e as base. This is natural log
    return - (1 / (x - 1)**2) * (log(x) - 2 * (x - 1) / (x + 1))


def func2(x):
    """
    Function used in the second example of Golden Section Method in the
    lecture notes.

    :param x: Function input
    :type x: float
    :return: function value
    :rtype: float
    """
    x = float(x)
    return exp(-x) - cos(x)


def golden_section(f, a, b, epsilon, t=0.618):
    """
    Golden Section method for constrained optimization of a univariate,
    unimodal function. f: R --> R

    :param f: function to be optimized.
    :type f: func
    :param a: left endpoint of the interval in which we are optimizing.
    :type a: float
    :param b: right endpoint of the interval in which we are optimizing.
    :type b: float
    :param epsilon: precision used as stopping criteria
    :type epsilon: float
    :param t: estimate of the golden ratio
    :type t: float
    :return: final left endpoint, final right endpoint of the interval
    :rtype: tuple
    """
    # Make sure everything is a float
    a = float(a)
    b = float(b)
    print("#"*40 + " Golden Section Method " + 40*"#")
    print("Initial Interval: [{}, {}]".format(a, b))
    # Calculate golden section point
    golden = a + t * (b - a)
    # Calculate reflection point
    reflection = b - t * (b - a)
    print("x_2 = {}".format(golden))
    print("x_1 = {}".format(reflection))
    # Count the number of iterations to print a pretty log
    i = 1
    # Stopping criteria is the length of the interval
    while abs(b - a) > epsilon:
        print("-"*40 + " Iteration {} ".format(i) + 40*"-")
        # Drop the part of the interval that has higher values (f unimodal!)
        if f(golden) > f(reflection):
            # Shift and contract to the left
            b = golden
            golden = reflection
            reflection = b - t * (b - a)
            print("f(x-2) > f(x_1)")
            print("a = {:.4}  \t\t\t Unchanged".format(a))
            print("x_1 = {:.4}\t\t\t Newly Calculated".format(reflection))
            print("x_2 = {:.4}\t\t\t Updated to old x_1".format(golden))
            print("b = {:.4}  \t\t\t Updated to old x_2".format(b))
        else:
            # Shift and contract to the right
            a = reflection
            reflection = golden
            golden = a + t * (b - a)
            print("f(x-2) <= f(x_1)")
            print("a = {:.4}  \t\t\t Updated to old x_1".format(a))
            print("x_1 = {:.4}\t\t\t Updated to old x_2".format(reflection))
            print("x_2 = {:.4}\t\t\t Newly Calculated".format(golden))
            print("b = {:.4}  \t\t\t Unchanged".format(b))
        i += 1
    print("<"*40 + " Final Results " + ">"*40)
    print("[a, b] = [{:.4}, {:.4}]".format(a, b))
    return a, b


if __name__ == "__main__":
    # Example 2.1
    golden_section(func, 1.5, 4.5, 0.2)
    # Example 2.2
    golden_section(func2, 0.0, 1.0, 0.5)
