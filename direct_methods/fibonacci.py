from math import log, exp, cos


def func(x):
    """
    Function used in the first example of Fibonacci Search Method in the
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


def fibonacci_greater_than(x):
    """
    Finds first fibonacci number greater than or equal to x. Notice that
    because all fibonacci numbers smaller than this will need to be used by
    the algorithm, so we return them in a list.

    :param x: Minimum value that our fibonacci number can take
    :type x: float
    :return: list of fibonacci numbers
    :rtype: int
    """
    fibonacci_numbers = []
    i = 0
    previous = 0
    current = 1
    while current < x:
        if i == 0 or i == 1:
            fibonacci_numbers.append(float(i))
        else:
            previous, current = current, previous + current
            fibonacci_numbers.append(float(current))
        i += 1
    return fibonacci_numbers


def fibonacci_search(f, a, b, epsilon):
    """
    Implements fibonacci search method for finding minimum of a unimodal and
    univariate function.

    :param f: function to be optimized.
    :type f: func
    :param a: left endpoint of the interval of optimization
    :type a: float
    :param b: right endpoint of the interval of optimization
    :type b: float
    :param epsilon: precision of the interval, stopping criteria
    :type epsilon: float
    :return: left endpoint, right endpoint of final interval
    :rtype: tuple
    """
    print("#"*40 + " Fibonacci Search Method " + "#"*40)
    # Transform inputs to float
    a, b, epsilon = float(a), float(b), float(epsilon)
    # Find F_N and N - 1 (number of function evaluations)
    fibo_numbers = fibonacci_greater_than(2 * (b - a) / epsilon)
    n_evaluations = len(fibo_numbers) - 3  # Used only for pretty logging
    print('-'*40 + ' Initial Settings ' + '-'*40)
    print("F_N = {}\nNumber of Function Evaluations: {}"
          .format(fibo_numbers[-1], n_evaluations))
    # Keep count of the iterations just for printing purposes
    i = 1
    while len(fibo_numbers) > 2:
        print("#"*40 + " Iteration {} ".format(i) + "#"*40)
        print("-"*40 + " Find Fibonacci Numbers " + "-"*40)
        # Store fibonacci numbers used to "bracket" the root.
        if i == 1:
            denom = fibo_numbers.pop()  # F_N-i+1 (i.e. F_N the first time)
            right = fibo_numbers.pop()  # F_N-i (i.e. F_{N-1} the first time)
            left = fibo_numbers.pop()  # F_N-i-1 (i.e. F_{N-2} the first time)
        else:
            denom = right
            right = left
            left = fibo_numbers.pop()
        print("F_(N-i+1) = {}".format(denom))
        print("F_(N-i) = {}".format(right))
        print("F_(N-i-1) = {}".format(left))
        if i == 1:
            print("-"*40 + " Initial brackets " + "-"*40)
            # find the bracket-ers (x_1 left and x_2 right)
            x_1 = a + (left / denom) * (b - a)
            x_2 = a + (right / denom) * (b - a)
            print("Initial x_1 = {:.4}".format(x_1))
            print("Initial x_2 = {:.4}".format(x_2))
            print("Initial Interval: [{}, {}]".format(a, b))
        # Decide which part of the interval to remove
        print("-"*40 + " Calculations " + "-"*40)
        if f(x_1) < f(x_2):
            b = x_2
            x_2 = x_1
            x_1 = a + (left / denom) * (b - a)
            print("f(x_1) < f(x_2)")
            print("a = {:.4} \t\t\t Unchanged".format(a))
            print("x_1 = {:.4} \t\t\t Calculated".format(x_1))
            print("x_2 = {:.4} \t\t\t Old x_1".format(x_2))
            print("b = {:.4} \t\t\t Old x_2".format(b))
        else:
            a = x_1
            x_1 = x_2
            x_2 = a + (right / denom) * (b - a)
            print("f(x_1) >= f(x_2)")
            print("a = {:.4} \t\t\t Old x_1".format(a))
            print("x_1 = {:.4} \t\t\t Old x_2".format(x_1))
            print("x_2 = {:.4} \t\t\t Calculated".format(x_2))
            print("b = {:.4} \t\t\t Unchanged".format(b))
        i += 1
    print("<"*40 + " Final Results " + ">"*40)
    print("[a, b] = [{:.4}, {:.4}]".format(a, b))
    return a, b


if __name__ == "__main__":
    # Example 3.1
    fibonacci_search(func, 1.5, 4.5, 2/7)
    # Example 3.2
    fibonacci_search(func2, 0, 1, 0.5)
