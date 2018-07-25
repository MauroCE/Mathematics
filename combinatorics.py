

def combinations(n: int, k: int) -> int:
    """
    Selection of items from a collection, such that the order of the selection
    does not matter. If the order does not matter, in general it means that
    there are fewer selections that when the order matters. Note that in
    general recursion is quite slow in Python, therefore recursive version of
    combinations will be much slower. To dig deeper, have a look here:
    https://www.quora.com/In-Python-which-is-more-efficient-to-use-
    recursion-or-iteration

    :param n: Size of the collection from which we are picking a selection from
    :type n: int
    :param k: Size of the selection we are drawing from the whole collection.
    :type k: int
    :return: Combinations of k from n, nCk
    :rtype: int
    """
    # Collection >= Selection
    if n < k:
        raise ValueError(
            "The size of the collection we are selecting items from must be "
            "larger than the size of the selection."
        )
    # Sizes > 0
    if n < 0 or k < 0:
        raise ValueError(
            "Cannot work with negative integers."
        )
    # Compute with standard python only
    numerator = 1
    for i in range(n + 1 - k, n+1):
        numerator *= i
    denominator = 1
    for i in range(1, k+1):
        denominator *= i
    return int(numerator / denominator)


def combinations2(n: int, k: int) -> int:
    """
    This function is similar to combinations, but uses the below-defined
    factorial function.

    :param n: Size of collections we are selecting items from.
    :type n: int
    :param k: Size of selection we are picking from the collection.
    :type k: int
    :return: Number of selections of size k from collection of size n
    :type: int
    """
    # Collection >= Selection
    if n < k:
        raise ValueError(
            "The size of the collection we are selecting items from must be "
            "larger than the size of the selection."
        )
    return int(factorial(n, n - k) / factorial(k))


def factorial(n: int, lower: int=-1) -> int:
    """
    This function calculates the factorial of a non-negative integer. The
    function is defined with a lower bound because this will make combinations2
    a bit faster.

    :param n: Number for which we want to calculate the factorial. If n is
              provided, n! will be calculated.
    :type n: int
    :param lower: Lower bound of the factorial. This is used to get a partial
                  factorial. The lower bound represents the last number < n
                  that is not used in the computation. For instance
                  factorial(5, 3) = 5 * 4 because 3 is the last (largest)
                  number that is not used in the calculation. This means that
                  factorial(5, 3) = factorial(5) / factorial(3)
    :type lower: int
    :return: Factorial of n, n!
    :rtype: int
    """
    # n > 0
    if n < 0:
        raise ValueError(
            "Cannot calculate factorial of a negative number."
        )
    # Recursive function up to n = 0 or up to lower bound
    if n - 1 >= 0 and n - 1 >= lower:
        return n * factorial(n - 1, lower)
    return 1


if __name__ == "__main__":
    # Try -1, will be undefined.
    print("5! = ", factorial(5))
    print("5! / 3! = ", factorial(5, 3))
    print("100C50 = ", combinations(100, 50))

