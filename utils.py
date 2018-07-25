

def sign(value: float) -> int:
    """
    Finds the sign of a number.
    :param value: Value for which we want to find the sign of.
    :type value: float
    :return: sign (1, 0, -1)
    :rtype: int
    """
    if value == 0:
        return 0
    return 1 if value > 0 else -1