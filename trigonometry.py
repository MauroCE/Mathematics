import math
from utils import sign

UNICODE_PI = "\u03C0"


def full_arctan(y: float, x: float, image='complex') -> float:
    """
    This function implements my version of the atan2. To read its definition go
    to:
    https://en.wikipedia.org/wiki/Atan2#Definition
    With this function you can choose whether you want to output angles to be
    within the (-pi, pi) range or not. Notice that this (-pi, pi) range is
    used with complex numbers. Specifically, the principal argument of a
    complex number is defined to be within that range.

    :param y: Represents the length of the opposite side, or imaginary part of
              a complex number.
    :type y: float
    :param x: Represents the length of the adjacent side, or real part of a
              complex number.
    :type x: float
    :param image: What type of range we want our output angles to be in. If
                  'complex' then it is between (-pi, pi), if 'real' it is
                  between (0, 2pi). Notice that by default it is 'complex'
                  because this function is more commonly used to compute the
                  principal argument of a complex number.
    :type image: str
    :return: Counter-clockwise angle from x axis to y axis describing the
             point (x, y) or the principal argument for a complex number.
    :rtype: float

    """
    if x > 0:
        # We are in the range (-pi/2, pi/2) thus arctan is safe
        return math.atan(y / x)
    elif x < 0:
        # (-pi, pi] range, correct range for principal argument
        if image == 'complex':
            if y >= 0:
                # move angle from (-pi/2, 0] to (pi/2, pi]
                return math.atan(y / x) + math.pi
            else:
                # move angle from (0, pi/ 2) to (-pi, -pi/2)
                return math.atan(y / x) - math.pi
            # [0, 2pi) range
        elif image == 'real':
            # from (-pi/2, 0] to (pi/2, pi] or from (0, pi/ 2) to (pi, 3pi/2)
            return math.atan(y / x) + math.pi
        else:
            raise NotImplemented("Image must be either 'complex' or 'real'.")
    elif x == 0 and y != 0:
        # pi/2 for positive, -pi/2 for negative
        return sign(y) * (math.pi / 2)
    else:
        raise ValueError("Arctan not defined for x=0 and y=0.")


def pretty_angle(angle):
    """
    This function tries to change the rad value of an angle to one of the usual
    angles (e.g. 2pi/3)
    :param angle:
    :return:
    """
    # Loop through the first 20 numbers and try it out
    for num_factor in range(1, 20):
        for den_factor in range(1, 20):
            if round(num_factor * math.pi / den_factor, 9) == abs(round(angle, 9)):
                return "{nu}\u03C0/{de}".format(nu=num_factor*sign(angle), de=den_factor)
    return str(angle)
