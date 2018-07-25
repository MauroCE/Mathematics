import numpy as np
import math


class Polynomial:

    def __init__(self, *args: tuple):
        """
        Args are the coefficients. The top left-most is the largest degree.
        The right-most is the constant.

        :param args: Coefficients of the polynomial. Each coefficient should
                     ideally be an int.
        :type args: tuple
        :return: Nothing to return
        :rtype: None

        """
        # Coefficients
        self.coeff = np.array(args)
        # Degree
        self.n = len(self.coeff)
        # Exponents
        self.powers = np.array([i for i in range(self.n - 1, -1, -1)])

    def __repr__(self):
        """
        Override the print output to make it easier to work with.

        :return: String representation to be printed to screen.
        :rtype: str

        """
        exp = ""
        try:
            # Add all but the constant
            exp += " + ".join("{coeff}x^{power}".format(coeff=c, power=p)
                              for c, p in zip(self.coeff[:-1],
                                              self.powers[:-1]))
            # Add the constant
            exp = exp + " + " if len(self.coeff) > 1 else ""
            exp += "{constant}".format(constant=self.coeff[-1])
            return exp
        except IndexError:
            # Means that it was just a constant. Derivative of a constant is 0
            return "0"

    def __call__(self, x: float) -> float:
        """
        Evaluates the value of the polynomial

        :param x: x value to be evaluated for the polynomial.
        :type x: float
        :return: result of the polynomial evaluation, i.e. f(x)
        :rtype: float

        """
        powered_x = np.array([math.pow(x, i) for i in self.powers])
        return (powered_x * self.coeff).sum()

    def differentiate(self, order: int=1) -> 'Polynomial':
        """
        This method differentiates the polynomial. Notice that the type hinting
        follows PEP-0484, to read more, go to:
        https://www.python.org/dev/peps/pep-0484/#forward-references

        :param order: Order of the differentiation. For instance if we want to
                      find the n-th derivative of a polynomial, you would use
                      order=n.
        :type order: int
        :return: Instance of Polynomial class representing the derivative
        :rtype: Polynomial

        """
        poly_derivative = self
        i = 1
        powers = self.powers
        coeff = self.coeff
        while i <= order:
            # Find multipliers due to exponent coming down
            factors = powers
            # New coefficients are factors * old coefficients
            coeff = (coeff * factors)[:-1]
            # Need to trim away the right-most coefficient as it was a constant
            powers = np.array([p for p in powers - 1 if p > -1])
            # update poly
            poly_derivative = Polynomial(*coeff)
            # Update counter
            i += 1
        return poly_derivative


if __name__ == "__main__":
    pol = Polynomial(3, 2, 1, 1)  # 2x^2 + 3x + 1
    print(pol(0))
    print(pol.differentiate())
    print(pol.differentiate(2))
    print(pol.differentiate(3))
    print(pol.differentiate(4))
