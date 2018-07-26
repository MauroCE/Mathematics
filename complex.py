"""
Even though there already is an implementation of complex number in Python, I
want to implement it myself, to better understand complex numbers.
"""
from trigonometry import full_arctan, pretty_angle
from utils import sign
import matplotlib.pyplot as plt
import math


class Complex:

    def __init__(self, real: float = 0, img: float = 0):
        """
        Constructor for Complex class. It initializes the class and creates an
        instance of the object.

        :param real: Real part of the complex number
        :type real: float
        :param img: Imaginary part of the complex number
        :type img: float

        """
        # Store real and imaginary parts separately just to have them as attr
        self.real = real
        self.img = img
        # Complex numbers are just ordered pairs
        self.num = (self.real, self.img)
        # Modulus
        self.modulus = self.__abs__()
        # Argument (counter-clockwise angle)
        try:
            self.arg = self._get_argument()
        except ValueError:
            self.arg = None
        # Now that you have the argument and the modulus, can find cos and sin
        if self.arg is not None:
            self.cos = math.cos(self.arg)
            self.sin = math.sin(self.arg)

    def __repr__(self) -> str:
        """
        Overwrite to have standard complex number representation in console.

        :return: String with standard complex number representation, i.e.
                 a + bi
        :rtype: str

        """

        string = "{real}".format(real=self.real) if self.real != 0 else ""
        # Want to have "a + bi", "a", "bi"
        # Want to avoid "+ bi" or "a + "
        if self.real != 0 and self.img < 0:
            string += " - "  # img will already have -
        elif self.real != 0 and self.img > 0:
            string += " + "
        elif self.real == 0 and self.img < 0:
            string += "- "
        string += "{img}i".format(img=abs(self.img)) if self.img != 0 else ""
        return string

    def __add__(self, other: 'Complex') -> 'Complex':
        """
        Defines what happens when we sum numbers. Recall that for two
        complex numbers
        x = a + bi
        y = c + di
        we have x + y = (a + c) + (b + d)i

        :param other: Other complex number that we are summing it with
        :type other: Complex
        :return: Complex sum of the two numbers
        :rtype: Complex

        """
        return Complex(self.real + other.real, self.img + other.img)

    def __sub__(self, other: 'Complex') -> 'Complex':
        """
        Defines what happens when we subtract two complex numbers.

        :param other: Other complex number that we are summing it with
        :type other: Complex
        :return: Complex sum of the two numbers
        :rtype: Complex

        """
        return Complex(self.real - other.real, self.img - other.img)

    def __matmul__(self, other: 'Complex') -> 'Complex':
        """

        :param other: Complex number that we are multiplying by
        :type other: Complex
        :return: Result of the multiplication
        :rtype: Complex

        """
        # (ac - bd)
        real = self.real * other.real - self.img * other.img
        # (ad + bc)
        img = self.real * other.img + self.img * other.real
        return Complex(real, img)

    def __mul__(self, other):
        """
        This implements:
            1) scalar multiplication with a Complex instance
            2) Complex * Complex multiplication
        For case 1) Recall that for complex numbers
        x = a + bi
        y = c + di
        their multiplication is
        x * y = ac + adi + cbi + - db = (ac - bd) + (ad + bc)i
        :param other:
        :return:
        """
        if isinstance(other, int) or isinstance(other, float):
            return Complex(self.real * other, self.img * other)
        elif isinstance(other, Complex):
            # real = (ac - bd) ; img = (ad + bc)
            real = self.real * other.real - self.img * other.img
            img = self.real * other.img + self.img * other.real
            return Complex(real, img)
        else:
            raise NotImplemented("Could not recognise the type of the factor.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        This is used when the class instance is divided by some other.

        :param other:
        :return:
        """
        if isinstance(other, float) or isinstance(other, int):
            return Complex(self.real / other, self.img / other)
        elif isinstance(other, Complex):
            # multiply above and below by conjugate
            return self * other.conjugate() / (other.modulus**2)

    def __abs__(self):
        """
        Implements what happens when we use the built-in abs() function on an
        instance of the Complex class.
        :return:
        """
        return math.sqrt(self.real**2 + self.img**2)

    def _get_argument(self) -> float:
        """
        This function finds the argument of the complex number.
        The argument is the counter-clockwise positive angle measured from the
        real axis. To calculate it we use math.atan2(y, x) which is defined
        here: https://en.wikipedia.org/wiki/Atan2
        Here is a basic explanation:

        Recall tanx = sinx / cosx where we have
        sinx = x - x^3/3! + x^5/5! - ...
        cosx = 1 - x^2/2! + x^4/4! - ...
        In order to invert a function we need it to be bijective, but tanx is
        not bijective. Therefore we restrict it. Usually we restrict tanx on
        (-pi/2, pi/2) interval with boundaries excluded because cosx = 0 there
        and would make the fraction undefined. The arctan is teh inverse
        function going from R to (-pi/2, pi/2)
        :return:
        """
        return full_arctan(self.img, self.real)

    def plot(self, ax=None):
        # Set up an axis and a canvas
        if ax is None:
            fig, ax = plt.subplots()
        # Scatter plot
        ax.scatter(self.real, self.img, label=self.__repr__())
        # Add the x and y axis as real and imaginary axes, show (0, 0)
        ax.set_xlabel("Real Axis")
        ax.set_ylabel("Imaginary Axis")
        ax.set_xlim(0, self.real + self.real * 0.1)
        ax.set_ylim(0, self.img + self.img * 0.1)
        # Need to plot a dotted line connecting point
        ax.plot([0, self.real], [0, self.img], linestyle='--', linewidth=0.5)
        # Add title
        ax.set_title(self.__repr__())
        return ax

    def conjugate(self) -> 'Complex':
        """
        Returns the conjugate complex number. Recall that for complex number
        z = x + iy
        its conjugate is the number
        z* = x - iy

        :return: Complex conjugate
        :rtype Complex

        """
        return Complex(self.real, -self.img)

    def reciprocal(self) -> 'Complex':
        """
        Given a complex number z, it finds its reciprocal z^-1 such that
        z * z^-1 = 1
        To find it, notice that given its conjugate z*, we have
        z \times z* = (x + iy) \times (x - iy) = x^2 + y^2 = |z|^2
        Therefore z^-1 = z* / |z|^2

        :return: Complex reciprocal
        :rtype Complex

        """
        return self.conjugate() / (self.modulus**2)

    def to_polar(self) -> str:
        """
        Prints a representation of the number in polar coordinates.

        :return: String representing the number in polar coordinates
        :rtype: str
        """
        return "{modulus}e^i{arg}".format(modulus=self.modulus,
                                          arg=pretty_angle(self.arg))

    @staticmethod
    def find_n_unity_roots(n: int):
        """
        This function solves the usual
        z^n = 1 for some n, i.e. it finds the n roots of unity

        :param n: Power of z, for z^2 = 1, use degree=2
        :type n: int
        :return:
        """
        roots = []
        for k in range(0, n):
            # 2kpi / n is the argument of the solutions
            root = Complex(real=math.cos(2 * k * math.pi / n),
                           img=math.sin(2 * k * math.pi / n))
            roots.append(root)
        return roots






if __name__ == "__main__":
    print("*" * 20, "Create Numbers", "*" * 20)
    a = Complex(1, 3)
    print('a = ', a)
    b = Complex(4, 5)
    print('b = ',  b)
    print("*" * 20, "Add, Sub, Mul", "*" * 20)
    print('a + b = ', a + b)
    print('a - b = ', a - b)
    print('a * 2 = ',  a * 2)
    print('2 * a = ', 2 * a)
    print('a * b = ', a * b)
    print('b * a = ', b * a)
    print('a * 2.5 = ', a * 2.5)
    print('2.5 * a = ', 2.5 * a)
    print("*" * 20, "Plot", "*" * 20)
    ax1 = a.plot()
    _ = b.plot(ax1)
    plt.legend()
    #plt.show()
    print("*" * 20, "Modulus", "*" * 20)
    print("|a| = ", abs(a))
    print("|a| = ", a.modulus)
    print("*" * 20, "Principal Argument", "*" * 20)
    print("arg(a) = ", a.arg)
    print("cos = ", a.cos)
    print("sin = ", a.sin)
    print("|a| * cos = ", a.cos * a.modulus)
    print("|a| * sin = ", a.sin * a.modulus)
    print("*" * 20, "Conjugate & reciprocal", "*" * 20)
    print("conj(a) = ", a.conjugate())
    print("recipr(a) = ", a.reciprocal())
    print("a * recipr(a) = ", a * a.reciprocal())
    print("*" * 20, "Polar", "*" * 20)
    print("a polar = ", a.to_polar())
    print("Example:")
    print((Complex(0, -3) / Complex(-math.sqrt(3), 1)).to_polar())
    print("*" * 20, "Unity Roots", "*" * 20)
    for r in Complex.find_n_unity_roots(3):
        print(r.to_polar())

