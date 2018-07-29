from math import sqrt, acos


class Vector:

    def __init__(self, *args):
        """
        Initializes Vector class.

        :param args: Ordered values representing coordinates of vector.

        """
        # Store values
        self.coords = args
        self.len = len(self)
        self.mod = abs(self)

    def __repr__(self):
        return "Vector{}".format(self.coords)

    def __len__(self):
        """
        Defines length of a vector.
        :return:
        """
        return len(self.coords)

    def __abs__(self):
        """
        Defines modulus of a vector.
        :return:
        """
        return sqrt(sum([i**2 for i in self.coords]))

    def __add__(self, other):
        """
        Defines addition between two vectors.
        :param other:
        :return:
        """
        if self.len == other.len:
            return Vector(*[i + j for i, j in zip(self.coords, other.coords)])
        else:
            raise ValueError(
                "Cannot add vectors with different lengths."
            )

    def __sub__(self, other):
        """
        Defines subtraction
        :param other:
        :return:
        """
        return self + Vector(*[-i for i in other.coords])

    def __mul__(self, other):
        """
        Defines multiplication
        :param other:
        :return:
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vector(*[i * other for i in self.coords])
        elif isinstance(other, Vector):
            return sum([i * j for i, j in zip(self.coords, other.coords)])

    def __rmul__(self, other):
        """
        Defines right multiplication
        :param other:
        :return:
        """
        return self * other

    def __eq__(self, other):
        """
        Checks whether two vectors are equal.

        :param other:
        :return:
        """
        return self.coords == other.coords

    def is_multiple(self, other):
        """
        Returns whether other is a multiple of self.
        :param other:
        :return:
        """
        # Notice that two vectors a_ and b_ are multiples if there exists a non
        # zero k such that a = kb
        if self.len != other.len:
            raise ValueError(
                "Cannot compare vectors with different lengths."
            )
        else:
            ix = 0
            while ix <= self.len - 1:
                try:
                    factor = self.coords[ix] / other.coords[ix]
                    if other * factor == self:
                        return True
                    else:
                        return False
                except ZeroDivisionError:
                    if self.coords[ix] == 0:
                        ix += 1
                    else:
                        return False

    def is_orthogonal(self, other):
        """
        Checks if two vectors are orthogonal
        :param self:
        :param other:
        :return:
        """
        return self * other == 0

    def is_unit(self):
        """
        Checks if self is a unit vector

        :return:
        """
        return self.mod == 1

    def component_along(self, other):
        """
        Finds the component of self along other.
        :param other:
        :return:
        """
        if other.mod != 0:
            return (self * other) / (other * other)
        else:
            raise ValueError(
                "Cannot calculate component along {} because it has modulus"
                " = 0".format(other)
            )

    def projection_along(self, other):
        """
        Calculates the projection of self along other.
        :param other:
        :return:
        """
        return self.component_along(other) * other

    def angle_between(self, other):
        """
        Calculates the angle between self and other. This angle is between
        [0, pi]
        :param other:
        :return:
        """
        return acos((self * other) / (self.mod * other.mod))

    def parametric_line_through(self, point):
        """
        Finds the equation of a line parallel to self and passing through point
        Parametric.

        :param point:
        :return:
        """
        # Check point and vector have same dimensions
        if len(point) != self.len:
            raise ValueError(
                "Point must have same dimensions as the vector."
            )
        # Get the point coordinates both if it is a position vector or a point
        if isinstance(point, tuple) or isinstance(point, list):
            point = tuple(point)
        elif isinstance(point, Vector):
            point = point.coords
        # Return parametric
        return "{p} + \u03BB{v}".format(p=point, v=self.coords)

    def cross(self, other):
        """
        Cross product between self and other, that is
        self x other

        :param other:
        :return:
        """
        if not isinstance(other, Vector):
            raise ValueError(
                "Cross product must be between two Vector instances."
            )
        if self.len != 3 and other.len != 3:
            raise ValueError(
                "Cannot evaluate cross product between vectors with dimensions"
                "different from 3."
            )

        return Vector(
            self.coords[1]*other.coords[2] - self.coords[2]*other.coords[1],
            self.coords[2]*other.coords[0] - self.coords[0]*other.coords[2],
            self.coords[0]*other.coords[1] - self.coords[1]*other.coords[0]
        )


if __name__ == "__main__":
    v = Vector(1, 2, 3)
    print(v)
    print(len(v))
    print(abs(v))
    v2 = Vector(4, 5, 6)
    print(v2)
    print("v + v2 = ", v + v2)
    print("v - v2 = ", v - v2)
    print("v * 2 = ", v * 2)
    print("2 * v = ", 2 * v)
    print("v * v2 = ", v * v2)
    print("v2 * v = ", v2 * v)
    v3 = v * 3
    print("v3 = v * 3 = ", v3)
    print("Is v3 multiple of v? ", v.is_multiple(v3))
    print("Are (1, 0, 0) and (0, 1, 0) orthogonal? ",
          Vector(1, 0, 0).is_orthogonal(Vector(0, 1, 0)))
    print("Component of (4, 5, -6) in y axis direction: ",
          Vector(4, 5, -6).component_along(Vector(0, 1, 0)))
    print("The projection is: ",
          Vector(4, 5, -6).projection_along(Vector(0, 1, 0)))
    print("Angle between (2, 0, 1, 4) and (-1, 3, -5, 7): ",
          Vector(2, 0, 1, 4).angle_between(Vector(-1, 3, -5, 7)))
    print("Line through (0, 1, 2, 3) parallel to (4, 5, 6, 7): ",
          Vector(4, 5, 6, 7).parametric_line_through((0, 1, 2, 3)))
    print("(-1,0,2)cross(3,-4,5) = ",
          Vector(-1, 0, 2).cross(Vector(3, -4, 5)))
