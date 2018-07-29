from linear_algebra.vectors import Vector


class Matrix:

    def __init__(self, *column_vectors):
        """
        Constructs Matrix class.

        :param column_vectors: Column vectors from left to right.
        """
        length = column_vectors[0].len
        if not all([v.len == length for v in column_vectors]):
            raise TypeError(
                "Column vectors must all have the same size."
            )
        # Store vectors as column vectors
        self.c_vecs = {
            index: vec for index, vec in enumerate(column_vectors)
        }
        # Store vectors as row vectors
        self.r_vecs = {
            index: vec for index, vec in enumerate(
                       [Vector(*[v.coords[index] for v in column_vectors])
                        for index in range(len(column_vectors))]
            )
        }

    def __repr__(self):
        return "Matrix{}".format(tuple(self.r_vecs.values()))

    def __str__(self):
        string = "["
        string += ",\n ".join([str(self.r_vecs[i])
                               for i in range(len(self.r_vecs))]) + "]"
        return string

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Matrix(*[self.c_vecs[i] * other
                            for i in range(len(self.c_vecs))])
        if isinstance(other, Vector):
            pass
        else:
            raise NotImplemented(
                "Not implemented."
            )


if __name__ == "__main__":
    m = Matrix(Vector(1, 2, 3), Vector(3, 4, 5), Vector(6, 7, 8))
    print(m)
    print("m * 2 = ", m * 2)
