"""Module for general matrix class and related unit tests."""

import random
import unittest

import exceptions as exc

__version__ = "0.1"


class Matrix:
    """A class to represent a general matrix."""

    def __init__(self, m, n, init=True):
        """Initalise matrix dimensions and contents."""
        # initialise matrix dimensions
        if not (isinstance(m, int) and isinstance(n, int)):
            raise TypeError("dimensions must be integral")
        if m <= 0 or n <= 0:
            raise ValueError("dimensions must be positive")
        self.m = m
        self.n = n

        # initalise matrix contents
        if not isinstance(init, bool):
            raise TypeError("init must be Boolean")
        if init:
            self.rows = [[0]*n for _ in range(m)]
        else:
            self.rows = []

    def __str__(self):
        """Generate text representation of matrix."""
        s = '\n'.join([' '.join([str(elem) for elem in row])
                      for row in self.rows])
        return s + '\n'

    def __repl__(self):
        """Generate reproducible representation of matrix."""
        s = "Matrix of dimension " + str(self.m) + " by " \
            + str(self.n) + '\n'
        s = s + "with data" + '\n'
        s = s + '\n'.join([' '.join([str(elem) for elem in row])
                          for row in self.rows])
        return s + '\n'

    def __eq__(self, mtrx):
        """Evaluate whether two matrices are equivalent."""
        return self.rows == mtrx.rows

    def __add__(self, obj):
        """Add a valid object to this matrix and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        and numeric scalars
        """
        if isinstance(obj, Matrix):
            if not (self.m == obj.m and self.n == obj.n):
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            res = Matrix(self.m, self.n)
            for i in range(self.m):
                for j in range(self.n):
                    val = self.get_ij(i, j) + obj.get_ij(i, j)
                    res.set_ij(i, j, val)
            return res
        elif self.isNumeric(obj):
            res = Matrix(self.m, self.n)
            for i in range(self.m):
                for j in range(self.n):
                    val = self.get_ij(i, j) + obj
                    res.set_ij(i, j, val)
            return res
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj) + " to matrix")

    def __sub__(self, obj):
        """Subtract a valid object from this matrix and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        and numeric scalars
        """
        if isinstance(obj, Matrix):
            if not (self.m == obj.m and self.n == obj.n):
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            res = Matrix(self.m, self.n)
            for i in range(self.m):
                for j in range(self.n):
                    val = self.get_ij(i, j) - obj.get_ij(i, j)
                    res.set_ij(i, j, val)
            return res
        elif self.isNumeric(obj):
            res = Matrix(self.m, self.n)
            for i in range(self.m):
                for j in range(self.n):
                    val = self.get_ij(i, j) - obj
                    res.set_ij(i, j, val)
            return res
        else:
            raise TypeError(
                    "cannot subtract object of type " + type(obj) +
                    "from matrix")

    def __mul__(self, obj):
        """Multiply this matrix by a valid object and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        and numeric scalars. In the case where the other object is a matrix,
        multiplication occurs with the current matrix on the left-hand side
        """
        if isinstance(obj, Matrix):
            if not self.n == obj.m:
                raise exc.ComformabilityError(
                        "column dimension of first matrix much match row " +
                        "dimension of second matrix")
            res = Matrix(self.m, obj.n)
            for i in range(self.m):
                for j in range(obj.n):
                    val = sum([self.get_ij(i, k) * obj.get_ij(k, j)
                               for k in range(self.m)])
                    res.set_ij(i, j, val)
            return res
        elif self.isNumeric(obj):
            res = Matrix(self.m, self.n)
            for i in range(self.m):
                for j in range(self.n):
                    val = self.get_ij(i, j) * obj
                    res.set_ij(i, j, val)
            return res
        else:
            raise TypeError(
                    "cannot multiply matrix by object of type " + type(obj))

    def __pos__(self):
        """Make all elements of matrix positive."""
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = +self.get_ij(i, j)
                res.set_ij(i, j, val)
        return res

    def __neg__(self):
        """Make all elements of matrix negative."""
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = -self.get_ij(i, j)
                res.set_ij(i, j, val)
        return res

    def __iadd__(self, mtrx):
        """Add a matrix to this matrix, modifying it in the process."""
        # calls __add__
        tmp = self + mtrx
        self.rows = tmp.rows
        return self

    def __isub__(self, mtrx):
        """Subtract a matrix from this matrix, modifying it in the process."""
        # calls __sub__
        tmp = self - mtrx
        self.rows = tmp.rows
        return self

    def __imul__(self, mtrx):
        """Right multiply this matrix by another and modify.

        Multiply this matrix by another matrix (on the right), modifying
        it in the process.
        """
        # calls __mul__
        tmp = self * mtrx
        self.rows = tmp.rows
        self.m, self.n = tmp.dim()
        return self

    def dim(self):
        """Get matrix dimensions as tuple."""
        return (self.m, self.n)

    def get_ij(self, i, j):
        """Get element in (i, j)th position."""
        return self.rows[i][j]

    def set_ij(self, i, j, val):
        """Set element in (i, j)th position."""
        self.rows[i][j] = val

    @classmethod
    def makeRandom(cls, m, n, min=0, max=1):
        """Create random matrix.

        Make a random matrix of dimension m by n with elements chosen
        independently and uniformly from the interval (min, max).
        """
        obj = Matrix(m, n, init=False)
        for _1 in range(m):
            obj.rows.append([random.randrange(min, max) for _2 in range(n)])
        return obj

    @classmethod
    def makeZero(cls, m, n):
        """Make a zero matrix of dimension m by n."""
        return Matrix(m, n, init=True)

    @classmethod
    def makeIdentity(cls, m):
        """Make an identity matrix of dimension m by m."""
        obj = Matrix(m, m, init=False)
        for i in range(m):
            obj.rows.append([1 if i == j else 1 for j in range(m)])
        return obj

    @classmethod
    def fromRows(cls, rows):
        """Make a matrix from a list of rows."""
        m = len(rows)
        n = len(rows[0])
        # check that list of rows is valid
        if any([len(row) != n for row in rows[1:]]):
            raise ValueError("inconsistent row lengths")
        obj = Matrix(m, n, init=False)
        obj.rows = rows
        return(obj)

    @classmethod
    def fromList(cls, elems, **kwargs):
        """Make matrix from list.

        Make a matrix from a list of elements, filling along rows,
        when given at least one dimension of the matrix.
        """
        if not ('m' in kwargs or 'n' in kwargs):
            raise ValueError("at least one of m and n must be specified")
        m = kwargs['m']
        n = kwargs['n']
        if m * n != len(elems):
            raise ValueError("dimension does not match number of elements in"
                             "list")

        obj = Matrix(m, m, init=False)
        for i in range(m):
            obj.rows.append(elems[i * m: i * (m + 1)])
        return obj

    @classmethod
    def isNumeric(cls, obj):
        """Check if a given object is of a numeric type.

        Note that since bool inherits from int, that this will accept
        Boolean values
        """
        return isinstance(obj, (int, float, complex))


class MatrixTests(unittest.TestCase):
    """Unit test functions."""

    def testAdd(self):
        """Test addition operator."""
        # test addition by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 + m2
        self.assertTrue(m3 == Matrix.fromRows([[6, 8], [10, 12]]))

        # test addition by scalar
        m4 = m1 + 1
        self.assertTrue(m4 == Matrix.fromRows([[2, 3], [4, 5]]))

        # test addition by non-conforming matrix
        m5 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 + m5

        # test addition by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 + 'spam'

    def testSub(self):
        """Test subtraction operator."""
        # test subtraction by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 - m2
        self.assertTrue(m3 == Matrix.fromRows([[-4, -4], [-4, -4]]))

        # test subtraction by scalar
        m4 = m1 - 1
        self.assertTrue(m4 == Matrix.fromRows([[0, 1], [2, 3]]))

        # test subtraction by non-conforming matrix
        m5 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 - m5

        # test subtraction by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 - 'spam'

    def testMul(self):
        """Test multiplication operator."""
        # test multiplication by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 * m2
        self.assertTrue(m3 == Matrix.fromRows([[19, 22], [43, 50]]))

        # test multiplication by scalar
        m4 = m1 * 2
        self.assertTrue(m4 == Matrix.fromRows([[2, 4], [6, 8]]))

        # test multiplication by non-conforming matrix
        m5 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 * m5

        # test multiplication by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 * 'spam'

    def testNeg(self):
        """Test matrix negation."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = -m1
        self.assertTrue(m2 == Matrix.fromRows([[-1, -2], [-3, -4]]))


if __name__ == "__main__":
    unittest.main()
