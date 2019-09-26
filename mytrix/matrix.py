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

    def __add__(self, mtrx):
        """Add a matrix to this matrix and return the result.

        Doesn't modify the current matrix
        """
        if not (self.m == mtrx.m and self.n == mtrx.n):
            raise exc.ComformabilityError(
                    "matrices must have the same dimensions")
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = self.get_ij(i, j) + mtrx.get_ij(i, j)
                res.set_ij(i, j, val)
        return res

    def __sub__(self, mtrx):
        """Subtract a matrix from this matrix and returns the result.

        Doesn't modify the current matrix
        """
        if not (self.m == mtrx.m and self.n == mtrx.n):
            raise exc.ComformabilityError(
                    "matrices must have the same dimensions")
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = self.get_ij(i, j) - mtrx.get_ij(i, j)
                res.set_ij(i, j, val)
        return res

    def __mul__(self, mtrx):
        """Right multiply this matrix by another and return.

        Multiply this matrix by another matrix (on the right) and return
        the result. Doesn't modify the current matrix
        """
        if not self.n == mtrx.m:
            raise exc.ComformabilityError(
                    "column dimension of first matrix much match row dimension"
                    "of second matrix")
        res = Matrix(self.m, mtrx.n)
        for i in range(self.m):
            for j in range(mtrx.n):
                val = sum([self.get_ij(i, k) * mtrx.get_ij(k, j)
                           for k in range(self.m)])
                res.set_ij(i, j, val)
        return res

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


class MatrixTests(unittest.TestCase):
    """Unit test functions."""

    def testAdd(self):
        """Test matrix addition."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 + m2
        self.assertTrue(m3 == Matrix.fromRows([[6, 8], [10, 12]]))

        m4 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 + m4

    def testSub(self):
        """Test matrix subtraction."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 - m2
        self.assertTrue(m3 == Matrix.fromRows([[-4, -4], [-4, -4]]))

        m4 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 - m4

    def testMul(self):
        """Test matrix multiplication."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 * m2
        self.assertTrue(m3 == Matrix.fromRows([[19, 22], [43, 50]]))

        m4 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 * m4

    def testNeg(self):
        """Test matrix negation."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = -m1
        self.assertTrue(m2 == Matrix.fromRows([[-1, -2], [-3, -4]]))


if __name__ == "__main__":
    unittest.main()
