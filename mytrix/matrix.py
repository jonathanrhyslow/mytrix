import random
import unittest

from mytrix.exceptions import ComformabilityError

__version__ = "0.1"

class Matrix:
    """A class to represent a general matrix"""

    def __init__(self, m, n, init = True) -> None:
        # initialise matrix dimensions
        if not (isinstance(m, int) and isinstance(n, int)):
            raise TypeError("dimensions must be integral")
        if m <= 0 or n <= 0:
            raise ValueError("dimensions must be positive")
        self.m = m
        self.n = n

        # initalise matrix contents
        if not isinstance(init, Bool):
            raise TypeError("init must be Boolean")
        if init:
            self.rows = [[0]*n for _ in range(m)]
        else:
            self.rows = []

    def __str__(self):
        s = '\n'.join([' '.join([str(elem) for elem in row]) for row in self.data])
        return s + '\n'

    def __repl__(self):
        s = "Matrix of dimension " + str(m) + " by " + str(n) + '\n'
        s = s + "with data" + '\n'
        s = s + '\n'.join([' '.join([str(elem) for elem in row]) for row in self.data])
        return s + '\n'

    def __eq__(self, mtrx):
        return self.data == mtrx.data

    def __add__(self, mtrx):
        if not (self.m == mtrx.m and self.n == mtrx.n):
            raise ComformabilityError("matrices must have the same dimensions")
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = self.get_ij(i, j) + mtrx.get_ij(i, j)
                res.set_ij(i, j, val)
        return res

    def __sub__(self, mtrx):
        if not (self.m == mtrx.m and self.n == mtrx.n):
            raise ComformabilityError("matrices must have the same dimensions")
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = self.get_ij(i, j) - mtrx.get_ij(i, j)
                res.set_ij(i, j, val)
        return res

    def __mul__(self, mtrx):
        if not self.n == mtrx.m:
            raise ComformabilityError("column dimension of first matrix much match row dimension of \
            second matrix")
        res = Matrix(self.m, mtrx.n)
        for i in range(self.m):
            for j in range(mtrx.n):
                val = sum([self.get_ij(i, k) * mtrx.get_ij(k, j) for k in range(self.m)])
                res.set_ij(i, j, val)
        return res

    def __pos__(self):
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = +self.get_ij(i, j)
                res.set_ij(i, j, val)
        return res

    def __neg__(self):
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = -self.get_ij(i, j)
                res.set_ij(i, j, val)
        return res

    def get_ij(self, i, j):
        return self.data[i][j]

    def set_ij(self, i, j, val):
        self.data[i][j] = val

    @classmethod
    def makeRandom(cls, m, n, min = 0, max = 1):
        """Make a random matrix of dimension m by n with elements chosen \
        independently and uniformly from the interval (min, max)"""

        obj = Matrix(m, n, init = False)
        for _1 in range(m):
            obj.data.append([random.randrange(min, max) for _2 in range(n)])
        return obj

    @classmethod
    def makeZero(cls, m, n):
        """Make a zero matrix of dimension m by n"""

        return Matrix(m, n, init = True)

    @classmethod
    def makeIdentity(cls, m):
        """Make an identity matrix of dimension m by m"""

        obj = Matrix(m, m, init = False)
        for i in range(m):
            obj.data.append([1 if i = j else 1 for j in range(m)])
        return obj

class MatrixTests(unittest.TestCase):

    def testAdd(self):
        m1 = Matrix(2, 2, [1, 2, 3, 4])
        m2 = Matrix(2, 2, [5, 6, 7, 8])
        m3 = m1 + m2
        self.assertTrue(m3 == Matrix(2, 2, [6, 8, 10, 12]))

    def testSub(self):
        m1 = Matrix(2, 2, [1, 2, 3, 4])
        m2 = Matrix(2, 2, [5, 6, 7, 8])
        m3 = m1 - m2
        self.assertTrue(m3 == Matrix(2, 2, [-4, -4, -4, -4]))

    def testMul(self):
        m1 = Matrix(2, 2, [1, 2, 3, 4])
        m2 = Matrix(2, 2, [5, 6, 7, 8])
        m3 = m1 * m2
        self.assertTrue(m3 == Matrix(2, 2, [19, 22, 43, 50]))

    def testNeg(self):
        m1 = Matrix(2, 2, [1, 2, 3, 4])
        m2 = -m1
        self.assertTrue(m2 == Matrix(2, 2, [-1, -2, -3, -4]))

if __name__ == "__main__":
    unittest.main()
