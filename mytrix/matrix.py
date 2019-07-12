from typing import List

import unittest

class Matrix:
    """A class to represent a general matrix"""

    def __init__(self, m: int, n: int, data: List[float] = None) -> None:
        # initialise matrix dimensions
        if not (isinstance(m, int) and isinstance(n, int)):
            raise TypeError("dimensions must be integral")
        if m < 0 or n < 0:
            raise ValueError("dimensions must be non-negative")
        self.m = m
        self.n = n

        # initalise matrix contents
        self.data = []
        if data:
            if len(data) != m * n:
                raise ValueError("data argument should be of length m x n")
            self.data = [[data[i * n + j % m] for j in range(n)] for i in range(m)]
        else:
            self.data = [[0 for j in range(n)] for i in range(n)]

    def __str__(self):
        s = '\n'.join([' '.join([str(elem) for elem in row]) for row in self.data])
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

if __name__ == "__main__":
    unittest.main()
