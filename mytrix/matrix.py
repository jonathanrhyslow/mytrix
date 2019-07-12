import unittest

class Matrix:
    """A class to represent a general matrix"""

    def __init__(self, m: int, n: int, data = None) -> None:
        # initialise matrix dimensions
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
        return ' '.join('\n'.join(data))

    def __eq__(self, M):
        return self.data == M.data

    def __add__(self, M):
        tmp = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = self.get_ij(i, j) + M.get_ij(i, j)
                tmp.set_ij(i, j, val)
        return tmp

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

if __name__ == "__main__":
    unittest.main()
