"""Module for testing general matrix class."""

from copy import copy, deepcopy
import unittest

from matrix import Matrix
import exceptions as exc


class MatrixTests(unittest.TestCase):
    """Unit test functions."""

    def testCopy(self):
        """Test shallow and deep copying."""
        # test shallow copying
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = copy(m1)
        self.assertTrue(m2 is not m1)
        m2[1, 1] = 5
        self.assertTrue(m1[1, 1] == 5)

        # test deep copying
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = deepcopy(m1)
        self.assertTrue(m2 is not m1)
        m2[1, 1] = 5
        self.assertTrue(m1[1, 1] == 4)

    def testStr(self):
        """Test string method."""
        raise NotImplementedError()

    def testRepl(self):
        """Test REPL method."""
        raise NotImplementedError()

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

        # test multiplication by non-square (but conforming) matrix
        m3 = Matrix.fromRows([[5, 6, 7], [8, 9, 10]])
        m4 = m1 * m3
        self.assertTrue(m4 == Matrix.fromRows([[21, 24, 27], [47, 54, 61]]))

        # test multiplication by scalar
        m5 = m1 * 2
        self.assertTrue(m5 == Matrix.fromRows([[2, 4], [6, 8]]))

        # test multiplication by non-conforming matrix
        m6 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 * m6

        # test multiplication by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 * 'spam'

    def testDiv(self):
        """Test division operator."""
        # test true division by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        with self.assertRaises(TypeError):
            m1 / m2

        # test true division by scalar
        m3 = m1 / 2
        self.assertTrue(m3 == Matrix.fromRows([[.5, 1.], [1.5, 2.]]))

        # test true division by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 / 'spam'

        # test floor division by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        with self.assertRaises(TypeError):
            m1 // m2

        # test floor division by scalar
        m3 = m1 // 2
        self.assertTrue(m3 == Matrix.fromRows([[0, 1], [1, 2]]))

        # test floor division by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 // 'spam'

    def testArithmeticAssignment(self):
        """Test matrix arithmetic using assignment magics."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])

        # test addition
        m1 += m2
        self.assertTrue(m1 == Matrix.fromRows([[6, 8], [10, 12]]))
        m1 += 1
        self.assertTrue(m1 == Matrix.fromRows([[7, 9], [11, 13]]))

        # test subtraction
        m1 = Matrix.fromRows([[1, 2], [3, 4]])  # reset m1
        m1 -= m2
        self.assertTrue(m1 == Matrix.fromRows([[-4, -4], [-4, -4]]))
        m1 -= 1
        self.assertTrue(m1 == Matrix.fromRows([[-5, -5], [-5, -5]]))

        # test multiplication
        m1 = Matrix.fromRows([[1, 2], [3, 4]])  # reset m1
        m1 *= m2
        self.assertTrue(m1 == Matrix.fromRows([[19, 22], [43, 50]]))
        m1 *= 2
        self.assertTrue(m1 == Matrix.fromRows([[38, 44], [86, 100]]))

        # test division
        m1 = Matrix.fromRows([[1, 2], [3, 4]])  # reset m1
        m1 //= 2
        self.assertTrue(m1 == Matrix.fromRows([[0, 1], [1, 2]]))
        m1 /= 2
        self.assertTrue(m1 == Matrix.fromRows([[0., .5], [.5, 1.]]))

    def testArithmeticReflection(self):
        """Test matrix arithmetic using reflection magics."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])

        # test addition
        m2 = 1 + m1
        self.assertTrue(m2 == Matrix.fromRows([[2, 3], [4, 5]]))

        # test subtraction
        m2 = 1 - m1
        self.assertTrue(m2 == Matrix.fromRows([[0, -1], [-2, -3]]))

        # test multiplication
        m2 = 2 * m1
        self.assertTrue(m2 == Matrix.fromRows([[2, 4], [6, 8]]))

    def testNeg(self):
        """Test matrix negation."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = -m1
        self.assertTrue(m2 == Matrix.fromRows([[-1, -2], [-3, -4]]))

    def testEq(self):
        """Test matrix equality."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1 == Matrix.fromRows([[1, 2], [3, 4]]))
        self.assertTrue(not m1 == 'spam')
        self.assertTrue(not m1 == Matrix.fromRows([[1, 2], [3, 4], [5, 6]]))

    def testGetItem(self):
        """Test getting of matrix element."""
        # test getting element using valid key
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1[1, 1] == 4)

        # test getting element using invalid key
        with self.assertRaises(TypeError):
            m1['spam']
        # TypeError check (must me tuple) is performed before ValueError
        # check (must be length two) so m1[1] raises TypeError
        with self.assertRaises(TypeError):
            m1[1]
        with self.assertRaises(ValueError):
            m1[1, 1, 1]
        with self.assertRaises(TypeError):
            m1[1, 'spam']
        with self.assertRaises(exc.OutOfBoundsError):
            m1[-1, 1]
        with self.assertRaises(exc.OutOfBoundsError):
            m1[1, -1]
        with self.assertRaises(exc.OutOfBoundsError):
            m1[2, 1]
        with self.assertRaises(exc.OutOfBoundsError):
            m1[1, 2]

    def testSetItem(self):
        """Test setting of matrix element."""
        # test setting element using valid key
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m1[1, 1] = 5
        self.assertTrue(m1 == Matrix.fromRows([[1, 2], [3, 5]]))

        # test setting element using invalid key
        with self.assertRaises(TypeError):
            m1['spam'] = 5
        # TypeError check (must me tuple) is performed before ValueError
        # check (must be length two) so m1[1] raises TypeError
        with self.assertRaises(TypeError):
            m1[1] = 5
        with self.assertRaises(ValueError):
            m1[1, 1, 1] = 5
        with self.assertRaises(TypeError):
            m1[1, 'spam'] = 5
        with self.assertRaises(exc.OutOfBoundsError):
            m1[-1, 1] = 5
        with self.assertRaises(exc.OutOfBoundsError):
            m1[1, -1] = 5
        with self.assertRaises(exc.OutOfBoundsError):
            m1[2, 1] = 5
        with self.assertRaises(exc.OutOfBoundsError):
            m1[1, 2] = 5

    def testSubset(self):
        """Test matrix subsetting."""
        # test subsetting matrix using valid rows/cols
        m1 = Matrix.fromRows([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m2 = m1.subset([0, 2], [1])
        self.assertTrue(m2 == Matrix.fromRows([[2], [8]]))

        # test subsetting matrix using invalid rows/cols
        with self.assertRaises(TypeError):
            m1.subset([0, 2], 'spam')
        with self.assertRaises(ValueError):
            m1.subset([0, 2], [])
        with self.assertRaises(exc.OutOfBoundsError):
            m1.subset([-1, 2], [1])
        with self.assertRaises(exc.OutOfBoundsError):
            m1.subset([0, 2], [-1])
        with self.assertRaises(exc.OutOfBoundsError):
            m1.subset([0, 3], [1])
        with self.assertRaises(exc.OutOfBoundsError):
            m1.subset([0, 2], [3])

    def testTranspose(self):
        """Test matrix transposition."""
        # test transposition
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.transpose() == Matrix.fromRows([[1, 3], [2, 4]]))

        # test involution property of transposition
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.transpose().transpose() == m1)

    def testSymmetry(self):
        """Test matrix symmetry."""
        # test symmetry
        m1 = Matrix.fromRows([[1, 2], [2, 4]])
        self.assertTrue(m1.is_symmetric())
        m2 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(not m2.is_symmetric())

        # test skew-symmetry
        m3 = Matrix.fromRows([[0, 2], [-2, 0]])
        self.assertTrue(m3.is_skew_symmetric())
        self.assertTrue(not m2.is_skew_symmetric())

    def testToeplitzDecomposition(self):
        """Test Toeplitz decomposition."""
        # test decomposition on square matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        sym, skew = m1.toeplitz_decomposition()
        self.assertTrue(sym.is_symmetric())
        self.assertTrue(skew.is_skew_symmetric())

        # test decomposition on non-square matrix
        m2 = Matrix.fromRows([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(exc.DecompositionError):
            m2.toeplitz_decomposition()

    def testRowReduction(self):
        """Test reduction to row-reduced and row-echelon form."""
        # test reduction to row-echelon form
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.row_echelon() == Matrix.fromRows([[1, 2], [0, 1]]))

        # test reduction to row-echelon form on the zero matrix
        m2 = Matrix.makeZero(2, 2)
        self.assertTrue(m2.row_echelon() == Matrix.makeZero(2, 2))

        # test reduction to row-echelon form on the matrix with only one row
        m3 = Matrix.fromRows([[1, 2, 3, 4]])
        self.assertTrue(m3.row_reduce() == Matrix.fromRows([[1, 2, 3, 4]]))

        # test reduction to row-echelon form on the matrix with only one column
        m4 = Matrix.fromRows([[1], [2], [3], [4]])
        self.assertTrue(m4.row_reduce() == Matrix.fromRows([[1], [0],
                                                            [0], [0]]))

        # test idompotency of reduction to row-echelon form
        self.assertTrue(m1.row_echelon() == m1.row_echelon().row_echelon())

        # test row reduction
        self.assertTrue(m1.row_reduce() == Matrix.makeIdentity(2))

        # test row reduction on the zero matrix
        self.assertTrue(m2.row_reduce() == Matrix.makeZero(2, 2))

        # test row reduction on the matrix with only one row
        m3 = Matrix.fromRows([[1, 2, 3, 4]])
        self.assertTrue(m3.row_reduce() == Matrix.fromRows([[1, 2, 3, 4]]))

        # test row reduction on the matrix with only one column
        self.assertTrue(m4.row_reduce() == Matrix.fromRows([[1], [0],
                                                            [0], [0]]))

        # test idompotency of reduction to row-echelon form
        self.assertTrue(m1.row_reduce() == m1.row_reduce().row_reduce())

    def testDeterminant(self):
        """Test calculation of determinant for square matrices."""
        # test determinant on square matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.determinant == -2)

        # test determinant on identity matrix
        m2 = Matrix.makeIdentity(2)
        self.assertTrue(m2.determinant == 1)

        # test determinant on non-square matrix
        m3 = Matrix.fromRows([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(exc.LinearAlgebraError):
            m3.determinant()

        # test determinant on a singular square matrix
        m1 = Matrix.fromRows([[1, 2], [2, 4]])
        self.assertTrue(m1.determinant == 0)

if __name__ == "__main__":
    unittest.main()
