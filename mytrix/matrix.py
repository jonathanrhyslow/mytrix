"""Module for general matrix class."""

from copy import deepcopy
from random import randrange

import exceptions as exc


class Matrix:
    """A class to represent a general matrix."""

    def __init__(self, m, n, data):
        """Initalise matrix dimensions and contents."""
        self.m = m
        self.n = n
        self.data = data

    def __str__(self):
        """Generate text representation of matrix."""
        s = '\n'.join([' '.join([str(elem) for elem in row])
                      for row in self.data])
        return s + '\n'

    def __repl__(self):
        """Generate reproducible representation of matrix."""
        s = "Matrix of dimension " + str(self.m) + " by " \
            + str(self.n) + '\n'
        s = s + "with data" + '\n'
        s = s + '\n'.join([' '.join([str(elem) for elem in row])
                          for row in self.data])
        return s + '\n'

    def __eq__(self, mtrx):
        """Evaluate whether two matrices are equal."""
        if not isinstance(mtrx, Matrix):
            return False
        if not (self.m == mtrx.m and self.n == mtrx.n):
            return False
        for i in range(self.m):
            for j in range(self.n):
                if self[i, j] != mtrx[i, j]:
                    return False
        return True

    def __copy__(self):
        """Create a shallow copy of this matrix.

        Creates a new instance of Matrix but with data referencing the data
        of the original matrix.
        """
        return Matrix(self.m, self.n, self.data)

    def __deepcopy__(self, memodict={}):
        """Create a deep copy of this matrix.

        Creates a new instance of Matrix with data copied from the original
        matrix.
        """
        return Matrix(self.m, self.n, deepcopy(self.data))

    def __add__(self, obj):
        """Add a valid object to this matrix and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        and numeric scalars
        """
        if isinstance(obj, Matrix):
            if self.m != obj.m or self.n != obj.n:
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            data = [[self[i, j] + obj[i, j]
                    for j in range(self.n)]
                    for i in range(self.m)]
        elif Matrix.is_numeric(obj):
            data = [[self[i, j] + obj
                    for j in range(self.n)]
                    for i in range(self.m)]
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj) + " to matrix")
        return Matrix(self.m, self.n, data)

    def __sub__(self, obj):
        """Subtract a valid object from this matrix and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        and numeric scalars
        """
        if isinstance(obj, Matrix):
            if self.m != obj.m or self.n != obj.n:
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            data = [[self[i, j] - obj[i, j]
                    for j in range(self.n)]
                    for i in range(self.m)]
        elif Matrix.is_numeric(obj):
            data = [[self[i, j] - obj
                    for j in range(self.n)]
                    for i in range(self.m)]
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj) + " to matrix")
        return Matrix(self.m, self.n, data)

    def __mul__(self, obj):
        """Multiply this matrix by a valid object and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        and numeric scalars. In the case where the other object is a matrix,
        multiplication occurs with the current matrix on the left-hand side
        """
        if isinstance(obj, Matrix):
            if self.n != obj.m:
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            data = [[sum([self[i, k] * obj[k, j] for k in range(self.n)])
                    for j in range(obj.n)]
                    for i in range(self.m)]
            return Matrix(self.m, obj.n, data)
        elif Matrix.is_numeric(obj):
            data = [[self[i, j] * obj
                    for j in range(self.n)]
                    for i in range(self.m)]
            return Matrix(self.m, self.n, data)
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj) + " to matrix")

    def __floordiv__(self, obj):
        """Divide this matrix by a scalar.

        Doesn't modify the current matrix
        """
        if Matrix.is_numeric(obj):
            data = [[self[i, j] // obj
                    for j in range(self.n)]
                    for i in range(self.m)]
            return Matrix(self.m, self.n, data)
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj) + " to matrix")

    def __truediv__(self, obj):
        """Divide this matrix by a scalar.

        Doesn't modify the current matrix
        """
        if Matrix.is_numeric(obj):
            data = [[self[i, j] / obj
                    for j in range(self.n)]
                    for i in range(self.m)]
            return Matrix(self.m, self.n, data)
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj) + " to matrix")

    def __pos__(self):
        """Unary positive. Included for symmetry only."""
        data = [[+self[i, j] for j in range(self.n)] for i in range(self.m)]
        return Matrix(self.m, self.n, data)

    def __neg__(self):
        """Negate all elements of the matrix."""
        data = [[-self[i, j] for j in range(self.n)] for i in range(self.m)]
        return Matrix(self.m, self.n, data)

    def __iadd__(self, obj):
        """Add a matrix to this matrix, modifying it in the process."""
        # calls __add__
        tmp = self + obj
        self.data = tmp.data
        return self

    def __isub__(self, obj):
        """Subtract a matrix from this matrix, modifying it in the process."""
        # calls __sub__
        tmp = self - obj
        self.data = tmp.data
        return self

    def __imul__(self, obj):
        """Right multiply this matrix by another and modify.

        Multiply this matrix by another matrix (on the right), modifying
        it in the process.
        """
        # calls __mul__
        tmp = self * obj
        self.data = tmp.data
        self.m, self.n = tmp.dim()
        return self

    def __ifloordiv__(self, obj):
        """Divide this matrix by a scalar, modifying it in the process."""
        # calls __floordiv__
        tmp = self // obj
        self.data = tmp.data
        return self

    def __itruediv__(self, obj):
        """Divide this matrix by a scalar, modifying it in the process."""
        # calls __truediv__
        tmp = self / obj
        self.data = tmp.data
        return self

    def __radd__(self, obj):
        """Implement reflected addition."""
        # calls __add__
        return self + obj

    def __rsub__(self, obj):
        """Implement reflected subtraction."""
        # calls __sub__
        return -self + obj

    def __rmul__(self, obj):
        """Implement reflected multiplication."""
        # calls __mul__
        # note, if two matrices are multiplied, __mul__ takes precedence
        return self * obj

    def dim(self):
        """Get matrix dimensions as tuple."""
        return (self.m, self.n)

    def __getitem__(self, key):
        """Get element in (i, j)th position."""
        self.__check_key_validity(key)
        return self.data[key[0]][key[1]]

    def __setitem__(self, key, val):
        """Set element in (i, j)th position."""
        self.__check_key_validity(key)
        self.data[key[0]][key[1]] = val

    def __check_key_validity(self, key):
        """Validate keys for __getitem__() and __setitem__() methods."""
        if not isinstance(key, tuple):
            raise TypeError("key must be a tuple")
        if len(key) != 2:
            raise ValueError("key must be of length two")
        if not (isinstance(key[0], int) and isinstance(key[1], int)):
            raise TypeError("elements of key must be integers")
        if not ((0 <= key[0] < self.m) and (0 <= key[1] < self.n)):
            raise exc.OutOfBoundsError("key is out of bounds")

    def subset(self, rows, cols):
        """Extract subset of data and columns and form into a new matrix."""
        # validation on data/cols
        if not (isinstance(rows, list) and isinstance(rows, list)):
            raise TypeError("arguments must be lists")
        if len(rows) == 0 or len(cols) == 0:
            raise ValueError("subset cannot be empty")
        # validation on elements of data/cols
        for i, elem in enumerate(rows + cols):
            if not isinstance(elem, int):
                raise TypeError("elements of data/cols must be integers")
            # if element represents a row
            if i < len(rows):
                if not 0 <= elem < self.m:
                    raise exc.OutOfBoundsError("key is out of bounds")
            else:
                if not 0 <= elem < self.n:
                    raise exc.OutOfBoundsError("key is out of bounds")
        # subset matrix
        data = [[self[r, c] for c in cols] for r in rows]
        return Matrix(len(data), len(cols), data)

    def transpose(self):
        """Transpose this matrix and return the result."""
        data = [list(col) for col in zip(*self.data)]
        return Matrix(self.n, self.m, data)

    def is_symmetric(self):
        """Return True if and only if this matrix is symmetric."""
        return self == self.transpose()

    def is_skew_symmetric(self):
        """Return True if and only if this matrix is skew-symmetric."""
        return self == -self.transpose()

    def toeplitz_decomposition(self):
        """Apply the Toeplitz decomposition to this matrix.

        Decompose this matrix into the sum of a symmetric and skew-symmetric
        matrix, returning the result as a tuple.
        """
        if self.m != self.n:
            raise exc.DecompositionError("non-square matrices do not have a " +
                                         "a Toeplitz decomposition")
        sym = (self + self.transpose()) * .5
        skew = (self - self.transpose()) * .5
        return sym, skew

    def row_reduce(self):
        """Return the row-reduced form of this matrix."""
        res = self.row_echelon()
        for i in range(1, res.m):
            for j in range(res.n):
                if res[i, j] == 1:
                    for k in range(i):
                        constant = res[k, j]
                        res.data[k] = [elem_k - elem_i * constant
                                       for elem_i, elem_k in
                                       zip(res.data[i], res.data[k])]
                    break
        return res

    def row_echelon(self):
        """Return the row-echelon form of this matrix."""
        # TODO: This can be refactored for better efficiency
        if all([all([self[i, j] == 0 for j in range(self.n)])
                for i in range(self.m)]):
            return Matrix.makeZero(self.m, self.n)
        res = deepcopy(self)
        i, j = 0, 0
        while i < res.m and j < res.n:
            # Use R2 to make pivot non-zero
            if res[i, j] == 0:
                found_non_zero = False
                for k in range(i, res.m):
                    if res[k, j] != 0:
                        found_non_zero = True
                        break
                if not found_non_zero:
                    j += 1
                    continue
                res.data[i], res.data[k] = res.data[k], res.data[i]
            # Use R3 to make pivot one
            if res[i, j] != 1:
                res.data[i] = [elem / res[i, j] for elem in res.data[i]]
            # Use R1 to eliminate entries below the pivot
            for k in range(i + 1, res.m):
                if res[k, j] != 0:
                    constant = res[k, j] / res[i, j]
                    res.data[k] = [elem_k - elem_i * constant
                                   for elem_i, elem_k in
                                   zip(res.data[i], res.data[k])]
            i, j = i + 1, j + 1
        return res

    @property
    def determinant(self):
        """Calculate the determinant of a square matrix.

        This method currently implements the Laplace expansion
        """
        if self.m != self.n:
            raise exc.LinearAlgebraError("cannot calculate the determinant of"
                                         "a non-square matrix")
        if self.m == 1:
            return self[0, 0]
        # TODO: can we choose a better row/column to improve efficiency
        return sum([self[0, j] * (-1 if j % 2 else 1) *
                    self.subset([i for i in range(1, self.m)],
                    [k for k in range(self.n) if k != j]).determinant
                   for j in range(self.n)])

    def invert(self):
        """Calculate the inverse of a non-singular matrix.

        This method currently implements Gaussian elimination
        """
        if self.m != self.n:
            raise exc.LinearAlgebraError("cannot invert a non-square matrix")
        if self.determinant == 0:
            raise exc.LinearAlgebraError("cannot invert a singular matrix")
        # TODO: implement block matrices in their own method
        block_rows = [r1 + r2 for r1, r2 in
                      zip(self.data, Matrix.makeIdentity(self.m).data)]
        inverse_block = Matrix.fromRows(block_rows).row_reduce()
        return inverse_block.subset([i for i in range(self.m)],
                                    [j + self.n for j in range(self.n)])

    @property
    def inverse(self):
        """Calculate the inverse of an invertable matrix as a property."""
        return self.invert()

    @classmethod
    def makeRandom(cls, m, n, min=0, max=1):
        """Create random matrix.

        Make a random matrix of dimension m by n with elements chosen
        independently and uniformly from the interval (min, max).
        """
        Matrix.validate_dimensions(m, n)
        data = [[randrange(min, max) for j in range(n)] for i in range(m)]
        return Matrix(m, n, data)

    @classmethod
    def makeZero(cls, m, n):
        """Make a zero matrix of dimension m by n."""
        Matrix.validate_dimensions(m, n)
        data = [[0 for j in range(n)] for i in range(m)]
        return Matrix(m, n, data)

    @classmethod
    def makeIdentity(cls, m):
        """Make an identity matrix of dimension m by m."""
        Matrix.validate_dimensions(m, m)
        data = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
        return Matrix(m, m, data)

    @classmethod
    def fromRows(cls, data):
        """Make a matrix from a list of data."""
        m = len(data)
        n = len(data[0])
        # check that list of data is valid
        if any([len(row) != n for row in data[1:]]):
            raise ValueError("inconsistent row lengths")
        return Matrix(m, n, data)

    @classmethod
    def fromList(cls, elems, **kwargs):
        """Make matrix from list.

        Make a matrix from a list of elements, filling along data,
        when given at least one dimension of the matrix.
        """
        if not ('m' in kwargs or 'n' in kwargs):
            raise ValueError("at least one of m and n must be specified")
        m = kwargs.get('m')
        n = kwargs.get('n')
        num_elems = len(elems)
        if m is None:
            m = num_elems // n
        elif n is None:
            n = num_elems // m
        elif m * n != num_elems:
            raise ValueError("dimension does not match number of elements in"
                             "list")

        data = [elems[i * n: i * (n + 1)] for i in range(m)]
        return Matrix(m, n, data)

    @staticmethod
    def is_numeric(obj):
        """Check if a given object is of a numeric type.

        Note that since bool inherits from int, that this will accept
        Boolean values
        """
        return isinstance(obj, (int, float, complex))

    @staticmethod
    def validate_dimensions(m, n):
        """Check whether a pair of matrix dimensions are valid."""
        if not (isinstance(m, int) and isinstance(n, int)):
            raise TypeError("dimensions must be integral")
        if m <= 0 or n <= 0:
            raise ValueError("dimensions must be positive")
