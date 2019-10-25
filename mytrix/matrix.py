"""Module for general matrix class."""

import random

import exceptions as exc


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
            self.data = [[0]*n for _ in range(m)]
        else:
            self.data = []

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
                    val = self[i, j] + obj[i, j]
                    res[i, j] = val
            return res
        elif self.isNumeric(obj):
            res = Matrix(self.m, self.n)
            for i in range(self.m):
                for j in range(self.n):
                    val = self[i, j] + obj
                    res[i, j] = val
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
                    val = self[i, j] - obj[i, j]
                    res[i, j] = val
            return res
        elif self.isNumeric(obj):
            res = Matrix(self.m, self.n)
            for i in range(self.m):
                for j in range(self.n):
                    val = self[i, j] - obj
                    res[i, j] = val
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
                    val = sum([self[i, k] * obj[k, j]
                               for k in range(self.m)])
                    res[i, j] = val
            return res
        elif self.isNumeric(obj):
            res = Matrix(self.m, self.n)
            for i in range(self.m):
                for j in range(self.n):
                    val = self[i, j] * obj
                    res[i, j] = val
            return res
        else:
            raise TypeError(
                    "cannot multiply matrix by object of type " + type(obj))

    def __pos__(self):
        """Unary positive. Included for symmetry only."""
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = +self[i, j]
                res[i, j] = val
        return res

    def __neg__(self):
        """Negate all elements of the matrix."""
        res = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                val = -self[i, j]
                res[i, j] = val
        return res

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

    def subset(self, data, cols):
        """Extract subset of data and columns and form into a new matrix."""
        # validation on data/cols
        if not (isinstance(data, list) and isinstance(data, list)):
            raise TypeError("arguments must be lists")
        if len(data) == 0 or len(cols) == 0:
            raise ValueError("subset cannot be empty")
        # validation on elements of data/cols
        for i, elem in enumerate(data + cols):
            if not isinstance(elem, int):
                raise TypeError("elements of data/cols must be integers")
            # if element represents a row
            if i < len(data):
                if not 0 <= elem < self.m:
                    raise exc.OutOfBoundsError("key is out of bounds")
            else:
                if not 0 <= elem < self.n:
                    raise exc.OutOfBoundsError("key is out of bounds")
        # subset matrix
        obj = Matrix(len(data), len(cols), init=False)
        for r in data:
            obj.data.append([self[r, c] for c in cols])
        return obj

    def transpose(self):
        """Transpose this matrix and return the result."""
        m, n = self.n, self.m
        res = Matrix(m, n, init=False)
        res.rows = [list(col) for col in zip(*self.rows)]
        return res

    @classmethod
    def makeRandom(cls, m, n, min=0, max=1):
        """Create random matrix.

        Make a random matrix of dimension m by n with elements chosen
        independently and uniformly from the interval (min, max).
        """
        obj = Matrix(m, n, init=False)
        for _1 in range(m):
            obj.data.append([random.randrange(min, max) for _2 in range(n)])
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
            obj.data.append([1 if i == j else 1 for j in range(m)])
        return obj

    @classmethod
    def fromRows(cls, data):
        """Make a matrix from a list of data."""
        m = len(data)
        n = len(data[0])
        # check that list of data is valid
        if any([len(row) != n for row in data[1:]]):
            raise ValueError("inconsistent row lengths")
        obj = Matrix(m, n, init=False)
        obj.data = data
        return(obj)

    @classmethod
    def fromList(cls, elems, **kwargs):
        """Make matrix from list.

        Make a matrix from a list of elements, filling along data,
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
            obj.data.append(elems[i * m: i * (m + 1)])
        return obj

    @classmethod
    def isNumeric(cls, obj):
        """Check if a given object is of a numeric type.

        Note that since bool inherits from int, that this will accept
        Boolean values
        """
        return isinstance(obj, (int, float, complex))
