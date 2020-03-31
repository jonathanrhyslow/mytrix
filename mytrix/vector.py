"""Module for general matrix class."""

from random import randrange
from math import sqrt

import mytrix.exceptions as exc
import mytrix.matrix as mat


class Vector:
    """A class to represent a general matrix."""

    def __init__(self, m, data):
        """Initalise vector dimensions and contents."""
        self.m = m
        self.data = data

    def __add__(self, obj):
        """Add a valid object to this vector and return the result.

        Doesn't modify the current vector. Valid objects include other vectors
        and numeric scalars
        """
        if isinstance(obj, Vector):
            if self.m != obj.m:
                raise exc.ComformabilityError(
                        "vectors must have the same length")
            data = [self[i] + obj[i] for i in range(self.m)]
        elif Vector.is_numeric(obj):
            data = [self[i] + obj for i in range(self.m)]
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj) + " to vector")
        return Vector(self.m, data)

    def __sub__(self, obj):
        """Subtract a valid object from this vector and return the result.

        Doesn't modify the current vector. Valid objects include other vectors
        and numeric scalars
        """
        if isinstance(obj, Vector):
            if self.m != obj.m:
                raise exc.ComformabilityError(
                        "vectors must have the same length")
            data = [self[i] - obj[i] for i in range(self.m)]
        elif Vector.is_numeric(obj):
            data = [self[i] - obj for i in range(self.m)]
        else:
            raise TypeError(
                    "cannot subtract object of type " + type(obj) +
                    " to vector")
        return Vector(self.m, data)

    def __mul__(self, obj):
        """Multiply this vector by a scalar.

        Doesn't modify the current matrix.
        """
        if Vector.is_numeric(obj):
            data = [self[i] * obj for i in range(self.m)]
            return Vector(self.m, data)
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj) + " to matrix")

    def __floordiv__(self, obj):
        """Divide this vector by a scalar.

        Doesn't modify the current vector
        """
        if Vector.is_numeric(obj):
            data = [self[i] // obj for i in range(self.m)]
            return Vector(self.m, data)
        else:
            raise TypeError(
                "cannot add object of type " + type(obj) + " to matrix")

    def __truediv__(self, obj):
        """Divide this vector by a scalar.

        Doesn't modify the current vector
        """
        if Vector.is_numeric(obj):
            data = [self[i] / obj for i in range(self.m)]
            return Vector(self.m, data)
        else:
            raise TypeError(
                "cannot add object of type " + type(obj) + " to matrix")

    def __pos__(self):
        """Unary positive. Included for symmetry only."""
        data = [+self[i] for i in range(self.m)]
        return Vector(self.m, data)

    def __neg__(self):
        """Negate all elements of the vector."""
        data = [-self[i] for i in range(self.m)]
        return Vector(self.m, data)

    def __iadd__(self, obj):
        """Add a vector to this vector, modifying it in the process."""
        # calls __add__
        tmp = self + obj
        self.data = tmp.data
        return self

    def __isub__(self, obj):
        """Subtract a vector from this vector, modifying it in the process."""
        # calls __sub__
        tmp = self - obj
        self.data = tmp.data
        return self

    def __imul__(self, obj):
        """Multiply this vector by a scalar, modifying it in the process."""
        # calls __mul__
        tmp = self * obj
        self.data = tmp.data
        return self

    def __ifloordiv__(self, obj):
        """Divide this vector by a scalar, modifying it in the process."""
        # calls __floordiv__
        tmp = self // obj
        self.data = tmp.data
        return self

    def __itruediv__(self, obj):
        """Divide this vector by a scalar, modifying it in the process."""
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
        return self * obj

    def __getitem__(self, key):
        """Get element in ith position."""
        self.__check_key_validity(key)
        return self.data[key]

    def __setitem__(self, key, val):
        """Set element in ith position."""
        self.__check_key_validity(key)
        self.data[key] = val

    def __check_key_validity(self, key):
        """Validate keys for __getitem__() and __setitem__() methods."""
        if not isinstance(key, int):
            raise TypeError("key must be an integer")
        if not 0 <= key < self.m:
            raise exc.OutOfBoundsError("key is out of bounds")

    def project_onto(self, v):
        """Project this vector onto another."""
        if not isinstance(v, Vector):
            raise TypeError("can only project onto a vector")
        if all([v[i] == 0 for i in range(v.m)]):
            raise exc.LinearAlgebraError("cannot project onto a zero vector")
        return (Vector.dot(self, v) / Vector.dot(v, v)) * v

    def normalise(self):
        """Normalise this vector."""
        return self / self.magnitude

    @property
    def magnitude(self):
        """Calculate the magnitude of this vector"""
        return sqrt(Vector.dot(self, self))

    @classmethod
    def makeRandom(cls, m, min=0, max=1):
        """Create random vector.

        Make a random vector of length m with elements chosen
        independently and uniformly from the interval (min, max).
        """
        Vector.validate_dimension(m)
        data = [randrange(min, max) for i in range(m)]
        return Vector(m, data)

    @classmethod
    def makeZero(cls, m):
        """Make a zero vector of dimension m."""
        Vector.validate_dimension(m)
        data = [0 for i in range(m)]
        return Vector(m, data)

    @classmethod
    def makeOne(cls, m):
        """Make a vector of ones of dimension m."""
        Vector.validate_dimension(m)
        data = [1 for i in range(m)]
        return Vector(m, data)

    @classmethod
    def fromList(cls, elems, **kwargs):
        """Make vector from list."""
        m = kwargs.get('m')
        num_elems = len(elems)
        if m is None:
            m = num_elems
        elif m != num_elems:
            raise ValueError("dimension does not match number of elements in"
                             "list")

        data = [elems[i] for i in range(m)]
        return Vector(m, data)

    @classmethod
    def fromMatrixColumn(cls, mtrx, col):
        """Extract a column of a matrix as a vector."""
        if not isinstance(mtrx, mat.Matrix):
            raise TypeError("can only extract a column from a matrix")
        if not isinstance(col, int):
            raise TypeError("col must be an integer")
        if not 0 <= col < mtrx.n:
            raise exc.OutOfBoundsError("column is out of bounds")
        data = [mtrx[i, col] for i in range(mtrx.m)]
        return Vector(mtrx.m, data)
    
    @staticmethod
    def validate_dimension(m):
        """Check whether a vector dimension is valid."""
        if not isinstance(m, int):
            raise TypeError("dimension must be integral")
        if m <= 0:
            raise ValueError("dimension must be positive")

    @staticmethod
    def is_numeric(obj):
        """Check if a given object is of a numeric type.

        Note that since bool inherits from int, that this will accept
        Boolean values
        """
        return isinstance(obj, (int, float, complex))

    @staticmethod
    def dot(u, v):
        """Calculate the dot product of two vectors"""
        if not (isinstance(u, Vector) and isinstance(v, Vector)):
            raise TypeError("can only dot two vectors")
        if u.m != v.m:
            raise ValueError("vector lengths do not match")
        return sum([e1 * e2 for e1, e2 in zip(u.data, v.data)])

    @staticmethod
    def hadamard(u, v):
        """Calculate the Hadamard product of two vectors"""
        if not (isinstance(u, Vector) and isinstance(v, Vector)):
            raise TypeError("can only Hadamard two vectors")
        if not u.m != v.m:
            raise ValueError("vector lengths do not match")
        data = [e1 * e2 for e1, e2 in zip(u.data, v.data)]
        return Vector(u.m, data)
