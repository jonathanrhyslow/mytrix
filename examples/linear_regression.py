"""An implementation of linear regression using Mytrix."""

from random import gauss

from mytrix.matrix import Matrix
from mytrix.vector import Vector


def least_squares_estimator(Z, y):
    # compute QR decomposition of Z
    Q, R = Z.qr_decomposition()
    b_hat = R.invert() * Q.transpose() * y
    return b_hat


if __name__ == '__main__':
    # parameters
    n = 100
    p = 5

    Z = Matrix.makeRandom(n, p)
    e = Vector.fromList([gauss(0, 1) for __ in range(n)])
    b = Vector.makeRandom(p)
    y = Z * b + e
