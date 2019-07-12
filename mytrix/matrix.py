class Matrix:
    """A simple matrix class containing attributes common to all
    matrices"""

    def __init__(self, m: int, n: int) -> None:
        if m < 0 or n < 0:
            raise ValueError("dimensions must be non-negative")
        self.m = m
        self.n = n
