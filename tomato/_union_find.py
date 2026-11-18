import numpy as np


def find(r: int, parent: dict = None | np.ndarray) -> int:
    while parent[r] != r:
        parent[r] = parent[parent[r]]
        r = parent[r]
    return r


class UnionFind:
    """
    Disjoint-set data structure with path compression and union by size.
    """

    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if self.size[a] < self.size[b]:
            a, b = b, a

        self.parent[b] = a
        self.size[a] += self.size[b]

        return True