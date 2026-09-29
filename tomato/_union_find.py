import numpy as np


def find(r: int, parent: dict = None | np.ndarray) -> int:
    while parent[r] != r:
        parent[r] = parent[parent[r]]
        r = parent[r]
    return r
