import numpy as np


def ordered_unique(arr):
    arr_unique, index, inverse = np.unique(arr, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(index)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))

    arr_ordered = arr_unique[order]
    index_ordered = index[order]
    inverse_ordered = rank[inverse]

    return arr_ordered, index_ordered, inverse_ordered