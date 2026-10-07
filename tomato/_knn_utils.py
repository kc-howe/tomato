from typing import Optional

import numpy as np
from numba import njit, prange
from pynndescent import NNDescent
from scipy.sparse import csr_matrix
from scipy.spatial.ckdtree import cKDTree
from sklearn.neighbors import NearestNeighbors


@njit(parallel=True, cache=True)
def _sort_rows(indices, data, indptr):
    for i in prange(indptr.shape[0] - 1):
        s, e = indptr[i], indptr[i + 1]
        order = np.argsort(indices[s:e])
        indices[s:e] = indices[s:e][order]
        data[s:e] = data[s:e][order]


def compute_core_distances(knn_graph=None) -> np.ndarray:
    """
    Compute core distances for each point.

    Parameters
    ----------
    knn_graph : csr_matrix
        Sparse k-nearest-neighbors distance matrix.

    Returns
    -------
    ndarray
        Core distance for each point.
    """
    core_distances = knn_graph.max(axis=1).toarray().flatten()
    return core_distances


def mutual_reachability_incidence(
    incidence_graph: csr_matrix,
    test_core_dists: np.ndarray,
    train_core_dists: np.ndarray,
) -> csr_matrix:
    """
    Computes mutual reachability distances for a (M x N) incidence graph.

    Parameters
    ----------
    incidence_graph : csr_matrix
        KNN incidence graph for test data, shape (M, N)
    test_core_dists: np.ndarray
        Core distances for test data.
    train_core_dists: np.ndarray
        Core distances for training data.

    Returns
    -------
    mrd_graph : csr_matrix
        Mutual reachability weighted incidence graph, shape (M, N)
    """
    mrd = incidence_graph.copy().astype(np.float64)
    rows, cols = mrd.nonzero()

    mrd.data = np.maximum(
        mrd.data,
        np.maximum(
            test_core_dists[rows],
            train_core_dists[cols],
        ),
    )

    return mrd


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = "euclidean",
    use_approximate_knn: bool = False,
    distance_weights: bool = False,
    query: np.ndarray = None,
    random_state: Optional[int] = None,
) -> csr_matrix:
    """
    Build a k-nearest-neighbors graph from data.

    Parameters
    ----------
    X : ndarray
        Data matrix of shape (n_samples, n_features).
    n_neighbors : int
        Number of nearest neighbors to use.
    metric : str
        Distance metric to use (e.g., 'euclidean', 'manhattan').
    use_approximate_knn : bool
        If True, use NNDescent for approximate kNN (faster on large datasets).
    distance_weights : bool
        If True, include distances as weights in the returned graph.
    query : np.ndarray
        Data matrix to query for nearest neighbors.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    csr_matrix
        Sparse kNN connectivity or distance graph.
    """
    if query is None:
        query_data = X
    else:
        query_data = query

    if use_approximate_knn:
        # Heuristic parameters for NNDescent (from UMAP)
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        nn = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            low_memory=False,
            n_jobs=-1,
            compressed=False,
            random_state=random_state,
        )
        if query is None:
            knn_indices = nn.neighbor_graph[0]
            knn_dists = nn.neighbor_graph[1]
        else:
            knn_indices, knn_dists = nn.query(query_data)
        graph = knn_to_csr(knn_dists, knn_indices, distance_weights, n_cols=X.shape[0])
    elif metric != "euclidean":
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1)
        nn.fit(X)
        knn_dists, knn_indices = nn.kneighbors(query_data)
        if query is None:
            graph = knn_to_csr(
                knn_dists[:, 1:],
                knn_indices[:, 1:],
                distance_weights,
                n_cols=X.shape[0],
            )
        else:
            graph = knn_to_csr(
                knn_dists[:, :-1],
                knn_indices[:, :-1],
                distance_weights,
                n_cols=X.shape[0],
            )
    else:
        tree = cKDTree(X)
        knn_dists, knn_indices = tree.query(query_data, k=n_neighbors + 1, workers=-1)
        if query is None:
            graph = knn_to_csr(
                knn_dists[:, 1:],
                knn_indices[:, 1:],
                distance_weights,
                n_cols=X.shape[0],
            )
        else:
            graph = knn_to_csr(
                knn_dists[:, :-1],
                knn_indices[:, :-1],
                distance_weights,
                n_cols=X.shape[0],
            )

    return graph


@njit(parallel=True, fastmath=True)
def knn_distances(X, knn_indices):
    n, k = knn_indices.shape
    _, d = X.shape
    knn_dists = np.empty((n, k), dtype=np.float64)
    for i in prange(n):
        xi = X[i]
        for j in range(k):
            idx = knn_indices[i, j]
            s = 0.0
            for t in range(d):
                diff = X[idx, t] - xi[t]
                s += diff * diff
            knn_dists[i, j] = np.sqrt(s)
    return knn_dists


def knn_to_csr(
    knn_dists: np.ndarray,
    knn_indices: np.ndarray,
    distance_weights: bool = False,
    n_cols: int = None,
) -> csr_matrix:
    """
    Convert KNN indices and distances to CSR matrix format.

    Parameters
    ----------
    knn_dists : ndarray
        Array of neighbor distances with shape (n_samples, k).
    knn_indices : ndarray
        Array of neighbor indices with shape (n_samples, k).
    distance_weights : bool
        If True use distances as weights, otherwise use 1.0 for each edge.
    n_cols : int
        Number of columns, equal to number of rows in KNN training data.

    Returns
    -------
    csr_matrix
        Sparse kNN matrix of shape (n_samples, n_cols).
    """
    n_samples, k = knn_indices.shape
    if n_cols is None:
        n_cols = n_samples

    nnz = n_samples * k
    idx_dtype = np.int32 if max(nnz, n_cols) < 2**31 else np.int64

    indices = np.ascontiguousarray(knn_indices, dtype=idx_dtype).ravel()

    if nnz and (indices.min() < 0 or indices.max() >= n_cols):
        raise ValueError("knn_indices contain out-of-range column indices")

    indptr = np.arange(0, nnz + 1, k, dtype=idx_dtype)
    if distance_weights:
        data = np.ascontiguousarray(knn_dists, dtype=np.float64).ravel()
    else:
        data = np.ones(nnz, dtype=np.float64)

    A = csr_matrix((data, indices, indptr), shape=(n_samples, n_cols))
    _sort_rows(A.indices, A.data, A.indptr)
    A.has_sorted_indices = True
    A.sum_duplicates()
    return A
