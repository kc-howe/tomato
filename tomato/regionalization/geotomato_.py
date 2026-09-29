from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from tomato.core_ import ToMAToCore
from tomato._numpy_utils import ordered_unique


def hop_balls(graph: csr_matrix, h_max: int) -> Dict[int, csr_matrix]:
    """
    Boolean graph of k-hop balls.
    """
    if h_max < 1:
        raise ValueError("h_max must be >= 1.")

    n = graph.shape[0]
    A1 = (graph.astype(bool) + sp.eye(n, format="csr", dtype=bool)).astype(bool)

    B = A1.copy()
    balls = {1: B.tocsr()}
    for h in range(2, h_max + 1):
        B = B @ A1
        balls[h] = B.tocsr()

    return balls


def local_dtm(
    X: np.ndarray,
    ball: csr_matrix,
    sample_weight: Optional[np.ndarray] = None,
    mass_fraction: float = 0.2,
    min_neighbors: int = 6,
    include_self: bool = True,
) -> np.ndarray:
    """
    Spatially-restricted, (optionally) measure-weighted Distance-To-Measure.

    For each vertex v, DTM is the weighted root-mean-square attribute-space
    distance (in X) from v to its nearest neighbors within v's row of `ball`,
    using enough of them to cover a fixed share of the neighborhood's weight.

    Returns
    -------
    f : np.ndarray, shape (n_samples,)
        f = DTM
    """
    n = X.shape[0]
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)

    if w.shape[0] != n:
        raise ValueError("sample_weight must have length n_samples.")
    if np.any(w <= 0):
        raise ValueError("sample_weight must be strictly positive.")

    ball = ball.tocsr()
    indptr, indices = ball.indptr, ball.indices

    f = np.empty(n)

    for v in range(n):
        idx = indices[indptr[v]:indptr[v + 1]]
        if not include_self:
            idx = idx[idx != v]

        if idx.size == 0:
            raise ValueError(
                f"Point {v} has an empty local neighborhood in `ball` "
                f"(check graph connectivity, n_hops, or include_self)."
            )

        d = np.linalg.norm(X[idx] - X[v], axis=1)
        order = np.argsort(d, kind="stable")
        d, wt = d[order], w[idx][order]

        cw = np.cumsum(wt)
        k_idx = min(min_neighbors, len(d)) - 1
        budget = max(mass_fraction * cw[-1], cw[k_idx])

        j = min(int(np.searchsorted(cw, budget)), len(d) - 1)

        used = wt[: j + 1].copy()
        used[j] -= cw[j] - budget

        f[v] = np.sqrt((used * d[: j + 1] ** 2).sum() / budget)

    return f


class GeoToMATo(ToMAToCore):
    """
    ToMATo clustering for geographic regionalization.

    Takes a user-supplied contiguity graph and attribute data, and
    clusters using a spatially-restricted Distance-To-Measure (DTM)
    vertex function.

    Parameters
    ----------
    n_hops : int
        Number of hops (h) defining each vertex's local neighborhood in
        `graph`. Larger n_hops gives a smoother, more regional landscape
        (fewer, larger clusters); smaller n_hops responds to finer local
        texture (more, smaller clusters).
    mass_fraction : float
        Fraction (m) of each local neighborhood's total mass to average
        over when computing DTM_m.
    min_neighbors : int
        Minimum number of neighbors' mass to guarantee in the DTM
        average regardless of mass_fraction. Protects small/sparse
        neighborhoods near graph boundaries or islands.
    include_self : bool
        Whether each vertex's own (zero-distance) entry is included in
        its local measure.
    sample_weight : np.ndarray or None
        Optional per-sample weights (e.g. population) defining the
        measure DTM is computed against. Must align with the rows of X
        passed to `fit`. Defaults to uniform weights.
    persistence_threshold : float or None
        Persistence threshold used directly.
    noise_aware : bool
        Detect noise points and assign a label of -1.
    n_clusters : int or None
        Target number of clusters to maintain.
    min_clusters : int or None
        Minimum number of clusters to maintain.
    drop_duplicates : bool
        Remove duplicate attribute rows before fitting. Not supported
        together with sample_weight (duplicate rows may carry different
        weights that would need explicit aggregation).
    verbose : bool
        Print debugging statements.
    """

    def __init__(
        self,
        n_hops: int = 3,
        mass_fraction: float = 0.5,
        min_neighbors: int = 6,
        include_self: bool = True,
        sample_weight: Optional[np.ndarray] = None,
        persistence_threshold: Optional[float] = None,
        noise_aware: bool = False,
        n_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        drop_duplicates: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            sig=None,
            persistence_threshold=persistence_threshold,
            noise_aware=noise_aware,
            n_clusters=n_clusters,
            min_clusters=min_clusters,
            ascending=False,
            use_edge_gradients=False,
            verbose=verbose,
        )

        if n_hops < 1:
            raise ValueError("n_hops must be >= 1.")

        self.n_hops = n_hops
        self.mass_fraction = mass_fraction
        self.min_neighbors = min_neighbors
        self.include_self = include_self
        self.sample_weight = sample_weight
        self.drop_duplicates = drop_duplicates

    @staticmethod
    def _validate_graph(
        graph: csr_matrix,
        n_samples: int,
    ) -> csr_matrix:
        graph = graph.tocsr()

        if graph.shape != (n_samples, n_samples):
            raise ValueError(
                f"graph must have shape ({n_samples}, {n_samples}), "
                f"got {graph.shape}."
            )

        return graph

    def _compute_vertex_function(
        self,
        graph: csr_matrix,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.verbose:
            print(f"[GeoToMATo] Building {self.n_hops}-hop balls...")

        balls = hop_balls(graph, self.n_hops)
        ball = balls[self.n_hops]
        self.hop_ball_ = ball

        if self.verbose:
            sizes = np.diff(ball.indptr)
            print(
                f"[GeoToMATo] Hop-ball sizes: min={sizes.min()}, "
                f"mean={sizes.mean():.1f}, max={sizes.max()}"
            )
            print("[GeoToMATo] Computing local DTM...")

        f = local_dtm(
            X,
            ball,
            sample_weight=self.sample_weight,
            mass_fraction=self.mass_fraction,
            min_neighbors=self.min_neighbors,
            include_self=self.include_self,
        )

        self.dtm_ = f
        self.vertex_function_ = f

        return self.vertex_function_

    def fit(self, graph: csr_matrix, X: np.ndarray) -> "GeoToMATo":
        """
        Fit GeoToMATo clustering.

        Parameters
        ----------
        graph : csr_matrix, shape (n_samples, n_samples)
            Contiguity graph (e.g. Queen adjacency). Defines both which
            vertices may be merged (every output cluster is guaranteed
            connected in this graph) and, via `n_hops`, the local
            neighborhoods used to compute the DTM vertex function.
        X : np.ndarray, shape (n_samples, n_features)
            Attribute data for each vertex (e.g. standardized / ILR-
            transformed demographic variables).
        """
        self._raw_data = X

        if self.sample_weight is not None and self.drop_duplicates:
            raise ValueError(
                "drop_duplicates=True is not supported together with "
                "sample_weight."
            )

        if self.drop_duplicates:
            X, self.duplicate_index_, self._duplicate_inverse = ordered_unique(X)

        n_samples = X.shape[0]

        graph = self._validate_graph(graph, n_samples)
        self.graph_ = graph

        if self.verbose:
            print(f"[GeoToMATo] Graph: {graph.nnz} edges, {graph.shape[0]} nodes")

        self._compute_vertex_function(graph, X)

        super().fit(
            graph,
            self.vertex_function_,
        )

        if self.drop_duplicates:
            self.labels_ = self.labels_[self._duplicate_inverse]
            self.probabilities_ = self.probabilities_[self._duplicate_inverse]
            self.dtm_ = self.dtm_[self._duplicate_inverse]
            self.vertex_function_ = self.vertex_function_[self._duplicate_inverse]

        return self

    def fit_predict(self, graph: csr_matrix, X: np.ndarray) -> np.ndarray:
        self.fit(graph, X)
        return self.labels_