from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from tomato.core_ import ToMAToCore
from tomato._knn_utils import build_knn_graph
from tomato.metric import check_metric
from tomato._graph_utils import merge_components
from tomato._numpy_utils import ordered_unique


class ToMATo(ToMAToCore):
    """
    ToMATo clustering with Distance-To-Measure (DTM) as the vertex function.

    DTM(x) is the root-mean-square distance from x to its k nearest
    neighbors. It is a stable (Wasserstein-Lipschitz) density surrogate:
    small DTM => dense region, large DTM => sparse region / outlier.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for the kNN graph and DTM (this is "k" in
        the DTM definition).
    metric : str
        Distance metric for kNN graph construction.
    dtm_weighted : bool
        If True, weight each neighbor's squared distance by the
        corresponding graph edge weight instead of using unweighted
        RMS distance. Only meaningful when a weighted precomputed_graph
        is supplied (e.g. attribute-distance weighted dual graph for
        regionalization). Ignored otherwise.
    persistence_threshold : float or None
        Persistence threshold used directly.
    noise_aware : bool
        Detect noise points and assign a label of -1.
    use_approximate_knn : bool
        If True, use NNDescent for approximate kNN.
    n_clusters : int or None
        Target number of clusters to maintain.
    min_clusters : int or None
        Minimum number of clusters to maintain.
    enforce_connected : bool
        Add approximately minimal spanning edges to ensure connectivity.
        NOTE: Does not apply to precomputed graphs.
    drop_duplicates : bool
        Remove duplicates from input data before graph construction.
    precomputed_graph : csr_matrix or None
        Optional precomputed graph. If provided, it is used directly as
        the graph passed to ToMATo and for computing DTM. For
        regionalization, pass a dual (contiguity) graph whose edge
        weights are attribute-space distances between adjacent regions,
        not raw geographic distances.
    random_state : int or None
        Random seed for reproducibility.
    verbose : bool
        Print debugging statements.
    """

    def __init__(
        self,
        n_neighbors: int = 25,
        metric: str = "euclidean",
        dtm_weighted: bool = False,
        cluster_selection_method: Optional[str] = None,
        persistence_threshold: Optional[float] = None,
        noise_aware: bool = False,
        use_approximate_knn: bool = False,
        n_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        enforce_connected: bool = False,
        drop_duplicates: bool = False,
        precomputed_graph: Optional[csr_matrix] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            cluster_selection_method=cluster_selection_method,
            sig=None,
            persistence_threshold=persistence_threshold,
            noise_aware=noise_aware,
            n_clusters=n_clusters,
            min_clusters=min_clusters,
            ascending=False,
            use_edge_gradients=False,
            verbose=verbose,
        )

        self.n_neighbors = n_neighbors
        self.metric = check_metric(metric)
        self.dtm_weighted = dtm_weighted
        self.use_approximate_knn = use_approximate_knn
        self.enforce_connected = enforce_connected
        self.drop_duplicates = drop_duplicates
        self.precomputed_graph = precomputed_graph
        self.random_state = random_state

    @staticmethod
    def _validate_graph(
        graph: csr_matrix,
        n_samples: int,
    ) -> csr_matrix:
        graph = graph.tocsr().astype(float)

        if graph.shape != (n_samples, n_samples):
            raise ValueError(
                f"precomputed_graph must have shape "
                f"({n_samples}, {n_samples}), got {graph.shape}."
            )

        if np.any(~np.isfinite(graph.data)):
            raise ValueError("precomputed_graph contains non-finite values.")

        if np.any(graph.data < 0):
            raise ValueError("precomputed_graph contains negative distances.")

        return graph

    def _compute_dtm(
        self,
        graph: csr_matrix,
    ) -> np.ndarray:
        """
        Compute DTM(x) = sqrt(mean(d(x, x_i)^2)) over the neighborhoods
        encoded by graph, then convert to a density-like vertex function
        (small DTM / dense => large vertex_function_, matching the sign
        convention ToMATo expects for mode-seeking).
        """
        graph = graph.tocsr()

        counts = np.diff(graph.indptr)

        if np.any(counts == 0):
            bad = np.where(counts == 0)[0][0]
            raise ValueError(f"Point {bad} has no neighbors in the graph.")

        sq_sums = np.add.reduceat(graph.data**2, graph.indptr[:-1])

        dtm = np.sqrt(sq_sums / counts)

        self.dtm_ = dtm
        self.vertex_function_ = dtm

        return self.vertex_function_

    def fit(self, X: np.ndarray) -> "ToMATo":
        """
        Fit the ToMATo clustering model.
        """

        self._raw_data = X

        if self.drop_duplicates:
            if self.precomputed_graph is not None:
                raise ValueError(
                    "drop_duplicates=True cannot be used with " "precomputed_graph."
                )

            X, self.duplicate_index_, self._duplicate_inverse = ordered_unique(X)

        n_samples = X.shape[0]

        if self.precomputed_graph is not None:
            if self.verbose:
                print("[ToMATo] Using precomputed graph...")

            graph = self._validate_graph(
                self.precomputed_graph,
                n_samples,
            )
        else:
            if self.verbose:
                print(
                    f"[ToMATo] Building kNN graph "
                    f"(k={self.n_neighbors}, metric={self.metric})..."
                )

            graph = build_knn_graph(
                X,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                use_approximate_knn=self.use_approximate_knn,
                distance_weights=True,
                random_state=self.random_state,
            )

            if self.enforce_connected:
                if self.verbose:
                    print("[ToMATo] Enforcing connectivity...")

                graph = merge_components(
                    graph,
                    X,
                    metric=self.metric,
                    random_state=self.random_state,
                )

                if self.verbose:
                    print(f"[ToMATo] Graph connectivity enforced: " f"{graph.nnz} edges")

        self.knn_graph_ = graph

        if self.verbose:
            print(
                f"[ToMATo] Graph built: "
                f"{graph.nnz} edges, {graph.shape[0]} nodes"
            )

        self._compute_dtm(graph)

        A = graph.copy()

        self.graph_ = A

        super().fit(
            A,
            self.vertex_function_,
        )

        if self.drop_duplicates:
            self.labels_ = self.labels_[self._duplicate_inverse]
            self.probabilities_ = self.probabilities_[self._duplicate_inverse]
            self.dtm_ = self.dtm_[self._duplicate_inverse]
            self.vertex_function_ = self.vertex_function_[self._duplicate_inverse]

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def predict(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if y is None:
            y = self.labels_

        mask = y > -1

        incidence_graph = build_knn_graph(
            self._raw_data[mask],
            n_neighbors=1,
            metric=self.metric,
            use_approximate_knn=self.use_approximate_knn,
            distance_weights=True,
            query=X,
            random_state=self.random_state,
        )

        label_indices = incidence_graph.indices[incidence_graph.indptr[:-1]]

        return y[mask][label_indices]
