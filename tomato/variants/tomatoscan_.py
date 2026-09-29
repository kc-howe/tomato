from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from tomato.core_ import ToMAToCore
from tomato._knn_utils import (
    build_knn_graph,
    compute_core_distances,
)
from tomato.metric import check_metric
from tomato._graph_utils import merge_components
from tomato._numpy_utils import ordered_unique


class ToMAToSCAN(ToMAToCore):
    """
    Topological Mode Analysis Tool with HDBSCAN-style mutual reachability distance.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for kNN graph.
    metric : str
        Distance metric for kNN graph.
    sig : float or None
        Significance level for exponential outlier detection (e.g., 0.01 for 1%).
        If provided, used to automatically determine persistence threshold.
    persistence_threshold : float or None
        If provided, used directly as the default persistence threshold.
    noise_aware : bool
        Detect noise points and assign a label of -1.
    use_approximate_knn : bool
        If True, use NNDescent for approximate kNN (faster on large data).
    n_clusters : int or None
        Minimum number of clusters to maintain (overrides threshold if needed).
    min_clusters: int or None
        Minimum number of clusters to maintain (overrides threshold if needed).
    enforce_connected : bool
        Add approximately minimal spanning edges to KNN graph to ensure connectivity.
    drop_duplicates : bool
        Remove duplicates from input data prior to KNN graph construction. Labels
        and assignment probabilities are re-duplicated to match input data.

        Intermediate data structures such as mode indices may not match indexing
        of input data. Use `duplicate_index_` attribute for exact indexing.
    random_state : int or None
            Random seed for reproducibility.
    verbose : bool
        Print debugging statements.
    """

    def __init__(
        self,
        n_neighbors: int = 25,
        metric: str = "euclidean",
        sig: Optional[float] = None,
        persistence_threshold: Optional[float] = None,
        noise_aware: bool = False,
        use_approximate_knn: bool = False,
        n_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        enforce_connected: bool = False,
        drop_duplicates: bool = False,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            sig=sig,
            persistence_threshold=persistence_threshold,
            noise_aware=noise_aware,
            n_clusters=n_clusters,
            min_clusters=min_clusters,
            ascending=False,
            use_edge_gradients=True,
            verbose=verbose,
        )
        self.n_neighbors = n_neighbors
        self.metric = check_metric(metric)
        self.use_approximate_knn = use_approximate_knn
        self.enforce_connected = enforce_connected
        self.drop_duplicates = drop_duplicates
        self.random_state = random_state
        return

    def _compute_mutual_reachability_distance(
        self, knn_graph: csr_matrix, core_distances: np.ndarray
    ) -> csr_matrix:
        """
        Compute a symmetric mutual-reachability distance graph.

        Every edge present in the input kNN graph is retained, including
        zero-distance edges produced by duplicate observations.

        Parameters
        ----------
        knn_graph : csr_matrix
            Sparse k-nearest-neighbors distance matrix.
        core_distances : ndarray
            Core distance for each point.

        Returns
        -------
        csr_matrix
            Symmetric mutual-reachability distance graph.
        """
        graph = knn_graph.tocoo()

        rows = graph.row
        cols = graph.col
        distances = graph.data

        rows = np.concatenate([rows, cols])
        cols = np.concatenate([cols, rows[: len(cols)]])
        distances = np.concatenate([distances, distances])

        mrd = np.maximum.reduce(
            [
                core_distances[rows],
                core_distances[cols],
                distances,
            ]
        )

        mrd_matrix = csr_matrix(
            (mrd, (rows, cols)),
            shape=knn_graph.shape,
        )

        return mrd_matrix

    def fit(self, X: np.ndarray) -> "ToMAToSCAN":
        """
        Fit the ToMAToSCAN clustering model to data.

        Parameters
        ----------
        X : ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        ToMAToSCAN
            The fitted estimator.
        """
        self._raw_data = X

        if self.drop_duplicates:
            X, self.duplicate_index_, self._duplicate_inverse = ordered_unique(X)

        if self.verbose:
            print(
                f"[ToMAToSCAN] Building kNN graph (k={self.n_neighbors}, metric={self.metric})..."
            )

        knn_graph = build_knn_graph(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            use_approximate_knn=self.use_approximate_knn,
            distance_weights=True,
            random_state=self.random_state,
        )

        if self.verbose:
            print(
                f"[ToMAToSCAN] kNN graph built: {knn_graph.nnz} edges, {knn_graph.shape[0]} nodes"
            )

        core_distances = compute_core_distances(knn_graph)

        if self.enforce_connected:
            if self.verbose:
                print(f"[ToMAToSCAN] Enforcing connectivity...")
            knn_graph = merge_components(
                knn_graph, X, metric=self.metric, random_state=self.random_state
            )
            if self.verbose:
                print(
                    f"[ToMAToSCAN] Graph connectivity enforced: {knn_graph.nnz} edges"
                )

        self.core_distances_ = core_distances
        self.knn_graph_ = knn_graph

        if self.verbose:
            print(f"[ToMAToSCAN] Computing mutual reachability distances...")

        A = self._compute_mutual_reachability_distance(knn_graph, self.core_distances_)

        super().fit(A, core_distances)

        if self.drop_duplicates:
            self.labels_ = self.labels_[self._duplicate_inverse]
            self.probabilities_ = self.probabilities_[self._duplicate_inverse]

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        X : ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Cluster labels for each sample.
        """
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Predict cluster labels for unseen data.

        Parameters
        ----------
        X : ndarray
            Data matrix.

        Returns
        -------
        ndarray
            Cluster labels for each sample.
        """
        if y is None:
            y = self.labels_
        
        incidence_graph = build_knn_graph(
            self._raw_data[y > -1],
            n_neighbors=1,
            metric=self.metric,
            use_approximate_knn=self.use_approximate_knn,
            distance_weights=True,
            query=X,
            random_state=self.random_state,
        )
        label_indices = incidence_graph.indices[incidence_graph.indptr[:-1]]
        pred_labels = y[y > -1][label_indices]
        return pred_labels


if __name__ == "__main__":

    from sklearn.datasets import make_blobs, make_moons
    from matplotlib import pyplot as plt

    from tomato.visualization.plotting import ToMAToVisualization

    """
    1. Generate Data
    """
    rng = np.random.RandomState(42)

    X0, y = make_blobs(n_samples=1_000, centers=2, cluster_std=1.0, random_state=42)
    X1, y = make_blobs(n_samples=1_000, centers=5, cluster_std=1.0, random_state=42)
    X2, y = make_moons(n_samples=1_000, noise=0.1, random_state=42)
    X2 *= 5
    X2 -= np.array([5, 5])

    X = np.vstack([X0, X1, X2])

    noise = rng.random((300, 2))

    X -= X.min(axis=0)
    X /= X.max()

    X -= X.mean(axis=0) - np.array([0.5, 0.5])

    X = np.vstack([X, noise])

    y = np.full(X.shape[0], True)
    y[-noise.shape[0] :] = False

    """
    2. Fit ToMAToSCAN
    """
    clf = ToMAToSCAN(n_clusters=7, noise_aware=True, random_state=42, verbose=True)
    clf.fit(X)

    """
    3. Visualize
    """
    vis = ToMAToVisualization(clf, X)

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
    axs[0, 0].set_title("Original Data")
    axs[0, 0].set_axis_off()

    vis.plot_graph(alpha=0.5, ax=axs[0, 1], cmap="coolwarm_r")
    axs[0, 1].set_title("MRD Graph")

    vis.plot_gradient(alpha=0.5, ax=axs[0, 2], cmap="coolwarm_r")
    axs[0, 2].set_title("Graph Gradients")

    vis.plot_mst(alpha=0.5, ax=axs[1, 0], cmap="coolwarm_r")
    axs[1, 0].set_title("Merge Tree")

    vis.plot_persistence_diagram(ax=axs[1, 1])
    axs[1, 1].set_title("Persistence Diagram")

    vis.plot_labels(alpha=0.5, ax=axs[1, 2])
    axs[1, 2].set_title("Cluster Assignments")

    plt.show()
