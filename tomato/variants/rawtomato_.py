from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from tomato._knn_utils import build_knn_graph
from tomato.metric import check_metric
from tomato.core_ import ToMAToCore
from tomato._graph_utils import merge_components
from tomato._numpy_utils import ordered_unique


class RaWToMATo(ToMAToCore):
    """
    Topological Mode Analysis Tool with Random Walk diffusion weights.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for kNN graph.
    metric : str
        Distance metric for kNN graph.
    tau : int
        Time scale for diffusion graph construction.
    sig : float or None
        Significance level for exponential outlier detection (e.g., 0.01 for 1%).
        If provided, used to automatically determine persistence threshold.
    persistence_threshold : float or None
        If provided, used directly as the default persistence threshold.
    noise_aware: bool
        Detect noise points and assign a label of -1.
    use_approximate_knn : bool
        If True, use NNDescent for approximate kNN (faster on large data).
    n_clusters : int or None
        Number of clusters to maintain (overrides threshold if needed).
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

    LARGE_DATA_SIZE = 200_000

    def __init__(
        self,
        n_neighbors: int = 25,
        metric: str = "euclidean",
        tau: int = 5,
        sig: Optional[float] = None,
        persistence_threshold: Optional[float] = None,
        noise_aware: bool = False,
        use_approximate_knn: bool = False,
        use_symmetric_knn: bool = False,
        n_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        enforce_connected: bool = False,
        drop_duplicates: bool = False,
        random_state: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        if sig is not None and persistence_threshold is not None:
            raise ValueError("Specify only one of sig or persistence_threshold.")
        super().__init__(
            sig=sig,
            persistence_threshold=persistence_threshold,
            noise_aware=noise_aware,
            n_clusters=n_clusters,
            min_clusters=min_clusters,
            ascending=True,
            verbose=verbose
        )
        self.n_neighbors = n_neighbors
        self.metric = check_metric(metric)
        self.tau = tau
        self.use_approximate_knn = use_approximate_knn
        self.use_symmetric_knn = use_symmetric_knn
        self.enforce_connected = enforce_connected
        self.drop_duplicates = drop_duplicates
        self.random_state = random_state
        return

    def _stationary_distribution(self, A: csr_matrix) -> np.ndarray:
        """
        Compute the stationary distribution of a graph adjacency matrix.

        Parameters
        ----------
        A : csr_matrix
            Sparse adjacency matrix.

        Returns
        -------
        ndarray
            Stationary distribution vector.
        """
        deg = np.asarray(A.sum(axis=1)).ravel()
        return deg / deg.sum()

    def _compute_total_displacement(self, A: csr_matrix, tau: int) -> csr_matrix:
        """
        Compute total displacement energy over tau steps for a diffusion graph.

        Parameters
        ----------
        A : csr_matrix
            Sparse adjacency/weight matrix.
        tau : int
            Number of diffusion steps to accumulate.

        Returns
        -------
        csr_matrix
            Symmetric energy matrix.
        """
        A = A.tocsr()

        row_sums = np.asarray(A.sum(axis=1)).ravel()
        P = csr_matrix(A.multiply(1.0 / row_sums[:, None]))

        psi = self._stationary_distribution(A)
        mask = (A != 0).astype(float)
        energy = csr_matrix(A.shape, dtype=float)
        Pt = P.copy()

        for _ in range(1, tau + 1):
            Pt_masked = Pt.multiply(mask)
            Pt_masked.data -= psi[Pt_masked.indices]
            energy += Pt_masked + Pt_masked.T
            Pt = Pt_masked.dot(P)

        energy.eliminate_zeros()

        return energy

    def fit(self, X: np.ndarray) -> "RaWToMATo":
        """
        Fit the RaWToMATo model to data X.

        Builds a diffusion-based energy graph from kNN, then applies
        ToMATo clustering.

        Parameters
        ----------
        X : ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        RaWToMATo
            The fitted estimator (self).
        """
        self._raw_data = X

        # Build diffusion graph
        if self.verbose:
            print(f"[RaWToMATo] Building kNN graph (k={self.n_neighbors}, metric={self.metric})...")

        if self.drop_duplicates:
            X, self.duplicate_index_, self._duplicate_inverse = ordered_unique(X)
        
        knn_graph = build_knn_graph(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            use_approximate_knn=self.use_approximate_knn,
            distance_weights=False,
            random_state=self.random_state,
        )

        if self.enforce_connected:
            if self.verbose:
                print(f"[MetricToMATo] Enforcing connectivity...")
            knn_graph = merge_components(
                knn_graph, X, metric=self.metric, random_state=self.random_state
            )
            if self.verbose:
                print(
                    f"[MetricToMATo] Graph connectivity enforced: {knn_graph.nnz} edges"
                )
        
        if self.verbose:
            print(f"[RaWToMATo] Computing random walk energy (tau={self.tau})...")

        A = self._compute_total_displacement(knn_graph, tau=self.tau)
        degree = self._degrees(A)

        super().fit(A, degree)

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
            Data matrix.

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
        pred_data = self._raw_data
        
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

    X -= X.min(axis=0)
    X /= X.max()

    X -= X.mean(axis=0) - np.array([0.5, 0.5])

    """
    2. Fit RaWToMATo
    """
    clf = RaWToMATo(n_neighbors=50, n_clusters=7, noise_aware=True, random_state=42, verbose=True)
    clf.fit(X)

    """
    3. Visualize
    """
    vis = ToMAToVisualization(clf, X)

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
    axs[0, 0].set_title("Original Data")
    axs[0, 0].set_axis_off()

    vis.plot_graph(alpha=0.5, ax=axs[0, 1], cmap="coolwarm")
    axs[0, 1].set_title("Diffusion Graph")

    vis.plot_gradient(alpha=0.5, ax=axs[0, 2], cmap="coolwarm")
    axs[0, 2].set_title("Graph Gradients")

    vis.plot_mst(alpha=0.5, ax=axs[1, 0], cmap="coolwarm")
    axs[1, 0].set_title("Merge Tree")

    vis.plot_persistence_diagram(ax=axs[1, 1])
    axs[1, 1].set_title("Persistence Diagram")

    vis.plot_labels(alpha=0.5, ax=axs[1, 2])
    axs[1, 2].set_title("Cluster Assignments")

    plt.show()
