import warnings
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, find
from scipy.sparse.csgraph import shortest_path
from scipy.signal import find_peaks
from scipy import stats

from tomato.graph_persistence import GraphPersistence
from tomato._graph_utils import (
    compute_incidence_gradients,
    get_assignment_parents,
    get_connected_components,
    merge_disconnected_components,
)


class ToMAToCore:
    """
    Topological Mode Analysis Tool (ToMAToCore) for clustering based on graph persistence.

    General clustering algorithm that accepts a graph and a function on its vertices.

    Parameters
    ----------
    sig : float or None
        Significance level for exponential outlier detection (e.g., 0.01 for 1%).
        If provided, used to automatically determine persistence threshold.
    persistence_threshold : float or None
        If provided, used directly as the default persistence threshold.
    noise_aware: bool
        Detect noise points and assign a label of -1.
    n_clusters : int or None
        Number of clusters to maintain (overrides threshold if needed).
    min_clusters: int or None
        Minimum number of clusters to maintain (overrides threshold if needed).
    ascending : bool
        If True, cluster ascending modes; if False, descending modes.
    use_stability : bool
        If True, weight persistence by cluster size (stability).
    use_edge_gradients : bool
        If True, treat graph edge weights as function gradients.
    verbose : bool
        If True, prints logging output to console.
    """

    def __init__(
        self,
        sig: Optional[float] = None,
        persistence_threshold: Optional[float] = None,
        noise_aware: bool = False,
        n_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        ascending: bool = True,
        use_stability: bool = False,
        use_edge_gradients: bool = False,
        verbose: bool = False,
    ) -> None:
        if sig is not None and persistence_threshold is not None:
            raise ValueError("Specify only one of sig or persistence_threshold.")
        self.sig = sig
        self.persistence_threshold = persistence_threshold
        self.noise_aware = noise_aware
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.ascending = ascending
        self.use_stability = use_stability
        self.use_edge_gradients = use_edge_gradients
        self.verbose = verbose
        return

    def _degrees(self, A: csr_matrix) -> np.ndarray:
        """
        Return node degrees from a sparse adjacency matrix.

        Parameters
        ----------
        A : csr_matrix
            Sparse adjacency matrix.

        Returns
        -------
        ndarray
            Degree vector.
        """
        return np.array(A.sum(axis=1)).flatten()

    def get_threshold_by_significance(self, sig: float) -> float:
        """
        Estimate a persistence threshold from an exponential fit.

        Parameters
        ----------
        sig : float
            Significance level (e.g., 0.01 for 1%).

        Returns
        -------
        float
            Estimated threshold value.
        """
        if self.graph_persistence_.persistences_ is None:
            raise ValueError("Must call fit() before get_threshold_by_significance()")

        scale = self.graph_persistence_.persistences_.mean()
        threshold = stats.expon.ppf(1 - sig, scale=scale)

        return float(threshold)

    def get_threshold_by_max_jump(self, z=3.0) -> float:
        """
        Estimate a persistence threshold by finding the largest gap in persistence values
        of clusters that actually get merged (appear as 'dead' in the merge tree).

        Returns
        -------
        float
            Estimated threshold value.
        """
        if self.graph_persistence_.persistences_ is None:
            raise ValueError("Must call fit() before get_threshold_by_max_jump()")

        sorted_persistences = np.sort(self.persistences_)[:-1]
        diffs = np.diff(sorted_persistences)

        if diffs.shape[0] < 2:
            return np.mean(sorted_persistences)

        peaks, properties = find_peaks(diffs, height=0)
        heights = properties["peak_heights"]

        if diffs[-1] > diffs[-2]:
            peaks = np.append(peaks, diffs.shape[0] - 1)
            heights = np.append(heights, diffs[-1])

        heights_z = (heights - diffs.mean()) / diffs.std()
        heights_z_gt = heights_z > z
        if not np.any(heights_z_gt):
            max_jump_index = np.argmax(diffs)
        else:
            max_jump_index = peaks[np.argmax(heights_z_gt)]

        threshold = 0.5 * (
            sorted_persistences[max_jump_index]
            + sorted_persistences[max_jump_index + 1]
        )

        return threshold

    def get_threshold_by_n_clusters(self, n_clusters: int) -> float:
        """
        Estimate a persistence threshold to yield at least n_clusters.

        Parameters
        ----------
        n_clusters : int
            Desired minimum number of clusters.

        Returns
        -------
        float
            Threshold value between two adjacent persistences.
        """
        if self.graph_persistence_.persistences_ is None:
            raise ValueError("Must call fit() before get_threshold_by_n_clusters()")

        n_modes = self.graph_persistence_.n_critical_points
        if n_clusters > n_modes:
            warnings.warn(
                "n_clusters exceeds number of detectable modes. "
                + f"Returning {n_modes} clusters."
            )
            threshold = 0.0
            return threshold

        if n_clusters == self.graph_persistence_.n_critical_points:
            threshold = 0.0
            return threshold

        sorted_persistences = np.flip(np.sort(self.graph_persistence_.persistences_))
        threshold_upper = sorted_persistences[n_clusters - 1]
        threshold_lower = sorted_persistences[n_clusters]
        threshold = 0.5 * (threshold_upper + threshold_lower)

        if self.verbose:
            print(f"[ToMATo] Cluster threshold for {n_clusters} clusters: {threshold}")
            print(
                f"[ToMATo] Clusters above threshold: {(sorted_persistences > threshold).sum()}"
            )

        return threshold

    def get_labels_by_threshold(
        self,
        persistence_threshold: float,
        merge_disconnected: bool = True,
        return_critical_points: bool = False,
    ) -> np.ndarray:
        """
        Generate cluster labels by cutting the merge tree at a persistence threshold.
        """
        max_critical = (
            int(self.graph_persistence_.critical_points.max())
            if len(self.graph_persistence_.critical_points) > 0
            else 0
        )
        parent = np.arange(max_critical + 1, dtype=np.int64)

        for dead, survivor, h, size in self.graph_persistence_.merge_tree_:
            if self.graph_persistence_.persistence[dead] < persistence_threshold:
                parent[dead] = survivor

        basin_assignment = self.graph_persistence_.basin_assignment.copy().astype(
            np.int64
        )
        basin_assignment = get_assignment_parents(parent, basin_assignment)

        unique_basins, labels = np.unique(basin_assignment, return_inverse=True)

        if merge_disconnected:
            labels = self._merge_disconnected_labels(labels)

        if return_critical_points:
            basin_map = {unique_basins[i]: i for i in range(len(unique_basins))}
            inv_basin_map = {v: int(k) for k, v in basin_map.items()}
            return labels, inv_basin_map

        return labels

    def _merge_disconnected_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Ensure that all points sharing a label are connected in the graph.
        When a label would merge into disconnected components, merge smaller
        components into their nearest neighboring clusters (by graph distance).

        Parameters
        ----------
        labels : ndarray
            Initial cluster labels (may contain disconnected points).

        Returns
        -------
        labels_fixed : ndarray
            Cluster labels where all points in a cluster are connected.
        """
        adjacency = self.graph_persistence_.adjacency
        adjacency = adjacency.maximum(adjacency.T)
        unique_labels = np.unique(labels[labels >= 0])

        if self.verbose:
            disconnected_info = {}
            for lab in unique_labels:
                mask = labels == lab
                if mask.sum() <= 1:
                    continue

                comp_id, ncomp = get_connected_components(
                    adjacency.indptr, adjacency.indices, mask
                )
                if ncomp > 1:
                    disconnected_info[lab] = ncomp
            if disconnected_info:
                print(
                    f"[ToMATo] Found {len(disconnected_info)} labels with disconnected components:"
                )
                for lab, n_comps in sorted(disconnected_info.items()):
                    print(f"[ToMATo]   Label {lab}: {n_comps} components")
                print(f"[ToMATo] Merging disconnected label components...")
            else:
                print(f"[ToMATo] All labels are connected")

        new_labels = merge_disconnected_components(
            labels, adjacency.indptr, adjacency.indices, unique_labels
        )

        if self.verbose:
            n_relabeled = np.sum(labels != new_labels)
            if n_relabeled > 0:
                print(f"[ToMATo] Relabeled {n_relabeled} points to ensure connectivity")

        return new_labels

    def _assign_default_labels(self):
        """
        Assign default cluster labels based on significance or threshold.

        Sets `self.labels_` and `self.modes_` according to the
        configured `persistence_threshold`, `sig`, or `n_clusters`.
        """
        if self.all_modes_.shape[0] == 1:
            self.threshold_ = np.inf
            labels, critical_points = np.zeros(
                self.f_.shape[0]
            ), self.all_modes_.astype(int)
        elif self.n_clusters is not None:
            if self.verbose:
                print(f"[ToMATo] Targeting n_clusters={self.n_clusters}")
            self.threshold_ = self.get_threshold_by_n_clusters(self.n_clusters)
            labels, critical_points = self.get_labels_by_threshold(
                self.threshold_, return_critical_points=True
            )
        elif self.persistence_threshold is not None:
            self.threshold_ = self.persistence_threshold
            if self.verbose:
                print(f"[ToMATo] Using persistence threshold: {self.threshold_:.6f}")
            labels, critical_points = self.get_labels_by_threshold(
                self.persistence_threshold, return_critical_points=True
            )
        elif self.sig is not None:
            self.threshold_ = self.get_threshold_by_significance(self.sig)
            if self.verbose:
                print(f"[ToMATo] Significance-based threshold: {self.threshold_:.6f}")
            labels, critical_points = self.get_labels_by_threshold(
                self.threshold_, return_critical_points=True
            )
        else:
            self.threshold_ = self.get_threshold_by_max_jump()
            if self.verbose:
                print(f"[ToMATo] Max-jump threshold: {self.threshold_:.6f}")
            labels, critical_points = self.get_labels_by_threshold(
                self.threshold_, return_critical_points=True
            )

        if self.min_clusters is not None and self.n_clusters is None:
            current_n = len(np.unique(labels[labels >= 0]))
            if current_n < self.min_clusters:
                if self.verbose:
                    print(
                        f"[ToMATo] Adjusting for min_clusters={self.min_clusters} (was {current_n})"
                    )
                self.threshold_ = self.get_threshold_by_n_clusters(self.min_clusters)
                labels, critical_points = self.get_labels_by_threshold(
                    self.threshold_, return_critical_points=True
                )

        probabilities = self.assignment_probabilities(labels)

        self.labels_ = labels
        self.probabilities_ = probabilities
        self.modes_ = critical_points

        if self.noise_aware:
            self.labels_ = self.noise_aware_labels()

        return

    def _standardize_labels(self):
        """
        Ensure that non-negative labels are contiguous in [0, 1, 2, ...].
        """
        if np.all(self.labels_ == -1) or np.all(self.labels_ == 0):
            return

        unique_labels = np.unique(self.labels_[self.labels_ >= 0])
        label_map = np.arange(len(unique_labels), dtype=self.labels_.dtype)

        max_label = unique_labels.max() + 1
        mapping = np.full(max_label, -1, dtype=self.labels_.dtype)
        mapping[unique_labels] = label_map

        mask = self.labels_ >= 0
        self.labels_[mask] = mapping[self.labels_[mask]]

        return

    def noise_aware_labels(
        self,
        labels: np.ndarray = None,
        probabilities: np.ndarray = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Return noise-aware labeling from a prior set of labels and probabilities.

        Parameters
        ----------
        labels : np.ndarray
            Array of cluster labels.
        probabilities : dict
            Array of cluster assignment probabilities.
        threshold : float
            Cluster assignment probability threshold. Points with lower assignment
            probability than threshold will return as noise.

        Returns
        -------
        new_labels : np.ndarray
            New labels with noise entries as -1.
        """
        if labels is None:
            if probabilities is not None:
                raise ValueError("Must supply both labels and label probabilities.")
            labels = self.labels_
            probabilities = self.probabilities_
        elif probabilities is None:
            raise ValueError("Must supply both labels and label probabilities.")

        new_labels = labels.copy()
        is_noise = probabilities < threshold
        new_labels[is_noise] = -1

        return new_labels

    def assignment_probabilities(self, labels: np.ndarray = None) -> np.ndarray:
        """
        Return assignment probabilities for a given set of labels.

        Parameters
        ----------
        labels : np.ndarray
            Array of cluster labels..

        Returns
        -------
        probabilities : np.ndarray
            Label assignment probabilities.
        """
        if labels is None:
            labels = self.labels_

        unique_labels = np.unique(labels)
        f_vals = self.graph_persistence_.f
        probabilities = np.zeros(labels.shape, dtype=np.float32)

        for label in unique_labels:
            if label == -1:
                continue
            label_mask = labels == label
            label_mode = f_vals[label_mask].mean()
            if self.ascending:
                probabilities[label_mask] = (
                    np.minimum(f_vals[label_mask], label_mode) / label_mode
                )
            else:
                probabilities[label_mask] = label_mode / np.maximum(
                    f_vals[label_mask], label_mode
                )

        return probabilities

    def full_assignment_probabilities(
        self, labels: np.ndarray = None, temperature: float = 1.0
    ) -> np.ndarray:
        """
        Compute full assignment probabilities for each sample to each cluster.

        Parameters
        ----------
        labels : np.ndarray
            Array of cluster labels.
        temperature : float
            Temperature parameter for softmax scaling of probabilities.

        Returns
        -------
        probabilities : np.ndarray
            Full assignment probabilities for each sample to each cluster.
        """
        if labels is None:
            labels = self.labels_

        is_deduplicated = hasattr(self, "duplicate_index_")

        if is_deduplicated:
            labels = labels[self.duplicate_index_]

        n = labels.shape[0]
        flat_probs = self.assignment_probabilities(labels)
        unique_labels = [l for l in np.unique(labels) if l != -1]
        n_labels = len(unique_labels)

        if temperature < 0.0:
            raise ValueError("Temperature must be non-negative.")
        if temperature == 0.0:
            unique_labels = np.unique(labels)
            probabilities = (labels[None, :] == unique_labels[:, None]).astype(float)
            return probabilities.T

        # Identify core points for each cluster
        cluster_cores = {}
        for label in unique_labels:
            mask = (labels == label) & np.isclose(flat_probs, 1.0)
            core_points = np.where(mask)[0]

            if len(core_points) == 0:
                core_points = [
                    np.argmax(self.graph_persistence_.f[labels == label])
                    + np.where(labels == label)[0][0]
                ]
            cluster_cores[label] = core_points

        # Build augmented graph with dummy nodes
        dummy_edges = []
        dummy_idx = n
        for label in unique_labels:
            core_points = cluster_cores[label]
            for p in core_points:
                dummy_edges.append((dummy_idx, p, 0.0))
            dummy_idx += 1

        rows, cols, data = find(self.graph_)
        if dummy_edges:
            drows, dcols, ddata = zip(*dummy_edges)
            rows = np.concatenate([rows, drows])
            cols = np.concatenate([cols, dcols])
            data = np.concatenate([data, ddata])

        aug_graph = csr_matrix((data, (rows, cols)), shape=(n + n_labels, n + n_labels))

        # Compute distances from dummy nodes
        dummy_indices = np.arange(n, n + n_labels)
        distances_from_dummies = shortest_path(
            csgraph=aug_graph,
            directed=False,
            indices=dummy_indices,
            return_predecessors=False,
        )

        # Extract distances to original nodes
        distances_to_cores = distances_from_dummies[:, :n].T

        # Softmax probabilities
        probabilities = np.exp(-distances_to_cores / temperature)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        if is_deduplicated:
            probabilities = probabilities[self._duplicate_inverse, :]

        return probabilities

    def fit(self, graph: csr_matrix, f: np.ndarray) -> "ToMAToCore":
        """
        Fit the ToMATo clustering algorithm to a graph and function.

        Parameters
        ----------
        graph : csr_matrix
            Sparse adjacency matrix of the graph.
        f : ndarray
            Function values on the vertices of the graph.

        Returns
        -------
        ToMATo
            The fitted estimator (self).
        """
        self.graph_ = graph
        self.f_ = f

        self.graph_persistence_ = GraphPersistence(
            graph,
            f,
            ascending=self.ascending,
            use_stability=self.use_stability,
            use_edge_gradients=self.use_edge_gradients,
            verbose=self.verbose,
        )
        if self.verbose:
            print("[ToMATo] Computing graph persistence...")
        self.graph_persistence_.fit()
        if self.verbose:
            print(
                f"[ToMATo] Found {self.graph_persistence_.n_critical_points} critical points"
            )

        self.persistences_ = self.graph_persistence_.persistences_
        self.diagram_ = self.graph_persistence_.diagram_
        self.all_modes_ = self.graph_persistence_.critical_points

        # Get default labels
        if self.verbose:
            print("[ToMATo] Assigning default labels...")
        self._assign_default_labels()
        self._standardize_labels()
        if self.verbose:
            n_clusters = len(np.unique(self.labels_[self.labels_ >= 0]))
            n_noise = np.sum(self.labels_ == -1)
            print(
                f"[ToMATo] Label assignment complete: {n_clusters} clusters, {n_noise} noise points"
            )

        return self

    def fit_predict(self, graph: csr_matrix, f: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        graph : csr_matrix
            Sparse adjacency matrix of the graph.
        f : ndarray
            Function values on the vertices of the graph.

        Returns
        -------
        ndarray
            Cluster labels for each sample.
        """
        self.fit(graph, f)
        return self.labels_

    def predict(self, incidence_graph: csr_matrix, f: np.ndarray) -> np.ndarray:

        gradients = compute_incidence_gradients(
            incidence_graph,
            f_train=self.f_,
            f_test=f,
            ascending=self.ascending,
            use_edge_gradients=self.use_edge_gradients,
        )
        gradients.eliminate_zeros()

        label_indices = gradients.indices[gradients.indptr[:-1]]
        pred_labels = self.labels_[label_indices]

        return pred_labels
