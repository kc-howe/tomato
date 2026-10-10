from typing import Optional

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, find
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import shortest_path

from tomato.cluster_selection import (
    DEFAULT_ALPHA,
    resolve_cluster_selection_method,
    threshold_by_max_jump,
    threshold_by_n_clusters,
    threshold_by_profile_likelihood,
    threshold_by_significance,
    validate_cluster_selection_method,
)
from tomato.graph_persistence import GraphPersistence
from tomato._graph_utils import (
    compute_incidence_gradients,
    get_assignment_parents,
    components_per_label_uf,
    merge_disconnected_components,
)


class ToMAToCore:
    """
    Topological Mode Analysis Tool (ToMAToCore) for clustering based on graph persistence.

    General clustering algorithm that accepts a graph and a function on its vertices.

    Parameters
    ----------
    cluster_selection_method : str or None
        Method used to choose the persistence threshold. One of:
            * "profile_likelihood": profile-likelihood split of the log-ratio
            persistences, accepted only if significant, otherwise the largest raw jump.
            * "max_jump": split at the largest gap by gap statistic.
            * "significance": exponential outlier threshold at significance level sig.
            * "persistence_threshold": use persistence_threshold directly.
            * "n_clusters": use the threshold that yields n_clusters clusters.
        An explicit method is used as given and the other parameters are not
        checked. If None (default), the method is inferred at fit time: n_clusters,
        persistence_threshold or sig select the matching method (in that order of
        precedence, with a warning if more than one is set), and
        "profile_likelihood" is used if none is set.
    sig : float or None
        Significance level for exponential outlier detection (e.g., 0.01 for 1%).
        Required when cluster_selection_method is "significance".
    persistence_threshold : float or None
        Persistence threshold used directly. Required when
        cluster_selection_method is "persistence_threshold".
    noise_aware: bool
        Detect noise points and assign a label of -1.
    n_clusters : int or None
        Number of clusters to maintain. Required when cluster_selection_method
        is "n_clusters".
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
        cluster_selection_method: Optional[str] = None,
        noise_aware: bool = False,
        n_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        ascending: bool = True,
        use_stability: bool = False,
        use_edge_gradients: bool = False,
        verbose: bool = False,
    ) -> None:
        self.sig = sig
        self.persistence_threshold = persistence_threshold
        if cluster_selection_method is not None:
            validate_cluster_selection_method(cluster_selection_method)
        self.cluster_selection_method = cluster_selection_method
        self.noise_aware = noise_aware
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.ascending = ascending
        self.use_stability = use_stability
        self.use_edge_gradients = use_edge_gradients
        self.verbose = verbose
        self._validate_selection_parameters()
        return

    def _validate_selection_parameters(self) -> None:
        """
        Check that the parameter required by cluster_selection_method is set.
        """
        required = {
            "significance": ("sig", self.sig),
            "persistence_threshold": (
                "persistence_threshold",
                self.persistence_threshold,
            ),
            "n_clusters": ("n_clusters", self.n_clusters),
        }
        name, value = required.get(self.cluster_selection_method, (None, True))
        if value is None:
            raise ValueError(
                f"cluster_selection_method='{self.cluster_selection_method}' "
                f"requires {name} to be set."
            )

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
        """
        return threshold_by_significance(self.graph_persistence_.persistences_, sig)

    def get_threshold_by_n_clusters(self, n_clusters: int) -> float:
        """
        Estimate a persistence threshold to yield at least n_clusters.
        """
        gp = self.graph_persistence_
        return threshold_by_n_clusters(
            gp.persistences_, n_clusters, gp.n_critical_points, verbose=self.verbose
        )

    def get_threshold_by_profile_likelihood(self, alpha: float = DEFAULT_ALPHA):
        """
        Profile-likelihood split of log-ratio persistences, accepted if its
        gap p-value is <= alpha; otherwise the largest raw jump.
        Sets split_method_ and split_pvalue_.
        """
        gp = self.graph_persistence_
        if gp.persistences_ is None:
            raise ValueError(
                "Must call fit() before get_threshold_by_profile_likelihood()"
            )
        res = threshold_by_profile_likelihood(gp.persistences_, gp.diagram_, alpha)
        self.split_method_ = res.method
        self.split_pvalue_ = res.pvalue
        return res.threshold

    def get_threshold_by_max_jump(self):
        """
        Split at the largest gap by gap statistic.
        Sets split_method_ and split_pvalue_ (always NaN: the null
        distribution is calibrated for the profile-likelihood split, not the max).
        """
        gp = self.graph_persistence_
        if gp.persistences_ is None:
            raise ValueError("Must call fit() before get_threshold_by_max_jump()")
        res = threshold_by_max_jump(gp.persistences_, gp.diagram_)
        self.split_method_ = res.method
        self.split_pvalue_ = res.pvalue
        return res.threshold

    def _select_threshold(self) -> float:
        """
        Return the persistence threshold given by the resolved selection method.
        """
        methods = {
            "profile_likelihood": self.get_threshold_by_profile_likelihood,
            "max_jump": self.get_threshold_by_max_jump,
            "significance": lambda: self.get_threshold_by_significance(self.sig),
            "persistence_threshold": lambda: self.persistence_threshold,
            "n_clusters": lambda: self.get_threshold_by_n_clusters(self.n_clusters),
        }
        return methods[self.cluster_selection_method_]()

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

        if basin_assignment.min() >= 0:
            present = np.zeros(max_critical + 1, dtype=bool)
            present[basin_assignment] = True
            unique_basins = np.flatnonzero(present)
            remap = np.cumsum(present) - 1
            labels = remap[basin_assignment]
        else:
            unique_basins, labels = np.unique(basin_assignment, return_inverse=True)

        if merge_disconnected:
            labels = self._merge_disconnected_labels(labels)

        if return_critical_points:
            basin_map = {unique_basins[i]: i for i in range(len(unique_basins))}
            inv_basin_map = {v: int(k) for k, v in basin_map.items()}
            return labels, inv_basin_map

        return labels

    def _get_sym_adjacency(self):
        """
        Symmetrized adjacency, computed once per fitted GraphPersistence.
        """
        gp = self.graph_persistence_
        cache = getattr(self, "_sym_adj_cache", None)
        if cache is None or cache[0] is not gp:
            A = gp.adjacency
            A = A.maximum(A.T).tocsr()
            self._sym_adj_cache = (gp, A)
        return self._sym_adj_cache[1]

    def _components_per_label(self, adjacency, labels):
        indptr, indices = adjacency.indptr, adjacency.indices
        lab_row = np.repeat(labels, np.diff(indptr))
        same = (lab_row == labels[indices]) & (lab_row >= 0)
        del lab_row

        sub = csr_matrix((same.astype(np.int8), indices, indptr), shape=adjacency.shape)
        sub.eliminate_zeros()

        _, comp = connected_components(sub, directed=True, connection="strong")

        comp_label = np.full(comp.max() + 1, -1, dtype=np.int64)
        comp_label[comp] = labels
        comp_label = comp_label[comp_label >= 0]
        return np.bincount(comp_label, minlength=labels.max() + 1)

    def _merge_disconnected_labels(self, labels: np.ndarray) -> np.ndarray:
        A = self.graph_persistence_.adjacency.tocsr()
        labels = np.asarray(labels, dtype=np.int64)
        n_comp = components_per_label_uf(
            A.indptr, A.indices, labels, int(labels.max()) + 1
        )
        bad_labels = np.flatnonzero(n_comp > 1)

        if self.verbose:
            if bad_labels.size:
                print(
                    f"[ToMATo] Found {bad_labels.size} labels with disconnected components:"
                )
                for lab in bad_labels:
                    print(f"[ToMATo]   Label {lab}: {n_comp[lab]} components")
                print("[ToMATo] Merging disconnected label components...")
            else:
                print("[ToMATo] All labels are connected")

        if bad_labels.size == 0:
            return labels

        adj = self._get_sym_adjacency()
        new_labels = merge_disconnected_components(
            labels, adj.indptr, adj.indices, bad_labels
        )

        if self.verbose:
            n_relabeled = np.sum(labels != new_labels)
            if n_relabeled > 0:
                print(f"[ToMATo] Relabeled {n_relabeled} points to ensure connectivity")

        return new_labels

    def _assign_default_labels(self):
        """
        Assign default cluster labels using the configured cluster selection method.

        Sets `self.labels_` and `self.modes_` from the threshold chosen by the
        resolved cluster selection method (`self.cluster_selection_method_`).
        """
        self.cluster_selection_method_ = resolve_cluster_selection_method(
            self.cluster_selection_method,
            n_clusters=self.n_clusters,
            persistence_threshold=self.persistence_threshold,
            sig=self.sig,
        )

        if self.all_modes_.shape[0] == 1:
            self.threshold_ = np.inf
            labels, critical_points = np.zeros(
                self.f_.shape[0]
            ), self.all_modes_.astype(int)
        else:
            self.threshold_ = self._select_threshold()
            if self.verbose:
                print(
                    f"[ToMATo] Threshold from {self.cluster_selection_method_}: "
                    f"{self.threshold_:.6f}"
                )
            labels, critical_points = self.get_labels_by_threshold(
                self.threshold_, return_critical_points=True
            )

        if (
            self.min_clusters is not None
            and self.cluster_selection_method_ != "n_clusters"
        ):
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
        labels = self.labels_
        mask = labels >= 0

        if not mask.any():
            return

        lab = labels[mask].astype(np.int64, copy=False)
        present = np.bincount(lab) > 0

        if present.all():
            return

        mapping = np.cumsum(present) - 1
        labels[mask] = mapping[lab]

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

    def assignment_probabilities(self, labels=None):
        if labels is None:
            labels = self.labels_
        f = self.graph_persistence_.f
        valid = labels >= 0
        lab = labels[valid].astype(np.int64, copy=False)
        sums = np.bincount(lab, weights=f[valid])
        counts = np.bincount(lab)
        mode = sums / np.maximum(counts, 1)
        m = mode[lab]
        p = np.zeros(labels.shape, dtype=np.float32)
        if self.ascending:
            p[valid] = np.minimum(f[valid], m) / m
        else:
            p[valid] = m / np.maximum(f[valid], m)
        return p

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