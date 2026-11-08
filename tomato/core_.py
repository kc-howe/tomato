from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from tomato.cluster_selection import (
    DEFAULT_ALPHA,
    resolve_cluster_selection_method,
    threshold_by_max_jump,
    threshold_by_n_clusters,
    threshold_by_profile_likelihood,
    validate_cluster_selection_method,
)
from tomato.graph_persistence import GraphPersistence
from tomato._graph_utils import (
    compute_incidence_gradients,
    get_assignment_parents,
    components_per_label_uf,
    merge_disconnected_components,
    first_entry_per_row,
)


DEFAULT_NOISE_THRESHOLD = 0.9


class ToMAToCore:
    """
    Topological Mode Analysis Tool (ToMAToCore) for clustering based on graph persistence.

    General clustering algorithm that accepts a graph and a function on its vertices.

    Parameters
    ----------
    cluster_selection_method : str or None
        If supplied, explicitly sets the method used to choose the persistence
        threshold. One of:
            * "profile_likelihood": profile-likelihood split of the log-ratio
            persistences, accepted only if significant, otherwise the largest raw jump.
            * "max_jump": split at the largest gap by gap statistic.
            * "persistence_threshold": use persistence_threshold directly.
            * "n_clusters": use the threshold that yields n_clusters clusters.
        An explicitly set method is used as given and the other parameters are not
        checked. If None (default), the method is inferred at fit time: n_clusters
        or persistence_threshold select the matching method (in that order of
        precedence, with a warning if more than one is set), and
        "profile_likelihood" is used if none is set.
    persistence_threshold : float or None
        Persistence threshold used directly. Required when
        cluster_selection_method is "persistence_threshold".
    filter_noise : bool
        If True, points whose assignment probability is below noise_threshold
        are labeled -1 (noise).
    noise_threshold : float
        Assignment probability below which points are labeled as noise.
        Only meaningful when filter_noise is True. Default 0.9.
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
        filter_noise: bool = False,
        noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
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
        self._check_noise_parameters(filter_noise, noise_threshold)
        self.filter_noise = filter_noise
        self.noise_threshold = noise_threshold
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.ascending = ascending
        self.use_stability = use_stability
        self.use_edge_gradients = use_edge_gradients
        self.verbose = verbose
        self._validate_selection_parameters(
            self.cluster_selection_method, self.persistence_threshold, self.n_clusters
        )
        return

    @staticmethod
    def _validate_selection_parameters(
        cluster_selection_method, persistence_threshold, n_clusters
    ) -> None:
        """
        Check that the parameter required by cluster_selection_method is set.
        """
        required = {
            "persistence_threshold": ("persistence_threshold", persistence_threshold),
            "n_clusters": ("n_clusters", n_clusters),
        }
        name, value = required.get(cluster_selection_method, (None, True))
        if value is None:
            raise ValueError(
                f"cluster_selection_method='{cluster_selection_method}' "
                f"requires {name} to be set."
            )

    @staticmethod
    def _check_noise_parameters(filter_noise, noise_threshold) -> None:
        if not filter_noise and noise_threshold != DEFAULT_NOISE_THRESHOLD:
            raise ValueError("noise_threshold has no effect unless filter_noise=True.")

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
        gp = getattr(self, "graph_persistence_", None)
        if gp is None:
            raise ValueError("Must call fit() first")
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
        gp = getattr(self, "graph_persistence_", None)
        if gp is None:
            raise ValueError("Must call fit() first")
        if gp.persistences_ is None:
            raise ValueError("Must call fit() before get_threshold_by_max_jump()")
        res = threshold_by_max_jump(gp.persistences_, gp.diagram_)
        self.split_method_ = res.method
        self.split_pvalue_ = res.pvalue
        return res.threshold

    def _select_threshold(
        self, method: str, persistence_threshold, n_clusters
    ) -> float:
        """
        Return the persistence threshold given by the resolved selection method.
        """
        methods = {
            "profile_likelihood": self.get_threshold_by_profile_likelihood,
            "max_jump": self.get_threshold_by_max_jump,
            "persistence_threshold": lambda: persistence_threshold,
            "n_clusters": lambda: self.get_threshold_by_n_clusters(n_clusters),
        }
        return methods[method]()

    def _labels_by_threshold(
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

        for dead, survivor, _, _ in self.graph_persistence_.merge_tree_:
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
            labels = self._merge_disconnected_labels(labels, unique_basins)

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

    def _merge_disconnected_labels(
        self, labels: np.ndarray, modes=None
    ) -> np.ndarray:
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

        A = self._get_sym_adjacency()

        new_labels = merge_disconnected_components(
            labels, A.indptr, A.indices, bad_labels, modes=modes
        )

        if self.verbose:
            n_relabeled = np.sum(labels != new_labels)
            if n_relabeled > 0:
                print(f"[ToMATo] Relabeled {n_relabeled} points to ensure connectivity")

        return new_labels

    def _assign_labels(
        self,
        sig,
        persistence_threshold,
        cluster_selection_method,
        n_clusters,
        min_clusters,
        noise_threshold,
    ):
        """
        Label assignment logic shared by fit() and get_labels().

        Uses the given parameters rather than the constructor's, and does not
        store anything on the estimator. Returns
        (labels, modes, full_labels, probabilities, method, threshold).
        """
        method = resolve_cluster_selection_method(
            cluster_selection_method,
            n_clusters=n_clusters,
            persistence_threshold=persistence_threshold,
            sig=sig,
        )

        if self.all_modes_.shape[0] == 1:
            threshold = np.inf
            labels = np.zeros(self.f_.shape[0], dtype=np.int64)
            critical_points = {0: int(self.all_modes_[0])}
        else:
            threshold = self._select_threshold(method, persistence_threshold, n_clusters)
            if self.verbose:
                print(f"[ToMATo] Threshold from {method}: {threshold:.6f}")
            labels, critical_points = self._labels_by_threshold(
                threshold, return_critical_points=True
            )

        if min_clusters is not None and method != "n_clusters":
            current_n = len(np.unique(labels[labels >= 0]))
            n_modes = self.all_modes_.shape[0]
            if current_n < min(min_clusters, n_modes):
                if self.verbose:
                    print(
                        f"[ToMATo] Adjusting for min_clusters={min_clusters} (was {current_n})"
                    )
                if min_clusters >= n_modes:
                    threshold = 0.0
                else:
                    threshold = self.get_threshold_by_n_clusters(min_clusters)
                labels, critical_points = self._labels_by_threshold(
                    threshold, return_critical_points=True
                )

        probabilities = self.assignment_probabilities(labels)

        full_labels = labels
        if noise_threshold is not None:
            labels = full_labels.copy()
            labels[probabilities < noise_threshold] = -1

        return labels, critical_points, full_labels, probabilities, method, threshold

    def _assign_default_labels(self):
        """
        Assign default cluster labels using the parameters given to the constructor.

        Sets `self.labels_` and `self.modes_`.
        """
        (
            labels,
            critical_points,
            full_labels,
            probabilities,
            method,
            threshold,
        ) = self._assign_labels(
            sig=self.sig,
            persistence_threshold=self.persistence_threshold,
            cluster_selection_method=self.cluster_selection_method,
            n_clusters=self.n_clusters,
            min_clusters=self.min_clusters,
            noise_threshold=self.noise_threshold if self.filter_noise else None,
        )

        self.cluster_selection_method_ = method
        self.threshold_ = threshold
        self._full_labels = full_labels
        self.labels_ = labels
        self._full_probabilities = probabilities
        self.probabilities_ = probabilities
        self.modes_ = critical_points

        return

    def _standardize_labels(self, labels, modes=None):
        """
        Ensure that non-negative labels are contiguous in [0, 1, 2, ...].

        Modifies `labels` in place and returns the updated `modes` (if given).
        """
        mask = labels >= 0

        if not mask.any():
            return modes

        lab = labels[mask].astype(np.int64, copy=False)
        present = np.bincount(lab) > 0

        if present.all():
            return modes

        mapping = np.cumsum(present) - 1
        labels[mask] = mapping[lab]
        if modes is None:
            return None
        return {
            int(mapping[k]): v
            for k, v in modes.items()
            if k < present.size and present[k]
        }

    def get_labels(
        self,
        sig: Optional[float] = None,
        persistence_threshold: Optional[float] = None,
        cluster_selection_method: Optional[str] = None,
        filter_noise: Optional[bool] = None,
        noise_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return cluster labels without modifying `labels_`.

        Parameters are as in the constructor, but None means "use the fitted
        value". With no arguments, returns the fit labels. With only
        filter_noise and/or noise_threshold, re-filters the fitted clustering
        (a threshold alone implies filter_noise=True). If any selection
        parameter (sig, persistence_threshold, cluster_selection_method,
        n_clusters, min_clusters) is given, selection is re-run; unset
        parameters fall back to the constructor's values, except that the first
        four replace the constructor's selection parameters as a group.
        """
        if getattr(self, "labels_", None) is None:
            raise ValueError("Must call fit() first")

        # Labels are computed on deduplicated data; expand to the input rows.
        inverse = getattr(self, "_duplicate_inverse", None)

        filter_given = filter_noise is not None
        threshold_given = noise_threshold is not None
        if filter_noise is False and threshold_given:
            raise ValueError("noise_threshold has no effect with filter_noise=False.")
        if filter_given or threshold_given:
            use_filter = filter_noise if filter_given else True
            threshold = self.noise_threshold if not threshold_given else noise_threshold
        else:
            use_filter, threshold = self.filter_noise, self.noise_threshold
        noise = threshold if use_filter else None

        selection = (sig, persistence_threshold, cluster_selection_method, n_clusters)
        selection_given = any(v is not None for v in selection)

        if not selection_given and min_clusters is None:
            if not (filter_given or threshold_given):
                return self.labels_
            labels = self._full_labels.copy()
            if noise is not None:
                labels[self._full_probabilities < noise] = -1
            self._standardize_labels(labels)
            return labels if inverse is None else labels[inverse]

        if selection_given:
            if cluster_selection_method is not None:
                validate_cluster_selection_method(cluster_selection_method)
            self._validate_selection_parameters(
                cluster_selection_method, persistence_threshold, n_clusters
            )
        else:
            sig = self.sig
            persistence_threshold = self.persistence_threshold
            cluster_selection_method = self.cluster_selection_method
            n_clusters = self.n_clusters

        labels, modes, *_ = self._assign_labels(
            sig=sig,
            persistence_threshold=persistence_threshold,
            cluster_selection_method=cluster_selection_method,
            n_clusters=n_clusters,
            min_clusters=self.min_clusters if min_clusters is None else min_clusters,
            noise_threshold=noise,
        )
        labels = labels.copy()
        self._standardize_labels(labels, modes)
        return labels if inverse is None else labels[inverse]

    def assignment_probabilities(self, labels=None):
        if labels is None:
            labels = self._full_labels

        labels = np.asarray(labels)
        sign = 1.0 if self.ascending else -1.0
        g = sign * np.asarray(self.graph_persistence_.f, dtype=np.float64)

        valid = labels >= 0
        p = np.zeros(labels.shape, dtype=np.float32)

        if not np.any(valid):
            return p

        lab = labels[valid].astype(np.int64, copy=False)
        x = g[valid]
        n_labels = int(lab.max()) + 1

        counts = np.bincount(lab, minlength=n_labels)
        means = np.bincount(lab, weights=x, minlength=n_labels) / np.maximum(counts, 1)

        deviations = np.abs(x - means[lab])
        scales = np.bincount(lab, weights=deviations, minlength=n_labels) / np.maximum(
            counts, 1
        )

        deficit = np.maximum(means[lab] - x, 0.0)

        scale = scales[lab]
        probs = np.ones_like(deficit)
        np.divide(
            scale,
            scale + deficit,
            out=probs,
            where=(scale + deficit) > 0,
        )

        p[valid] = probs.astype(np.float32)
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
        if temperature < 0.0:
            raise ValueError("Temperature must be non-negative.")
        if labels is None:
            labels = self.labels_

        dup = getattr(self, "duplicate_index_", None)
        is_dedup = dup is not None
        n_original = labels.shape[0]
        if is_dedup:
            labels = labels[dup]

        n = labels.shape[0]
        unique_labels = np.unique(labels[labels != -1])
        n_labels = len(unique_labels)

        if n_labels == 0:
            return np.zeros((n_original, 0), dtype=float)

        if temperature == 0.0:
            probabilities = (
                labels[:, None] == unique_labels[None, :]
            ).astype(float)
        else:
            flat_probs = self.assignment_probabilities(labels)
            cluster_cores = {}
            for label in unique_labels:
                mask = (labels == label) & np.isclose(flat_probs, 1.0)
                core_points = np.where(mask)[0]

                if len(core_points) == 0:
                    members = np.flatnonzero(labels == label)
                    fm = self.f_[members] if self.ascending else -self.f_[members]
                    core_points = [members[np.argmax(fm)]]
                cluster_cores[label] = core_points

            dummy_edges = []
            dummy_idx = n
            for label in unique_labels:
                core_points = cluster_cores[label]
                for p in core_points:
                    dummy_edges.append((dummy_idx, p, 0.0))
                dummy_idx += 1

            graph_coo = self.graph_.tocoo()
            drows, dcols, ddata = zip(*dummy_edges)
            rows = np.concatenate((graph_coo.row, np.asarray(drows)))
            cols = np.concatenate((graph_coo.col, np.asarray(dcols)))
            data = np.concatenate((graph_coo.data, np.asarray(ddata)))
            aug_graph = csr_matrix(
                (data, (rows, cols)), shape=(n + n_labels, n + n_labels)
            )

            dummy_indices = np.arange(n, n + n_labels)
            distances_from_dummies = shortest_path(
                csgraph=aug_graph,
                directed=False,
                indices=dummy_indices,
                return_predecessors=False,
            )
            distances_to_cores = distances_from_dummies[:, :n].T

            d = distances_to_cores
            dmin = d.min(axis=1, keepdims=True)
            reachable = np.isfinite(dmin)
            w = np.exp(
                -(d - np.where(reachable, dmin, 0.0)) / temperature
            )
            w[~reachable[:, 0]] = 1.0
            probabilities = w / w.sum(axis=1, keepdims=True)

        if is_dedup:
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
        self.modes_ = self._standardize_labels(self.labels_, self.modes_)
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
        """Predict labels; -1 indicates the test point is its own local mode."""
        gradients = compute_incidence_gradients(
            incidence_graph,
            f_train=self.f_,
            f_test=f,
            ascending=self.ascending,
            use_edge_gradients=self.use_edge_gradients,
        )
        gradients.eliminate_zeros()

        idx, has = first_entry_per_row(gradients)
        pred = np.full(idx.shape[0], -1, dtype=self.labels_.dtype)
        pred[has] = self.labels_[idx[has]]
        return pred