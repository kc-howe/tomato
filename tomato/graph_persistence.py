import numpy as np
from scipy.sparse import spmatrix

from tomato._graph_utils import (
    compute_gradients,
    compute_saddles,
    build_merge_tree,
    compute_persistence,
    compute_stabilities,
    get_assignment_parents
)


class GraphPersistence:

    def __init__(
        self,
        adjacency: spmatrix,
        f: np.ndarray,
        ascending: bool = True,
        use_stability: bool = False,
        use_edge_gradients: bool = False,
        verbose: bool = False,
    ):
        self.adjacency = adjacency
        self.f = f
        self.ascending = ascending
        self.use_stability = use_stability
        self.use_edge_gradients = use_edge_gradients
        self.verbose = verbose

        self.gradient = None
        self.critical_points = None
        self.basin_assignment = None
        self.saddle = None
        self.saddles = None
        self.merge_tree_ = None
        self.death = None
        self.persistence = None
        self.diagram_ = None
        self.persistences_ = None
        self.cluster_sizes_ = None
        return

    def _compute_gradients(self):
        if self.verbose:
            print("[GraphPersistence] Computing gradients...")
        self.gradient = compute_gradients(
            adjacency=self.adjacency,
            f=self.f,
            ascending=self.ascending,
            use_edge_gradients=self.use_edge_gradients,
        )
        return

    def _find_critical_points(self, deg: int = 0):
        if self.verbose:
            print("[GraphPersistence] Finding critical points...")
        out_f = np.array(self.gradient.getnnz(axis=1)).flatten()
        critical_points = np.where(out_f == deg)[0]
        self.critical_points = critical_points.astype(int)
        self.n_critical_points = critical_points.shape[0]
        if self.verbose:
            print(f"[GraphPersistence] Found {self.n_critical_points} critical points")
        return

    def _assign_critical_points(self):
        if self.verbose:
            print("[GraphPersistence] Assigning critical points to basins...")
        n = self.gradient.shape[0]
        parent = np.arange(n, dtype=np.int64)

        indptr = self.gradient.indptr
        indices = self.gradient.indices
        has_edge = np.diff(indptr) > 0
        rows = np.where(has_edge)[0]
        cols = indices[indptr[rows]]
        parent[rows] = cols

        basin_assignment = get_assignment_parents(parent, np.arange(n, dtype=np.int64))
        self.basin_assignment = basin_assignment

        return

    def _compute_saddles(self):
        if self.verbose:
            print("[GraphPersistence] Computing saddles...")
        self.saddles = compute_saddles(
            self.adjacency, self.f, self.basin_assignment, self.ascending
        )
        if self.verbose:
            print(f"[GraphPersistence] Found {len(self.saddles)} saddle pairs")
        return

    def _build_merge_tree(self):
        if self.verbose:
            print("[GraphPersistence] Building merge tree...")
        saddle, mst, merge_tree, death, cluster_size_map = build_merge_tree(
            self.critical_points,
            self.basin_assignment,
            self.f,
            self.saddles,
            self.ascending,
        )
        self.saddle = saddle
        self.mst = mst
        self.merge_tree_ = merge_tree
        self.death = death
        self.cluster_size_map = cluster_size_map
        if self.verbose:
            print(f"[GraphPersistence] Merge tree built: {len(merge_tree)} merges")
        return

    def _compute_persistence(self):
        if self.verbose:
            print("[GraphPersistence] Computing persistence diagram...")
        if self.use_stability:
            self.persistence, self.diagram_, self.persistences_ = compute_stabilities(
                self.f,
                self.death,
                self.critical_points,
                self.merge_tree_,
                self.basin_assignment,
                self.ascending,
            )
            if self.verbose:
                print("[GraphPersistence] Using stability-weighted persistence")
        else:
            self.persistence, self.diagram_, self.persistences_ = compute_persistence(
                self.f,
                self.death,
                self.critical_points,
                self.ascending,
            )
        return

    def fit(self):
        """
        Run the full pipeline: gradient, critical points, saddles, and persistence.

        Populates attributes like `gradient`, `critical_points`, `basin_assignment`,
        `saddles`, `merge_tree_`, `diagram_`, and `persistences_`.
        """
        if self.verbose:
            print("[GraphPersistence] Starting persistence analysis...")
        self._compute_gradients()
        self._find_critical_points()
        self._assign_critical_points()
        self._compute_saddles()
        self._build_merge_tree()
        self._compute_persistence()
        if self.verbose:
            print(f"[GraphPersistence] Persistence analysis complete")
        return
