from typing import Optional

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import cdist

from tomato._union_find import find


@njit(parallel=True)
def get_connected_components(indptr, indices, mask):
    """
    Return comp_id array (len n) where nodes not in mask have -1,
    and comp_id >= 0 for nodes in mask. Also returns n_components.
    """
    n = mask.shape[0]
    comp_id = -1 * np.ones(n, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    comp = 0
    for i in range(n):
        if not mask[i] or comp_id[i] != -1:
            continue
        top = 0
        stack[top] = i
        top += 1
        comp_id[i] = comp
        while top > 0:
            top -= 1
            node = stack[top]
            start = indptr[node]
            end = indptr[node + 1]
            for ptr in range(start, end):
                nbr = indices[ptr]
                if mask[nbr] and comp_id[nbr] == -1:
                    comp_id[nbr] = comp
                    stack[top] = nbr
                    top += 1
        comp += 1
    return comp_id, comp


@njit
def get_assignment_parents(parent, basin_assignment):
    """
    For each entry in `basin_assignment` return the root parent using
    path compression on `parent`. Negative assignments (e.g. noise) are preserved.
    Both arrays must be integer numpy arrays.
    """
    n = basin_assignment.shape[0]
    out = np.empty(n, dtype=parent.dtype)
    psize = parent.shape[0]
    for i in range(n):
        r = basin_assignment[i]

        if r < 0 or r >= psize:
            out[i] = r
            continue

        root = r
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        out[i] = root
    return out


def compute_gradients(adjacency, f, ascending=True, use_edge_gradients=False):
    """
    Build a gradient graph where each node points to its best neighbor by free.
    """
    adjacency = adjacency.tocoo()
    n = adjacency.shape[0]
    rows, cols, data = adjacency.row, adjacency.col, adjacency.data

    if ascending:
        mask = f[cols] > f[rows]
    else:
        mask = f[cols] < f[rows]

    rows_m, cols_m, data_m = rows[mask], cols[mask], data[mask]

    df = pd.DataFrame({"row": rows_m, "col": cols_m, "f": f[cols_m], "w": data_m})

    sort_col = "w" if use_edge_gradients else "f"

    if ascending:
        idx = df.groupby("row")[sort_col].idxmax()
    else:
        idx = df.groupby("row")[sort_col].idxmin()

    best_edges = df.loc[idx]

    gradient = csr_matrix(
        (
            best_edges["w"].values,
            (best_edges["row"].values, best_edges["col"].values),
        ),
        shape=adjacency.shape,
    )

    return gradient


def compute_incidence_gradients(
    adjacency, f_train, f_test, ascending=True, use_edge_gradients=False
):
    """
    Build a gradient graph where each node points to its best neighbor by free.
    """
    adjacency = adjacency.tocoo()
    rows, cols, data = adjacency.row, adjacency.col, adjacency.data

    if ascending:
        mask = f_train[cols] > f_test[rows]
    else:
        mask = f_train[cols] < f_test[rows]

    rows_m, cols_m, data_m = rows[mask], cols[mask], data[mask]

    df = pd.DataFrame({"row": rows_m, "col": cols_m, "f": f_train[cols_m], "w": data_m})

    sort_col = "w" if use_edge_gradients else "f"

    if ascending:
        idx = df.groupby("row")[sort_col].idxmax()
    else:
        idx = df.groupby("row")[sort_col].idxmin()

    best_edges = df.loc[idx]

    gradient = csr_matrix(
        (
            best_edges["w"].values,
            (best_edges["row"].values, best_edges["col"].values),
        ),
        shape=adjacency.shape,
    )

    return gradient


def compute_saddles(adjacency, f, basin_assignment, ascending=True):
    """
    Collect candidate saddles between critical points based on shared edges.

    Returns
    -------
    dict
        Mapping from critical point pair to merge weight.
    """
    adjacency_csr = adjacency.tocsr()
    n = adjacency_csr.shape[0]
    row_counts = np.diff(adjacency_csr.indptr)
    rows = np.repeat(np.arange(n), row_counts)
    cols = adjacency_csr.indices

    ru = basin_assignment[rows]
    rv = basin_assignment[cols]

    mask = ru != rv
    ru, rv = ru[mask], rv[mask]
    rows, cols = rows[mask], cols[mask]

    f_u = f[rows]
    f_v = f[cols]

    if ascending:
        h = np.minimum(f_u, f_v)
    else:
        h = np.maximum(f_u, f_v)

    critical_point_pairs = np.column_stack([np.minimum(ru, rv), np.maximum(ru, rv)])

    df = pd.DataFrame(
        {"r1": critical_point_pairs[:, 0], "r2": critical_point_pairs[:, 1], "h": h}
    )

    if ascending:
        result = df.groupby(["r1", "r2"], sort=False)["h"].max()
    else:
        result = df.groupby(["r1", "r2"], sort=False)["h"].min()

    saddles = {(int(r1), int(r2)): float(val) for (r1, r2), val in result.items()}

    return saddles


def _saddles_to_csr(saddles) -> csr_matrix:
    """
    Convert `saddles` dictionary to a symmetric undirected sparse graph.

    Parameters
    ----------
    saddles : dict
        Dictionary mapping (node1, node2) -> weight (float).

    Returns
    -------
    csr_matrix
        Symmetric sparse adjacency matrix of shape (n_nodes, n_nodes),
        where n_nodes is the largest node index + 1.
    """
    if not saddles:
        return csr_matrix((0, 0))

    rows, cols, data = zip(*[(u, v, w) for (u, v), w in saddles.items()])
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    n_nodes = cols.max() + 1

    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])

    mat = csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n_nodes, n_nodes))
    mat.sort_indices()

    return mat


def build_merge_tree(critical_points, basin_assignment, f, saddles, ascending=True):
    """
    Construct the final merge tree from condensed graph edges.

    Parameters
    ----------
    critical_points : ndarray
        Indices of critical points
    basin_assignment : ndarray
        Basin assignment of each node to critical point
    f : ndarray
        Function values per node
    saddles : dict
        Mapping of (cp1, cp2) to saddle height
    ascending : bool, default=True
        If True, merge tree for ascending case; else descending

    Returns
    -------
    saddle : csr_matrix
        Symmetric saddle matrix
    mst : csr_matrix
        Minimum spanning tree of saddles
    merge_tree : list
        List of merge tuples (dead, survivor, h, merged_size)
    death : dict
        Mapping of critical point to death value (saddle height)
    cluster_size_map : dict
        Final cluster sizes after all merges
    """

    saddle = _saddles_to_csr(saddles)
    if ascending:
        saddle_sym = saddle.maximum(saddle.T)
        saddle_sym.data = -saddle_sym.data
        saddle = saddle_sym

    mst = minimum_spanning_tree(saddle)

    if ascending:
        mst.data = -mst.data

    parent = {r: r for r in critical_points}

    merge_tree = []
    death = dict()

    mst_coo = mst.tocoo()
    edges = [
        (mst_coo.row[i], mst_coo.col[i], mst_coo.data[i])
        for i in range(len(mst_coo.data))
    ]

    if ascending:
        edges.sort(key=lambda x: x[2], reverse=True)
    else:
        edges.sort(key=lambda x: x[2])

    unique_basins, cluster_sizes = np.unique(basin_assignment, return_counts=True)

    cluster_size_map = {
        int(basin): int(size) for basin, size in zip(unique_basins, cluster_sizes)
    }

    birth = {r: f[r] for r in critical_points}

    for r1, r2, h in edges:
        r1_basin = find(r1, parent)
        r2_basin = find(r2, parent)
        if r1_basin == r2_basin:
            continue

        if ascending:
            if birth[r1_basin] >= birth[r2_basin]:
                survivor, dead = r1_basin, r2_basin
            else:
                survivor, dead = r2_basin, r1_basin
            birth[survivor] = max(birth[survivor], birth[dead])
        else:
            if birth[r1_basin] <= birth[r2_basin]:
                survivor, dead = r1_basin, r2_basin
            else:
                survivor, dead = r2_basin, r1_basin
            birth[survivor] = min(birth[survivor], birth[dead])

        dead_size = cluster_size_map[dead]
        survivor_size = cluster_size_map[survivor]
        merged_size = dead_size + survivor_size

        cluster_size_map[survivor] = merged_size
        del cluster_size_map[dead]

        parent[dead] = survivor
        death[dead] = h
        merge_tree.append((dead, survivor, h, merged_size))

    return saddle, mst, merge_tree, death, cluster_size_map


def build_merge_tree_old(critical_points, basin_assignment, f, saddles, ascending=True):
    """
    Construct the final merge tree from condensed graph edges.

    Parameters
    ----------
    critical_points : ndarray
        Indices of critical points
    basin_assignment : ndarray
        Basin assignment of each node to critical point
    f : ndarray
        Function values per node
    saddles : dict
        Mapping of (cp1, cp2) to saddle height
    ascending : bool, default=True
        If True, merge tree for ascending case; else descending

    Returns
    -------
    saddle : csr_matrix
        Symmetric saddle matrix
    mst : csr_matrix
        Minimum spanning tree of saddles
    merge_tree : list
        List of merge tuples (dead, survivor, h, merged_size)
    death : dict
        Mapping of critical point to death value (saddle height)
    cluster_size_map : dict
        Final cluster sizes after all merges
    """

    saddle = _saddles_to_csr(saddles)
    if ascending:
        saddle_sym = saddle.maximum(saddle.T)
        saddle_sym.data = -saddle_sym.data
        saddle = saddle_sym

    mst = minimum_spanning_tree(saddle)

    if ascending:
        mst.data = -mst.data

    parent = {r: r for r in critical_points}

    merge_tree = []
    death = dict()

    mst_coo = mst.tocoo()
    edges = [
        (mst_coo.row[i], mst_coo.col[i], mst_coo.data[i])
        for i in range(len(mst_coo.data))
    ]

    if ascending:
        edges.sort(key=lambda x: x[2], reverse=True)
    else:
        edges.sort(key=lambda x: x[2])

    unique_basins, cluster_sizes = np.unique(basin_assignment, return_counts=True)

    cluster_size_map = {
        int(basin): int(size) for basin, size in zip(unique_basins, cluster_sizes)
    }

    birth = {r: f[r] for r in critical_points}

    for r1, r2, h in edges:
        r1_basin = find(r1, parent)
        r2_basin = find(r2, parent)
        if r1_basin == r2_basin:
            continue

        if birth[r1_basin] >= birth[r2_basin]:
            survivor, dead = r1_basin, r2_basin
        else:
            survivor, dead = r2_basin, r1_basin

        dead_size = cluster_size_map[dead]
        survivor_size = cluster_size_map[survivor]
        merged_size = dead_size + survivor_size

        cluster_size_map[survivor] = merged_size
        del cluster_size_map[dead]

        parent[dead] = survivor
        birth[survivor] = max(birth[survivor], birth[dead])
        death[dead] = h
        merge_tree.append((dead, survivor, h, merged_size))

    return saddle, mst, merge_tree, death, cluster_size_map


def compute_persistence(f, death, critical_points, ascending=True):
    """
    Compute a persistence diagram from critical_point degrees and persistence.

    Parameters
    ----------
    f : ndarray
        Function values per node
    death : dict
        Mapping of critical point to death value (saddle height)
    critical_points : ndarray
        Indices of critical points
    ascending : bool, default=True
        If True, compute persistence for ascending case; else descending

    Returns
    -------
    persistence : dict
        Mapping of critical point to persistence value
    diagram : ndarray
        Persistence diagram, shape (n_critical, 2)
    persistences : ndarray
        Persistence values array
    cluster_sizes : ndarray
        Cluster sizes at each critical point
    weighted_persistences : ndarray
        Persistence weighted by cluster size
    """

    birth = f[critical_points]

    if ascending:
        death_vals = np.array(
            [death.get(r, 0.0) for r in critical_points], dtype=np.float64
        )
        persistence_values = birth - death_vals
        persistence_values = np.maximum(persistence_values, 0.0)
        diagram = np.column_stack([death_vals, birth])
    else:
        default_death = f.max()
        death_vals = np.array(
            [death.get(r, default_death) for r in critical_points], dtype=np.float64
        )
        persistence_values = death_vals - birth
        persistence_values = np.maximum(persistence_values, 0.0)
        death = birth + persistence_values
        diagram = np.column_stack([birth, death])

    persistence = dict(zip(critical_points, persistence_values))

    lifetimes = diagram[:, 1] - diagram[:, 0]

    return persistence, diagram, lifetimes


def compute_stabilities(
    f, death, critical_points, merge_tree, basin_assignment, ascending=True
):
    """
    Compute cluster stability (persistence weighted by cluster size).

    Parameters
    ----------
    f : ndarray
        Function values per node
    death : dict
        Mapping of critical point to death value (saddle height)
    critical_points : ndarray
        Indices of critical points
    merge_tree : list
        List of merge tuples (dead, survivor, h, merged_size)
    basin_assignment : ndarray
        Basin assignment of each node to critical point
    ascending : bool, default=True
        If True, compute persistence for ascending case; else descending

    Returns
    -------
    stabilities : ndarray
        Persistence weighted by cluster size
    """

    persistence, diagram, lifetimes = compute_persistence(
        f, death, critical_points, ascending
    )

    cluster_size_dict = {
        dead: merged_size for dead, survivor, h, merged_size in merge_tree
    }

    unique_basins, counts = np.unique(basin_assignment, return_counts=True)
    initial_sizes = dict(zip(unique_basins, counts))

    cluster_sizes = np.array(
        [
            cluster_size_dict.get(cp_id, initial_sizes.get(cp_id, 0))
            for cp_id in critical_points
        ],
        dtype=int,
    )

    normalized_sizes = cluster_sizes / cluster_sizes.max()
    stabilities = normalized_sizes * lifetimes
    diagram = normalized_sizes[:, None] * diagram
    persistence = {
        cp: normalized_sizes[i] * pv for i, (cp, pv) in enumerate(persistence.items())
    }

    return persistence, diagram, stabilities


def _find_bridge_nodes(
    c1: int,
    c2: int,
    pos: np.ndarray,
    graph: csr_matrix,
    metric: str,
) -> tuple[int, int]:
    """
    Hill-climb from both sides until neither endpoint can move closer
    to the other along graph edges.

    Parameters
    ----------
    c1, c2 : int
        Seed nodes in two different components.
    pos : np.ndarray
        Node positions, shape (n, d).
    graph : csr_matrix
        Adjacency matrix of the graph.
    metric : str
        Any distance metric accepted by `scipy.spatial.distance.cdist`
        (e.g. "euclidean", "cosine", "minkowski").

    Returns
    -------
    (c1, c2) : tuple[int, int]
        Locally optimal bridge node pair.
    """
    while True:
        moved = False

        # Try to advance c1 toward c2
        nbrs1 = graph.indices[graph.indptr[c1] : graph.indptr[c1 + 1]]
        if len(nbrs1):
            dists = cdist(pos[nbrs1], pos[[c2]], metric=metric).ravel()
            best_n1 = nbrs1[dists.argmin()]
            if dists.min() < cdist(pos[[c1]], pos[[c2]], metric=metric).item():
                c1 = best_n1
                moved = True

        # Try to advance c2 toward c1
        nbrs2 = graph.indices[graph.indptr[c2] : graph.indptr[c2 + 1]]
        if len(nbrs2):
            dists = cdist(pos[nbrs2], pos[[c1]], metric=metric).ravel()
            best_n2 = nbrs2[dists.argmin()]
            if dists.min() < cdist(pos[[c2]], pos[[c1]], metric=metric).item():
                c2 = best_n2
                moved = True

        if not moved:
            break

    return c1, c2


def _find_best_bridge(
    nodes_a: np.ndarray,
    nodes_b: np.ndarray,
    pos: np.ndarray,
    graph: csr_matrix,
    n_seeds: int,
    metric: str,
    rng: np.random.Generator,
) -> tuple[int, int, float]:
    """
    Run hill-climb from multiple random seed pairs between two components,
    returning the bridge with the smallest final distance.

    Parameters
    ----------
    nodes_a, nodes_b : np.ndarray
        Node indices belonging to each component.
    pos : np.ndarray
        Node positions, shape (n, d).
    graph : csr_matrix
        Adjacency matrix of the graph.
    n_seeds : int
        Number of random seed pairs to try.
    metric : str
        Any distance metric accepted by `scipy.spatial.distance.cdist`
        (e.g. "euclidean", "cosine", "minkowski").
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    (c1, c2, distance) : tuple[int, int, float]
        Best bridge node pair and its distance.
    """
    actual_seeds = min(n_seeds, len(nodes_a), len(nodes_b))
    seeds_a = rng.choice(nodes_a, size=actual_seeds, replace=False)
    seeds_b = rng.choice(nodes_b, size=actual_seeds, replace=False)

    best = None
    for sa, sb in zip(seeds_a, seeds_b):
        c1, c2 = _find_bridge_nodes(sa, sb, pos, graph, metric)
        w = cdist(pos[[c1]], pos[[c2]], metric=metric).item()
        if best is None or w < best[2]:
            best = (c1, c2, w)

    return best


def merge_components(
    graph: csr_matrix,
    pos: np.ndarray,
    n_seeds: int = 5,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> csr_matrix:
    """
    Merge disconnected components of a sparse CSR geometric graph with
    MST-like bridge edges found via Borůvka-style greedy hill-climbing.

    Each round, every component finds its cheapest bridge to any other
    component using multiple random seed pairs. The cheapest bridge per
    component is committed, components are recomputed, and the process
    repeats until the graph is fully connected.

    Parameters
    ----------
    graph : csr_matrix
        Adjacency matrix of the geometric graph (n x n). Treated as
        undirected; the matrix need not be symmetric on input, but
        bridge edges are added symmetrically.
    pos : np.ndarray
        Node positions, shape (n, d).
    n_seeds : int, optional
        Number of random seed pairs to try per component pair per round.
        Higher values reduce the chance of poor local minima at the cost
        of more hill-climb calls. Default is 5.
    metric : str
        Any distance metric accepted by `scipy.spatial.distance.cdist`
        (e.g. "euclidean", "cosine", "minkowski").
    random_state : int, optional
        Seed for the random number generator. Use for reproducibility.

    Returns
    -------
    csr_matrix
        Copy of the input graph with bridge edges added.
    """
    rng = np.random.default_rng(random_state)
    graph = graph.copy()

    while True:
        n_components, labels = connected_components(graph, directed=False)
        if n_components == 1:
            break

        label_to_nodes = {
            label: np.where(labels == label)[0] for label in np.unique(labels)
        }
        label_list = list(label_to_nodes.keys())

        best_bridge = dict()
        for i, label_a in enumerate(label_list):
            for label_b in label_list[i + 1 :]:
                c1, c2, w = _find_best_bridge(
                    nodes_a=label_to_nodes[label_a],
                    nodes_b=label_to_nodes[label_b],
                    pos=pos,
                    graph=graph,
                    n_seeds=n_seeds,
                    metric=metric,
                    rng=rng,
                )

                if label_a not in best_bridge or w < best_bridge[label_a][0]:
                    best_bridge[label_a] = (w, c1, c2)
                if label_b not in best_bridge or w < best_bridge[label_b][0]:
                    best_bridge[label_b] = (w, c1, c2)

        seen = set()
        bridge_rows, bridge_cols, bridge_weights = [], [], []

        for _, (w, c1, c2) in best_bridge.items():
            key = frozenset((c1, c2))
            if key in seen:
                continue
            seen.add(key)
            bridge_rows += [c1, c2]
            bridge_cols += [c2, c1]
            bridge_weights += [w, w]

        bridge = csr_matrix(
            (bridge_weights, (bridge_rows, bridge_cols)),
            shape=graph.shape,
        )
        graph = graph + bridge

    return graph


@njit
def _find_nearest_cluster(comp_nodes, cluster_lookup, indptr, indices, n_nodes):
    """
    Use BFS from component nodes to find the nearest cluster using a lookup array.
    Fast Numba-compiled version.

    Parameters
    ----------
    comp_nodes : ndarray
        Indices of nodes in the disconnected component.
    cluster_lookup : ndarray
        Array where cluster_lookup[node] = cluster_label (for nodes not in component, -1).
    indptr : ndarray
        CSR matrix indptr array.
    indices : ndarray
        CSR matrix indices array.
    n_nodes : int
        Total number of nodes (for array bounds).

    Returns
    -------
    nearest_label : int
        Label of the nearest cluster, or -1 if none found.
    distance : int
        Distance to the nearest cluster.
    """
    visited = np.zeros(n_nodes, dtype=np.bool_)
    queue = np.empty(n_nodes, dtype=np.int64)
    queue_size = 0

    for node in comp_nodes:
        visited[node] = True
        queue[queue_size] = node
        queue_size += 1

    distance = 0

    while queue_size > 0:
        next_queue_size = 0
        next_queue = np.empty(n_nodes, dtype=np.int64)
        distance += 1

        for i in range(queue_size):
            node = queue[i]
            start = indptr[node]
            end = indptr[node + 1]

            for ptr in range(start, end):
                neighbor = indices[ptr]
                if visited[neighbor]:
                    continue
                visited[neighbor] = True

                label = cluster_lookup[neighbor]
                if label >= 0:
                    return label, distance

                next_queue[next_queue_size] = neighbor
                next_queue_size += 1

        queue = next_queue
        queue_size = next_queue_size

    return -1, -1


@njit
def _find_adjacent_cluster(
    comp_nodes,
    cluster_lookup,
    indptr,
    indices,
):
    """
    Find the neighboring cluster with the largest number of
    boundary edges from a disconnected component.
    """
    n_nodes = cluster_lookup.shape[0]

    counts = np.zeros(n_nodes, dtype=np.int64)

    for i in range(comp_nodes.shape[0]):
        node = comp_nodes[i]
        start = indptr[node]
        end = indptr[node + 1]

        for ptr in range(start, end):
            nbr = indices[ptr]
            label = cluster_lookup[nbr]

            if label >= 0:
                counts[label] += 1

    best_label = -1
    best_count = 0

    for label in range(n_nodes):
        if counts[label] > best_count:
            best_count = counts[label]
            best_label = label

    return best_label


def merge_disconnected_components(
    labels,
    indptr,
    indices,
    unique_labels,
):
    new_labels = labels.copy()
    n_nodes = labels.shape[0]

    for lab in unique_labels:
        mask = new_labels == lab

        if mask.sum() <= 1:
            continue

        comp_id, ncomp = get_connected_components(indptr, indices, mask)

        if ncomp <= 1:
            continue

        sizes = np.bincount(
            comp_id[comp_id >= 0],
            minlength=ncomp,
        )

        keep_comp = sizes.argmax()

        # Map every node outside this label to its current label.
        cluster_lookup = np.full(
            n_nodes,
            -1,
            dtype=np.int64,
        )

        for node in range(n_nodes):
            node_label = new_labels[node]
            if node_label != lab:
                cluster_lookup[node] = node_label

        # Process smaller components.
        order = np.argsort(sizes)

        for comp_idx in order:
            if comp_idx == keep_comp:
                continue

            comp_nodes = np.where(comp_id == comp_idx)[0]

            target = _find_adjacent_cluster(
                comp_nodes,
                cluster_lookup,
                indptr,
                indices,
            )

            if target >= 0:
                new_labels[comp_nodes] = target

    return new_labels


def merge_disconnected_components_old(labels, indptr, indices, unique_labels):
    """
    Process a label that has disconnected components. Keep the largest
    component and merge smaller ones into their nearest neighboring clusters.

    Parameters
    ----------
    labels : ndarray
        Current cluster labels.
    indptr : ndarray
        CSR matrix indptr array.
    indices : ndarray
        CSR matrix indices array.
    unique_labels : ndarray
        Array of unique labels to process.

    Returns
    -------
    new_labels : ndarray
        Modified labels with disconnected components merged to neighbors.
    """
    new_labels = labels.copy()
    n_nodes = labels.shape[0]

    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() <= 1:
            continue

        comp_id, ncomp = get_connected_components(indptr, indices, mask)
        if ncomp <= 1:
            continue

        valid = comp_id >= 0
        sizes = np.bincount(comp_id[valid], minlength=ncomp)
        keep_comp = sizes.argmax()

        is_valid = comp_id != -1
        is_keep = comp_id == keep_comp
        new_labels[is_valid & is_keep] = lab

        cluster_lookup = np.full(n_nodes, -1, dtype=np.int64)
        for other_lab in unique_labels:
            if other_lab == lab:
                continue
            other_mask = new_labels == other_lab
            if other_mask.any():
                other_nodes = np.where(other_mask)[0]
                for node in other_nodes:
                    cluster_lookup[node] = other_lab

        for comp_idx in range(ncomp):
            if comp_idx == keep_comp:
                continue

            comp_mask = comp_id == comp_idx
            if not comp_mask.any():
                continue

            comp_nodes = np.where(comp_mask)[0]
            nearest_label, _ = _find_nearest_cluster(
                comp_nodes, cluster_lookup, indptr, indices, n_nodes
            )

            if nearest_label >= 0:
                new_labels[comp_mask] = nearest_label
            else:
                new_labels[comp_mask] = lab

    return new_labels


def csr_nonzero_argmin(matrix: csr_matrix) -> np.ndarray:
    """
    Returns the column index of the minimum non-zero value in each row,
    ignoring structural and explicit zeros.

    Returns -1 for empty rows.
    """
    n_rows = matrix.shape[0]
    result = np.full(n_rows, -1, dtype=np.intp)

    for i in range(n_rows):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        if start == end:
            continue
        row_data = matrix.data[start:end]
        nonzero_mask = row_data != 0
        if not nonzero_mask.any():
            continue
        local_idx = row_data[nonzero_mask].argmin()
        result[i] = matrix.indices[start:end][nonzero_mask][local_idx]

    return result
