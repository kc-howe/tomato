from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from tomato.core_ import ToMAToCore


def clf_to_tree(clf: ToMAToCore) -> "MergeTree":
    """
    Build a MergeTree from a fitted ToMAToCore classifier.

    Parameters
    ----------
    clf : ToMAToCore
        A fitted classifier exposing graph_persistence_ and labels_.

    Returns
    -------
    MergeTree
        The merge tree with node labels set from clf.labels_.
    """
    tree = MergeTree(graph_persistence=clf.graph_persistence_)
    return tree.set_labels(clf.labels_)


class MergeTreeNode:
    """
    A single node in a topological merge tree.

    Leaf nodes correspond to critical points (basins) identified during
    0-th order persistence computation. Internal nodes correspond to
    merge events between two child basins, occurring at the function
    value of the connecting saddle.

    Attributes
    ----------
    node_id : int
        Unique identifier for this node. For leaves, this is the
        original critical point index. For internal (merged) nodes,
        this is a synthetic id assigned during tree construction.
    birth : float or None
        Function value at which this component was "born" (the extremal
        value of the surviving critical point for this subtree).
    death : float or None
        Function value at which this component was absorbed into a
        larger one. None for the root (it never dies).
    value : float or None
        Function value at which this specific merge occurred. None
        for leaves.
    indices : np.ndarray
        Original point indices belonging to this node's subtree.
    children : list[MergeTreeNode]
        Child nodes (empty for leaves).
    parent : MergeTreeNode or None
        Parent node (None for the root).
    labels : np.ndarray or None
        Labels for the original points belonging to this node.
    canonical_label : int
        Label of the mode represented by this node, or -1 when the node
        is not a mode or descendant of one.
    is_leaf : bool

    Notes
    -----
    __slots__ is used because trees built from large point clouds can
    have many thousands of nodes; the per-instance __dict__ that a
    normal class would carry is unnecessary overhead here. This does mean
    the class is not directly subclassable without also declaring
    __slots__ (or __dict__) on the subclass.
    """

    __slots__ = (
        "node_id",
        "birth",
        "death",
        "value",
        "indices",
        "children",
        "parent",
        "labels",
        "canonical_label",
        "is_leaf",
    )

    def __init__(self, node_id, birth=None, death=None, value=None, indices=None):
        self.node_id = node_id
        self.birth = birth
        self.death = death
        self.value = value
        self.indices = indices if indices is not None else np.array([], dtype=int)
        self.children = []
        self.parent = None
        self.labels = None
        self.canonical_label = -1
        self.is_leaf = True

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.is_leaf = False

    @property
    def persistence(self):
        """
        abs(death - birth), or None if this node has no death (i.e. root).
        """
        if self.birth is None or self.death is None:
            return None
        return abs(self.death - self.birth)

    @property
    def size(self):
        return len(self.indices)

    def __repr__(self):
        kind = "Leaf" if self.is_leaf else "Internal"
        return (
            f"<MergeTreeNode {kind} id={self.node_id} size={self.size} "
            f"birth={self.birth} death={self.death}>"
        )


class MergeTree:
    """
    Tree representation of a fitted GraphPersistence's merge structure.

    Parameters
    ----------
    graph_persistence : GraphPersistence
        A GraphPersistence instance that has already been fit().

    Attributes
    ----------
    nodes : dict[int, MergeTreeNode]
        All nodes in the tree, keyed by node_id.
    root : MergeTreeNode or None
        The root of the tree (the final surviving merged component).
        None only if the tree has not been built.

    Notes
    -----
    Leaf construction relies only on critical_points, basin_assignment,
    and f, which are well-defined by GraphPersistence.fit().

    Internal-node construction relies on graph_persistence.merge_tree_,
    whose exact shape depends on _graph_utils.build_merge_tree. This
    class normalizes that structure through _parse_merge_events; if
    your actual merge_tree_ format differs from the assumed
    (basin_a, basin_b, merge_value) record shape, adjust that one
    method — the rest of the class only depends on getting a clean,
    time-ordered list of triples out of it.
    """

    def __init__(self, graph_persistence):
        gp = graph_persistence
        required = ("critical_points", "basin_assignment", "f")
        for attr in required:
            if getattr(gp, attr, None) is None:
                raise ValueError(
                    f"GraphPersistence object must be fit before building a "
                    f"MergeTree (missing attribute '{attr}'). Call gp.fit() first."
                )
        self.gp = gp
        self.nodes = {}
        self.root = None
        self._build()

    def _build(self):
        for cp in self.gp.critical_points:
            cp = int(cp)
            leaf_indices = np.where(self.gp.basin_assignment == cp)[0]
            self.nodes[cp] = MergeTreeNode(
                node_id=cp,
                birth=float(self.gp.f[cp]),
                indices=leaf_indices,
            )

        if not self.nodes:
            return

        merge_events = self._parse_merge_events()
        next_internal_id = max(self.nodes) + 1

        self._active = {node_id: node_id for node_id in self.nodes}

        for dead_id, survivor_id, merge_value in merge_events:
            root_a = self._find_active_ancestor(dead_id)
            root_b = self._find_active_ancestor(survivor_id)
            node_a = self.nodes[root_a]
            node_b = self.nodes[root_b]
            if node_a is node_b:
                continue

            parent_birth = (
                min(node_a.birth, node_b.birth)
                if self.gp.ascending
                else max(node_a.birth, node_b.birth)
            )
            parent = MergeTreeNode(
                node_id=next_internal_id,
                birth=parent_birth,
                value=merge_value,
                indices=np.concatenate([node_a.indices, node_b.indices]),
            )

            node_a.death = merge_value
            parent.add_child(node_a)
            parent.add_child(node_b)

            self.nodes[parent.node_id] = parent
            self._active[parent.node_id] = parent.node_id
            self._active[root_a] = parent.node_id
            self._active[root_b] = parent.node_id
            self.root = parent

            next_internal_id += 1

        roots = [node for node in self.nodes.values() if node.parent is None]
        if len(roots) == 1:
            self.root = roots[0]
        elif len(roots) > 1:
            roots.sort(key=lambda node: node.indices.min())
            dummy_root = MergeTreeNode(
                node_id=next_internal_id,
                indices=np.concatenate([node.indices for node in roots]),
            )
            for node in roots:
                dummy_root.add_child(node)
            self.nodes[dummy_root.node_id] = dummy_root
            self.root = dummy_root

        del self._active

    def _find_active_ancestor(self, node_id):
        """
        Return the id of the current top-most (not yet merged) ancestor
        of node_id, using union-find with path compression.

        Only valid while self._active exists, i.e. during _build.
        """
        root = node_id
        while self._active[root] != root:
            root = self._active[root]
        while self._active[node_id] != root:
            self._active[node_id], node_id = root, self._active[node_id]
        return root

    def _parse_merge_events(self):
        """
        Normalize graph_persistence.merge_tree_ into a flat list of
        (dead_id, survivor_id, merge_value) triples.
        """
        return [
            (int(dead), int(survivor), float(h))
            for dead, survivor, h, _ in self.gp.merge_tree_
        ]

    def get_node(self, node_id):
        """
        Return the node with the given id.
        """
        return self.nodes[node_id]

    def get_leaves(self):
        """
        Return all leaf nodes (original critical points/basins).
        """
        if self.root is None:
            return []

        leaves = []

        def _walk(node):
            if node.is_leaf:
                leaves.append(node)
                return
            for child in node.children:
                _walk(child)

        _walk(self.root)
        return leaves

    def indices_at(self, node_id):
        """
        Return the array of original point indices under a given node.
        """
        return self.nodes[node_id].indices

    def set_labels(self, labels):
        """
        Set the labels associated with every node in the tree.

        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            A label for every point in the dataset used to build the tree.
        Returns
        -------
        MergeTree
            This tree, to allow method chaining.
        """
        labels = np.asarray(labels)
        self._labels = labels

        n_samples = self.gp.f.shape[0]
        if labels.ndim != 1 or labels.shape[0] != n_samples:
            raise ValueError(
                "labels must be a one-dimensional array with one label "
                f"for each dataset point ({n_samples})"
            )

        for node in self.nodes.values():
            node.labels = labels[node.indices]
            node.canonical_label = -1

        for node in self.postorder():
            if node.is_leaf:
                if node.labels.size and np.all(node.labels == node.labels[0]):
                    node.canonical_label = node.labels[0]
                elif node.labels.size:
                    node.canonical_label = -1
                continue
            child_labels = [child.canonical_label for child in node.children]
            if child_labels and all(label == child_labels[0] for label in child_labels):
                node.canonical_label = child_labels[0]
        return self

    def _is_extended(self):
        critical_set = {int(cp) for cp in self.gp.critical_points}

        singleton_vertices = {
            int(node.indices[0])
            for node in self.nodes.values()
            if len(node.indices) == 1
            and node.death is None
            and int(node.indices[0]) not in critical_set
        }

        represented_vertices = {
            int(v)
            for node in self.nodes.values()
            for v in node.indices
            if int(v) not in critical_set
        }

        return singleton_vertices == represented_vertices

    def extend_to_vertices(self):
        """
        Extend this tree with a singleton leaf for every graph vertex.

        For every vertex not already a critical point, a new leaf node is
        inserted as a child of the basin leaf it is assigned to (via
        gp.basin_assignment), with birth equal to that vertex's own
        function value gp.f[v] and no death. If self already has
        labels set (via set_labels), the returned tree has labels set on
        every node, including the new singletons, so canonical_label and
        labels are populated the same way as after set_labels; if not,
        the returned tree is unlabeled just like self.

        Returns
        -------
        MergeTree
            A new tree with one additional leaf per non-critical-point
            vertex. self is left unmodified.
        """
        if self._is_extended():
            return self

        had_labels = getattr(self, "_labels", None) is not None
        labels = self._labels if had_labels else None

        extended = MergeTree.__new__(MergeTree)
        extended.gp = self.gp
        extended.nodes = {}
        extended.root = None

        next_id = max(self.nodes) + 1

        copies = {}
        for node in self.postorder():
            copy = MergeTreeNode(
                node_id=node.node_id,
                birth=node.birth,
                death=node.death,
                value=node.value,
                indices=node.indices,
            )
            copies[node.node_id] = copy
            for child in node.children:
                copy.add_child(copies[child.node_id])
            extended.nodes[copy.node_id] = copy
            if node is self.root:
                extended.root = copy

        critical_set = set(int(cp) for cp in self.gp.critical_points)
        for leaf in self.get_leaves():
            parent = copies[leaf.node_id]
            for v in leaf.indices:
                v = int(v)
                if v in critical_set:
                    continue
                singleton = MergeTreeNode(
                    node_id=next_id,
                    birth=float(self.gp.f[v]),
                    indices=np.array([v], dtype=int),
                )
                next_id += 1
                parent.add_child(singleton)
                extended.nodes[singleton.node_id] = singleton

        if had_labels:
            extended.set_labels(labels)

        return extended

    def depth(self, node):
        """
        Distance from node up to the root.
        """
        d = 0
        cur = node
        while cur.parent is not None:
            cur = cur.parent
            d += 1
        return d

    def _depths(self):
        """
        Compute depth for every node in a single O(n) BFS pass.
        """
        depths = {}
        if self.root is None:
            return depths
        depths[self.root.node_id] = 0
        for node in self.bfs():
            if node.parent is not None:
                depths[node.node_id] = depths[node.parent.node_id] + 1
        return depths

    def height(self):
        """
        Length of the longest root-to-leaf path.
        """
        if self.root is None:
            return 0
        depths = self._depths()
        leaf_ids = (leaf.node_id for leaf in self.get_leaves())
        return max((depths[node_id] for node_id in leaf_ids), default=0)

    def layout(self, use_function_value=False):
        """
        Calculate xy-coordinates for visualizing the tree.

        Leaves are assigned consecutive x-values in child traversal order.
        Each internal node is placed at the mean x-value of its children. By
        default, y-values place the root at tree height and leaves at zero,
        with one unit per tree level. If use_function_value is true,
        internal nodes use their merge value and leaves use their birth value
        for y instead.

        Children at each internal node are ordered by height (tallest first)
        before leaf x-values are assigned, so that short subtrees end up
        adjacent to their parent's x-position. This keeps the long, steep
        edges from short children from visually crossing over taller
        sibling subtrees.

        Parameters
        ----------
        use_function_value : bool, default=False
            Whether to use function values instead of tree levels for y.

        Returns
        -------
        dict[int, tuple[float, float]]
            Mapping from node id to its (x, y) coordinate.
        """
        if self.root is None:
            return {}

        coordinates = {}

        # First past: Postorder traversal
        node_y = {}
        for leaf in self.get_leaves():
            node_y[leaf.node_id] = leaf.birth if use_function_value else 0.0

        for node in self.postorder():
            if node.is_leaf:
                continue
            if use_function_value:
                y = node.value
                if y is None:
                    child_y = [node_y[child.node_id] for child in node.children]
                    y = max(child_y) if self.gp.ascending else min(child_y)
            else:
                y = 1 + max(node_y[child.node_id] for child in node.children)
            node_y[node.node_id] = y

            # Order children tallest first
            node.children.sort(key=lambda c: node_y[c.node_id], reverse=True)

        # Second pass: Assign leaf x-values using new child order
        leaf_order = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaf_order.append(node)
            else:
                stack.extend(reversed(node.children))

        for x, leaf in enumerate(leaf_order):
            coordinates[leaf.node_id] = (float(x), float(node_y[leaf.node_id]))

        # Third pass: Compute x-values as mean of children
        for node in self.postorder():
            if node.is_leaf:
                continue
            x = np.mean([coordinates[child.node_id][0] for child in node.children])
            coordinates[node.node_id] = (float(x), float(node_y[node.node_id]))

        return coordinates

    def plot(
        self,
        use_function_value=False,
        color_by_label=False,
        show_singletons=False,
        ax=None,
        marker=None,
        marker_size=None,
        marker_color=None,
        line_color=None,
        line_width=None,
        line_style="-",
    ):
        """
        Plot the merge tree.

        Parameters
        ----------
        use_function_value : bool, default=False
            Whether to use function values instead of tree levels for y.
        color_by_label : bool, default=False
            Whether to color nodes according to their canonical label. Nodes
            with canonical label -1 are colored light grey.
        show_singletons : bool, default=False
            Whether to include per-vertex singleton leaves added by
            extend_to_vertices in the plot. When False (the default),
            only basin leaves and merge nodes are drawn, i.e. the tree is
            rendered as if singletons were never added, even if self has
            been extended. When True, singleton leaves are drawn as regular
            leaf nodes, which can be dense for large datasets.
        ax : matplotlib.axes.Axes or None, default=None
            Axes on which to draw the tree. If None, a new figure is created.
        marker : str, default="o"
            Marker style for nodes, passed to Axes.scatter.
        marker_size : float, default=36
            Marker size for nodes, passed to Axes.scatter as s.
        marker_color : str, default="tab:blue"
            Marker color for nodes, used when color_by_label=False.
            Ignored when color_by_label=True.
        line_color : str, default="k"
            Color of the edges connecting nodes.
        line_width : float, default=1.0
            Width of the edges connecting nodes.
        line_style : str, default="-"
            Line style of the edges connecting nodes (e.g. "-", "--",
            ":", "-.").

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure, or None when an axes object was supplied.
        """
        coordinates = self.layout(use_function_value=use_function_value)

        def _is_singleton(node):
            return (
                node.is_leaf
                and node.size == 1
                and node.node_id not in (int(cp) for cp in self.gp.critical_points)
            )

        if show_singletons:
            nodes = [
                node for node in self.nodes.values() if node.node_id in coordinates
            ]
        else:
            nodes = [
                node
                for node in self.nodes.values()
                if node.node_id in coordinates and not _is_singleton(node)
            ]

        visible_ids = {node.node_id for node in nodes}
        segments = [
            [coordinates[node.node_id], coordinates[child.node_id]]
            for node in nodes
            for child in node.children
            if child.node_id in visible_ids
        ]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        ax.add_collection(
            LineCollection(
                segments,
                colors=line_color,
                linewidths=line_width,
                linestyles=line_style,
                zorder=1,
            )
        )
        colors = marker_color
        if color_by_label:
            canonical_labels = [node.canonical_label for node in nodes]
            unique_labels = sorted(
                list(dict.fromkeys(label for label in canonical_labels if label != -1))
            )
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            label_colors = {
                label: color_cycle[index % len(color_cycle)]
                for index, label in enumerate(unique_labels)
            }
            colors = [
                "lightgrey" if label == -1 else label_colors[label]
                for label in canonical_labels
            ]
        ax.scatter(
            [coordinates[node.node_id][0] for node in nodes],
            [coordinates[node.node_id][1] for node in nodes],
            c=colors,
            marker=marker,
            s=marker_size,
            zorder=2,
        )

        ax.xaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_ylabel("Function Value" if use_function_value else "Height")
        ax.autoscale_view()

        if fig is not None:
            return fig

    def preorder(self, start=None):
        """
        Yield nodes in preorder (parent before children).
        """
        start = start if start is not None else self.root
        if start is None:
            return
        stack = [start]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def postorder(self, start=None):
        """
        Yield nodes in postorder (children before parent).
        """
        start = start if start is not None else self.root
        if start is None:
            return

        def _walk(node):
            for child in node.children:
                yield from _walk(child)
            yield node

        yield from _walk(start)

    def bfs(self, start=None):
        """
        Yield nodes in breadth-first order.
        """
        start = start if start is not None else self.root
        if start is None:
            return
        queue = deque([start])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def __iter__(self):
        return self.preorder()

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, node_id):
        return self.nodes[node_id]

    def __repr__(self):
        return (
            f"<MergeTree n_nodes={len(self.nodes)} "
            f"n_leaves={len(self.get_leaves())} height={self.height()}>"
        )

    def prune_mapping(self, min_persistence):
        """
        Return the ids of nodes that survive a persistence threshold cut:
        every leaf whose own persistence (death - birth) is below
        min_persistence is dropped in favor of its nearest surviving
        ancestor, mirroring how ToMATo collapses low-persistence basins.

        Returns
        -------
        dict[int, int]
            Mapping from every leaf node id to the id of the surviving
            (pruned) node it belongs to.
        """
        surviving = set()
        for node in self.preorder():
            if node is self.root:
                surviving.add(node.node_id)
                continue
            if node.persistence is not None and node.persistence >= min_persistence:
                surviving.add(node.node_id)

        mapping = {}
        for leaf in self.get_leaves():
            node = leaf
            while node.node_id not in surviving and node.parent is not None:
                node = node.parent
            mapping[leaf.node_id] = node.node_id
        return mapping

    def prune(self, min_persistence):
        """
        Collapse low-persistence leaves and return the result as a new
        MergeTree instance.

        Parameters
        ----------
        min_persistence : float
            Leaves with persistence below this threshold are pruned.

        Returns
        -------
        MergeTree
            A new tree containing only the surviving nodes. self is
            left unmodified.
        """
        mapping = self.prune_mapping(min_persistence)
        surviving_ids = set(mapping.values())

        pruned = MergeTree.__new__(MergeTree)
        pruned.gp = self.gp
        pruned.nodes = {}
        pruned.root = None

        for node in self.postorder():
            if node.node_id not in surviving_ids:
                continue

            surviving_children = [
                pruned.nodes[child.node_id]
                for child in node.children
                if child.node_id in surviving_ids
            ]

            if not surviving_children:
                collapsed_indices = (
                    np.concatenate(
                        [
                            self.nodes[leaf_id].indices
                            for leaf_id, survivor_id in mapping.items()
                            if survivor_id == node.node_id
                        ]
                    )
                    if any(
                        survivor_id == node.node_id for survivor_id in mapping.values()
                    )
                    else node.indices
                )
                new_node = MergeTreeNode(
                    node_id=node.node_id,
                    birth=node.birth,
                    death=node.death,
                    indices=collapsed_indices,
                )
            else:
                new_node = MergeTreeNode(
                    node_id=node.node_id,
                    birth=node.birth,
                    death=node.death,
                    value=node.value,
                    indices=np.concatenate(
                        [child.indices for child in surviving_children]
                    ),
                )
                for child in surviving_children:
                    new_node.add_child(child)

            pruned.nodes[new_node.node_id] = new_node
            if node is self.root:
                pruned.root = new_node

        return pruned

    def _node_f(self, node):
        """
        The function value associated with a node, for edge-weighting.
        """
        if node.is_leaf:
            return node.birth
        if node.value is not None:
            return node.value
        child_f = [self._node_f(child) for child in node.children]
        return max(child_f) if self.gp.ascending else min(child_f)

    def _edge_weight(self, u, v, weighted):
        """
        Weight of the tree edge between adjacent nodes u and v.
        """
        if not weighted:
            return 1.0
        return abs(self._node_f(u) - self._node_f(v))

    def lca(self, node_a, node_b):
        """
        Return the lowest common ancestor of two nodes.

        Parameters
        ----------
        node_a, node_b : MergeTreeNode

        Returns
        -------
        MergeTreeNode
            The deepest node that is an ancestor of (or equal to) both
            node_a and node_b.
        """
        depths = self._depths()
        a, b = node_a, node_b
        while depths[a.node_id] > depths[b.node_id]:
            a = a.parent
        while depths[b.node_id] > depths[a.node_id]:
            b = b.parent
        while a is not b:
            a = a.parent
            b = b.parent
        return a

    def _path(self, node_a, node_b):
        """
        Return the ordered list of nodes on the tree path from
        node_a to node_b, inclusive, passing through their LCA.
        """
        ancestor = self.lca(node_a, node_b)

        up_a = []
        node = node_a
        while node is not ancestor:
            up_a.append(node)
            node = node.parent
        up_a.append(ancestor)

        down_b = []
        node = node_b
        while node is not ancestor:
            down_b.append(node)
            node = node.parent
        down_b.reverse()

        return up_a + down_b

    def distance(self, node_a, node_b, weighted=False):
        """
        Path distance between two nodes along the tree.

        Parameters
        ----------
        node_a, node_b : MergeTreeNode
        weighted : bool, default=False
            If False, each edge counts as 1 (hop distance). If True,
            each edge is weighted by abs(f(u) - f(v)), the absolute
            difference in function value between its endpoints (birth
            for leaves, merge value for internal nodes).

        Returns
        -------
        float
            The summed edge weight along the unique path between the
            two nodes.
        """
        path = self._path(node_a, node_b)
        return sum(
            self._edge_weight(path[i], path[i + 1], weighted)
            for i in range(len(path) - 1)
        )

    def distances_from(self, node, weighted=False):
        """
        Distance from node to every other node in the tree.

        Parameters
        ----------
        node : MergeTreeNode
        weighted : bool, default=False

        Returns
        -------
        dict[int, float]
            Mapping from node id to its distance from node.
        """
        adjacency = {}
        for n in self.nodes.values():
            for child in n.children:
                adjacency.setdefault(n.node_id, []).append(child)
                adjacency.setdefault(child.node_id, []).append(n)

        distances = {node.node_id: 0.0}
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current.node_id, []):
                if neighbor.node_id in distances:
                    continue
                distances[neighbor.node_id] = distances[current.node_id] + (
                    self._edge_weight(current, neighbor, weighted)
                )
                queue.append(neighbor)
        return distances

    def distance_matrix(self, weighted=False, nodes=None):
        """
        Pairwise path distances between a set of nodes.

        Parameters
        ----------
        weighted : bool, default=False
        nodes : list[MergeTreeNode] or None, default=None
            Nodes to compute pairwise distances between. Defaults to all
            leaves (the usual "distance between basins/modes" query);
            pass an explicit list to include internal nodes as well.

        Returns
        -------
        matrix : np.ndarray of shape (len(nodes), len(nodes))
        node_ids : list[int]
            node_ids[i] is the id of the node corresponding to row/
            column i of matrix.
        """
        if nodes is None:
            nodes = self.get_leaves()
        node_ids = [node.node_id for node in nodes]

        n = len(nodes)
        matrix = np.zeros((n, n), dtype=float)
        for i, node in enumerate(nodes):
            dists = self.distances_from(node, weighted=weighted)
            for j, other_id in enumerate(node_ids):
                matrix[i, j] = dists[other_id]

        return matrix, node_ids

    def mode_of(self, node):
        """
        Return the nearest ancestor of node (including itself) that
        represents a single mode, i.e. has canonical_label != -1.

        Parameters
        ----------
        node : MergeTreeNode

        Returns
        -------
        MergeTreeNode or None
        """
        current = node
        while current is not None:
            if current.canonical_label != -1:
                return current
            current = current.parent
        return None

    def nearest_common_mode(self, node_a, node_b):
        """
        Return the smallest single-mode ancestor shared by two nodes.

        Parameters
        ----------
        node_a, node_b : MergeTreeNode

        Returns
        -------
        MergeTreeNode or None
        """
        return self.mode_of(self.lca(node_a, node_b))

    def to_networkx(self):
        """
        Convert this tree to a networkx DiGraph.

        Returns
        -------
        networkx.DiGraph
        """
        import networkx as nx

        graph = nx.DiGraph()
        for node in self.nodes.values():
            graph.add_node(
                node.node_id,
                birth=node.birth,
                death=node.death,
                value=node.value,
                size=node.size,
                canonical_label=node.canonical_label,
                is_leaf=node.is_leaf,
            )
        for node in self.nodes.values():
            for child in node.children:
                graph.add_edge(
                    node.node_id,
                    child.node_id,
                    weight=self._edge_weight(node, child, weighted=True),
                )
        return graph

    def to_scipy_sparse(self, weighted=True):
        """
        Convert this tree to a scipy sparse adjacency matrix.

        Parameters
        ----------
        weighted : bool, default=True
            If True, edge entries are abs(f(u) - f(v)); if False,
            edge entries are 1.

        Returns
        -------
        matrix : scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        node_ids : list[int]
        """
        from scipy.sparse import coo_matrix

        node_ids = list(self.nodes.keys())
        index_of = {node_id: i for i, node_id in enumerate(node_ids)}

        rows, cols, weights = [], [], []
        for node in self.nodes.values():
            for child in node.children:
                w = self._edge_weight(node, child, weighted)
                i, j = index_of[node.node_id], index_of[child.node_id]
                rows += [i, j]
                cols += [j, i]
                weights += [w, w]

        n = len(node_ids)
        matrix = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        return matrix, node_ids
