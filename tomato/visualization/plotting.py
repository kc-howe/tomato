from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.sparse import triu
from sklearn.manifold import SpectralEmbedding

from tomato.core_ import ToMAToCore
from tomato.postprocessing.merge_tree import clf_to_tree


def _get_line_segments(x, y, graph):
    from_idx, to_idx = graph.nonzero()
    segments = []
    for i, j in zip(from_idx, to_idx):
        segments.append([[x[i], y[i]], [x[j], y[j]]])
    return segments


def _get_line_color(weights, color, cmap):
    if cmap is not None:
        array = weights
        color = None
    else:
        array = None
        if color is None:
            color = list(plt.rcParams["axes.prop_cycle"])[0]["color"]
    return array, color


def plot_network(
    graph, X, color=None, cmap="viridis", alpha=1.0, linewidth=1.0, ax=None
):
    if ax is None:
        fig, ax_ = plt.subplots(1)
    else:
        ax_ = ax

    graph = triu(graph, k=1, format="csr")
    segments = _get_line_segments(X[:, 0], X[:, 1], graph)
    weights = np.asarray(graph[graph.nonzero()]).reshape(-1)
    array, color = _get_line_color(weights, color, cmap)

    ax_.scatter(X[:, 0], X[:, 1], color="#eeeeee")
    ax_.add_collection(
        LineCollection(
            segments,
            colors=color,
            alpha=alpha,
            array=array,
            linewidth=linewidth,
            zorder=1,
            cmap=cmap,
        )
    )

    if ax is not None:
        return

    return fig


class ToMAToVisualization:

    def __init__(self, clf: ToMAToCore) -> None:
        self.clf = clf
        self.X = clf._raw_data

        if clf.drop_duplicates:
            self.X = self.X[clf.duplicate_index_]
        
        return

    def _get_data_2d(self) -> np.ndarray:
        if hasattr(self, "_data_low_dim"):
            return self._data_low_dim
        if self.X.shape[1] == 2:
            self._data_low_dim = self.X
            return self._data_low_dim
        self._data_low_dim = SpectralEmbedding(affinity="precomputed").fit(
            self.clf.graph_persistence_.gradient
        )
        return self._data_low_dim

    def plot_cluster_persistences(
        self, threshold: Optional[float] = None, ax=None
    ) -> plt.Figure:
        """
        Plot the cluster persistences for the fitted model.

        Parameters
        ----------
        threshold : float or None
            Persistence threshold used to highlight persistent critical points. If
            None, uses `self.threshold_` computed during `fit()`.
        ax : matplotlib.axes.Axes or None
            Matplotlib axes object to plot on. If None, creates a new figure and axes.

        Returns
        -------
        matplotlib.figure.Figure
            The created matplotlib Figure containing the diagram.
        """

        if threshold is None:
            threshold = self.clf.threshold_

        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        sort_index = np.argsort(self.clf.persistences_)
        sorted_persistences = self.clf.persistences_[sort_index]
        sorted_modes = self.clf.all_modes_[sort_index]
        index = np.arange(sorted_persistences.shape[0])

        mask = sorted_persistences < threshold

        low_persistences = sorted_persistences[mask]
        low_index = index[mask]
        high_persistences = sorted_persistences[~mask]
        high_index = index[~mask]

        ax.bar(low_index, low_persistences)
        ax.bar(high_index, high_persistences)
        ax.axhline(threshold, color="grey", linestyle="--", zorder=-1)
        ax.set_xticks(sort_index, sorted_modes)
        ax.set_xlabel("Cluster Mode Index")
        ax.set_ylabel("Cluster Persistence")

        return fig

    def plot_persistence_diagram(
        self, threshold: Optional[float] = None, ax=None
    ) -> plt.Figure:
        """
        Plot the persistence diagram for the fitted model.

        Points correspond to critical points with coordinates (birth, death) where
        birth is the root degree and death = birth - persistence.

        Parameters
        ----------
        threshold : float or None
            Persistence threshold used to highlight persistent critical points. If
            None, uses `self.threshold_` computed during `fit()`.
        ax : matplotlib.axes.Axes or None
            Matplotlib axes object to plot on. If None, creates a new figure and axes.

        Returns
        -------
        matplotlib.figure.Figure
            The created matplotlib Figure containing the diagram.
        """

        if threshold is None:
            threshold = self.clf.threshold_

        diagram = self.clf.graph_persistence_.diagram_
        x = np.linspace(0, diagram.max())
        y = x

        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        if threshold > 0.0:
            y_t = x + threshold
            mask = self.clf.graph_persistence_.persistences_ > threshold

            if not mask.all():
                ax.scatter(diagram[~mask, 0], diagram[~mask, 1])
            else:
                ax.scatter([], [])
            if mask.any():
                ax.scatter(diagram[mask, 0], diagram[mask, 1])

            ax.plot(x, y, color="k")
            ax.plot(x, y_t, linestyle="--", color="lightgrey")

        else:
            ax.scatter(diagram[:, 0], diagram[:, 1])
            ax.plot(x, y, linestyle="--", color="k")

        ax.set_xlabel("birth")
        ax.set_ylabel("death")

        if fig is not None:
            return fig

    def plot_graph(self, color=None, cmap="viridis", alpha=1.0, linewidth=1.0, ax=None):

        X = self._get_data_2d()
        graph = self.clf.graph_persistence_.adjacency

        graph = triu(graph.maximum(graph.T), k=1, format="csr")
        segments = _get_line_segments(X[:, 0], X[:, 1], graph)
        weights = np.asarray(graph[graph.nonzero()]).reshape(-1)
        array, color = _get_line_color(weights, color, cmap)

        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        ax.scatter(X[:, 0], X[:, 1], color="#eeeeee")
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                alpha=alpha,
                array=array,
                linewidth=linewidth,
                zorder=1,
                cmap=cmap,
            )
        )

        ax.set_axis_off()

        if fig is not None:
            return fig

    def plot_gradient(
        self, color=None, cmap="viridis", alpha=1.0, linewidth=1.0, ax=None
    ):

        X = self._get_data_2d()
        graph = self.clf.graph_persistence_.gradient
        critical_points = self.clf.graph_persistence_.critical_points
        f = self.clf.graph_persistence_.f

        graph = triu(graph.maximum(graph.T), k=1, format="csr")
        segments = _get_line_segments(X[:, 0], X[:, 1], graph)
        weights = np.asarray(graph[graph.nonzero()]).reshape(-1)
        array, color = _get_line_color(weights, color, cmap)

        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        ax.scatter(X[:, 0], X[:, 1], color="#eeeeee")
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                alpha=alpha,
                array=array,
                linewidth=linewidth,
                zorder=1,
                cmap=cmap,
            )
        )
        ax.scatter(
            X[critical_points, 0],
            X[critical_points, 1],
            c=f[critical_points],
            cmap=cmap,
            edgecolors="#222222",
        )

        ax.set_axis_off()

        if fig is not None:
            return fig

    def plot_mst(self, color=None, cmap="viridis", alpha=1.0, linewidth=1.0, ax=None):

        X = self._get_data_2d()
        graph = self.clf.graph_persistence_.mst
        critical_points = self.clf.graph_persistence_.critical_points
        f = self.clf.graph_persistence_.f

        graph = triu(graph.maximum(graph.T), k=1, format="csr")
        segments = _get_line_segments(X[:, 0], X[:, 1], graph)
        weights = np.asarray(graph[graph.nonzero()]).reshape(-1)
        array, color = _get_line_color(weights, color, cmap)

        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        ax.scatter(X[:, 0], X[:, 1], color="#eeeeee")
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                alpha=alpha,
                array=array,
                linewidth=linewidth,
                zorder=1,
                cmap=cmap,
            )
        )
        ax.scatter(
            X[critical_points, 0],
            X[critical_points, 1],
            c=f[critical_points],
            cmap=cmap,
            edgecolors="#222222",
        )

        ax.set_axis_off()

        if fig is not None:
            return fig

    def plot_labels(
        self,
        labels=None,
        critical_points=None,
        show_critical_points=False,
        alpha=1.0,
        size=None,
        ax=None,
    ):

        if size is None:
            size = 20
        if labels is None:
            labels = self.clf.labels_
        if critical_points is None:
            critical_points = self.clf.modes_

        X = self._get_data_2d()

        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        for label in np.unique(self.clf.labels_):
            label_mask = self.clf.labels_ == label

            if label == -1:
                sc = ax.scatter(
                    X[label_mask, 0], X[label_mask, 1], c="#bbbbbb", alpha=alpha, s=size
                )
            else:
                sc = ax.scatter(X[label_mask, 0], X[label_mask, 1], alpha=alpha, s=size)

            if show_critical_points:
                r = critical_points[label]
                ax.scatter(
                    X[r, 0],
                    X[r, 1],
                    c=sc.get_facecolor(),
                    marker="*",
                    alpha=alpha,
                    s=10 * size,
                    edgecolors="#222222",
                )

        ax.set_axis_off()

        if fig is not None:
            return fig

    def plot_merge_tree(
            self,
            use_function_value=False,
            color_by_label=True,
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
            if not hasattr(self, "_tree"):
                tree = clf_to_tree(self.clf)
                self._tree = tree
            else:
                tree = self._tree
            
            if show_singletons:
                tree = tree.extend_to_vertices()

            fig = tree.plot(
                use_function_value=use_function_value,
                color_by_label=color_by_label,
                show_singletons=show_singletons,
                ax=ax,
                marker=marker,
                marker_size=marker_size,
                marker_color=marker_color,
                line_color=line_color,
                line_width=line_width,
                line_style=line_style
            )

            return fig