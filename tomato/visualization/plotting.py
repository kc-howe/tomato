from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.sparse import triu
from sklearn.manifold import SpectralEmbedding

from tomato.core_ import ToMAToCore
from tomato.postprocessing.merge_tree import clf_to_tree


def _get_line_color(weights, color, cmap):
    if color is None and cmap is not None:
        return weights, None
    if color is None:
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    return None, color


def _draw_edges(ax, X, graph, color, cmap, alpha, linewidth):
    graph = triu(graph.maximum(graph.T), k=1, format="csr")
    graph.eliminate_zeros()
    coo = graph.tocoo()
    segments = np.stack([X[coo.row, :2], X[coo.col, :2]], axis=1)
    array, color = _get_line_color(coo.data, color, cmap)
    ax.add_collection(
        LineCollection(
            segments,
            colors=color,
            alpha=alpha,
            array=array,
            linewidth=linewidth,
            zorder=1,
            cmap=cmap if array is not None else None,
        )
    )
    return


def plot_network(
    graph, X, color=None, cmap="viridis", alpha=1.0, linewidth=1.0, ax=None
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.scatter(X[:, 0], X[:, 1], color="#eeeeee")
    _draw_edges(ax, X, graph, color, cmap, alpha, linewidth)
    ax.set_axis_off()

    return fig


class ToMAToVisualization:
    """
    Plotting helpers for a fitted ToMATo model.
    """

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

        A = self.clf.graph_persistence_.adjacency
        A = A.maximum(A.T).tocsr()
        A.data[:] = 1.0
        self._data_low_dim = SpectralEmbedding(
            n_components=2,
            affinity="precomputed",
            random_state=getattr(self.clf, "random_state", None),
        ).fit_transform(A)

        return self._data_low_dim

    def plot_cluster_persistences(
        self, threshold: Optional[float] = None, ax=None
    ) -> Optional[plt.Figure]:
        """
        Plot the cluster persistences for the fitted model.

        Parameters
        ----------
        threshold : float or None
            Persistence threshold used to highlight persistent critical points.
            If None, uses `clf.threshold_` computed during `fit()`.
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure and axes.

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure, or None when `ax` was supplied.
        """
        if threshold is None:
            threshold = self.clf.threshold_

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1)

        sort_index = np.argsort(self.clf.persistences_)
        sorted_persistences = self.clf.persistences_[sort_index]
        sorted_modes = self.clf.all_modes_[sort_index]
        index = np.arange(sorted_persistences.shape[0])

        mask = sorted_persistences < threshold

        ax.bar(index[mask], sorted_persistences[mask], color="C0")
        ax.bar(index[~mask], sorted_persistences[~mask], color="C1")
        if np.isfinite(threshold):
            ax.axhline(threshold, color="grey", linestyle="--", zorder=-1)
        ax.set_xticks(index, sorted_modes)
        ax.set_xlabel("Cluster Mode Index")
        ax.set_ylabel("Cluster Persistence")

        return fig

    def plot_persistence_diagram(
        self, threshold: Optional[float] = None, ax=None
    ) -> Optional[plt.Figure]:
        """
        Plot the persistence diagram for the fitted model.

        Parameters
        ----------
        threshold : float or None
            Persistence threshold used to highlight persistent critical points.
            If None, uses `clf.threshold_` computed during `fit()`.
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure and axes.

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure, or None when `ax` was supplied.
        """
        if threshold is None:
            threshold = self.clf.threshold_

        diagram = self.clf.graph_persistence_.diagram_
        x = np.linspace(diagram.min(), diagram.max())

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1)

        if np.isfinite(threshold) and threshold > 0.0:
            mask = self.clf.graph_persistence_.persistences_ >= threshold
            ax.scatter(diagram[~mask, 0], diagram[~mask, 1], color="C0")
            ax.scatter(diagram[mask, 0], diagram[mask, 1], color="C1")
            ax.plot(x, x, color="k")
            ax.plot(x, x + threshold, linestyle="--", color="lightgrey")
        else:
            ax.scatter(diagram[:, 0], diagram[:, 1], color="C0")
            ax.plot(x, x, linestyle="--", color="k")

        if self.clf.ascending:
            ax.set_xlabel("death")
            ax.set_ylabel("birth")
        else:
            ax.set_xlabel("birth")
            ax.set_ylabel("death")

        return fig

    def plot_graph(
        self, color=None, cmap="viridis", alpha=1.0, linewidth=1.0, ax=None
    ) -> Optional[plt.Figure]:
        X = self._get_data_2d()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1)

        ax.scatter(X[:, 0], X[:, 1], color="#eeeeee")
        _draw_edges(
            ax,
            X,
            self.clf.graph_persistence_.adjacency,
            color,
            cmap,
            alpha,
            linewidth,
        )
        ax.set_axis_off()

        return fig

    def _plot_critical_graph(self, graph, color, cmap, alpha, linewidth, ax):
        X = self._get_data_2d()
        gp = self.clf.graph_persistence_
        cp, f = gp.critical_points, gp.f

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1)

        ax.scatter(X[:, 0], X[:, 1], color="#eeeeee")
        _draw_edges(ax, X, graph, color, cmap, alpha, linewidth)
        ax.scatter(
            X[cp, 0],
            X[cp, 1],
            c=f[cp],
            cmap=cmap,
            edgecolors="#222222",
            zorder=2,
        )
        ax.set_axis_off()

        return fig

    def plot_gradient(
        self, color=None, cmap="viridis", alpha=1.0, linewidth=1.0, ax=None
    ) -> Optional[plt.Figure]:
        return self._plot_critical_graph(
            self.clf.graph_persistence_.gradient, color, cmap, alpha, linewidth, ax
        )

    def plot_mst(
        self, color=None, cmap="viridis", alpha=1.0, linewidth=1.0, ax=None
    ) -> Optional[plt.Figure]:
        return self._plot_critical_graph(
            self.clf.graph_persistence_.mst, color, cmap, alpha, linewidth, ax
        )

    def plot_labels(
        self,
        labels=None,
        critical_points=None,
        show_critical_points=False,
        alpha=1.0,
        size=None,
        ax=None,
    ) -> Optional[plt.Figure]:

        size = 20 if size is None else size
        X = self._get_data_2d()

        labels = np.asarray(self.clf.labels_ if labels is None else labels)
        if labels.shape[0] != X.shape[0]:
            labels = labels[self.clf.duplicate_index_]
        if critical_points is None:
            critical_points = self.clf.modes_

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1)

        for label in np.unique(labels):
            label_mask = labels == label

            if label == -1:
                ax.scatter(
                    X[label_mask, 0],
                    X[label_mask, 1],
                    c="#bbbbbb",
                    alpha=alpha,
                    s=size,
                )
                continue

            sc = ax.scatter(X[label_mask, 0], X[label_mask, 1], alpha=alpha, s=size)

            if show_critical_points and label in critical_points:
                r = critical_points[label]
                ax.scatter(
                    X[r, 0],
                    X[r, 1],
                    c=sc.get_facecolor(),
                    marker="*",
                    alpha=alpha,
                    s=10 * size,
                    edgecolors="#222222",
                    zorder=3,
                )

        ax.set_axis_off()

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
        color_by_label : bool, default=True
            Whether to color nodes according to their canonical label. Nodes
            with canonical label -1 are colored light grey.
        show_singletons : bool, default=False
            Whether to include per-vertex singleton leaves added by
            `extend_to_vertices`. When False, only basin leaves and merge
            nodes are drawn. When True, singleton leaves are drawn as regular
            leaf nodes, which can be dense for large datasets.
        ax : matplotlib.axes.Axes or None, default=None
            Axes on which to draw the tree. If None, a new figure is created.
        marker, marker_size, marker_color, line_color, line_width : optional
            Style overrides. If None, the tree's own defaults are used.
            `marker_color` is ignored when `color_by_label=True`.
        line_style : str, default="-"
            Line style of the edges connecting nodes ("-", "--", ":", "-.").

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure, or None when an axes object was supplied.
        """
        if not hasattr(self, "_tree"):
            self._tree = clf_to_tree(self.clf)

        tree = self._tree

        if show_singletons:
            tree = tree.extend_to_vertices()

        return tree.plot(
            use_function_value=use_function_value,
            color_by_label=color_by_label,
            show_singletons=show_singletons,
            ax=ax,
            marker=marker,
            marker_size=marker_size,
            marker_color=marker_color,
            line_color=line_color,
            line_width=line_width,
            line_style=line_style,
        )
