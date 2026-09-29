import numpy as np
import matplotlib.pyplot as plt


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_cluster_colors(clf):
    unique_labels = [label for label in np.unique(clf.labels_) if label != -1]
    color_map = plt.rcParams["axes.prop_cycle"].by_key().get("color")
    colors = {
        label: color_map[i % len(color_map)] for i, label in enumerate(unique_labels)
    }
    return colors


def get_soft_cluster_colors(clf, cluster_colors, temperature=1.0):
    full_probas = clf.full_assignment_probabilities(
        clf.labels_, temperature=temperature
    )
    color_arr = np.vstack([hex_to_rgb(color) for color in cluster_colors.values()])
    soft_colors = full_probas @ color_arr
    soft_colors /= 255.0
    return list(tuple(color) for color in soft_colors)


if __name__ == "__main__":

    from sklearn.datasets import make_blobs
    from tomato.tomato_ import ToMATo

    rng = np.random.RandomState(42)
    X, _ = make_blobs(n_samples=5_000, centers=10, cluster_std=1.5, random_state=rng)

    fig, axs = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(8)

    n_clusters = [2, 4, 7]
    for i, n in enumerate(n_clusters):
        clf = ToMATo(n_clusters=n, random_state=42)
        clf.fit(X)

        cluster_colors = get_cluster_colors(clf)
        flat_colors = get_soft_cluster_colors(clf, cluster_colors, temperature=0.0)
        soft_colors = get_soft_cluster_colors(clf, cluster_colors, temperature=1.0)

        axs[0, i].scatter(X[:, 0], X[:, 1], alpha=0.5, c=flat_colors)
        axs[1, i].scatter(X[:, 0], X[:, 1], alpha=0.5, c=soft_colors)
        axs[0, i].set_title(f"{n} Clusters, Flat Clustering")
        axs[1, i].set_title(f"{n} Clusters, Soft Clustering")
        axs[0, i].axis("off")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()
