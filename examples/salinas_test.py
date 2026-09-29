from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
from scipy.ndimage import label as ndimage_label
from scipy.sparse import coo_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from tomato.regionalization.geotomato_ import GeoToMATo

DATA_DIR = Path("./examples/data")
BLOCK_SIZE = 2
N_HOPS = 1
MASS_FRACTION = 1.0
NOISE_THRESHOLD = 0.75
N_NEIGHBORS = 16
N_CLUSTERS = 32
SAVE_IMAGE = True
SAVE_RESULTS = False


def read_mat_file(path, key):
    f = loadmat(path)
    data = f[key]
    return data


def _load_salinas(data_path=None, gt_path=None):
    if data_path is None:
        data_path = DATA_DIR / "Salinas_corrected.mat"
    if gt_path is None:
        gt_path = DATA_DIR / "Salinas_gt.mat"

    data = read_mat_file(data_path, "salinas_corrected")
    gt = read_mat_file(gt_path, "salinas_gt")

    data = data.astype(np.float32)
    gt = gt.astype(int)

    return data, gt


def pool_data(data, gt, block_size: int = 1):
    """
    Apply mean pooling to Salinas data and max-vote pooling to
    ground-truth labels.
    """
    if block_size == 1:
        return data, gt

    assert len(data.shape) == 3

    a0, a1, a2 = data.shape  # Original data dimensions
    b0, b1 = a0 // block_size, a1 // block_size  # Target data dimensions

    data_pooled = (
        data[: b0 * block_size, : b1 * block_size]
        .reshape(b0, block_size, b1, block_size, a2)
        .mean(axis=(1, 3))
    )

    gt_block = gt[: b0 * block_size, : b1 * block_size].reshape(
        b0, block_size, b1, block_size
    )
    gt_counts = np.stack(
        [(gt_block == k).sum(axis=(1, 3)) for k in range(1, gt.max() + 1)], axis=-1
    )
    gt_pooled = np.where(gt_counts.sum(-1) > 0, gt_counts.argmax(-1) + 1, 0)

    return data_pooled, gt_pooled


def queen_graph_grid(height, width):
    """
    Construct the queen contiguity graph for a regular grid.
    """
    n_vertices = height * width
    idx = np.arange(n_vertices).reshape(height, width)

    rows, cols = [], []
    for drow, dcol in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        row0, row1 = max(0, -drow), height - max(0, drow)
        col0, col1 = max(0, -dcol), width - max(0, dcol)
        rows.append(idx[row0:row1, col0:col1].ravel())
        cols.append(idx[row0 + drow : row1 + drow, col0 + dcol : col1 + dcol].ravel())

    row, col = np.concatenate(rows), np.concatenate(cols)

    graph = coo_matrix(
        (np.ones(len(row), dtype=bool), (row, col)), shape=(n_vertices, n_vertices)
    )
    graph = (graph + graph.T).tocsr().astype(bool)

    return graph


def knn_graph_grid(height, width, n_neighbors=8):
    """
    Construct the queen contiguity graph for a regular grid.
    """
    X = np.column_stack(np.divmod(np.arange(height * width), width)).astype(float)
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False).astype(
        bool
    )
    graph = (graph + graph.T).tocsr().astype(bool)
    return graph


def contiguous_gt_regions(gt):
    """
    Label the contiguous regions in ground-truth labels.
    """
    res = np.zeros_like(gt)
    label_next = 1
    for label in np.unique(gt[gt > 0]):
        labels, n = ndimage_label(gt == label, structure=np.ones((3, 3)))
        res[labels > 0] = labels[labels > 0] + label_next - 1
        label_next += n
    return res


def fragmentation(label_img):
    ks = np.unique(label_img)
    frag = sum(
        ndimage_label(label_img == k, structure=np.ones((3, 3)))[1] for k in ks
    ) / len(ks)
    return frag


def ssd(X, labels):
    res = sum(
        ((X[labels == k] - X[labels == k].mean(0)) ** 2).sum()
        for k in np.unique(labels)
    )
    return res


def score(pred_labels, gt_flat, gt_cc_flat, X, shape):

    m = gt_flat > 0

    k = len(np.unique(pred_labels))
    homogeneity, completeness, _ = homogeneity_completeness_v_measure(
        gt_cc_flat[m], pred_labels[m]
    )
    ari_class = adjusted_rand_score(gt_flat[m], pred_labels[m])
    ari_contig = adjusted_rand_score(gt_cc_flat[m], pred_labels[m])
    ssd_score = ssd(X, pred_labels)
    frag = fragmentation(pred_labels.reshape(shape))

    res = {
        "k": k,
        "ARI_class": ari_class,
        "ARI_contig": ari_contig,
        "homog_contig": homogeneity,
        "compl_contig": completeness,
        "SSD": ssd_score,
        "frag": frag,
    }

    return res


def show_labels(ax, labels, title, noise_value=-1, random_state=0):

    labels = np.asarray(labels)
    is_noise = labels == noise_value
    n_unique, inv = np.unique(labels[~is_noise], return_inverse=True)
    idx = np.full(labels.shape, -1)
    idx[~is_noise] = inv

    palette = [
        c for cm in (plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c) for c in cm.colors
    ]
    order = np.random.default_rng(random_state).permutation(len(n_unique))

    cmap = ListedColormap([palette[i % len(palette)] for i in order] or [(0, 0, 0)])
    cmap.set_bad("white")

    ax.imshow(
        np.ma.masked_where(is_noise, idx),
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=max(len(n_unique) - 1, 0),
    )

    ax.set_title(title)
    ax.axis("off")

    return


def plot_tests(runs, height, width, gt, gt_cc, out_path=None):
    fig, axes = plt.subplots(len(runs), 5, figsize=(20, 4 * len(runs)), squeeze=False)

    for row, run in zip(axes, runs):
        show_labels(row[0], gt, "Class GT", noise_value=0)
        show_labels(row[1], gt_cc, "Contiguous Regions GT", noise_value=0)

        row[2].imshow(-run["f"].reshape(height, width), cmap="coolwarm")
        row[2].set_title("f = -DTM")
        row[2].axis("off")

        show_labels(
            row[3], run["tomato"].reshape(height, width), "GeoToMATo", noise_value=-1
        )
        show_labels(row[4], run["ward"].reshape(height, width), "Ward")

        row[0].text(
            -0.05,
            0.5,
            f"{run['graph']}, k={run['k']}",
            transform=row[0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
        print(f"Saved: {out_path}")

    return fig


def load_salinas(data_path=None, gt_path=None, block_size: int = 1):
    if data_path is None:
        data_path = DATA_DIR / "Salinas_corrected.mat"
    if gt_path is None:
        gt_path = DATA_DIR / "Salinas_gt.mat"

    data, gt = _load_salinas(data_path, gt_path)
    data, gt = pool_data(data, gt, block_size=block_size)

    shape = data.shape

    data = StandardScaler().fit_transform(data.reshape(-1, data.shape[2]))
    data = np.clip(data, -3.0, 3.0)

    gt_cc = contiguous_gt_regions(gt)
    gt_flat, gt_cc_flat = gt.ravel(), gt_cc.ravel()

    return data, gt_flat, gt_cc_flat, shape


def _run_test(clf, data, graph=None):
    start = time()
    if graph is None:
        labels = clf.fit_predict(data)
    else:
        labels = clf.fit_predict(graph, data)
    end = time()
    elapsed = end - start
    return labels, elapsed


def run_salinas_tests(
    data_path=None,
    gt_path=None,
    block_size: int = 1,
    n_neighbors: int = 16,
    n_clusters: int = 16,
    n_hops: int = 2,
    mass_fraction: float = 0.5,
    noise_threshold: float = 0.5,
    save_results: bool = False,
    save_image: bool = False,
    results_out: str = None,
    image_out: str = None,
):

    data, gt_flat, gt_cc_flat, shape = load_salinas(
        data_path, gt_path, block_size=block_size
    )
    height, width, bands = shape

    graphs = [
        queen_graph_grid(height, width),
        knn_graph_grid(height, width, n_neighbors=n_neighbors),
    ]
    graph_names = ["Queen", "KNN"]

    clf_geotomato = GeoToMATo(
        n_hops=n_hops, mass_fraction=mass_fraction, n_clusters=n_clusters
    )

    info, runs = [], []
    for graph, name in zip(graphs, graph_names):
        clf_ward = AgglomerativeClustering(
            n_clusters=n_clusters, connectivity=graph, linkage="ward"
        )

        labels_tomato, time_tomato = _run_test(clf_geotomato, data, graph=graph)
        labels_ward, time_ward = _run_test(clf_ward, data)

        if noise_threshold > 0.0:
            labels_tomato = clf_geotomato.noise_aware_labels(threshold=noise_threshold)

        res_tomato = score(labels_tomato, gt_flat, gt_cc_flat, data, (height, width))
        res_ward = score(labels_ward, gt_flat, gt_cc_flat, data, (height, width))

        res_tomato.update(
            graph=name, method="GeoToMATo", target_k=n_clusters, seconds=time_tomato
        )
        res_ward.update(
            graph=name, method="GeoToMATo", target_k=n_clusters, seconds=time_ward
        )

        info.append(res_tomato)
        info.append(res_ward)

        runs.append(
            dict(
                graph=name,
                k=n_clusters,
                f=clf_geotomato.vertex_function_,
                tomato=labels_tomato,
                ward=labels_ward,
            )
        )

    cols = [
        "graph",
        "method",
        "target_k",
        "k",
        "ARI_class",
        "ARI_contig",
        "homog_contig",
        "compl_contig",
        "SSD",
        "frag",
        "seconds",
    ]

    res = pd.DataFrame(info)[cols].round(3)

    print(res.to_string(index=False))

    if save_results:
        if results_out is None:
            results_out = "salinas_test_results.csv"
        res.to_csv(results_out)

    gt = gt_flat.reshape(height, width)
    gt_cc = gt_cc_flat.reshape(height, width)

    if save_image:
        if image_out is None:
            image_out = "salinas_results.png"
        fig = plot_tests(runs, height, width, gt, gt_cc, out_path=image_out)
    else:
        fig = plot_tests(runs, height, width, gt, gt_cc)

    return res, fig


if __name__ == "__main__":
    run_salinas_tests(
        block_size=BLOCK_SIZE,
        n_neighbors=N_NEIGHBORS,
        n_clusters=N_CLUSTERS,
        n_hops=N_HOPS,
        mass_fraction=MASS_FRACTION,
        noise_threshold=NOISE_THRESHOLD,
        save_results=SAVE_RESULTS,
        save_image=SAVE_IMAGE,
    )
