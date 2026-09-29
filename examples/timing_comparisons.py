import time

import hdbscan
import numpy as np
import pandas as pd
import scipy.cluster
import seaborn as sns
import sklearn.cluster
from sklearn.datasets import make_blobs

from tomato.tomato_ import ToMATo


def benchmark_algorithm(
    dataset_sizes,
    cluster_function,
    function_args,
    function_kwds,
    dataset_dimension=2,
    max_time=45,
    sample_size=2,
):

    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result = np.nan * np.ones((len(dataset_sizes), sample_size))
    for index, size in enumerate(dataset_sizes):
        dataset_n_clusters = int(np.log2(size))
        for s in range(sample_size):
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            data, _ = make_blobs(
                n_samples=size, n_features=dataset_dimension, centers=dataset_n_clusters
            )

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time

            # If we are taking more than max_time then abort -- we don't
            # want to spend excessive time on slow algorithms
            if time_taken > max_time:
                result[index, s] = time_taken
                return pd.DataFrame(
                    np.vstack([dataset_sizes.repeat(sample_size), result.flatten()]).T,
                    columns=["x", "y"],
                )
            else:
                result[index, s] = time_taken

    # Return the result as a dataframe for easier handling with seaborn afterwards
    return pd.DataFrame(
        np.vstack([dataset_sizes.repeat(sample_size), result.flatten()]).T,
        columns=["x", "y"],
    )


def run_test_large():
    base = 100_000
    huge_dataset_sizes = np.arange(1, 16) * base

    print("Sklearn K-Means")
    k_means = sklearn.cluster.KMeans(10)
    huge_k_means_data = benchmark_algorithm(
        huge_dataset_sizes, k_means.fit, (), {}, max_time=120, sample_size=2
    )

    print("Scipy K-Means")
    huge_scipy_k_means_data = benchmark_algorithm(
        huge_dataset_sizes,
        scipy.cluster.vq.kmeans,
        (10,),
        {},
        max_time=240,
        sample_size=2,
    )

    print("HDBSCAN")
    hdbscan_boruvka = hdbscan.HDBSCAN(algorithm="boruvka_kdtree")
    huge_hdbscan_data = benchmark_algorithm(
        huge_dataset_sizes, hdbscan_boruvka.fit, (), {}, max_time=240, sample_size=4
    )

    print("ToMATo")
    tomatoscan = ToMATo()
    huge_tomatoscan_data = benchmark_algorithm(
        huge_dataset_sizes, tomatoscan.fit, (), {}, max_time=240, sample_size=4
    )

    results_dict = {
        "base_size": base,
        "max_size": huge_dataset_sizes.max(),
        "K-Means": huge_k_means_data,
        "Scipy K-Means": huge_scipy_k_means_data,
        "HDBSCAN": huge_hdbscan_data,
        "ToMATo": huge_tomatoscan_data,
    }

    return results_dict


def run_test_small():
    base = 20_000
    huge_dataset_sizes = np.arange(1, 16) * base

    print("Sklearn K-Means")
    k_means = sklearn.cluster.KMeans(10)
    huge_k_means_data = benchmark_algorithm(
        huge_dataset_sizes, k_means.fit, (), {}, max_time=120, sample_size=2
    )

    print("Scipy K-Means")
    huge_scipy_k_means_data = benchmark_algorithm(
        huge_dataset_sizes,
        scipy.cluster.vq.kmeans,
        (10,),
        {},
        max_time=240,
        sample_size=2,
    )

    print("HDBSCAN")
    hdbscan_boruvka = hdbscan.HDBSCAN(algorithm="boruvka_kdtree")
    huge_hdbscan_data = benchmark_algorithm(
        huge_dataset_sizes, hdbscan_boruvka.fit, (), {}, max_time=240, sample_size=4
    )

    print("ToMATo")
    tomatoscan = ToMATo()
    huge_tomatoscan_data = benchmark_algorithm(
        huge_dataset_sizes, tomatoscan.fit, (), {}, max_time=240, sample_size=4
    )

    print("DBSCAN")
    dbscan = sklearn.cluster.DBSCAN()
    huge_dbscan_data = benchmark_algorithm(
        huge_dataset_sizes, dbscan.fit, (), {}, max_time=240, sample_size=2
    )

    results_dict = {
        "base_size": base,
        "max_size": huge_dataset_sizes.max(),
        "K-Means": huge_k_means_data,
        "Scipy K-Means": huge_scipy_k_means_data,
        "HDBSCAN": huge_hdbscan_data,
        "ToMATo": huge_tomatoscan_data,
        "DBSCAN": huge_dbscan_data,
    }

    return results_dict


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    RUN_LARGE = True

    if RUN_LARGE:
        results = run_test_large()
        title_prefix = "large"
        plot_max = 110
    else:
        results = run_test_small()
        title_prefix = "small"
        plot_max = 10

    for alg_name, alg_data in results.items():
        if alg_name not in ["base_size", "max_size"]:
            sns.regplot(
                x="x",
                y="y",
                data=alg_data,
                order=2,
                label=alg_name,
                x_estimator=np.mean,
            )

    plt.gca().axis([0, results["max_size"] + results["base_size"], 0, plot_max])
    plt.gca().set_xlabel("Number of data points")
    plt.gca().set_ylabel("Time taken to cluster (s)")
    plt.title("Performance Comparison of K-Means, DBSCAN, and HDBSCAN")
    plt.legend()
    plt.savefig(f"output/{title_prefix}_clustering_performance_comparison.png", dpi=300)
    plt.show()
