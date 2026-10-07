<p align="center">
  <img src="https://raw.githubusercontent.com/kc-howe/tomato/main/docs/images/tomato-logo.svg", height=400 />
</p>

**Author**: Kenneth Howe

**Version**: 0.4.4

</br>

# ToMATo

#### *ToMATo*: Topological Mode Analysis Tool.

ToMATo quickly finds intuitive clusters in data. It does so by choosing the most topologically persistent clusters in a neighborhood graph with respect to a vertex function, typically a smooth density estimate. Its clusters therefore correspond to persistent modes of some density estimate (or other vertex function), rather than to components of an arbitrary distance or density threshold.

ToMATo was originally introduced as an algorithm for finding clusters in any graph equipped with any vertex function. This repository preserves that generic form via the `ToMAToCore` class, which takes a graph and vertex function directly. For general-purpose clustering of point cloud data, `ToMATo` fixes the choice of graph and vertex function, using the RMS distance estimator of *distance to a measure* (Chazal et al., 2011) on a KNN graph, as used in the original ToMATo paper.

This addreses the needs of the most common spatial clustering use cases, however the underlying ToMATo algorithm is a highly extensible, general framework for graph clustering, with applications extending to network analysis, geographic regionalization, and other graph-based modeling tasks.

ToMATo was introduced in 2011 by Chazal, et al.:

> Chazal, F., Guibas, L. J., Oudot, S., & Skraba, P. (2011). Persistence-based clustering in Riemannian manifolds. Proceedings of the 27th Annual ACM Symposium on Computational Geometry (SCG '11), 97–106. https://geometry.stanford.edu/paper/cgos-pbcrm-11/cgos-pbcrm-11.pdf


<p align="center">
  <figure style="text-align: center;">
    <img src="https://raw.githubusercontent.com/kc-howe/tomato/main/docs/images/tomato-example.png", width=800 />
    <figcaption>ToMATo Example Clustering.</figcaption>
  </figure>
</p>

## Installation

To install, navigate to the package directory and run:

``` bash
pip install -r requirements.txt
pip install .
```

## How to Use

The `ToMATo` class is designed to conform to the sklearn API.

``` python
from tomato import ToMATo
from sklearn.datasets import make_blobs

X, _ = make_blobs(1_000, random_state=42)

clf = ToMATo()
labels = clf.fit_predict(X)
```

## Features

The ToMATo algorithm is grounded in a mathematical theory that provides a concise framework for a number of features valuable to a clustering tool:
- Fast, hierarchical, density-based clustering
- *Optional* noise-awareness
- *Optional* number of clusters specification
- Assignment probabilities
- Soft clustering
- Out-of-sample prediction
- Spatially constrained regionalization

### 1. General Graph Persistence

ToMATo calculates the 0th-order persistence of piecewise-linear functions on graphs. This enables ToMATo to support a several features applicable to a wide range of data sources. The persistence diagram underlying a ToMATo clustering can also be obtained from the exposed `ToMATo.graph_persistence_` object and used directly.

**Note:** For readers unfamiliar with topological persistence, [this blog post](https://christian.bock.ml/posts/persistent_homology/) by Christian Bock provides a concise introduction.

#### Precomputed Graphs and Functions

The `ToMAToCore` class exposes a `fit` method that accepts two inputs: `graph` (of type `csr_matrix`) and `f` (of type `np.ndarray`). This interface accommodates users who already have data in graph form, who prefer a custom method of graph construction, or who have a custom descriptor function.

#### Persistence Diagram Inspection

The `ToMATo.graph_persistence_` attribute stores a `GraphPersistence` object providing access to the persistence diagram, spanning tree, and other persistence data. This can be used to characterize the cluster function or to determine custom persistence thresholds for labeling.

#### Optional Number of Clusters Specification

Because the spanning tree can be cut at a depth that guarantees an exact number of surviving nodes, `ToMATo` can return a fixed number of clusters when required. Alternatively, a minimum number of clusters can be specified, for example to prevent `ToMATo` from returning a single cluster.

#### Persistence Threshold Selection Options

Several methods are available for selecting persistence thresholds:
- User-specified threshold.
- Maximum jump statistic (default).
- Exponential model z-score.
- Cut at a specified number of clusters.

### 2. Assignment Probabilities

ToMATo applies a fast post-processing step to determine pointwise assignment strength for a given clustering, by comparing each point's local function value to the cluster average. This supports two downstream capabilities:

#### Optional Noise-Aware Labeling

An assignment strength threshold (50% by default) can be specified to classify points as noise when their assignment strength falls below it; such points receive a uniform label of `-1`.

#### Soft Clustering

Assignment probabilities can also be computed for every cluster in a labeling, enabling soft clustering in near-linear time.

<p align="center">
  <figure style="text-align: center;">
    <img src="https://raw.githubusercontent.com/kc-howe/tomato/main/docs/images/tomato-soft-clustering-example.png", width=800 />
    <figcaption>ToMATo Soft Clustering Example with Color Blending.</figcaption>
  </figure>
</p>

#### Geographic (Spatially Constrained) Regionalization

Because the core ToMATo algorithm can cluster any graph equipped with any vertex function, variants of the algorithm can be designed to perform spatially constrained clustering, commonly called "geographic regionalization" in geographic applications. The supplied graph should encode the spatial contiguity structure to preserve in the returned regions, and the passed data should encode attributes to cluster on.

This repository contains a `GeoToMATo` class implementing this kind of regionalization.

```python
from tomato.regionalization import GeoToMATo

clf = GeoToMATo()
clf.fit(graph, X)
```

<p align="center">
  <figure style="text-align: center;">
    <img src="https://raw.githubusercontent.com/kc-howe/tomato/main/docs/images/tomato-regionalization-example.png", width=800 />
    <figcaption>GeoToMATo Regionalization Example on Salinas Valley, CA Hyperspectral Imagery.</figcaption>
  </figure>
</p>

### 3. Visualization

The `ToMAToVisualization` class offers methods for visualizing:
- Cluster labels
- Input graph and vertex function
- Graph gradients
- Merge tree
- Persistence diagram

The class is initialized using a fitted instance of the ToMATo clusterer:

``` python
from tomato.visualization import ToMAToVisualization

# Assuming `clf` has already been fit on `X`
viz = ToMAToVisualization(clf)
viz.plot_labels(X)
viz.plot_persistence_diagram()
```

<p align="center">
  <figure style="text-align: center;">
    <img src="https://raw.githubusercontent.com/kc-howe/tomato/main/docs/images/tomato-labels.png", width=800 />
    <figcaption>Labels plot.</figcaption>
  </figure>
</p>

<p align="center">
  <figure style="text-align: center;">
    <img src="https://raw.githubusercontent.com/kc-howe/tomato/main/docs/images/tomato-pd.png", width=800 />
    <figcaption>Persistence diagram plot.</figcaption>
  </figure>
</p>

## Performance

ToMATo runs in near-linear time, making it significantly faster than comparable algorithms such as HDBSCAN, whose complexity is typically worse than $O(n \log n)$.

The following timing results compare ToMATo against DBSCAN, HDBSCAN, and two implementations of K-Means:

<p align="center">
  <img src="https://raw.githubusercontent.com/kc-howe/tomato/main/docs/images/tomato-small-timings.png", width=600 />
</p>

Extended to 1.5 million data points, ToMATo maintains near-linear scaling with relatively fast clustering times:

<p align="center">
  <img src="https://raw.githubusercontent.com/kc-howe/tomato/main/docs/images/tomato-large-timings.png", width=600 />
</p>

*Timing and plotting code adapted from the [HDBSCAN repository](https://github.com/scikit-learn-contrib/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations-v0.7.ipynb).*

## License

ToMATo is released under the MIT License. See `LICENSE` for the full license text.