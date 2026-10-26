from tomato.regionalization.geotomato_ import GeoToMATo


def _run_compilation_test():
    import numpy as np
    from scipy.sparse import csr_matrix
    from tomato._graph_utils import merge_disconnected_components

    X = np.random.random((50, 2))
    graph = csr_matrix(np.eye(50, k=-1) + np.eye(50, k=0) + np.eye(50, k=1))

    GeoToMATo().fit(graph, X)

    # Warm up connected components merging
    repair_graph = csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    )
    merge_disconnected_components(
        np.array([0, 1, 0], dtype=np.int64),
        repair_graph.indptr,
        repair_graph.indices,
        np.array([0], dtype=np.int64),
    )

    return


_run_compilation_test()
