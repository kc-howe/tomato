from tomato.regionalization.geotomato_ import GeoToMATo


def _run_compilation_test():
    import numpy as np
    from scipy.sparse import csr_matrix

    X = np.random.random((50, 2))
    graph = csr_matrix(np.eye(50, k=-1) + np.eye(50, k=0) + np.eye(50, k=1))

    GeoToMATo().fit(graph, X)

    return


_run_compilation_test()
