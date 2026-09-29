from tomato.core_ import ToMAToCore
from tomato.tomato_ import ToMATo


def _run_compilation_test():
    import numpy as np
    X = np.random.rand(50, 2)
    ToMATo().fit(X)
    return

_run_compilation_test()