from math import sqrt
import numpy as np

def test_fraccion(frac_func):
    np.random.seed(1234)
    for N in [10, 50, 100, 500]:
        assert np.allclose(frac_func(N, 1234),
        _baseline_fraccion_ad(N, 1234)), N
    print(u"☺︎")

def _baseline_fraccion_ad(n, seed=None):

    # Fija la semilla
    if seed is not None:
        np.random.seed(seed)

    # Crea la muestra.
    sample = np.random.rand(n, 2)
    sample = sample * 2 - 1

    n_ad = np.sum((sample[:, 0] > 0) * (sample[:, 1] > 0))
    f_ad = n_ad / n
    sd_f = sqrt(n_ad)/len(sample)

    return f_ad, sd_f
