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


def test_samples():

    # Crea un arreglo con los tamaños de las muestras que vamos a usar. Generamos valores desde 2^4 hasta 2^13.
    # Vamos a espacirlo en log, para tener más resolución en los tamaños pequeños.
    sizes = 2**np.arange(4, 14)
    
    results = np.empty([4, len(sizes), 2])
    # Itera sobre las muestras
    for i in range(4):
        # lee muestra i
        sample = np.loadtxt('../datasets/01_Aleatoreidad'
                            '_muestra{}.txt'.format(i+1), delimiter=',')
        
        f = []
        sd = []

        # Iteramos en los tamaños (usamos siempre el mismo seed?)
        for size in sizes:
            n_ad = np.sum((sample[:size, 0] > 0) * (sample[:size, 1] > 0))
            f_ad = n_ad / len(sample[:size])
            sd_f = sqrt(n_ad) / len(sample[:size])
            
            f.append(f_ad)
            sd.append(sd_f)
        
        results[i] = np.array([f, sd]).T
    return sizes, results


def plot_test_samples():
    from matplotlib import pyplot as plt
    
    sizes, results = test_samples()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i, r in enumerate(results):
        ax.errorbar(sizes, r[:, 0], r[:, 1], fmt='o', 
                    label='Muestra {}'.format(i+1))
        
    ax.legend()
    ax.axhline(0.25, color='0.5')
    ax.set_xscale('log')
    
    return
    
    