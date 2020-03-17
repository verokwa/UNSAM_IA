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

def plot_posterior_ex4():
    import time
    from IPython import display
    import numpy.random as rr

    # El paquete en el que están implementadas muchas funciones de distribución es scipy.stats
    import scipy.stats as st

    # Genera los puntos para graficar la curva.
    x = np.linspace(0, 1, 1000)

    # Define el mu verdadero
    mu_t = 0.45

    # Define número de tiradas
    N = 500

    # Genera al azar 100 tiradas de una moneda con mu = 0.5.
    t = np.where(rr.rand(N) < mu_t, 1, 0)

    # Inicializa los contadores de las caras y las cecas
    m = 0
    l = 0

    # Define el tiempo de espera máximo y mínimo
    tmin = 10/N
    tmax = 0.5
    tsleep = np.logspace(log(1), log(0.1), len(t))

    # Grafica el prior
    plt.plot(0, st.beta(1,1).cdf(0.5), 'o')
    plt.axhline(0.95, color='0.5', ls='--')

    ax = plt.gca()
    ax.text(0.8, 0.8,'N=0; c=0; e=0', transform=ax.transAxes, va='center', ha='center')

    # sleep
    time.sleep(tmax)

    for i, tt in enumerate(t):

        # Force plot display
        display.clear_output(wait=True)
        display.display(plt.gcf())

        if tt == 1:
            m += 1
        elif tt == 0:
            l += 1

        posterior = st.beta(1+m, 1+l)
        ax = plt.gca()
        ydata = list(ax.lines[0].get_ydata())

        # Compute mass around maximum
        mu_ml = m/(i+1)
        delta = abs(mu_ml - 0.5)
        mass = posterior.cdf(mu_ml + delta) - posterior.cdf(mu_ml - delta)

        ydata.append(mass)

        ax.lines[0].set_data([np.arange(i+2), ydata])

        # Update scale
        ax.relim()
        ax.autoscale_view(True,True,True)

        # Update text
        ax.texts[0].set_text('N={}; c={}; e={}'.format(str(i+1), str(m), str(l)))

        time.sleep(tsleep[i])
        #plt.clf()
    return
