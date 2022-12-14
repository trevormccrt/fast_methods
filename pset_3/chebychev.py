import numpy as np
from scipy import fftpack


def extrema_grid(N):
    n = np.arange(start=0, stop=N, step=1)
    return np.cos(np.pi * n/(N-1))


def cheb(x, axis=-1):
    N = np.shape(x)[axis]
    return 1/(N-1) * fftpack.dct(x, axis=axis, type=1)


def icheb(x, axis=-1):
    return 1/2 * fftpack.idct(x, axis=axis, type=1)

