import numpy as np
from scipy import fftpack


def dst_1(signal, axis=-1):
    N = signal.shape[axis]
    return fftpack.dst(signal, type=1, axis=axis)/N


def idst_1(signal, axis=-1):
    N = signal.shape[axis]
    return N/(2 * (N+1)) * fftpack.dst(signal, type=1, axis=axis)


def dst_1_physical(signal, axis=-1):
    N = signal.shape[axis]
    return np.pi/(2 * (N+1)) * fftpack.dst(signal, type=1, axis=axis)


def idst_1_physical(signal, axis=-1):
    return 1/np.pi * fftpack.dst(signal, type=1, axis=axis)
