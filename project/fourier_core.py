import numpy as np


def fourier_series_coeffs(x, axes_from=None):
    axes = None
    if axes_from is not None:
        axes = list(range(len(x.shape)))[axes_from:]
    return np.fft.fftn(x, norm="forward", axes=axes)


def nifft(x, axes_from=None):
    axes = None
    if axes_from is not None:
        axes = list(range(len(x.shape)))[axes_from:]
    return np.fft.ifftn(x, norm="forward", axes=axes)


def fft_convolution_integral(x_series_coeffs, y_series_coeffs, axes_from=None):
    n = len(np.shape(x_series_coeffs))
    return ((2 * np.pi) ** n) * nifft(x_series_coeffs * y_series_coeffs, axes_from=axes_from)
