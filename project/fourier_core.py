import numpy as np


def fourier_series_coeffs(x, axes_from=None):
    axes = None
    if axes_from is not None:
        axes = list(range(len(x.shape)))[axes_from:]
    return np.fft.fftn(x, norm="forward", axes=axes)


def fft_convolution_integral(x_series_coeffs, y_series_coeffs):
    n = len(np.shape(x_series_coeffs))
    return ((2 * np.pi) ** n) * np.fft.ifftn(x_series_coeffs * y_series_coeffs, norm="forward")
