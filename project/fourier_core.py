import numpy as np


def fourier_grid(k):
    grid = np.linspace(start=0, stop=2 * np.pi, num=2 ** k + 1)
    return grid[:-1]


def get_k_vals(n):
    return np.fft.fftfreq(n, 1/n).astype(int)


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


def evaluate_fourier_series(k_vals, series_coeffs, x):
    x_tiled_k = np.tile(np.expand_dims(x, -1), (*np.ones_like(np.shape(x)), len(k_vals)))
    exponentials = np.exp(1j * k_vals * x_tiled_k)
    return np.einsum("...k, k -> ...", exponentials, series_coeffs)

