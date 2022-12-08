import numpy as np


def fourier_series_coeffs(x):
    return np.fft.fftn(x, norm="forward")


def fft_convolution_integral(x_series_coeffs, y_series_coeffs):
    n = len(np.shape(x_series_coeffs))
    return ((2 * np.pi) ** n) * np.fft.ifftn(x_series_coeffs * y_series_coeffs, norm="forward")
