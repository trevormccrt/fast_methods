import numpy as np
import torch


def fourier_grid(k):
    grid = torch.linspace(start=0, end=2 * torch.pi, steps=2 ** k + 1)
    return grid[:-1]


def fourier_series_coeffs(x, axes_from=None):
    axes = None
    if axes_from is not None:
        axes = tuple(np.arange(start=0, stop=x.dim(), step=1)[axes_from:])
    return torch.fft.fftn(x, norm="forward", dim=axes)


def nifft(x, axes_from=None):
    axes = None
    if axes_from is not None:
        axes = tuple(np.arange(start=0, stop=x.dim(), step=1)[axes_from:])
    return torch.fft.ifftn(x, norm="forward", dim=axes)


def fft_convolution_integral(x_series_coeffs, y_series_coeffs, axes_from=None):
    n = x_series_coeffs.dim()
    return ((2 * np.pi) ** n) * nifft(x_series_coeffs * y_series_coeffs, axes_from=axes_from)
