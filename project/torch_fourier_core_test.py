import numpy as np
import torch

from project import fourier_core, torch_fourier_core


def test_transform_equivalence():
    x_dims = (2**6, 2**7, 2**5, 2**4)
    axis_from = np.random.randint(0, len(x_dims))
    x = np.random.uniform(-1, 1, x_dims)
    y = np.random.uniform(-1, 1, x_dims)
    scipy_coeffs = fourier_core.fourier_series_coeffs(x, axis_from)
    scipy_ifft = fourier_core.nifft(x, axis_from)
    scipy_conv = fourier_core.fft_convolution_integral(x, y, axis_from)
    with torch.no_grad():
        torch_coeffs = torch_fourier_core.fourier_series_coeffs(torch.from_numpy(x), axis_from)
        torch_ifft = torch_fourier_core.nifft(torch.from_numpy(x), axis_from)
        torch_conv = torch_fourier_core.fft_convolution_integral(torch.from_numpy(x), torch.from_numpy(y), axis_from)
    np.testing.assert_allclose(scipy_coeffs, torch_coeffs)
    np.testing.assert_allclose(scipy_ifft, torch_ifft)
    np.testing.assert_allclose(scipy_conv, torch_conv)
