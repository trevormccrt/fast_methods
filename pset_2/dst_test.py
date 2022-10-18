import numpy as np
from scipy import integrate, fft

import dst, fd_core


def test_inverse():
    signal = np.random.uniform(-1, 1, 50)
    forward = dst.dst_1(signal)
    backward = dst.idst_1(forward)
    np.testing.assert_allclose(backward, signal)


def test_physical_inverse():
    signal = np.random.uniform(-1, 1, 50)
    forward = dst.dst_1_physical(signal)
    backward = dst.idst_1_physical(forward)
    np.testing.assert_allclose(backward, signal)


def test_physical_normalization():
    def true_f(x):
        return x**2

    def true_integrand(x, k):
        return true_f(x) * np.sin((k+1) * x)

    k=3
    true_result = integrate.quad(lambda x: true_integrand(x, k), 0, np.pi)
    grid = np.arange(start=0, stop=np.pi, step=0.0001)
    f_grid = true_f(grid)
    dst_result = dst.dst_1_physical(f_grid)
    np.testing.assert_allclose(true_result[0], dst_result[k], rtol=1e-5)
