import numpy as np
from scipy import integrate

from project import fourier_core

# i have trust issues with the FFT


def _complex_quadrature(func, a, b, **kwargs):

    real_integral = integrate.quad(lambda x: np.real(func(x)), a, b, **kwargs)
    imag_integral = integrate.quad(lambda x: np.imag(func(x)), a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]


def _double_complex_quadrature(func, a, b, gfun, hfun, **kwargs):
    real_integral = integrate.dblquad(lambda y, x: np.real(func(y, x)), a, b, gfun, hfun, **kwargs)
    imag_integral = integrate.dblquad(lambda y, x: np.imag(func(y, x)), a, b, gfun, hfun, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]


def test_fourier_series_coeffs_1d():
    test_f = lambda t: np.exp(np.cos(10 * t) ** 4 - np.sin(2 * t))
    fourier_integrand = lambda t, k, f: 1/(2 * np.pi) * np.exp(-1j * k * t) * f(t)
    t_vals = np.linspace(start=0, stop=2 * np.pi, num=2**10 + 1)
    t_vals = t_vals[:-1]
    f_vals = test_f(t_vals)
    fft_coeffs = fourier_core.fourier_series_coeffs(f_vals)
    k_vals = np.arange(start=0, stop=50, step=1)
    exact_coeffs = []
    for k in k_vals:
        exact_coeff = _complex_quadrature(lambda t: fourier_integrand(t, k, test_f), 0, 2 * np.pi)
        exact_coeffs.append(exact_coeff)
    np.testing.assert_allclose(fft_coeffs[:len(k_vals)], exact_coeffs, atol=1e-11)


def test_fourier_series_coeffs_2d():
    test_f = lambda t_1, t_2: np.exp(np.cos(2 * (t_1 + t_2)) ** 4 - np.sin(2 * t_1) + np.cos(2 * t_2))
    fourier_integrand = lambda t_1, t_2, k_1, k_2, f: 1 / (2 * np.pi)**2 * np.exp(-1j * k_1 * t_1) * np.exp(-1j * k_2 * t_2) * f(t_1, t_2)
    t_vals = np.linspace(start=0, stop=2 * np.pi, num=2 ** 10 + 1)
    t_vals = t_vals[:-1]
    t_1_grid, t_2_grid = np.meshgrid(t_vals, t_vals)
    f_vals = test_f(t_1_grid, t_2_grid)
    fft_coeffs = fourier_core.fourier_series_coeffs(f_vals)
    k_vals = np.arange(start=0, stop=5, step=1)
    for k_1 in k_vals:
        for k_2 in k_vals:
            exact_coeff = _double_complex_quadrature(lambda t_1, t_2: fourier_integrand(t_1, t_2, k_1, k_2, test_f), 0, 2 * np.pi, 0, 2 * np.pi)
            np.testing.assert_allclose(exact_coeff, fft_coeffs[k_2, k_1], atol=1e-11)


def test_1d_conv():
    def convolution_integrand(x, y, t, t_p):
        return x(t_p) * y(t - t_p)

    def x_periodic(t):
        return  np.exp(np.cos(10 * t) ** 2 * np.sin(t))

    def y_periodic(t):
        return np.exp(np.cos(t) ** 2 - np.sin(t))

    all_t_vals = np.linspace(start=0, stop=2*np.pi, num=2**10+1)
    all_t_vals = all_t_vals[:-1]
    x_vals = x_periodic(all_t_vals)
    y_vals = y_periodic(all_t_vals)
    x_coeffs = fourier_core.fourier_series_coeffs(x_vals)
    y_coeffs = fourier_core.fourier_series_coeffs(y_vals)
    fft_conv_vals = fourier_core.fft_convolution_integral(x_coeffs, y_coeffs)
    compare_every = 20
    true_vals = []
    for t in all_t_vals[::compare_every]:
        true_val = integrate.quad(lambda tp: convolution_integrand(x_periodic, y_periodic, t, tp), 0, 2 * np.pi)
        true_vals.append(true_val[0])
    compare_vals = fft_conv_vals[::compare_every]
    np.testing.assert_allclose(compare_vals, true_vals, atol=1e-8)


def test_2d_conv():
    def convolution_integrand(x, y, t_1, t_p_1, t_2, t_p_2):
        return x(t_p_1, t_p_2) * y(t_1 - t_p_1, t_2 - t_p_2)

    def x_periodic(t_1, t_2):
        return np.exp(np.cos(10 * (t_1 + t_2)) ** 2 * np.sin(t_1) - np.cos(t_1 + t_2)**5)

    def y_periodic(t_1, t_2):
        return np.exp(np.cos(t_1 + t_2) ** 2 - np.sin(t_1) + np.cos(3 * t_2))

    all_t_vals = np.linspace(start=0, stop=2 * np.pi, num=2 ** 10 + 1)
    all_t_vals = all_t_vals[:-1]
    t_1_grid, t_2_grid = np.meshgrid(all_t_vals, all_t_vals)
    x_vals = x_periodic(t_1_grid, t_2_grid)
    y_vals = y_periodic(t_1_grid, t_2_grid)
    x_coeffs = fourier_core.fourier_series_coeffs(x_vals)
    y_coeffs = fourier_core.fourier_series_coeffs(y_vals)
    fft_conv_vals = fourier_core.fft_convolution_integral(x_coeffs, y_coeffs)
    compare_every = 200
    for i, t_1 in enumerate(all_t_vals[::compare_every]):
        for j, t_2 in enumerate(all_t_vals[::compare_every]):
            true_val = integrate.dblquad(lambda t_p_1, t_p_2: convolution_integrand(x_periodic, y_periodic, t_1, t_p_1, t_2, t_p_2), 0, 2 * np.pi, 0, 2 * np.pi)
            compare_val = fft_conv_vals[j * compare_every, i * compare_every]
            np.testing.assert_allclose([true_val[0]], [compare_val], atol=1e-11)
