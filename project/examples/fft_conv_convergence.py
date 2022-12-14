import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy import integrate, interpolate

from project import fourier_core


def convolution_integrand(x, y, t, t_p):
    return x(t - t_p) * y(t_p)

def x_line(t):
    return t


def y_periodic(t):
    return np.exp(np.cos(t) ** 2 - np.sin(5 * t))


def z_periodic(t):
    return np.exp(np.sin(10 * t))

fig, axs = plt.subplots(nrows=1, ncols=3)

max_order = 14
t_vals = np.linspace(start=0, stop=2 * np.pi, num=(2**max_order) + 1)
t_vals = t_vals[:-1]
x_vals = x_line(t_vals)
y_vals = y_periodic(t_vals)
z_vals = z_periodic(t_vals)
axs[0].plot(t_vals, x_vals, label="x", color="C0")
axs[0].plot(t_vals, y_vals, label="y", color="C1")
axs[0].plot(t_vals, z_vals, label="z", color="C2")
axs[0].legend()
axs[0].set_xlabel("t")
axs[0].set_ylabel("Function Value")
axs[0].set_title("Test Functions")

x_fft = np.fft.fft(x_vals)
y_fft = np.fft.fft(y_vals)
z_fft = np.fft.fft(z_vals)
freqs = np.array(list(range(len(x_fft)))) + 1


axs[1].scatter(freqs[:int(len(freqs)/2)], np.abs(x_fft[:int(len(freqs)/2)]), s=2, color="C0")
axs[1].scatter(freqs[:int(len(freqs)/2)], np.abs(y_fft[:int(len(freqs)/2)]), s=2, color="C1")
axs[1].scatter(freqs[:int(len(freqs)/2)], np.abs(z_fft[:int(len(freqs)/2)]), s=2, color="C2")
axs[1].set_yscale("log")
axs[1].set_xscale("log")
axs[1].set_ylim([1e-4, 1e5])
axs[1].set_xlabel("Mode Index")
axs[1].set_ylabel("Coeffecient Magnitude")
axs[1].set_title("Fourier Series Convergence")


def aperiod_integration_driver(t):
    return integrate.quad(lambda t_p: convolution_integrand(y_periodic, x_line, t, t_p), 0, 2 * np.pi)[0]

p = mp.Pool()
true_aperiod_conv_results = p.map(aperiod_integration_driver, t_vals)
aperiod_interp = interpolate.interp1d(t_vals, true_aperiod_conv_results)
p.close()
p.join()

def period_integration_driver(t):
    return integrate.quad(lambda t_p: convolution_integrand(z_periodic, y_periodic, t, t_p), 0, 2 * np.pi)[0]

p = mp.Pool()
true_period_conv_results = p.map(period_integration_driver, t_vals)
period_interp = interpolate.interp1d(t_vals, true_period_conv_results)
p.close()
p.join()

all_k_test = np.arange(start=0, stop=max_order + 1, step=1)
all_ap_errors = []
all_p_errors = []
for k_test in all_k_test:
    this_t_vals = np.linspace(start=0, stop=2 * np.pi, num=(2 ** k_test) + 1)
    this_t_vals = this_t_vals[:-1]
    true_ap_vals = aperiod_interp(this_t_vals)
    true_p_vals = period_interp(this_t_vals)
    this_x_vals = x_line(this_t_vals)
    this_y_vals = y_periodic(this_t_vals)
    this_z_vals = z_periodic(this_t_vals)
    this_x_coeffs = fourier_core.fourier_series_coeffs(this_x_vals)
    this_y_coeffs = fourier_core.fourier_series_coeffs(this_y_vals)
    this_z_coeffs = fourier_core.fourier_series_coeffs(this_z_vals)
    this_ap_conv = fourier_core.fft_convolution_integral(this_x_coeffs, this_y_coeffs)
    this_p_conv = fourier_core.fft_convolution_integral(this_y_coeffs, this_z_coeffs)
    error_ap = np.max(np.abs(this_ap_conv - true_ap_vals))/np.max(np.abs(true_ap_vals))
    error_p = np.max(np.abs(this_p_conv - true_p_vals))/np.max(np.abs(true_p_vals))
    all_ap_errors.append(error_ap)
    all_p_errors.append(error_p)

axs[2].plot(2**all_k_test, all_ap_errors, label="x∗y", color="C3")
axs[2].plot(2**all_k_test, all_p_errors, label="z∗y", color="C4")
axs[2].set_yscale("log")
axs[2].legend()
axs[2].set_ylabel("Relative Error")
axs[2].set_xlabel("Fourier Series Length")
axs[2].set_title("Accuracy of FFT-Based Convolution")
axs[2].set_xscale("log")

fig.tight_layout()
plt.show()
