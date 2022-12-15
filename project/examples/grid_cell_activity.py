import matplotlib.pyplot as plt
import numpy as np


from project import fourier_core, conv_solver


def conv_kernel(x1, x2, a, gamma, beta):
    x1 = np.arctan2(np.sin(x1), np.cos(x1))
    x2 = np.arctan2(np.sin(x2), np.cos(x2))
    r_sq = x1**2 + x2**2
    return a * np.exp(-gamma * r_sq) - np.exp(-beta * r_sq)


def relu(x):
    return np.max(np.stack([x, np.zeros_like(x)]), axis=0)

k = 8
g = 1
a = 1
beta = 3/(13**2)
gamma = 1.05 * beta
f_nonlin = np.tanh
fourier_grid = fourier_core.fourier_grid(k)
x1_grid, x2_grid = np.meshgrid(fourier_grid, fourier_grid, indexing="ij")
kernel = conv_kernel(x1_grid, x2_grid, a, gamma, beta)
kernel_coeffs = fourier_core.fourier_series_coeffs(kernel)
static_input = fourier_core.fourier_series_coeffs(np.ones_like(kernel_coeffs))
input_func = lambda t: static_input

init_state = fourier_core.fourier_series_coeffs(np.random.uniform(-0.2, 0.2, np.shape(kernel)))

soln_t, soln_coeffs = conv_solver.solve_coeff_space(kernel_coeffs, init_state, input_func, g, f_nonlin, 50, rtol=1e-7)
soln_grid = np.real_if_close(fourier_core.nifft(soln_coeffs, axes_from=1))

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)

axs[0].imshow(kernel, origin="lower", extent=(0, 2 * np.pi, 0, 2 * np.pi))
axs[1].imshow(soln_grid[-1, ...], origin="lower", extent=(0, 2 * np.pi, 0, 2 * np.pi))
plt.show()

print("")
