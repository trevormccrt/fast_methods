import matplotlib.pyplot as plt
import numpy as np


from project import fourier_core, conv_solver


def conv_kernel(x1, x2):
    x1 = np.arctan2(np.sin(x1), np.cos(x1))
    x2 = np.arctan2(np.sin(x2), np.cos(x2))
    r_sq = x1**2 + x2**2
    return np.exp(-r_sq * 1.5) * np.cos(3 * r_sq)


def relu(x):
    return np.max(np.stack([x, np.zeros_like(x)]), axis=0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))

k = 8
g = 1
f_nonlin = sigmoid
fourier_grid = fourier_core.fourier_grid(k)
x1_grid, x2_grid = np.meshgrid(fourier_grid, fourier_grid, indexing="ij")
kernel = conv_kernel(x1_grid, x2_grid)
kernel_coeffs = fourier_core.fourier_series_coeffs(kernel)
static_input = fourier_core.fourier_series_coeffs(np.zeros_like(kernel_coeffs))
input_func = lambda t: static_input

kernel_grid = np.arange(start=-np.pi, stop=np.pi, step=0.01)
kernel_x1_grid, kernel_x2_grid = np.meshgrid(kernel_grid, kernel_grid, indexing="ij")
kernel_plot = conv_kernel(kernel_x1_grid, kernel_x2_grid)

init_state = fourier_core.fourier_series_coeffs(np.random.uniform(-1, 1, np.shape(kernel)))

soln_t, soln_coeffs = conv_solver.solve_coeff_space(kernel_coeffs, init_state, input_func, g, f_nonlin, 2000)
soln_grid = np.real_if_close(fourier_core.nifft(soln_coeffs, axes_from=1))

fig, axs = plt.subplots(nrows=1, ncols=3)
kernel_slice = conv_kernel(0, kernel_grid)
axs[0].plot(kernel_grid, kernel_slice)
axs[0].set_xlabel("x1")
axs[0].set_ylabel("W(x1, 0)")
axs[0].set_title("Slice of Kernel")
axs[1].imshow(kernel_plot, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi))
axs[1].set_xlabel("x1")
axs[1].set_ylabel("x2")
axs[1].set_title("Kernel")
axs[2].imshow(soln_grid[-1, ...] - np.mean(soln_grid[-1, ...]), origin="lower", extent=(0, 2 * np.pi, 0, 2 * np.pi))
axs[2].set_ylabel("x1")
axs[2].set_ylabel("x2")
axs[2].set_title("Steady-State Response")
fig.tight_layout()
plt.show()

print("")
