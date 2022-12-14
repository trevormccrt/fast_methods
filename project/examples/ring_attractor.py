import matplotlib.pyplot as plt
import numpy as np

from project import fourier_core, conv_solver


def cosine_kernel(w0, w1, f, x):
    return -w0 + w1 * np.cos(f * x)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


k = 9
g = 2
w_0 = 1
w_1 = 1
f = 3
g = 0.5
soln_time = 7
t_solve = np.arange(start=0, stop=soln_time, step=0.1)
f_nonlin = np.tanh
fourier_grid = fourier_core.fourier_grid(k)
kernel_grid = cosine_kernel(w_0, w_1, f, fourier_grid)
kernel_coeffs = fourier_core.fourier_series_coeffs(kernel_grid)
init_state_coeffs = fourier_core.fourier_series_coeffs(np.random.uniform(-0.5, 0.5, len(fourier_grid)))
f_b = lambda t: np.zeros_like(init_state_coeffs)

soln_t, soln_y = conv_solver.solve_coeff_space(kernel_coeffs, init_state_coeffs, f_b, g, f_nonlin, soln_time, t_solve)

soln_y_grid = np.real_if_close(fourier_core.nifft(soln_y, axes_from=1))
fig, axs = plt.subplots(ncols=2)
axs[0].plot(fourier_grid/(2 * np.pi), kernel_grid)
axs[0].set_xlabel(r"$r/2\pi$")
axs[0].set_ylabel("W(r)")
axs[0].set_title("Kernel")
axs[1].imshow(soln_y_grid, aspect="auto", origin="lower", extent=(0, 1, 0, soln_time))
axs[1].set_ylabel("t")
axs[1].set_xlabel(r"$r/2\pi$")
axs[1].set_title(r"$x(r,t)$")

plt.show()
print("")
