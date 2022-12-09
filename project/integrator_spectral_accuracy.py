import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy import integrate, interpolate

from project import chebychev_core


def f_x(x):
    return np.exp(np.sin(2 * np.pi * x + x**2)) ** 2 - x


def f_w(x1, x2):
    return np.sin(2 * np.pi * x1 * x2 * 5) * np.tanh(x1 + x2) + np.log(x1**2 + x2**2 + 1) * np.sin(x2 * 20)


def exact_integrand(r, x):
    return f_w(x, r) * f_x(x)


def exact_integrator(r):
    return integrate.quad(lambda x: exact_integrand(r, x), -1, 1, epsrel=1e-11, limit=2000)[0]

exact_k = 12
exact_grid_size = 2 ** exact_k
extrema_grid = chebychev_core.extrema_grid(exact_grid_size)
p = mp.Pool()
exact_results = p.map(exact_integrator, extrema_grid)
p.close()
p.join()
exact_interp = interpolate.interp1d(extrema_grid, exact_results)

print("running FFTs")
k_vals = np.arange(start=1, stop=7, step=1)
errors = []
for k in k_vals:
    this_grid = chebychev_core.extrema_grid(2**k)
    grid_x, grid_y = np.meshgrid(this_grid, this_grid)
    x_grid = f_x(this_grid)
    w_grid = f_w(grid_x, grid_y)
    x_cheb = chebychev_core.ncheb(x_grid)
    w_cheb = chebychev_core.ncheb(w_grid)
    cheb_int_mat = chebychev_core.generate_cheb_integral_matrix(len(this_grid) - 1)
    int_cheb = chebychev_core.chebychev_kernel_integral(x_cheb, w_cheb, cheb_int_mat)
    int_grid = chebychev_core.nicheb(int_cheb)
    exact_int = exact_interp(this_grid)
    error = np.max(np.abs(int_grid - exact_int))/np.max(np.abs(exact_int))
    errors.append(error)

fig, axs = plt.subplots(nrows=1, ncols=3)

plot_grid = np.linspace(start=-1, stop=1, num=100)
grid_x, grid_y = np.meshgrid(plot_grid, plot_grid)
x_grid = f_x(plot_grid)
w_grid = f_w(grid_x, grid_y)

axs[0].imshow(w_grid, origin="lower", extent=(-1, 1, -1, 1))
axs[0].set_xlabel("r'")
axs[0].set_ylabel("r")
axs[0].set_title("W(r,r')")
axs[1].plot(plot_grid, x_grid)
axs[1].set_xlabel("r")
axs[1].set_ylabel("x(r)")
axs[2].plot(2**k_vals, errors)
axs[2].set_xlabel("Basis Size")
axs[2].set_ylabel("Relative Error")
axs[2].set_title("Integration Error")
axs[2].set_yscale("log")
#axs[2].set_xscale("log")
fig.tight_layout()
plt.show()
print("")
