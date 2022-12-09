import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy import integrate, interpolate

from project import chebychev_core

np.random.seed(420420)

test_function_grid = np.linspace(start=-1, stop=1, num=10)
test_function_grid_x, test_function_grid_y = np.meshgrid(test_function_grid, test_function_grid)
test_function_samples = np.random.uniform(-1,1, np.shape(test_function_grid_x))
test_function = interpolate.interp2d(test_function_grid, test_function_grid, test_function_samples, kind="cubic")


fig, axs = plt.subplots(nrows=1, ncols=3)

plot_function_grid = np.linspace(start=-1, stop=1, num=10000)
function_plot = test_function(plot_function_grid, plot_function_grid)

axs[0].imshow(function_plot, origin="lower", extent=(-1, 1, -1, 1))

x_test = np.random.uniform(-2, 2, len(test_function_grid))/2
x_test_function = interpolate.interp1d(test_function_grid, x_test, kind="cubic")

axs[1].plot(plot_function_grid, x_test_function(plot_function_grid))

k_max = 8
exact_grid = chebychev_core.extrema_grid(2**k_max)


def exact_integrand(x1, x2):
    return test_function(x2, x1) * x_test_function(x1)


def compute_exact(x2):
    return integrate.quad(lambda x1: exact_integrand(x1, x2), -1, 1)[0]

p = mp.Pool()
true_results = p.map(compute_exact, exact_grid)
p.close()
p.join()

true_interp = interpolate.interp1d(exact_grid, true_results)
for k in range(1, k_max):
    this_grid = chebychev_core.extrema_grid(2**k)
    cheb_int_mat = chebychev_core.generate_cheb_integral_matrix(len(this_grid)-1)
    this_w = np.transpose(test_function(this_grid, this_grid))
    this_x = x_test_function(this_grid)
    this_w_coeff = chebychev_core.ncheb(this_w)
    this_x_coeff = chebychev_core.ncheb(this_x)
    approx_int = chebychev_core.chebychev_kernel_integral(this_x_coeff, this_w_coeff, cheb_int_mat)
    approx_int_grid = chebychev_core.nicheb(approx_int)
    true_vals = true_interp(this_grid)

    print("")




plt.show()
print("")
