import matplotlib.pyplot as plt
import numpy as np

from pset_3 import chebychev
import cgle_solver

N = 1024
h = 0.02
L = 300
b = 0.5
c = -1.76
n_iter = 30000


grid = chebychev.extrema_grid(N)
grid_x = 1/2 * L * (grid + 1)
a_o_grid = 10**-3 * np.sin(2 * np.pi * grid_x/L)

solns = cgle_solver.solve_iteratively(n_iter, N, h, b, c, L, a_o_grid)

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(np.abs(solns))
axs[1].imshow(np.arctan2(np.imag(solns), np.real(solns)))
plt.show()
