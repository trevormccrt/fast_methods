import matplotlib.pyplot as plt
import numpy as np

from pset_3 import chebychev, newton_solver, colocation_core

N = 64
n = 3

grid = chebychev.extrema_grid(N)
grid_x = (grid + 1)/2
init_u = 5 * (1 - grid_x)

this_u = init_u
all_resids = []
for i in range(100000):
    this_v = newton_solver.solve_update(grid, this_u, n)
    resid = np.max(np.abs(this_v))
    all_resids.append(resid)
    this_u = this_u + this_v
    if resid < 1e-10:
        break


fig, axs = plt.subplots(ncols=2)
axs[0].plot(all_resids)
axs[0].set_yscale("log")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Residual")
axs[1].plot(grid_x, this_u)
axs[1].set_xlabel("x")
axs[1].set_ylabel("u(x)")

kg = 6.34e29
deriv_mat = colocation_core.extrema_diff_mat(grid)
soln_deriv = deriv_mat.dot(this_u)[0]
m = -1 * 4/np.sqrt(np.pi) * kg * soln_deriv
m_rel = m/1.99e30
print(m_rel)
plt.show()


print("")

