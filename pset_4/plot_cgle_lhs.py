import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import cgle_solver

N = 32
h = 0.02
L = 300
b = 0.5
c = -1.76

lhs = cgle_solver.construct_lhs(h, b, L, N)

fig, axs = plt.subplots()
im = axs.imshow(np.abs(lhs.todense()), norm=LogNorm())
fig.colorbar(im)
plt.show()
