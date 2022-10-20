import matplotlib.pyplot as plt
import numpy as np

from pset_3 import chebychev


N = 1000
grid = chebychev.extrema_grid(N)
mode_nos = np.arange(start=0, stop=N, step=1)

f1 = np.cos(2 * np.pi * grid) + np.sin(2 * np.pi * grid)
f2 = np.cos(200 * np.pi * grid) + np.sin(200 * np.pi * grid)
f3 = np.sqrt(1 - grid**2)
f4 = 1/(1 + 100 * (grid - 1/10)**2)
f5 = np.abs(grid - 1/2)**3
f6 = f2 * f4

funcs = np.stack([f1, f2, f3, f4, f5, f6])
transforms = chebychev.cheb(funcs)

fig, axs = plt.subplots(nrows=np.shape(funcs)[0], ncols=2)
for i, (f, trns, ax) in enumerate(zip(funcs, transforms, axs)):
    ax[0].plot(grid, f)
    ax[0].set_ylabel("f{}".format(i+1))
    ax[1].plot(mode_nos, np.abs(trns), label="f{}".format(i))
    ax[1].set_yscale("log")
axs[-1, 1].set_xlabel("Mode Number")
axs[-1, 0].set_xlabel("x")
plt.show()
