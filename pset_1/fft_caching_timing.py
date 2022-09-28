import matplotlib.pyplot as plt
import numpy as np
import time

from pset_1 import dft, fft, util

M = 200
n_test = 13
n_repeats = 50
N_test = 2 ** n_test
all_N_min = 2 ** np.arange(start=0, stop=n_test, step=1)
apply_cached_DFT = dft.generate_cached_dft(all_N_min)

x_vals = [util.random_complex(M, N_test) for _ in range(n_repeats)]
times = []

for N_min in all_N_min:
    print(N_min)
    my_start_time = time.time()
    _ = [fft.apply_fft_cached(x, apply_cached_DFT, N_min) for x in x_vals]
    my_end_time = time.time()
    times.append((my_end_time - my_start_time) / n_repeats)


plt.plot(all_N_min, times)
plt.xlabel("N Min")
plt.ylabel("Time Per Transform (s)")
plt.xscale("log")
plt.yscale("log")
plt.show()

