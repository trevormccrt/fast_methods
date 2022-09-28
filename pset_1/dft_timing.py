import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt

from pset_1 import util, dft, fft

M = 200
n_vals = 2 ** np.arange(start=0, stop=11, step=1)
apply_cached_DFT = dft.generate_cached_dft(n_vals)
n_repeats = 50
cached_fft_nmin = 2**6

my_dft_times = []
my_cached_dft_times = []
numpy_fft_times = []
my_fft_times = []

for N in n_vals:
    print(N)
    x_vals = [util.random_complex(M, N) for _ in range(n_repeats)]
    my_start_time = time.time()
    _ = [dft.apply_DFT(x) for x in x_vals]
    my_end_time = time.time()
    my_cached_start_time = time.time()
    _ = [apply_cached_DFT(x) for x in x_vals]
    my_cached_end_time = time.time()
    my_fft_start_time = time.time()
    _ = [fft.apply_FFT(x) for x in x_vals]
    my_fft_end_time = time.time()
    numpy_start_time = time.time()
    _ = [np.fft.fft(x) for x in x_vals]
    numpy_end_time = time.time()
    my_dft_times.append((my_end_time - my_start_time)/n_repeats)
    my_cached_dft_times.append((my_cached_end_time - my_cached_start_time)/n_repeats)
    my_fft_times.append((my_fft_end_time - my_fft_start_time)/ n_repeats)
    numpy_fft_times.append((numpy_end_time - numpy_start_time) / n_repeats)

plt.plot(n_vals, my_dft_times, label="My DFT")
plt.plot(n_vals, my_cached_dft_times, label="My Cached DFT")
plt.plot(n_vals, my_fft_times, label="My FFT")
plt.plot(n_vals, numpy_fft_times, label="Numpy FFT")
plt.legend()
plt.xlabel("N")
plt.ylabel("Time Per Transform (s)")
plt.xscale("log")
plt.yscale("log")
plt.show()
