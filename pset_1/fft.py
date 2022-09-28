import numpy as np


def apply_FFT(x):
    N = np.shape(x)[1]
    if N == 1:
        return x
    k_vals = np.arange(start=0, stop=N, step=1)
    even_fft = apply_FFT(x[:, 0::2])
    odd_fft = apply_FFT(x[:, 1::2])
    return np.concatenate([even_fft, even_fft], axis=-1) +\
           np.exp(-2 * np.pi * 1j * k_vals/N) * np.concatenate([odd_fft, odd_fft], axis=-1)


def apply_fft_cached(x, cached_fft_fn, N_min):
    N = np.shape(x)[1]
    if N == 1:
        return x
    if N == N_min:
        return cached_fft_fn(x)
    k_vals = np.arange(start=0, stop=N, step=1)
    even_fft = apply_fft_cached(x[:, 0::2], cached_fft_fn, N_min)
    odd_fft = apply_fft_cached(x[:, 1::2], cached_fft_fn, N_min)
    return np.concatenate([even_fft, even_fft], axis=-1) + \
           np.exp(-2 * np.pi * 1j * k_vals / N) * np.concatenate([odd_fft, odd_fft], axis=-1)