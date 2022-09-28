import numpy as np


def DFT_matrix(N):
    axis_vals = np.arange(start=0, stop=N, step=1)
    return np.exp(-2 * np.pi * 1j * 1/N * np.einsum("i, j -> ij", axis_vals, axis_vals))


def apply_DFT(x):
    N = np.shape(x)[1]
    return np.einsum("ij, kj -> ki", DFT_matrix(N), x)


def generate_cached_dft(n_vals):
    dft_mats = {}
    for N in n_vals:
        dft_mats[N] = DFT_matrix(N)

    def apply_cached_DFT(x):
        N = np.shape(x)[1]
        return np.einsum("ij, kj -> ki", dft_mats[N], x)

    return apply_cached_DFT