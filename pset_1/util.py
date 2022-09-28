import numpy as np


def random_complex(M, N):
    return np.random.uniform(-1, 1, (M, N)) + 1j * np.random.uniform(-1, 1, (M, N))
