import numpy as np
from scipy import sparse


def differentiation_matrix(lamb, N):
    const = (2 ** (lamb - 1)) * np.math.factorial(lamb - 1)
    row = const * (np.arange(start=0, stop=N-lamb, step=1) + lamb)
    return sparse.diags([row], [lamb])


def conversion_matrix(alpha, N):
    if alpha > 0:
        row = np.arange(start=1, stop=N, step=1)
        row = np.concatenate([[1], alpha/(alpha + row)])
        return sparse.diags([row, -1 * row[2:]], [0, 2])
    elif alpha == 0:
        row = 1/2 * np.ones(N)
        return sparse.diags([row, -1 * row[2:]], [0, 2])
    raise ValueError("alpha must be non-negative")





