import numpy as np


def mse(table, U, SV):
    rows, cols = np.nonzero(table)
    suma = 0
    for i, j in zip(rows, cols):
        suma += (table[i, j] - U[i, :].dot(SV[:, j])) ** 2
    #     arr = (table[np.nonzero(table)] - U.dot(SV)[])
    return np.sqrt(suma / len(rows))
