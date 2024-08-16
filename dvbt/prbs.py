import numpy as np

def prbs(init: np.ndarray) -> np.ndarray:
    z = init
    n = 13
    for i in range(n + 1, 2**n - 1):
        q = np.logical_xor(z[-11], z[-9])
        z = np.append(z, q)
    return z
