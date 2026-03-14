import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.array(x, dtype=float)
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


print(gelu( [-1.0, 0.0, 1.0]))
print(gelu( [[-2., -1.],[0., 1.]]))