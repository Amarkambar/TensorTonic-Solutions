import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A = np.array(A)
    return np.trace(A)

print(matrix_trace([[1, 2], [3, 4]]))