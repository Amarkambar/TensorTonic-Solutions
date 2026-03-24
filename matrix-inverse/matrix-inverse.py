import numpy as np

def matrix_inverse(A):
    A = np.array(A)
    
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None   # or return "Singular matrix"