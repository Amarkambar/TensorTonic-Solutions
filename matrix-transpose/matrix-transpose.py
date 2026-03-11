import numpy as np

def matrix_transpose(A):
    A = np.array(A)
    return A.T

print(matrix_transpose([[1, 2, 3], [4, 5, 6]]))
print(matrix_transpose( [[1, 2], [3, 4]]))
print(matrix_transpose([[1, 2, 3, 4]]))