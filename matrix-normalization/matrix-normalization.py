import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        matrix = np.array(matrix, dtype=float)

        if matrix.ndim != 2:
            return None
        if axis not in (None, 0, 1):
            return None
        if norm_type == 'l1':
            norms = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'l2':
            norms = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
        elif norm_type == 'max':
            norms = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            return None

        # Prevent division by zero
        norms[norms == 0] = 1

        return matrix / norms

    except:
        return None