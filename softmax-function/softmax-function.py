import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x , dtype = float)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis = -1, keepdims=True  )

print(softmax(np.array([1, 2, 3])))
print(softmax(np.array([[1, 2, 3], [0, 0, 0]])))