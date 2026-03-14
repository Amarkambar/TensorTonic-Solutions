import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.array(x)
    return np.tanh(x)

print(tanh([0, 1, -1, 3]))
print(tanh(0.0))
print(tanh([[0, 1], [-1, 2]]))