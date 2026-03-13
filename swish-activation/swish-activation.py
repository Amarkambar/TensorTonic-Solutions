import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.array(x)
    return x * (1 / ( 1 + np.exp(-x)))

print(swish( [0, 1, -1, 3]))
print(swish(0.0))
print(swish( [[1, -1], [2, -2]]))