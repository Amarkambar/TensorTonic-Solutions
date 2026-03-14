import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.array(x)
    return np.where(x > 0, x, alpha * x)

print(leaky_relu([-2, -1, 0, 1, 2], alpha = 0.1))
print(leaky_relu([-5, 5], alpha = 0.01))