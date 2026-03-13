import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.array(x)
    return np.maximum(0 , x)

print(relu([-2, -1, 0, 3]))
print(relu(5.0))
print(relu([[-1, 2],[3, -4]]))