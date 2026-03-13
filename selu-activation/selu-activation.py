import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    # Write code here
    x = np.array(x)
    return lam * np.where(x > 0,x, alpha * (np.exp(x) - 1))

print(selu([1.0, -1.0, 0.0]))
print(selu([0.5, 1.5, 2.5]))
