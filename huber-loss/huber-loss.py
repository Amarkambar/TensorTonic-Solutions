import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic

    loss = 0.5 * quadratic ** 2 + linear * delta

    return np.mean(loss)

print(huber_loss([1, 2 , 3], [1.5, 1.7, 2.5]))