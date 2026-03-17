import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p = np.array(p)
    y = np.array(y)

    loss_pos = -y * (1 - p) ** gamma * np.log(p)
    loss_neg = -(1 - y) * (p ** gamma) * np.log(1 - p)

    loss = loss_pos + loss_neg

    return np.mean(loss)


# Test
print(focal_loss([0.9, 0.2, 0.7, 0.1], [1, 0, 1, 0]))
