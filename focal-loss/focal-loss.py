import numpy as np

def focal_loss(p, y, gamma=2.0, eps=1e-9):
    # Convert to numpy arrays FIRST
    p = np.array(p)
    y = np.array(y)

    # Avoid log(0)
    p = np.clip(p, eps, 1 - eps)

    # Correct class probability
    pt = y * p + (1 - y) * (1 - p)

    # Focal loss
    loss = -(1 - pt) ** gamma * np.log(pt)

    return np.mean(loss)


print(focal_loss([0.9, 0.2, 0.7, 0.1],[1, 0, 1, 0]))