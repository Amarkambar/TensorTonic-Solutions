import math

def binary_focal_loss(predictions, targets, alpha=1.0, gamma=2.0, eps=1e-9):
    total_loss = 0.0

    for p, y in zip(predictions, targets):
        p = min(max(p , eps),1 - eps)
        pt = p if y == 1 else (1 - p)
        loss = -alpha * ( 1 - pt) ** gamma * math.log(pt)
        total_loss += loss
    return total_loss / len(predictions)

print(binary_focal_loss([0.9],[1]))