import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)

    # Ensure 2D (handles both single + batch)
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
        positive = positive.reshape(1, -1)
        negative = negative.reshape(1, -1)

    # Squared Euclidean distance (row-wise)
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)

    # Triplet loss per sample
    losses = np.maximum(pos_dist - neg_dist + margin, 0)

    # Mean loss
    return float(np.mean(losses))