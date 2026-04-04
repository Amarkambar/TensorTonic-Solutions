import numpy as np

def k_means_assignment(points, centroids):
    points = np.array(points)
    centroids = np.array(centroids)

    distances = np.sum((points[:, None] - centroids) ** 2, axis=2)

    return np.argmin(distances, axis=1).tolist()


# Example
points = [[1, 1], [1, 2], [10, 10], [10, 11]]
centroids = [[0, 0], [11, 11]]

print(k_means_assignment(points, centroids))