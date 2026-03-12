def k_means_centroid_update(points, assignments, k):
    dim = len(points[0])

    centroids = [[0]*dim for _ in range(k)]
    counts = [0]*k

    # add points to clusters
    for p, cluster in zip(points, assignments):
        counts[cluster] += 1
        for i in range(dim):
            centroids[cluster][i] += p[i]

    # compute mean
    for j in range(k):
        if counts[j] > 0:
            for i in range(dim):
                centroids[j][i] /= counts[j]

    return centroids


# -------- Input --------
points = [[0,0],[2,2],[10,10],[12,12]]
assignments = [0,0,1,1]
k = 2

print(k_means_centroid_update(points, assignments, k))