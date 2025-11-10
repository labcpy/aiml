# 3. Apply K-means clustering on the following data.
#  x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
#  y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([4, 5, 10, 4, 3, 11, 14, 6, 10, 12])
y = np.array([21, 19, 24, 17, 16, 25, 24, 22, 21, 21])

# Combine into one dataset (each point = [x, y])
data = np.column_stack((x, y))

# Number of clusters
k = 2  # You can change to 3 to see more clusters

# Step 1: Randomly choose k initial centroids from data
np.random.seed(0)
centroids = data[np.random.choice(len(data), k, replace=False)]

for i in range(10):  # Run for 10 iterations
    # Step 2: Assign each point to the nearest centroid
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    cluster_labels = np.argmin(distances, axis=0)

    # Step 3: Update centroids (mean of points in each cluster)
    new_centroids = np.array([data[cluster_labels == j].mean(axis=0) for j in range(k)])

    # If centroids do not change → stop
    if np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids

# Step 4: Plot the final clusters
colors = ['red', 'blue', 'green']
for i in range(k):
    plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], color=colors[i], label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='*', s=200, label='Centroids')
plt.title("K-Means Clustering (k=2)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
