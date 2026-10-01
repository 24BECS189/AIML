# K-Means Clustering and Cluster Validation

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Dataset
X = np.array([
    [1, 2],
    [1, 3],
    [2, 2],
    [2, 3],
    [8, 7],
    [8, 8],
    [9, 7],
    [9, 8],
    [5, 5],
    [6, 5]
])


# Number of clusters
k = 3

# Create K-Means model
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

# Train the model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_

# Calculate Silhouette Score
silhouette = silhouette_score(X, labels)


# Display results
print("K-Means Clustering")
print("------------------")

print("\nData Points:")
print(X)

print("\nCluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

print("\nSilhouette Score:")
print(silhouette)