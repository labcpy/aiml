# 1. Apply KNN and predict the class for customer John.Find the error with changed value of k.

import numpy as np
from collections import Counter

# Sample dataset (you can modify with your dataset)
# Example: [Age, Income, No_of_credit_cards, Label]
data = np.array([
    [25, 50000, 1, 0],
    [30, 60000, 2, 0],
    [35, 65000, 2, 1],
    [40, 70000, 3, 1],
    [45, 80000, 4, 1]
])

# Split features and labels
X = data[:, :-1]  # all columns except last (features)
y = data[:, -1]   # last column (labels)

# John's data
john = np.array([37, 67000, 2])

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# KNN function
def knn_predict(X, y, query, k):
    distances = [euclidean_distance(query, point) for point in X]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y[i] for i in k_indices]
    prediction = Counter(k_labels).most_common(1)[0][0]
    return prediction

# Try with different k values
for k in [1, 3, 5]:
    pred = knn_predict(X, y, john, k)
    print(f"For k={k}, predicted class for John: {pred}")
