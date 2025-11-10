
# 4. Perform Hierarchical Clustering on: a. all_Customers_data.csv and draw the dendogram. b. On the data points[18, 22, 25, 27, 42,43] and draw the dendogram.

# Q4(a): Hierarchical Clustering on all_Customers_data.csv
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Load dataset
data = pd.read_csv("C:\Users\ASUS\Downloads\Chrome\Customers.csv")

# Display first few rows
print(data.head())

# Extract numerical columns for clustering
X = data.select_dtypes(include=['float64', 'int64'])

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram for all_Customers_data.csv')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Q4(b): Hierarchical Clustering on custom data points
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Data
X = np.array([[18], [22], [25], [27], [42], [43]])

# Plot dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram for given data points')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()
