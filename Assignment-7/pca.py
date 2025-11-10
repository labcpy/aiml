import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define dataset
data = np.array([
    [5.702, 4.386],
    [9.884, 1.020],
    [2.089, 1.613],
    [6.531, 2.533],
    [4.663, 2.444],
    [1.590, 1.104],
    [6.563, 1.382],
    [1.966, 3.687],
    [8.210, 0.971],
    [8.379, 0.961]
])

# Step 2: Standardize data (mean centering)
mean = np.mean(data, axis=0)
data_centered = data - mean

# Step 3: Calculate covariance matrix
cov_matrix = np.cov(data_centered.T)

# Step 4: Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort eigenvalues (largest → smallest)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 6: Select top 1 eigenvector (principal component)
pc1 = eigenvectors[:, 0]

# Step 7: Project data onto this component
projected_data = data_centered.dot(pc1)

# Step 8: Reconstruct points back to 2D (for visualization)
reconstructed = np.outer(projected_data, pc1) + mean

# Step 9: Plot original vs reduced data
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Original Data')
plt.scatter(reconstructed[:, 0], reconstructed[:, 1], color='red', label='PCA (1D Projection)')

# Draw lines from original to projected points
for i in range(len(data)):
    plt.plot([data[i, 0], reconstructed[i, 0]], [data[i, 1], reconstructed[i, 1]], 'gray', linestyle='--')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA - Dimensionality Reduction to 1 Component')
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Display important details
print("Mean of features:\n", mean)
print("\nCovariance Matrix:\n", cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
print("\nPrincipal Component 1:\n", pc1)
print("\nProjected 1D data:\n", projected_data)
