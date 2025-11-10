# 1. Find the optimal hyperplane for SVM use the following data set
# positive class:(3,1),(3,-1),(6,1),(6,-1)
# Negative Class:(1,0),(0,1),(0,-1),(-1,0)

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define data points and their labels
positive = np.array([[3, 1], [3, -1], [6, 1], [6, -1]])
negative = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])

X = np.vstack((positive, negative))
y = np.array([1]*len(positive) + [-1]*len(negative))

# Step 2: Initialize weights and bias (w1, w2, b)
w = np.zeros(2)
b = 0
learning_rate = 0.01
epochs = 1000

# Step 3: Train SVM using basic gradient descent (linear hard-margin)
for epoch in range(epochs):
    for i in range(len(X)):
        if y[i] * (np.dot(X[i], w) + b) < 1:   # Misclassified point
            w = w + learning_rate * (y[i] * X[i] - 2 * (1/epochs) * w)
            b = b + learning_rate * y[i]
        else:  # Correctly classified
            w = w - learning_rate * 2 * (1/epochs) * w

# Step 4: Print final equation
print("Optimal Weight Vector (w):", w)
print("Bias (b):", b)
print(f"Equation of hyperplane: {w[0]:.3f}x + {w[1]:.3f}y + {b:.3f} = 0")

# Step 5: Plotting
plt.scatter(positive[:, 0], positive[:, 1], color='blue', label='Positive (+1)')
plt.scatter(negative[:, 0], negative[:, 1], color='red', label='Negative (-1)')

# Function to plot line
def plot_line(w, b, color):
    x = np.linspace(-2, 8, 100)
    y = -(w[0]*x + b)/w[1]
    plt.plot(x, y, color=color)

# Plot the hyperplane
plot_line(w, b, 'green')
plot_line(w, b+1, 'black')   # Positive margin
plot_line(w, b-1, 'black')   # Negative margin

plt.title("SVM Optimal Hyperplane (from scratch)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.show()
