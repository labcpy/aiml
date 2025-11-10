# 1. Apply simple linear regression on X=1,2,3,4,5 and actual Y=3,4,2,4,5. Calculate MSE, plot the regression line.

import numpy as np
import matplotlib.pyplot as plt

# Given data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([3, 4, 2, 4, 5])

# Step 1: Calculate mean of X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Step 2: Calculate slope (m) and intercept (c)
numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x) ** 2)
m = numerator / denominator
c = mean_y - m * mean_x

# Step 3: Predicted Y values
Y_pred = m * X + c

# Step 4: Mean Squared Error (MSE)
MSE = np.mean((Y - Y_pred) ** 2)

print("Slope (m):", m)
print("Intercept (c):", c)
print("Mean Squared Error (MSE):", MSE)

# Step 5: Plot regression line
plt.scatter(X, Y, color='blue', label='Actual points')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
