# 2.Apply multi linear regression on
# X1=3,4,5,6,2
# X2=8,5,7,3,1
# Y=-3.7,3.5,2.5,11.5,5.7 Calculate MSE and plot the regression line.

import numpy as np
import matplotlib.pyplot as plt

# Given data
X1 = np.array([3, 4, 5, 6, 2])
X2 = np.array([8, 5, 7, 3, 1])
Y = np.array([-3.7, 3.5, 2.5, 11.5, 5.7])

# Step 1: Combine X1 and X2 with a column of 1s for intercept
X = np.column_stack((np.ones(len(X1)), X1, X2))

# Step 2: Calculate coefficients using Normal Equation: B = (X^T X)^-1 X^T Y
B = np.linalg.inv(X.T @ X) @ X.T @ Y

# Step 3: Predicted Y values
Y_pred = X @ B

# Step 4: Calculate MSE
MSE = np.mean((Y - Y_pred) ** 2)

print("Coefficients [Intercept, b1, b2]:", B)
print("Mean Squared Error (MSE):", MSE)

# Step 5: Plot (we’ll plot predicted vs actual since it’s multi-dimensional)
plt.scatter(Y, Y_pred, color='green')
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linestyle='--')
plt.show()
