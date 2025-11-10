# 5.Implement Polinomial regression on Data1.csv. Display the coefficients.¶


# Q5: Polynomial Regression (Without sklearn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("Data1.csv")
print("First few rows:\n", data.head())

# Assume the file has columns like: 'X' (input) and 'Y' (output)
X = data.iloc[:, 0].values.reshape(-1, 1)  # independent variable
y = data.iloc[:, 1].values                 # dependent variable

# Degree of the polynomial
degree = 3

# --- Step 1: Create polynomial features manually ---
# [1, X, X^2, X^3, ...]
X_poly = np.ones((len(X), degree + 1))
for d in range(1, degree + 1):
    X_poly[:, d] = X[:, 0] ** d

# --- Step 2: Train model using Normal Equation ---
# θ = (XᵀX)^(-1) Xᵀy
theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

# --- Step 3: Predictions ---
y_pred = X_poly @ theta

# --- Step 4: Display coefficients ---
print("Polynomial Coefficients:", theta[1:])
print("Intercept:", theta[0])

# --- Step 5: Visualization ---
# Sort for smooth plotting
sort_idx = np.argsort(X[:, 0])
X_sorted = X[sort_idx]
y_pred_sorted = y_pred[sort_idx]

plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X_sorted, y_pred_sorted, color='red', label=f'Polynomial Regression (Degree {degree})')
plt.title('Polynomial Regression on Data1.csv (Manual Implementation)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()