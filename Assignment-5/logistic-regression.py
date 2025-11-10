# 6.Apply logistic regression on the following dataset predict the result for study hour 30. Display the regression coefficients .


# Q6: Logistic Regression (Without sklearn)
import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Study_Hours': [29, 15, 33, 28, 39],
    'Result': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Split into X and y
X = df[['Study_Hours']].values
y = df['Result'].values.reshape(-1, 1)

# Normalize features for better convergence
X_mean = np.mean(X)
X_std = np.std(X)
X_norm = (X - X_mean) / X_std

# Add intercept term (bias)
X_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
theta = np.zeros((X_b.shape[1], 1))
learning_rate = 0.1
iterations = 1000

# Gradient descent
for _ in range(iterations):
    z = X_b @ theta
    h = sigmoid(z)
    gradient = (1 / len(y)) * X_b.T @ (h - y)
    theta -= learning_rate * gradient

# Extract coefficients
intercept = theta[0][0]
coef = theta[1][0]

# Predict result for 30 hours of study
new_data = np.array([[30]])
new_data_norm = (new_data - X_mean) / X_std
new_data_b = np.c_[np.ones((new_data_norm.shape[0], 1)), new_data_norm]

# Compute probability and prediction
prob = sigmoid(new_data_b @ theta)[0][0]
predicted = 1 if prob >= 0.5 else 0

print("Predicted Result for 30 Study Hours:", predicted)
print("Probability (Fail, Pass):", [1 - prob, prob])
print("Regression Coefficient:", coef)
print("Intercept:", intercept)
