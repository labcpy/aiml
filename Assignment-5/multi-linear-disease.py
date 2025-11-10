# 2. Apply multi linear regression and plot the regression line . Find the error and also predict the disease occurrence for the new input details.Use heart data .csv

import numpy as np
import matplotlib.pyplot as plt
x=np.array([[54, 1, 188],[49, 1, 176],[54, 1, 198],[65, 1, 168],[57, 1, 207],[63, 1, 223],[35, 1, 192],[41, 1, 157],[62, 0, 204],[43, 1, 223],[42, 1, 211],[67, 1, 212],[56, 0, 223],[64, 1, 246]])
y=np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])
x=np.hstack((np.ones((x.shape[0],1)),x))
coff=np.linalg.inv(x.T@x)@(x.T@y)
print("coefficients:",coff)
y_pred=x@coff
print("Prediction:",y_pred)
mse=np.mean((y-y_pred)**2)
print(f"Mean Square Error(MSE):{mse}")
plt.figure(figsize=(8,6))
plt.scatter(range(len(y)),y,color='blue',label='Actual Data')
plt.scatter(range(len(y_pred)),y_pred,color='black',label='Predicted Data')
plt.plot(range(len(y)),y_pred, color='red', linewidth=2,label='Regression Line')
plt.vlines(range(len(y)),y_pred,y,colors='green',linestyle='dashed',label='Residual')
plt.title("Simple Linear Regeression")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.legend(loc="upper left")
plt.show()