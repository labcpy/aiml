# 1. Use Salary Dataset - Simple linear regression from kaggle and apply simple linear regression and plot the regression line . Find the error and also predict the salary for given experience details.


import numpy as np
import matplotlib.pyplot as plt
x=np.array([1.1,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,8.2,8.7,9,9.5,9.6,10.3,10.5])
y=np.array([39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940,91738,98273,101302,113812,109431,105582,116969,112635,122391,121872])
x_mean=np.mean(x)
y_mean=np.mean(y)
B1=np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
B0=y_mean-B1*x_mean
y_pred=B0+B1*x
mse=np.mean((y-y_pred)**2)
print(f"Mean Square Error(MSE):{mse}")
plt.figure(figsize=(8,6))
plt.scatter(x,y,color='blue',label='Actual Data')
plt.plot(x,y_pred, color='red', linewidth=2,label='Regression Line')
plt.vlines(x,y_pred,y,colors='green',linestyle='dashed',label='Residual')
plt.title("Simple Linear Regeression")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.legend(loc="upper left")
plt.show()
experience_values = np.array([3, 5, 10])
predicted_salaries = B0 + B1 * experience_values
print("\nPredicted Salaries:")
for exp, sal in zip(experience_values, predicted_salaries):
    print(f"Experience = {exp} years → Predicted Salary = {sal:.2f}")