# Import required libraries
import pandas as pd   # for handling CSV data
import numpy as np    # for mathematical operations like sqrt, square, etc.

# Step 1: Load the dataset from a CSV file
df = pd.read_csv("knn_dataset.csv")   # CSV should have columns: Age, Income, No_of_credit_cards, Class

# Step 2: Take user input for the new customer's data (the test point)
age = float(input("Enter age: "))                    # example: 37
income = float(input("Enter income: "))              # example: 67000
no_of_credit_cards = float(input("No of credit cards: "))  # example: 2

# Step 3: Store the test data in a dictionary for easy reference
test_data = {"Age": age, "Income": income, "No_of_credit_cards": no_of_credit_cards}

# Step 4: Calculate the Euclidean distance between the test data and each record in the dataset
# Formula: sqrt((x1 - a1)^2 + (x2 - a2)^2 + (x3 - a3)^2)
df["Distances"] = np.sqrt(
    (df["Age"] - test_data["Age"])**2 +
    (df["Income"] - test_data["Income"])**2 +
    (df["No_of_credit_cards"] - test_data["No_of_credit_cards"])**2
)

# Step 5: Sort the dataset by distance in ascending order
# (the smallest distance = closest neighbor)
sorted_values = df.sort_values(by="Distances")

# Step 6: Choose the top 'k' nearest neighbors
k = 3                         # number of nearest neighbors to consider
top_k = sorted_values.head(k) # select first k rows (smallest distances)

# Step 7: Count how many neighbors belong to each class
# (e.g., Class 0 appears twice, Class 1 appears once)
class_count = top_k["Class"].value_counts()

# Step 8: Pick the class that appears most frequently among the top k neighbors
predicted_class = class_count.idxmax()  # returns the class with the highest count

# Step 9: Display the prediction
print("Predicted class is:", predicted_class)
