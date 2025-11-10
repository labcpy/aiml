# 2. Apply Naive Bayesian Classifier on
import pandas as pd

# Step 1: Create the dataset
data = {
    'Age': ['Youth','Youth','Middle','Senior','Senior','Senior','Middle','Youth','Youth','Senior'],
    'Income': ['High','High','High','Medium','Low','Low','Low','Medium','Low','Medium'],
    'Student': ['No','No','No','No','Yes','Yes','Yes','No','Yes','Yes'],
    'Credit_Rating': ['Fair','Excellent','Fair','Fair','Fair','Excellent','Excellent','Fair','Fair','Excellent'],
    'Buys_Computer': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes']
}

df = pd.DataFrame(data)

# Step 2: Calculate prior probabilities
total = len(df)
yes_count = len(df[df['Buys_Computer'] == 'Yes'])
no_count = len(df[df['Buys_Computer'] == 'No'])
P_yes = yes_count / total
P_no = no_count / total

print(f"P(Yes) = {P_yes:.2f}")
print(f"P(No) = {P_no:.2f}")

# Step 3: Example: Predict for this new customer
# Example: Age=Senior, Income=Medium, Student=Yes, Credit_Rating=Fair
test = {'Age': 'Senior', 'Income': 'Medium', 'Student': 'Yes', 'Credit_Rating': 'Fair'}

# Step 4: Compute conditional probabilities
def get_conditional_prob(attribute, value, label):
    subset = df[df['Buys_Computer'] == label]
    return len(subset[subset[attribute] == value]) / len(subset)

P_age_yes = get_conditional_prob('Age', test['Age'], 'Yes')
P_income_yes = get_conditional_prob('Income', test['Income'], 'Yes')
P_student_yes = get_conditional_prob('Student', test['Student'], 'Yes')
P_credit_yes = get_conditional_prob('Credit_Rating', test['Credit_Rating'], 'Yes')

P_age_no = get_conditional_prob('Age', test['Age'], 'No')
P_income_no = get_conditional_prob('Income', test['Income'], 'No')
P_student_no = get_conditional_prob('Student', test['Student'], 'No')
P_credit_no = get_conditional_prob('Credit_Rating', test['Credit_Rating'], 'No')

# Step 5: Apply Naive Bayes formula
P_yes_given_X = P_yes * P_age_yes * P_income_yes * P_student_yes * P_credit_yes
P_no_given_X  = P_no  * P_age_no  * P_income_no  * P_student_no  * P_credit_no

print("\n--- Probabilities ---")
print(f"P(Yes|X) = {P_yes_given_X:.6f}")
print(f"P(No|X) = {P_no_given_X:.6f}")

# Step 6: Prediction
if P_yes_given_X > P_no_given_X:
    print("\n✅ Prediction: Buys_Computer = YES")
else:
    print("\n❌ Prediction: Buys_Computer = NO")
