# 2.Apply decision tree for the following dataset

import math

# --- Step 1: Sample "Play Tennis" dataset ---
dataset = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# --- Step 2: Calculate Entropy ---
def entropy(data):
    labels = [row[-1] for row in data]
    yes = labels.count('Yes')
    no = labels.count('No')
    total = len(labels)
    if yes == 0 or no == 0:
        return 0
    p_yes = yes / total
    p_no = no / total
    return -p_yes * math.log2(p_yes) - p_no * math.log2(p_no)

# --- Step 3: Information Gain ---
def info_gain(data, attr_index):
    total_entropy = entropy(data)
    values = set([row[attr_index] for row in data])
    weighted_entropy = 0
    for val in values:
        subset = [row for row in data if row[attr_index] == val]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted_entropy

# --- Step 4: Find the best attribute to split ---
def best_attribute(data, attributes):
    gains = [info_gain(data, i) for i in range(len(attributes))]
    best_index = gains.index(max(gains))
    return best_index

# --- Step 5: Build the Decision Tree recursively ---
def build_tree(data, attributes):
    labels = [row[-1] for row in data]
    
    # If all labels are the same → return that label
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    
    # If no more attributes left → return majority label
    if not attributes:
        return max(set(labels), key=labels.count)
    
    # Choose the best attribute
    best_idx = best_attribute(data, attributes)
    best_attr = attributes[best_idx]
    
    tree = {best_attr: {}}
    attr_values = set([row[best_idx] for row in data])
    
    for val in attr_values:
        subset = [row[:best_idx] + row[best_idx+1:] for row in data if row[best_idx] == val]
        sub_attrs = attributes[:best_idx] + attributes[best_idx+1:]
        subtree = build_tree(subset, sub_attrs)
        tree[best_attr][val] = subtree
    
    return tree

# --- Step 6: Build and print the tree ---
decision_tree = build_tree(dataset, attributes)

print("Decision Tree Structure:")
print(decision_tree)
