# ID3


import numpy as np
import pandas as pd
import math

# Define the node structure for the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for binary splitting
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Majority class value for leaf nodes

# Calculate the entropy of a given dataset
def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = sum(-p * math.log2(p) for p in probabilities)
    return entropy

# Calculate the information gain for a given split
def calculate_information_gain(X, y, feature, threshold):
    left_indices = X[:, feature] < threshold
    y_left = y[left_indices]
    y_right = y[~left_indices]
    
    entropy_parent = calculate_entropy(y)
    entropy_left = calculate_entropy(y_left)
    entropy_right = calculate_entropy(y_right)
    
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    
    information_gain = entropy_parent - (weight_left * entropy_left + weight_right * entropy_right)
    return information_gain

# Recursively build the decision tree
def build_tree(X, y, max_depth):
    # If all labels are the same or maximum depth reached, create a leaf node
    if len(set(y)) == 1 or max_depth == 0:
        value = max(set(y), key=list(y).count)
        return Node(value=value)
    
    n_features = X.shape[1]
    best_feature = None
    best_threshold = None
    best_info_gain = -1
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            info_gain = calculate_information_gain(X, y, feature, threshold)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
                best_threshold = threshold
    
    if best_info_gain == 0:
        value = max(set(y), key=list(y).count)
        return Node(value=value)
    
    left_indices = X[:, best_feature] < best_threshold
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[~left_indices], y[~left_indices]
    
    left_subtree = build_tree(X_left, y_left, max_depth - 1)
    right_subtree = build_tree(X_right, y_right, max_depth - 1)
    
    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Make predictions using the built tree
def predict_tree(tree, x):
    if tree.value is not None:
        return tree.value
    
    if x[tree.feature] < tree.threshold:
        return predict_tree(tree.left, x)
    else:
        return predict_tree(tree.right, x)

# Evaluate the accuracy of the tree on a test set
def evaluate_tree(tree, X_test, y_test):
    y_pred = [predict_tree(tree, x) for x in X_test]
    accuracy = sum(y_pred == y_test) / len(y_test)
    return accuracy

# Load the weather dataset from the local file
file_path = r"C:\Users\Ramachandra\OneDrive\Desktop\ML Lab\weather_forecast.csv"
df = pd.read_csv(file_path)

# Preprocessing: convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Splitting the dataset into features and target variable
X = df.drop('Play_Yes', axis=1)  # Modify the column name here if needed
y = df['Play_Yes']  # Modify the column name here if needed

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Build the decision tree using our implementation
tree_id3 = build_tree(X_train, y_train, max_depth=5)

# Evaluate the tree on the test set
accuracy_id3 = evaluate_tree(tree_id3, X_test, y_test)
print("ID3 Algorithm Results:")
print(f"Accuracy: {accuracy_id3}")




#  CART





import numpy as np
import pandas as pd
import math

# Define the node structure for the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for binary splitting
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Majority class value for leaf nodes

# Calculate the Gini impurity of a given dataset
def calculate_gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - sum(p**2 for p in probabilities)
    return gini

# Calculate the Gini impurity gain for a given split
def calculate_gini_gain(X, y, feature, threshold):
    left_indices = X[:, feature] < threshold
    y_left = y[left_indices]
    y_right = y[~left_indices]
    
    gini_parent = calculate_gini(y)
    gini_left = calculate_gini(y_left)
    gini_right = calculate_gini(y_right)
    
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    
    gini_gain = gini_parent - (weight_left * gini_left + weight_right * gini_right)
    return gini_gain

# Recursively build the decision tree using CART algorithm
def build_tree_cart(X, y, max_depth):
    if len(set(y)) == 1 or max_depth == 0:
        value = max(set(y), key=list(y).count)
        return Node(value=value)
    
    n_features = X.shape[1]
    best_feature = None
    best_threshold = None
    best_gini_gain = -1
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            gini_gain = calculate_gini_gain(X, y, feature, threshold)
            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_feature = feature
                best_threshold = threshold
    
    if best_gini_gain == 0:
        value = max(set(y), key=list(y).count)
        return Node(value=value)
    
    left_indices = X[:, best_feature] < best_threshold
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[~left_indices], y[~left_indices]
    
    left_subtree = build_tree_cart(X_left, y_left, max_depth - 1)
    right_subtree = build_tree_cart(X_right, y_right, max_depth - 1)
    
    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Make predictions using the built tree
def predict_tree(tree, x):
    if tree.value is not None:
        return tree.value
    
    if x[tree.feature] < tree.threshold:
        return predict_tree(tree.left, x)
    else:
        return predict_tree(tree.right, x)

# Evaluate the accuracy of the tree on a test set
def evaluate_tree(tree, X_test, y_test):
    y_pred = [predict_tree(tree, x) for x in X_test]
    accuracy = sum(y_pred == y_test) / len(y_test)
    return accuracy

# Load the weather dataset from the local file
file_path = r"C:\Users\Ramachandra\OneDrive\Desktop\ML Lab\weather_forecast.csv"
df = pd.read_csv(file_path)

# Preprocessing: convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Splitting the dataset into features and target variable
X = df.drop('Play_Yes', axis=1)  # Modify the column name here if needed
y = df['Play_Yes']  # Modify the column name here if needed

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Build the decision tree using CART algorithm
tree_cart = build_tree_cart(X_train, y_train, max_depth=5)

# Evaluate the tree on the test set
accuracy_cart = evaluate_tree(tree_cart, X_test, y_test)
print("CART Algorithm Results:")
print(f"Accuracy: {accuracy_cart}")
