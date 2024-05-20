#ID3

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load the weather dataset from the local file
file_path = r"C:\Users\Ramachandra\OneDrive\Desktop\ML Lab\weather_forecast.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Preprocessing: convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Splitting the dataset into features and target variable
X = df.drop('Play_Yes', axis=1)  # Modify the column name here if needed
y = df['Play_Yes']  # Modify the column name here if needed

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier with ID3 algorithm
clf_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Fit the ID3 classifier to the training data
clf_id3.fit(X_train, y_train)

# Visualize the ID3 decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf_id3, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

# Predict the labels for the test set using ID3
y_pred_id3 = clf_id3.predict(X_test)

# Evaluate the ID3 model
accuracy_id3 = accuracy_score(y_test, y_pred_id3)
report_id3 = classification_report(y_test, y_pred_id3)

print("ID3 Algorithm Results:")
print(f"Accuracy: {accuracy_id3}")
print(f"Classification Report:\n{report_id3}")



#  CART



# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load the weather dataset from the local file
file_path = r"C:\Users\Ramachandra\OneDrive\Desktop\ML Lab\weather_forecast.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Preprocessing: convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Splitting the dataset into features and target variable
X = df.drop('Play_Yes', axis=1)  # Modify the column name here if needed
y = df['Play_Yes']  # Modify the column name here if needed

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier with CART algorithm
clf_cart = DecisionTreeClassifier(criterion='gini', random_state=42)

# Fit the CART classifier to the training data
clf_cart.fit(X_train, y_train)

# Visualize the CART decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf_cart, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

# Predict the labels for the test set using CART
y_pred_cart = clf_cart.predict(X_test)

# Evaluate the CART model
accuracy_cart = accuracy_score(y_test, y_pred_cart)
report_cart = classification_report(y_test, y_pred_cart)

print("CART Algorithm Results:")
print(f"Accuracy: {accuracy_cart}")
print(f"Classification Report:\n{report_cart}")
