import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and preprocess the Iris dataset
iris = pd.read_csv('Iris.csv')
species = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris['Species'] = iris['Species'].map(species)

# Function to separate the dataset according to their label and store in a dictionary
def divide_by_label(dataset):
    label_divided_data = {}
    for row in dataset:
        label = row[-1]
        if label not in label_divided_data:
            label_divided_data[label] = []
        label_divided_data[label].append(row[:-1])
    return label_divided_data

# Calculate the mean of a column
def calculate_mean(column):
    return np.mean(column)

# Calculate the standard deviation of a column
def calculate_std_div(column):
    return np.std(column)

# Calculate mean and standard deviation for each column in the dataset based on class labels
def calculate_mean_std_div_by_class(data):
    divided_dataset = divide_by_label(data)
    mean_std_by_label = {}
    for label, rows in divided_dataset.items():
        mean_std_by_label[label] = [(calculate_mean(col), calculate_std_div(col)) for col in np.array(rows).T]
    return mean_std_by_label

# Calculate Gaussian probability density
def calculate_prob_density(x, mean, std_div):
    if std_div == 0:
        return 0  # Handle division by zero gracefully
    exponent = math.exp(-((x - mean) ** 2) / (2 * std_div ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std_div)) * exponent

# Calculate probability that X (a test case) belongs to a class Ci
def calculate_class_prob(mean_std_by_label, test_case):
    probabilities = {}
    for label, mean_std in mean_std_by_label.items():
        probabilities[label] = 1
        for i in range(len(mean_std)):
            mean, std_div = mean_std[i]
            x = test_case[i]
            probabilities[label] *= calculate_prob_density(x, mean, std_div)
    return probabilities

# Predict class labels
def predict_label(mean_std_by_label, test_case):
    probabilities = calculate_class_prob(mean_std_by_label, test_case)
    best_label, best_prob = None, -1
    for key, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = key
    return best_label

# Naive Bayes classifier
def naive_bayesian_classifier(training_set, test_set):
    mean_std_each_label = calculate_mean_std_div_by_class(training_set)
    predictions = []
    for test_case in test_set:
        pred = predict_label(mean_std_each_label, test_case)
        predictions.append(pred)
    return predictions

# Prepare the dataset and perform train-test split
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = iris['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Run the Naive Bayes classifier
predicted_labels = naive_bayesian_classifier(X_train, X_test)

# Convert predicted labels to integers
predicted_labels = np.array(predicted_labels, dtype=int)

# Evaluate the model
accuracy = accuracy_score(predicted_labels, y_test)
print("Accuracy:", accuracy)
cm = confusion_matrix(predicted_labels, y_test)
print("Confusion Matrix:\n", cm)
