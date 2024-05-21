import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('glass.csv')
y = df['Type'].values
X = df.drop('Type', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Using scikit-learn with Euclidean distance
clf_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
clf_euclidean.fit(X_train, y_train)
predictions_euclidean = clf_euclidean.predict(X_test)
accuracy_euclidean = accuracy_score(y_test, predictions_euclidean)
print("Accuracy with Euclidean distance:", accuracy_euclidean)

# Using scikit-learn with Manhattan distance
clf_manhattan = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
clf_manhattan.fit(X_train, y_train)
predictions_manhattan = clf_manhattan.predict(X_test)
accuracy_manhattan = accuracy_score(y_test, predictions_manhattan)
print("Accuracy with Manhattan distance:", accuracy_manhattan)
