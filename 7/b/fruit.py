# Using SK Learn


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('fruits.csv')
y = df['fruit_label'].values
X = df[['mass', 'width', 'height', 'color_score']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Using scikit-learn with Euclidean distance
clf_euclidean = KNeighborsClassifier(n_neighbors=5, metric=euclidean_distance)
clf_euclidean.fit(X_train, y_train)
predictions_euclidean = clf_euclidean.predict(X_test)

accuracy_euclidean = accuracy_score(y_test, predictions_euclidean)
print("Accuracy with Euclidean distance (using sklearn):", accuracy_euclidean)

# Using scikit-learn with Manhattan distance
clf_manhattan = KNeighborsClassifier(n_neighbors=5, metric=manhattan_distance)
clf_manhattan.fit(X_train, y_train)
predictions_manhattan = clf_manhattan.predict(X_test)



# Without using SK Learn




import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

def manhattan_distance(x1, x2):
    distance = np.sum(np.abs(x1 - x2))
    return distance

class KNN:
    def __init__(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [self.distance_metric(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

df = pd.read_csv('fruits.csv')
y = df['fruit_label'].values
X = df[['mass', 'width', 'height', 'color_score']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Without using scikit-learn with Euclidean distance
clf_euclidean_custom = KNN(k=5, distance_metric=euclidean_distance)
clf_euclidean_custom.fit(X_train, y_train)
predictions_euclidean_custom = clf_euclidean_custom.predict(X_test)

accuracy_euclidean_custom = np.sum(predictions_euclidean_custom == y_test) / len(y_test)
print("Accuracy with Euclidean distance (without sklearn):", accuracy_euclidean_custom)

# Without using scikit-learn with Manhattan distance
clf_manhattan_custom = KNN(k=5, distance_metric=manhattan_distance)
clf_manhattan_custom.fit(X_train, y_train)
predictions_manhattan_custom = clf_manhattan_custom.predict(X_test)

accuracy_manhattan_custom = np.sum(predictions_manhattan_custom == y_test) / len(y_test)
print("Accuracy with Manhattan distance (without sklearn):", accuracy_manhattan_custom)
accuracy_manhattan = accuracy_score(y_test, predictions_manhattan)
print("Accuracy with Manhattan distance (using sklearn):", accuracy_manhattan)




