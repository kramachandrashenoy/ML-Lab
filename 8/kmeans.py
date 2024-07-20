# Do kmeans without using sklearn for lab

#Without sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def kmeans(X, K):
    # Step 1: Initialize centroids with the first K samples
    centroids = X[:K]
    
    # Step 1: Assign the remaining n-K samples to the nearest centroid and update centroids
    pointsPerCentroid=[[] for _ in range(3)]
    for i in range(K, len(X)):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        nearest_centroid = np.argmin(distances)
        pointsPerCentroid[nearest_centroid].append(X[i])
        centroids[nearest_centroid] = np.mean(pointsPerCentroid[nearest_centroid],axis=0)

    labels = np.zeros(X.shape[0])
    print(labels)
    # Step 2: Assign each sample to the nearest centroid without updating centroids
    for i in range(len(X)):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        nearest_centroid = np.argmin(distances)
        labels[i] = nearest_centroid
    
    return labels, centroids

# Load the iris dataset and preprocess it
iris = load_iris()
X = iris.data
y = iris.target


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate and plot the correlation matrix
correlation_matrix = np.corrcoef(X_scaled.T)
plt.figure(figsize=(6, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

K = 3
labels, centroids = kmeans(X_scaled, K)
print("Labels:", labels)
print("Centroids:", centroids)

# Plot the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('K-means Clustering of Iris Dataset')
plt.show()

# Calculate and plot the confusion matrix
# the X-axis in the plot is for the predicted class and the Y-axis is for the true class. 
conf_matrix = confusion_matrix(labels, y)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

print("The accuracy is: ", accuracy_score(labels,y))


#using sklearn


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
X = load_iris().data

# K-means using scikit-learn
K = 3
kmeans_sklearn = KMeans(n_clusters=K, random_state=0)
labels_sklearn = kmeans_sklearn.fit_predict(X)
centroids_sklearn = kmeans_sklearn.cluster_centers_

print("Scikit-learn K-means Labels:", labels_sklearn)
print("Scikit-learn K-means Centroids:", centroids_sklearn)

# Plotting K-means results using sklearn
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_sklearn)
plt.scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scikit-learn K-means Clustering of Iris Dataset')
plt.show()
