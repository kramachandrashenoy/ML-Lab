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
