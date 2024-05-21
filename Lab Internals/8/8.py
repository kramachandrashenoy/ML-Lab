import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Number of clusters
K = 3

# K-means using scikit-learn
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Print results
print("K-means Labels:", labels)
print("K-means Centroids:", centroids)

# Plotting K-means results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering of Iris Dataset')
plt.show()
