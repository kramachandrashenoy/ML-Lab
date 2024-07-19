import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns

def kmeans(X, K, max_iters=100):
    # Step 1: Initialize centroids with the first K samples
    centroids = X[:K]
    
    # Step 1: Assign the remaining n-K samples to the nearest centroid and update centroids
    for i in range(K, len(X)):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        nearest_centroid = np.argmin(distances)
        centroids[nearest_centroid] = (centroids[nearest_centroid] + X[i]) / 2
    
    labels = np.zeros(X.shape[0])
    
    # Step 2: Assign each sample to the nearest centroid without updating centroids
    for _ in range(max_iters):
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
conf_matrix = confusion_matrix(y, labels)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

