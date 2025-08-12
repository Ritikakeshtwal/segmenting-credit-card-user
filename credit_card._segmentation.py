import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 0: Create 'plots' folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Step 1: Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 2: Select features (Annual Income, Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Find optimal number of clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig("plots/elbow_method.png")  # Save elbow plot
plt.show()

# Step 4: Apply KMeans with optimal clusters (let's say 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Step 5: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, cmap='rainbow', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='black', marker='X', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig("plots/customer_segments.png")  # Save segmentation plot
plt.show()
