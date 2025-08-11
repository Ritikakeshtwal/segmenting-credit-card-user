# 1. Libraries Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 2. Load Dataset
df = pd.read_csv("CC GENERAL.csv")

# 3. Basic Info
print(df.head())
print(df.info())

# 4. Handle Missing Values
df = df.drop('CUST_ID', axis=1)  # Customer ID not useful for clustering
df = df.fillna(df.mean())        # Fill missing with mean

# 5. Scaling Data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# 6. Find Optimal K (Elbow Method)
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()

# 7. Fit KMeans with chosen K (example: 4)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_df)

# Add cluster column to original data
df['Cluster'] = clusters

# 8. Cluster Analysis
print(df.groupby('Cluster').mean())

# 9. Silhouette Score (Cluster Quality)
sil_score = silhouette_score(scaled_df, clusters)
print("Silhouette Score:", sil_score)

# 10. PCA for 2D Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_df)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=clusters, palette='viridis', s=60)
plt.title('Credit Card Users Segmentation (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()