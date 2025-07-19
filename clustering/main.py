import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan nilai K optimal dengan Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 4. Plot Elbow
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Jumlah Klaster (K)')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method untuk Menentukan K')
plt.grid()
plt.show()

# Pelatihan K-Means dengan K = 3 (karena kita tahu ada 3 spesies)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Evaluasi menggunakan Silhouette Score
score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score untuk K={k_optimal}: {score:.3f}")

# Visualisasi Hasil Klasterisasi (PCA 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_result = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_result['Cluster'] = cluster_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_result, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100)
plt.title('Visualisasi Klaster Data Iris (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Klaster')
plt.grid()
plt.show()
