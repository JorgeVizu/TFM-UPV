import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skfuzzy import cmeans
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Lee los datos desde el archivo Excel
data = pd.read_excel("datos_final_sin_minimo_keyword_sin_roa.xlsx")

# Escala los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Realiza Fuzzy C-Means (FCM)
fcm_centers, fcm_u, _, _, _, _, _ = cmeans(scaled_data.T, c=4, m=2, error=0.005, maxiter=1000)

# Obtiene las asignaciones de cluster para cada punto
fcm_clusters = fcm_u.argmax(axis=0)

# Realiza K-Means
kmeans_model = KMeans(n_clusters=4, random_state=42)
kmeans_clusters = kmeans_model.fit_predict(scaled_data)

# Visualiza los resultados de Fuzzy C-Means
plt.figure(figsize=(10, 6))
cmap = get_cmap('tab10')
for cluster in range(4):
    plt.scatter(data.iloc[fcm_clusters == cluster, 0], data.iloc[fcm_clusters == cluster, 1], label=f'Cluster {cluster}', color=cmap(cluster), alpha=0.5)
plt.scatter(fcm_centers[0], fcm_centers[1], marker='o', s=200, color='k', label='Centroides')
plt.title('Fuzzy C-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Visualiza los resultados de K-Means
plt.figure(figsize=(10, 6))
for cluster in range(4):
    plt.scatter(data.iloc[kmeans_clusters == cluster, 0], data.iloc[kmeans_clusters == cluster, 1], label=f'Cluster {cluster}', color=cmap(cluster), alpha=0.5)
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], marker='o', s=200, color='k', label='Centroides')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()