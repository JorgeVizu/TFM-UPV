import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer

# Cargar los datos desde el archivo Excel
data = pd.read_excel("datos_final_sin_minimo_keyword_sin_roa.xlsx")

# Asegurarse de que todos los datos sean numéricos
data = data.select_dtypes(include=[np.number])

# Convertir a numpy array
data_np = data.values

# Calcular el estadístico de Hopkins
def hopkins_statistic(X):
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)  # Subsample size (10% of data)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    
    # Generar muestras aleatorias en el mismo rango que los datos originales
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    rand_X = np.random.rand(m, d) * (max_vals - min_vals) + min_vals
    
    ujd = []
    wjd = []
    for j in range(m):
        u_dist, _ = nbrs.kneighbors([rand_X[j]], 2, return_distance=True)
        w_dist, _ = nbrs.kneighbors([X[np.random.randint(0, n)]], 2, return_distance=True)
        ujd.append(u_dist[0][1])
        wjd.append(w_dist[0][1])
    H = np.sum(ujd) / (np.sum(ujd) + np.sum(wjd))
    return H

hopkins_value = hopkins_statistic(data_np)
print(f"Hopkins statistic: {hopkins_value}")

# Normalizar los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_np)

# Determinar el número óptimo de clusters usando el método del codo
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(data_scaled)
visualizer.show()

# Aplicar K-Means
kmeans = KMeans(n_clusters=4)
kmeans_clusters = kmeans.fit_predict(data_scaled)

# Aplicar Fuzzy C-Means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_scaled.T, 4, 2, error=0.005, maxiter=1000, init=None)

# Obtener las asignaciones de cluster
fcm_clusters = np.argmax(u, axis=0)

# Visualizar K-Means
plt.figure(figsize=(10, 7))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=kmeans_clusters, palette='viridis')
plt.title("K-Means Clustering")
plt.show()

# Visualizar Fuzzy C-Means
plt.figure(figsize=(10, 7))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=fcm_clusters, palette='viridis')
plt.title("Fuzzy C-Means Clustering")
plt.show()
