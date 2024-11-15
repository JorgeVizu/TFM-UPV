import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
from sklearn.experimental import enable_iterative_imputer  # Importa esta línea para habilitar IterativeImputer
from sklearn.impute import IterativeImputer


# Paso 1: Carga de datos y preprocesamiento
data = pd.read_excel("data.xlsx")  # Reemplaza "datos.xlsx" con el nombre de tu archivo Excel


# Paso 2: Tratamiento de outliers usando T^2 de Hotelling
def hotelling_t2_test(data):
    n, p = data.shape
    print("Número de observaciones:", n)
    print("Número de variables:", p)
    
    center = np.mean(data, axis=0)
    print("Centro de los datos:")
    print(center)
    
    cov_matrix = np.cov(data, rowvar=False)
    print("Matriz de covarianza:")
    print(cov_matrix)
    
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    print("Inversa de la matriz de covarianza:")
    print(inv_cov_matrix)
    
    t_squared = np.zeros(n)
    for i in range(n):
        t_squared[i] = np.dot(np.dot((data.iloc[i] - center).T, inv_cov_matrix), (data.iloc[i] - center))
    
    print("Estadístico T^2 para cada observación:")
    print(t_squared)
    
    alpha = 0.05  # Nivel de significancia
    critical_value = (n - 1) * (p ** 2) / (n * (n - p)) * chi2.ppf(1 - alpha, p)
    print("Valor crítico del estadístico T^2:", critical_value)
    
    outliers = np.where(t_squared > critical_value)[0]
    print("Índices de outliers detectados:", outliers)
  
    return outliers

# Elimina los outliers
#data = data.drop(outliers_indices)

# Paso 3: Tratamiento de datos faltantes con MICE
#imputer = IterativeImputer(max_iter=10, random_state=0)
#data_imputed = imputer.fit_transform(data)

# Paso 4: Estandarización de datos
#scaler = StandardScaler()
#data_scaled = scaler.fit_transform(data_imputed)

# Paso 5: Aplicación del PCA
#pca = PCA()
#pca.fit(data_scaled)

# Paso 6: Selección de componentes principales
#variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
#n_components = np.argmax(variance_ratio_cumulative >= 0.9) + 1  # Mantener componentes que expliquen al menos el 90% de la varianza

# Paso 7: Transformación de datos
#data_transformed = pca.transform(data_scaled)[:, :n_components]

# Uso del primer componente principal como variable dependiente
#y = data_transform#ed[:, 0]

# Paso 8: Validación y análisis de resultados
# Aquí podrías agregar más análisis, como la interpretación de los componentes principales o la evaluación de modelos.