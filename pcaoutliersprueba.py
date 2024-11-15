import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Columnas de interés
columnas_de_interes = ['ROA 2021', 'ROA 2020', 'ROA 2019', 'ROA 2018', 'ROA 2017', 'ROA 2016', 'ROA 2015', 'ROA 2014', 'ROA 2013', 'ROA 2012'] 

# 1. Cargar los datos desde el archivo xlsx
data = pd.read_excel('archivo_filtrado.xlsx', usecols=columnas_de_interes)
#data = pd.read_excel('datos_sin_atipicos_con_resto_columnas.xlsx', usecols=columnas_de_interes)


# 2. Preprocesamiento de datos (opcional)
# Crear un imputador simple para rellenar los valores faltantes con la media de cada característica
imputer = SimpleImputer(strategy='mean')

# Aplicar el imputador a tus datos
data_imputed = imputer.fit_transform(data)

# 3. Aplicar NIPALS (PCA en este caso)
pca_model = PCA(n_components=2)  # Especifica el número de componentes principales
pca_model.fit(data_imputed)
principal_components = pca_model.components_
projected_data = pca_model.transform(data_imputed)

# 4. Calcular SPE y T2 de Hotelling
# Calcular Squared Prediction Error (SPE)
reconstructed_data = np.dot(projected_data, principal_components) + pca_model.mean_
squared_errors = np.sum((data_imputed - reconstructed_data)**2, axis=1)

# Calcular T2 de Hotelling
T2 = np.sum((projected_data / np.sqrt(pca_model.explained_variance_))**2, axis=1)

# Histograma para evaluar la normalidad de los datos
plt.figure(figsize=(10, 6))
plt.hist(data.values.flatten(), bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de los Datos')
plt.grid(True)
plt.show()

# Definir umbrales para SPE y T2
umbral_SPE = np.mean(squared_errors) + 3 * np.std(squared_errors)
umbral_T2 = np.mean(T2) + 3 * np.std(T2)

# Inicializar lista para almacenar los puntos atípicos eliminados
puntos_atipicos_eliminados = []

# Iniciar bucle para la identificación y eliminación de puntos atípicos
iteracion = 0
while True:
    # Incrementar el contador de iteraciones
    iteracion += 1
    
    # Encontrar puntos atípicos basados en los umbrales actuales
    nuevos_puntos_atipicos_SPE = data[squared_errors > umbral_SPE]
    nuevos_puntos_atipicos_T2 = data[T2 > umbral_T2]
    
    # Combinar los nuevos puntos atípicos identificados
    nuevos_puntos_atipicos = pd.concat([nuevos_puntos_atipicos_SPE, nuevos_puntos_atipicos_T2])
    
    # Detener el bucle si no se encontraron nuevos puntos atípicos
    if nuevos_puntos_atipicos.empty:
        break
    
    # Imprimir el número de puntos atípicos encontrados en esta iteración
    print(f"Iteración {iteracion}: Se encontraron {len(nuevos_puntos_atipicos)} puntos atípicos.")
    
    # Imprimir los puntos atípicos encontrados en esta iteración
    print(f"Puntos atípicos encontrados en la iteración {iteracion}:")
    print(nuevos_puntos_atipicos)
    
    # Agregar los nuevos puntos atípicos a la lista de puntos atípicos eliminados
    puntos_atipicos_eliminados.append(nuevos_puntos_atipicos)
    
    # Eliminar los puntos atípicos identificados de los datos
    data = data[~data.index.isin(nuevos_puntos_atipicos.index)]
    
    # Re-calcular SPE y T2 con los datos actualizados
    data_imputed = imputer.fit_transform(data)
    projected_data = pca_model.transform(data_imputed)
    reconstructed_data = np.dot(projected_data, principal_components) + pca_model.mean_
    squared_errors = np.sum((data_imputed - reconstructed_data)**2, axis=1)
    T2 = np.sum((projected_data / np.sqrt(pca_model.explained_variance_))**2, axis=1)
    
    # Re-calcular los umbrales con los nuevos valores de SPE y T2
    umbral_SPE = np.mean(squared_errors) + 3 * np.std(squared_errors)
    umbral_T2 = np.mean(T2) + 3 * np.std(T2)

# Imprimir el número total de puntos atípicos eliminados
total_puntos_atipicos = sum(len(puntos) for puntos in puntos_atipicos_eliminados)
print(f"\nNúmero total de puntos atípicos eliminados: {total_puntos_atipicos}")

# Imprimir las dimensiones del DataFrame después de eliminar los valores atípicos
print(f"\nDimensiones del DataFrame después de eliminar los valores atípicos: {data.shape}")

# 5. Visualización de resultados
# Gráfico de dispersión SPE
plt.figure(figsize=(10, 6))
plt.scatter(range(len(squared_errors)), squared_errors, color='blue')
plt.axhline(np.mean(squared_errors) + 3 * np.std(squared_errors), color='red', linestyle='--', label='Umbral de alerta (3 sigma)')
plt.xlabel('Índice de observación')
plt.ylabel('Squared Prediction Error (SPE)')
plt.title('Gráfico de Dispersión SPE')
plt.legend()
plt.show()

# Gráfico de dispersión T2 de Hotelling
plt.figure(figsize=(10, 6))
plt.scatter(range(len(T2)), T2, color='green')
plt.axhline(np.mean(T2) + 3 * np.std(T2), color='red', linestyle='--', label='Umbral de alerta (3 sigma)')
plt.xlabel('Índice de observación')
plt.ylabel('T2 de Hotelling')
plt.title('Gráfico de Dispersión T2 de Hotelling')
plt.legend()
plt.show()

# Histograma para evaluar la normalidad de los datos
plt.figure(figsize=(10, 6))
plt.hist(data.values.flatten(), bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de los Datos')
plt.grid(True)
plt.show()

# Plotear el biplot
plt.figure(figsize=(10, 8))
plt.scatter(projected_data[:, 0], projected_data[:, 1])
for i, (component1, component2) in enumerate(zip(principal_components[0], principal_components[1])):
    plt.arrow(0, 0, component1 * max(projected_data[:, 0]), component2 * max(projected_data[:, 1]), color='r', alpha=0.5)
    plt.text(component1 * max(projected_data[:, 0]), component2 * max(projected_data[:, 1]), columnas_de_interes[i], color='g', ha='right', va='bottom', fontsize=10)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Biplot de las Componentes Principales')
plt.grid(True)
plt.show()

# Plotear los loadings de todas las variables
plt.figure(figsize=(10, 8))
plt.scatter(pca_model.components_[0], pca_model.components_[1], color='b')
plt.xlabel('Loading en PC1')
plt.ylabel('Loading en PC2')
plt.title('Loadings de todas las variables')
for i, columna in enumerate(columnas_de_interes):
    plt.text(pca_model.components_[0, i], pca_model.components_[1, i], columna, color='b', ha='right', va='bottom', fontsize=10)
plt.grid(True)
plt.show()


# Guardar el DataFrame sin valores atípicos en un archivo Excel, manteniendo todas las columnas del archivo original
data_original = pd.read_excel('archivo_filtrado.xlsx')
data_final = data_original[data_original.index.isin(data.index)]
data_final.to_excel('c:/Users/Usuario/Desktop/UPV/TFM/BASEDATOS/prueba/datos_sin_atipicos_con_resto_columnas.xlsx', index=False)
print("Se han guardado los datos sin valores atípicos junto con el resto de columnas en 'datos_sin_atipicos_con_resto_columnas.xlsx'")

