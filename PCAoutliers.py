import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

"A cotinuacuón se definen las columnas de interés en las que buscar los valores atípicos"
columnas_de_interes = ['ROA 2021', 'ROA 2020', 'ROA 2019', 'ROA 2018', 'ROA 2017', 'ROA 2016', 'ROA 2015', 'ROA 2014', 'ROA 2013', 'ROA 2012'] 


"Cargamos los datos desde el excel"
data = pd.read_excel('archivo_filtrado.xlsx', usecols=columnas_de_interes)

#Cargar los datos desde el archivo xlsx
#data = pd.read_excel('archivo_filtrado_solo_columnas_de_interes.xlsx')
#data = pd.read_excel('roa_sin_atipicos.xlsx')


#Preprocesamiento de datos (opcional)
# Crear un imputador simple para rellenar los valores faltantes con la media de cada característica
imputer = SimpleImputer(strategy='mean')

# Aplicar el imputador a tus datos
data_imputed = imputer.fit_transform(data)

#Aplicar NIPALS (PCA en este caso)
pca_model = PCA(n_components=2)  # Especifica el número de componentes principales
pca_model.fit(data_imputed)
principal_components = pca_model.components_
projected_data = pca_model.transform(data_imputed)


# Visualización del biplot antes de eliminar outliers
components_df = pd.DataFrame(pca_model.components_.T, columns=['PC1', 'PC2'], index=data.columns)

plt.figure(figsize=(10, 8))
plt.scatter(projected_data[:, 0], projected_data[:, 1], c='blue', alpha=0.5, label='Datos proyectados')
for feature in components_df.index:
    plt.arrow(0, 0, components_df.loc[feature, 'PC1'], components_df.loc[feature, 'PC2'], color='red', alpha=0.75, head_width=0.1)
    plt.text(components_df.loc[feature, 'PC1']*1.1, components_df.loc[feature, 'PC2']*1.1, feature, color='black', ha='center', va='center')

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Biplot de Componentes Principales antes de eliminar outliers')
plt.grid()
plt.legend()
plt.show()



# 4. Calcular SPE y T2 de Hotelling
# Calcular Squared Prediction Error (SPE)
reconstructed_data = np.dot(projected_data, principal_components) + pca_model.mean_
squared_errors = np.sum((data_imputed - reconstructed_data)**2, axis=1)

# Calcular T2 de Hotelling
T2 = np.sum((projected_data / np.sqrt(pca_model.explained_variance_))**2, axis=1)

# Imprimir los resultados del PCA
print("Valor propio de cada componente principal:", pca_model.explained_variance_)
print("Cantidad de varianza explicada por cada componente principal:", pca_model.explained_variance_ratio_)
print("Varianza acumulada explicada por los componentes principales:", np.cumsum(pca_model.explained_variance_ratio_))


# Crear un DataFrame para las componentes principales y los nombres de las características
components_df = pd.DataFrame(pca_model.components_.T, columns=['PC1', 'PC2'], index=data.columns)
print(components_df)

# Plotear el biplot
plt.figure(figsize=(10, 8))
plt.scatter(projected_data[:, 0], projected_data[:, 1], c='blue', alpha=0.5, label='Datos proyectados')
for feature in components_df.index:
    plt.arrow(0, 0, components_df.loc[feature, 'PC1'], components_df.loc[feature, 'PC2'], color='red', alpha=0.75, head_width=0.1)
    plt.text(components_df.loc[feature, 'PC1']*1.1, components_df.loc[feature, 'PC2']*1.1, feature, color='black', ha='center', va='center')

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Biplot de Componentes Principales')
plt.grid()
plt.legend()
plt.show()

print("Forma de principal_components:", principal_components.shape)
print("Forma de projected_data:", projected_data.shape)

# 4. Calcular SPE y T2 de Hotelling
# Calcular Squared Prediction Error (SPE)
reconstructed_data = np.dot(projected_data, principal_components) + pca_model.mean_
squared_errors = np.sum((data_imputed - reconstructed_data)**2, axis=1)
print(squared_errors)

# Calcular T2 de Hotelling
T2 = np.sum((projected_data / np.sqrt(pca_model.explained_variance_))**2, axis=1)
print(T2)

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

# Guardar el DataFrame sin valores atípicos en un archivo Excel
data.to_excel('datos_sin_atipicos.xlsx', index=False)
print("Se han guardado los datos sin valores atípicos en 'datos_sin_atipicos.xlsx'.")