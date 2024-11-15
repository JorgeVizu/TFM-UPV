import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que tu DataFrame se llama 'data' y la columna de la variable respuesta es 'ROA_mean'
# Asumiendo que las variables binarias están etiquetadas como 'binary_1', 'binary_2', etc.
# Cambia el nombre 'data.csv' por el nombre de tu archivo CSV que contiene tus datos
#data = pd.read_excel('datos_final.xlsx')
data = pd.read_excel('datos_fusion.xlsx')

# Visualización de las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(data.head())

# Descripción estadística de las variables
print("\nDescripción estadística de las variables:")
print(data.describe())

# Visualización de la distribución de la variable dependiente (ROA_mean)
plt.figure(figsize=(8, 6))
sns.histplot(data['roa_mean'], kde=True, color='blue')
plt.title('Distribución de roa_mean')
plt.xlabel('roa_mean')
plt.ylabel('Frecuencia')
plt.show()

# Análisis de correlación entre variables binarias y la variable dependiente
correlation_matrix = data.corr()
binary_ROA_correlation = correlation_matrix['roa_mean'].drop('roa_mean')
print("\nCorrelación entre variables binarias y roa_mean:")
print(binary_ROA_correlation)

# Visualización de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()