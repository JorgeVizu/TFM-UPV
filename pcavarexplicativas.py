import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# 1. Cargar los datos desde el archivo Excel
#data = pd.read_excel('datos_final.xlsx') #con dato mínimo
#data = pd.read_excel('datos_final_sin_minimo.xlsx') #sin dato mínimo
#data = pd.read_excel('datos_final_sin_minimo_sin_roa.xlsx') #sin dato mínimo y sin roa
#data = pd.read_excel('datos_final_sin_minimo_keyword.xlsx') #sin dato mínimo y sin roa y sin kvstems
data = pd.read_excel('datos_final_sin_minimo_keyword_sin_roa.xlsx') #sin dato mínimo y sin roa y sin kvstems

#data = pd.read_excel('datos_fusion.xlsx')

# 2. Aplicar PCA a los datos
pca_model = PCA()
pca_model.fit(data)
print(pca_model)

# 3. Obtener las componentes principales y la varianza explicada
principal_components = pca_model.components_
explained_variance = pca_model.explained_variance_ratio_
eigenvalues = pca_model.explained_variance_
print(eigenvalues)

# Tomar solo las tres primeras componentes principales y su varianza explicada
explained_variance_10 = explained_variance[:10]
eigenvalues_10 = eigenvalues[:10]


# 4. Calcular el porcentaje acumulado de la varianza explicada
cumulative_variance = np.cumsum(explained_variance)

# 5. Plotear el biplot de las componentes principales
plt.figure(figsize=(10, 8))

# Scatter plot de las observaciones
plt.scatter(pca_model.transform(data)[:, 0], pca_model.transform(data)[:, 1], alpha=0.5)

# Loop para agregar las líneas de los loadings
for i, (component1, component2) in enumerate(zip(principal_components[0], principal_components[1])):
    plt.arrow(0, 0, component1, component2, color='r', alpha=0.5)
    # Ajuste manual de la posición de las etiquetas de texto para evitar solapamientos
    offset = 0.1
    plt.text(component1 + offset, component2 + offset, data.columns[i], color='g', ha='right', va='bottom', fontsize=10)

# Etiquetas y título
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Biplot de las Componentes Principales')
plt.grid(True)

# Mostrar el gráfico
plt.show()


# 5. Plotear el Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Porcentaje acumulado de Varianza Explicada')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# 6. Visualizar la varianza explicada por cada componente
fig, ax = plt.subplots(figsize=(10, 6))
bar_plot = ax.bar(range(len(explained_variance_10)), explained_variance_10, alpha=0.5, align='center', label='Porcentaje de Varianza Explicada')

# Mostrar los valores propios y la cantidad de varianza explicada en las etiquetas de las tres primeras barras
for i, (eigenvalue, explained_var) in enumerate(zip(eigenvalues_10, explained_variance_10)):
    ax.text(i, explained_var, f'Eigenvalor: {eigenvalue:.2f}\nVarianza Explicada: {explained_var:.2f}', ha='center', va='bottom')

plt.xlabel('Componentes Principales')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Componente Principal (3 primeras componentes)')
plt.legend()
plt.show()


# 7. Mostrar los valores propios de cada componente principal
plt.figure(figsize=(10, 6))
plt.bar(range(len(eigenvalues)), eigenvalues, alpha=0.5, align='center')
plt.xlabel('Componentes Principales')
plt.ylabel('Valor Propio')
plt.title('Valores Propios de los Componentes Principales')
plt.show()

# 5. Plotear el biplot de las componentes principales
plt.figure(figsize=(10, 8))
plt.scatter(principal_components[0], principal_components[1], alpha=0.5)
for i, (component1, component2) in enumerate(zip(principal_components[0], principal_components[1])):
    plt.arrow(0, 0, component1, component2, color='r', alpha=0.5)
    plt.text(component1, component2, data.columns[i], color='g', ha='right', va='bottom', fontsize=10)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Biplot de las Componentes Principales')
plt.grid(True)
plt.show()

# 6. Plotear el gráfico de loadings para ver la relación de las variables
plt.figure(figsize=(10, 8))
plt.scatter(pca_model.components_[0], pca_model.components_[1], color='b')
plt.xlabel('Loading en PC1')
plt.ylabel('Loading en PC2')
plt.title('Loadings de las Variables')
for i, columna in enumerate(data.columns):
    plt.text(pca_model.components_[0, i], pca_model.components_[1, i], columna, color='b', ha='right', va='bottom', fontsize=10)
plt.grid(True)
plt.show()

# 8. Plotear el gráfico de scores T1-T2
plt.figure(figsize=(10, 8))
plt.scatter(pca_model.transform(data)[:, 0], pca_model.transform(data)[:, 1], alpha=0.5)
plt.xlabel('Score en T1')
plt.ylabel('Score en T2')
plt.title('Gráfico de Scores T1-T2')
plt.grid(True)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Transformar los datos a las primeras tres componentes principales
scores_3d = pca_model.transform(data)[:, :3]

# Crear un gráfico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(scores_3d[:, 0], scores_3d[:, 1], scores_3d[:, 2], alpha=0.5)

ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('Gráfico de Scores 3D (T1-T2-T3)')

plt.show()


