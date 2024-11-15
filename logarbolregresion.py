import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import plot_tree
from sklearn.tree import export_text
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo xlsx
data = pd.read_excel("datos_final_sin_minimo_keyword.xlsx")

# Aplicar la transformación logarítmica a la variable dependiente
data['logroa'] = np.log(data['roa_mean'] + abs(min(data['roa_mean'])) + 1)

# Identificar las variables con desviación estándar igual a cero
zero_sd_vars = data.std() == 0
zero_sd_var_names = zero_sd_vars[zero_sd_vars].index.tolist()

# Mostrar la lista de variables con desviación estándar igual a cero
if len(zero_sd_var_names) > 0:
    print("Las siguientes", len(zero_sd_var_names), "variables tienen desviación estándar igual a cero:")
    print(", ".join(zero_sd_var_names))
else:
    print("No hay variables con desviación estándar igual a cero.")

# Eliminar variables con desviación estándar igual a cero del conjunto de datos
data = data.drop(columns=zero_sd_var_names)
print(data)

# Eliminar la columna 'roa_mean' antes de dividir los datos
data = data.drop(columns=['roa_mean'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(columns=['logroa'])
y = data['logroa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Verificar valores faltantes en los conjuntos de datos de entrenamiento
print("Valores faltantes en el conjunto de entrenamiento:")
print(X_train.isnull().sum())

# Verificar la forma de los conjuntos de entrenamiento
print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)

# Ajustar el modelo de árbol de regresión si no hay valores faltantes y los conjuntos tienen la misma forma
if X_train.isnull().sum().sum() == 0 and X_train.shape[0] == y_train.shape[0]:
    modelo_arbol = DecisionTreeRegressor(min_samples_split=10, ccp_alpha=0.01)
    modelo_arbol.fit(X_train, y_train)

    # Obtener predicciones del modelo de árbol de regresión
    predictions_tree = modelo_arbol.predict(X_test)
else:
    print("Error: Valores faltantes en el conjunto de entrenamiento o dimensiones incompatibles.")

# Calcular MAE
mae_tree = mean_absolute_error(y_test, predictions_tree)

# Calcular MSE
mse_tree = mean_squared_error(y_test, predictions_tree)

# Calcular NMSE
nmse_tree = mse_tree / y_test.var()

# Imprimir resultados del modelo de árbol de regresión
print("Mean Absolute Error (MAE) - Regression Tree:", mae_tree)
print("Mean Squared Error (MSE) - Regression Tree:", mse_tree)
print("Normalized Mean Squared Error (NMSE) - Regression Tree:", nmse_tree)

# Visualizar el árbol resultante
plt.figure(figsize=(20,10))  # Ajustar el tamaño de la figura según tus preferencias
plot_tree(modelo_arbol, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=8)
plt.show()

# Obtener la representación de texto del árbol
tree_rules = export_text(modelo_arbol, feature_names=X.columns.tolist(), decimals=2)

# Mostrar los resultados de los nodos del árbol
print(tree_rules)

# Imprimir estadísticas descriptivas de la variable objetivo
print(data['logroa'].describe())

# Graficar un histograma de la variable objetivo
plt.hist(data['logroa'], bins=20)
plt.title('Distribución de logroa')
plt.xlabel('logroa')
plt.ylabel('Frecuencia')
plt.show()