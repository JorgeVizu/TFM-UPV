import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_text
import numpy as np


#CODIGO PARA EL MODELO ARBOL DE REGRESION
"""En primer lugar creamos las diferentes variables para
cada uno de los conjuntos de datos de entrenamiento y testing"""

X_train = pd.read_csv("x_train.csv", index_col=False)
X_test = pd.read_csv("x_test.csv", index_col=False)
y_train = pd.read_csv("y_train.csv", index_col=False)
y_test = pd.read_csv("y_test.csv", index_col=False)


'''Posteriormente ajustamos el arbol de regresión'''
modelo_arbol = DecisionTreeRegressor(min_samples_split=10, ccp_alpha=0.01, random_state=123)
modelo_arbol.fit(X_train, y_train)

'''Obtenemos las prediciones del modelo'''
y_pred = modelo_arbol.predict(X_test)


'''pedimos la importancia de las variables'''
importance = modelo_arbol.feature_importances_
feature_names = X_train.columns.tolist()

'''Creamos el dataframe para la importancia de las variables'''
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

'''Visualizamos por pantalla el dataframe y sus 10 más importantes'''
print(importance_df)
importance_top10 = importance_df.head(10)
print(importance_top10)


'''Calculamos las diferentes medidas de bondad de ajuste'''
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
nmse = mse / np.var(y_test)
nmae = mae / np.ptp(y_test)
mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
r2 = r2_score(y_test, y_pred)

'''Mostramos por pantalla los diferentes resultados de las medidas de bondad de ajuste'''
print("Error Cuadrático Medio (MSE):", mse)
print("Error Absoluto Medio (MAE):", mae)
print("Raíz del Error Cuadrático Medio (RMSE):", rmse)
print("Normalized Mean Squared Error (NMSE):", nmse)
print("Normalized Mean Absolute Error (NMAE):", nmae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Coeficiente de Determinación (R^2):", r2)


'''Visualizamos gráficamente el árbol resultante'''
plt.figure(figsize=(300,50))  # Ajustar el tamaño de la figura según tus preferencias
plot_tree(modelo_arbol, feature_names=X_train.columns.tolist(), filled=True, rounded=True, fontsize=2)
plt.show()

# Obtener la representación de texto del árbol
#tree_rules = export_text(modelo_arbol, feature_names=X_train.columns.tolist(), decimals=2)

# Mostrar los resultados de los nodos del árbol
#print(tree_rules)

