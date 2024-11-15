import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Leer los datos desde un archivo xlsx
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

# Eliminar la columna 'roa_mean' antes de dividir los datos
data = data.drop(columns=['roa_mean'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(columns=['logroa'])
y = data['logroa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

## Ajustar el modelo de bosques aleatorios
model_rf = RandomForestRegressor(random_state=123)
model_rf.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba para el modelo de bosques aleatorios
predictions_rf = model_rf.predict(X_test)

# Calcular las métricas de evaluación para el modelo de bosques aleatorios
mse_rf = mean_squared_error(y_test, predictions_rf)
mae_rf = mean_absolute_error(y_test, predictions_rf)
rmse_rf = np.sqrt(mse_rf)
r_squared_rf = r2_score(y_test, predictions_rf)
correlation_rf = np.corrcoef(predictions_rf, y_test)[0, 1]

# Visualizar los resultados del modelo de bosques aleatorios
print("Resultados del modelo de bosques aleatorios:")
print("Error Cuadrático Medio (MSE):", mse_rf)
print("Error Absoluto Medio (MAE):", mae_rf)
print("Raíz del Error Cuadrático Medio (RMSE):", rmse_rf)
print("Coeficiente de Determinación (R^2):", r_squared_rf)
print("Coeficiente de Correlación de Pearson:", correlation_rf)

# Visualizar la importancia de las variables
importances_rf = model_rf.feature_importances_
indices = np.argsort(importances_rf)[::-1]

plt.figure()
plt.title("Importancia de las variables")
plt.bar(range(X.shape[1]), importances_rf[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# Visualizar la importancia de las características utilizando Mean Decrease Gini
plt.figure()
plt.title("Importancia de las variables (Mean Decrease Gini)")
plt.bar(range(X.shape[1]), importances_rf[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()