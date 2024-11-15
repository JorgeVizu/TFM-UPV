import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


# Cargar los datos desde un archivo xlsx
X_train = pd.read_csv("x_train.csv", index_col=False)
X_test = pd.read_csv("x_test.csv", index_col=False)
y_train = pd.read_csv("y_train.csv", index_col=False)
y_test = pd.read_csv("y_test.csv", index_col=False)

## Ajustar el modelo de bosques aleatorios
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba para el modelo de bosques aleatorios
y_pred = model_rf.predict(X_test)

# Calcular las métricas de evaluación para el modelo de bosques aleatorios
mse_rf = mean_squared_error(y_test, y_pred)
mae_rf = mean_absolute_error(y_test, y_pred)
rmse_rf = np.sqrt(mse_rf)
r_squared_rf = r2_score(y_test, y_pred)
nmse = mse_rf / np.var(y_test)
nmae = mae_rf / np.ptp(y_test)
mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
r2 = r2_score(y_test, y_pred)

# Visualizar los resultados del modelo de bosques aleatorios
print("Error Cuadrático Medio (MSE):", mse_rf)
print("Error Absoluto Medio (MAE):", mae_rf)
print("Raíz del Error Cuadrático Medio (RMSE):", rmse_rf)
print("Normalized Mean Squared Error (NMSE):", nmse)
print("Normalized Mean Absolute Error (NMAE):", nmae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Coeficiente de Determinación (R^2):", r2)




# Visualizar la importancia de las variables
importances_rf = model_rf.feature_importances_
indices = np.argsort(importances_rf)[::-1]

plt.figure()
plt.title("Importancia de las variables")
plt.bar(range(X_train.shape[1]), importances_rf[indices],
       color="r", align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Visualizar la importancia de las características utilizando Mean Decrease Gini
plt.figure()
plt.title("Importancia de las variables (Mean Decrease Gini)")
plt.bar(range(X_train.shape[1]), importances_rf[indices],
       color="r", align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

