import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


# Cargar los datos desde un archivo xlsx
X_train = pd.read_csv("x_train.csv", index_col=False)
X_test = pd.read_csv("x_test.csv", index_col=False)
y_train = pd.read_csv("y_train.csv", index_col=False)
y_test = pd.read_csv("y_test.csv", index_col=False)

# Convertir y_train y y_test a arrays unidimensionales
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Ajustar el modelo de máquinas de vectores de soporte (SVM)
model_svm = SVR(kernel='linear')
model_svm.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
predictions_svm = model_svm.predict(X_test)

# Calcular las métricas de evaluación para el modelo SVM
mse_svm = mean_squared_error(y_test, predictions_svm)
mae_svm = mean_absolute_error(y_test, predictions_svm)
rmse_svm = mse_svm**0.5
nmse = mse_svm / np.var(y_test)
nmae = mae_svm / np.ptp(y_test)
mape = np.mean(np.abs((y_test - predictions_svm) / y_test)) * 100
r2 = r2_score(y_test, predictions_svm)

# Visualizar los resultados del modelo de máquinas de vectores de soporte
print("\nResultados del modelo de máquinas de vectores de soporte:")
print("Error Cuadrático Medio (MSE):", mse_svm)
print("Error Absoluto Medio (MAE):", mae_svm)
print("Raíz del Error Cuadrático Medio (RMSE):", rmse_svm)
print("Normalized Mean Squared Error (NMSE):", nmse)
print("Normalized Mean Absolute Error (NMAE):", nmae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Coeficiente de Determinación (R^2):", r2)


# Gráfico de dispersión de predicciones vs. valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions_svm, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs. Valores Reales')
plt.grid(True)
plt.show()

# Gráfico de residuos vs. valores predichos
residuos = y_test - predictions_svm
plt.figure(figsize=(8, 6))
plt.scatter(predictions_svm, residuos, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Residuos vs. Valores Predichos')
plt.grid(True)
plt.show()
