import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(columns=['logroa'])
y = data['logroa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Ajustar el modelo de máquinas de vectores de soporte (SVM)
model_svm = SVR(kernel='linear')
model_svm.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
predictions_svm = model_svm.predict(X_test)

# Calcular las métricas de evaluación para el modelo SVM
mse_svm = mean_squared_error(y_test, predictions_svm)
mae_svm = mean_absolute_error(y_test, predictions_svm)
rmse_svm = mse_svm**0.5
r_squared_svm = r2_score(y_test, predictions_svm)
correlation_svm = pd.Series(y_test).corr(pd.Series(predictions_svm))

# Visualizar los resultados del modelo de máquinas de vectores de soporte
print("\nResultados del modelo de máquinas de vectores de soporte:")
print("Error Cuadrático Medio (MSE):", mse_svm)
print("Error Absoluto Medio (MAE):", mae_svm)
print("Raíz del Error Cuadrático Medio (RMSE):", rmse_svm)
print("Coeficiente de Determinación (R^2):", r_squared_svm)
print("Coeficiente de Correlación de Pearson:", correlation_svm)

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

