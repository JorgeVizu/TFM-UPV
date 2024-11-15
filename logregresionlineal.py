import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import shapiro, kstest


import statsmodels.api as sm

# Leer los datos desde un archivo xlsx y TRANSFORMACION LOG
data = pd.read_excel("datos_final_sin_minimo_keyword.xlsx")
data['logroa'] = np.log(data['roa_mean'] + abs(min(data['roa_mean'])) + 1)

# Crear el boxplot de logroa
plt.boxplot(data['logroa'])
plt.title('Boxplot de logroa')
plt.show()

# Crear el histograma de logroa
plt.hist(data['logroa'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histograma de logroa')
plt.xlabel('logroa')
plt.ylabel('Frecuencia')
plt.show()

print(data['logroa'].describe())

# Identificar las variables con desviación estándar igual a cero
zero_sd_vars = data.std() == 0
zero_sd_var_names = zero_sd_vars[zero_sd_vars].index.tolist()

if len(zero_sd_var_names) > 0:
    print("Las siguientes", len(zero_sd_var_names), "variables tienen desviación estándar igual a cero:")
    print(", ".join(zero_sd_var_names))
else:
    print("No hay variables con desviación estándar igual a cero.")

# Eliminar variables con desviación estándar igual a cero del conjunto de datos
data = data.drop(zero_sd_var_names, axis=1)
print(data)

# Eliminar la columna 'roa_mean' antes de dividir los datos
data = data.drop(columns=['roa_mean'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(columns=['logroa'])
y = data['logroa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Ajustar el modelo de regresión lineal
model_lm = LinearRegression()
model_lm.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba para el modelo de regresión lineal
predictions_lm = model_lm.predict(X_test)

# Calcular métricas para el modelo de regresión lineal
mse = mean_squared_error(y_test, predictions_lm)
mae = mean_absolute_error(y_test, predictions_lm)
rmse = np.sqrt(mse)
r_squared = model_lm.score(X_test, y_test)
r2_lm = r2_score(y_test, predictions_lm)

print("Métricas para el modelo de regresión lineal:")
print("Error Cuadrático Medio (MSE):", mse)
print("Error Absoluto Medio (MAE):", mae)
print("Raíz del Error Cuadrático Medio (RMSE):", rmse)
print("Coeficiente de Determinación (R^2):", r_squared)
print("Coeficiente de Determinación (R^2):", r2_lm)

# Ajustar el modelo de regresión lineal utilizando StatsModels para realizar el análisis de varianza
X_train_sm = sm.add_constant(X_train)
model_lm_sm = sm.OLS(y_train, X_train_sm).fit()

# Crear un objeto PandasData a partir de los datos de entrenamiento
data_train_pd = sm.add_constant(X_train)
data_train_pd['logroa'] = y_train

# Realizar análisis de varianza para el modelo de regresión lineal
anova_result_lm = sm.stats.anova_lm(sm.OLS.from_formula('logroa ~ ' + ' + '.join(X.columns), data=data_train_pd).fit())
print("Resultados del análisis de varianza para el modelo de regresión lineal:")
print(anova_result_lm)

# Visualizar observados vs predichos con diagonal
plt.scatter(y_test, predictions_lm)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal
plt.xlabel('Observados')
plt.ylabel('Predichos')
plt.title('Observados vs. Predichos - Regresión Lineal')
plt.show()



# Imprimir la tabla ANOVA actualizada
print("Resultados del análisis de varianza para el modelo de regresión lineal:")
print(anova_result_lm)


#---------------------------------
# Definir una función para realizar la regresión stepwise
def stepwise_selection(X, y):
    features = list(X.columns)
    selected_features = []
    while len(features) > 0:
        remaining_features = list(set(features) - set(selected_features))
        p_values = []
        for feature in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
            p_values.append((feature, model.pvalues[feature]))
        p_values = pd.DataFrame(p_values, columns=['feature', 'p_value'])
        min_p_value = p_values['p_value'].min()
        if min_p_value < 0.05:
            best_feature = p_values.loc[p_values['p_value'].idxmin()]['feature']
            selected_features.append(best_feature)
        else:
            break
    return selected_features

# Aplicar la regresión stepwise
selected_features = stepwise_selection(X_train, y_train)

# Ajustar el modelo con las características seleccionadas
X_train_stepwise = X_train[selected_features]
X_train_stepwise = sm.add_constant(X_train_stepwise)  # Añadir una columna de unos para el intercepto
model_stepwise = sm.OLS(y_train, X_train_stepwise).fit()

# Ver el resumen del modelo stepwise
print(model_stepwise.summary())

# Residual standard error
residual_std_error = model_stepwise.mse_resid ** 0.5
print("Residual standard error:", residual_std_error)

# R cuadrado múltiple y R cuadrado ajustado
r_squared = model_stepwise.rsquared
adjusted_r_squared = model_stepwise.rsquared_adj
print("Multiple R-squared:", r_squared)
print("Adjusted R-squared:", adjusted_r_squared)

# Estadísticas F
f_statistic = model_stepwise.fvalue
f_p_value = model_stepwise.f_pvalue
print("F-statistic:", f_statistic)
print("p-value:", f_p_value)

# Obtener los residuos del modelo de regresión
residuos = model_stepwise.resid

#NORMALIDAD
# Realizar la prueba de Shapiro-Wilk para evaluar la normalidad de los residuos
shapiro_test_statistic, shapiro_p_value = shapiro(model_stepwise.resid)
print("Resultado de la prueba de Shapiro-Wilk:")
print("Estadístico de prueba:", shapiro_test_statistic)
print("P-valor:", shapiro_p_value)
if shapiro_p_value > 0.05:
    print("Los residuos parecen provenir de una distribución normal (no se rechaza H0)")
else:
    print("Los residuos no parecen provenir de una distribución normal (se rechaza H0)")

# Realizar la prueba de Kolmogorov-Smirnov para evaluar la normalidad de los residuos
ks_test_statistic, ks_p_value = kstest(model_stepwise.resid, 'norm')
print("Resultado de la prueba de Kolmogorov-Smirnov:")
print("Estadístico de prueba:", ks_test_statistic)
print("P-valor:", ks_p_value)
if ks_p_value > 0.05:
    print("Los residuos parecen provenir de una distribución normal (no se rechaza H0)")
else:
    print("Los residuos no parecen provenir de una distribución normal (se rechaza H0)")


#HOMOCEDASTICIDAD
# Graficar los residuos vs. los valores predichos
plt.scatter(model_stepwise.fittedvalues, residuos)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Residuos vs. Valores Predichos')
plt.axhline(y=0, color='r', linestyle='--')  # Línea horizontal en y=0 para referencia
plt.show()

# Calcular los residuos
residuos = y_test - predictions_lm

# Graficar los residuos vs. los valores predichos
plt.scatter(predictions_lm, residuos)
plt.xlabel('Predichos')
plt.ylabel('Residuos')
plt.title('Residuos vs. Predichos - Regresión Lineal')
plt.axhline(y=0, color='red', linestyle='--')  # Línea de referencia en y=0
plt.show()



#AUTOCORRELACION
import statsmodels.stats.api as sms

# Calcular el estadístico de Durbin-Watson
durbin_watson_statistic = sms.durbin_watson(model_stepwise.resid)

# Imprimir el estadístico de Durbin-Watson
print("Estadístico de Durbin-Watson:", durbin_watson_statistic)

# Interpretar el resultado
if durbin_watson_statistic < 1.5:
    print("Los residuos tienen autocorrelación positiva.")
elif durbin_watson_statistic > 2.5:
    print("Los residuos tienen autocorrelación negativa.")
else:
    print("Los residuos no tienen autocorrelación significativa.")


#MULTICOLINEALIDAD
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ajustar el modelo de regresión lineal utilizando todas las variables explicativas
model = sm.OLS(y_train, X_train_stepwise)
result = model.fit()

# Calcular el VIF para cada variable explicativa
vif = pd.DataFrame()
vif["Variable"] = X_train_stepwise.columns
vif["VIF"] = [variance_inflation_factor(X_train_stepwise.values, i) for i in range(X_train_stepwise.shape[1])]

# Imprimir los resultados
print(vif)

