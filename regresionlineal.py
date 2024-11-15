import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest


#CODIGO PARA EL MODELO DE REGRESIÓN LINEAL
'''En primer lugar creamos las diferentes variables para
cada uno de los conjuntos de datos de entrenamiento y testing'''
X_train = pd.read_csv("x_train.csv", index_col=False)
X_test = pd.read_csv("x_test.csv", index_col=False)
y_train = pd.read_csv("y_train.csv", index_col=False)
y_test = pd.read_csv("y_test.csv", index_col=False)


'''Posteriormente inicializamos y entrenamos el modelo'''
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

'''Obtenemos los coeficientes'''
coeficientes = model.coef_

'''Obtenemos las prediciones del modelo y las mostramos por pantalla'''
y_pred = model.predict(X_test)
print(y_pred)

'''Calculamos las diferentes medidas de bondad de ajuste'''
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
nmse = mse / np.var(y_test, ddof=1)
nmae = mae / np.ptp(y_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)


# Calcular el coeficiente de correlación de Pearson
corr = np.corrcoef(y_test, y_pred)[0, 1]

'''Mostramos por pantalla los resultados de las medidas de bondad de ajuste'''
print("Coeficientes de regresión lineal:", coeficientes)
print("Error Cuadrático Medio (MSE):", mse)
print("Error Absoluto Medio (MAE):", mae)
print("Raíz del Error Cuadrático Medio (RMSE):", rmse)
print("Normalized Mean Squared Error (NMSE):", nmse)
print("Normalized Mean Absolute Error (NMAE):", nmae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Coeficiente de Determinación (R^2):", r2)
print("Coeficiente de Correlación de Pearson:", corr)

'''Visualizas los valores observados vs los valores predichos'''
plt.scatter(y_test, y_pred)
plt.xlabel('Observados')
plt.ylabel('Predichos')
plt.title('Observados vs. Predichos - Regresión Lineal')
plt.show()



'''Definimos una función para realizar la regresión stepwise'''
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

'''Aplicamos la regresión stepwise'''
selected_features = stepwise_selection(X_train, y_train)

'''Ajustamos el modelo con las características seleccionadas'''
X_train_stepwise = X_train[selected_features]
X_train_stepwise = sm.add_constant(X_train_stepwise)  # Añadir una columna de unos para el intercepto
model_stepwise = sm.OLS(y_train, X_train_stepwise).fit()

'''Mostramos por pantalla el resumen del modelo stepwise'''
print(model_stepwise.summary())

'''Calculamos el Residual standard error, el R cuadrado múltiple y R cuadrado ajustado
y los estadisticos F y los mostramos por pantalla'''
residual_std_error = model_stepwise.mse_resid ** 0.5
print("Residual standard error:", residual_std_error)
r_squared = model_stepwise.rsquared
print("Multiple R-squared:", r_squared)
adjusted_r_squared = model_stepwise.rsquared_adj
print("Adjusted R-squared:", adjusted_r_squared)

# Estadísticas F
f_statistic = model_stepwise.fvalue
f_p_value = model_stepwise.f_pvalue
print("F-statistic:", f_statistic)
print("p-value:", f_p_value)

# Obtener los residuos del modelo de regresión
residuos = model_stepwise.resid

#NORMALIDAD
'''Realizamos la prueba de Shapiro-Wilk para evaluar la normalidad de los residuos
y mostramos por pantalla los resultados'''
shapiro_test_statistic, shapiro_p_value = shapiro(model_stepwise.resid)
print("Resultado de la prueba de Shapiro-Wilk:")
print("Estadístico de prueba:", shapiro_test_statistic)
print("P-valor:", shapiro_p_value)
if shapiro_p_value > 0.05:
    print("Los residuos parecen provenir de una distribución normal (no se rechaza H0)")
else:
    print("Los residuos no parecen provenir de una distribución normal (se rechaza H0)")

'''Realizamos la prueba de Kolmogorov-Smirnov para evaluar la normalidad de los residuos
y mostramos por pantalla los resultados'''
ks_test_statistic, ks_p_value = kstest(model_stepwise.resid, 'norm')
print("Resultado de la prueba de Kolmogorov-Smirnov:")
print("Estadístico de prueba:", ks_test_statistic)
print("P-valor:", ks_p_value)
if ks_p_value > 0.05:
    print("Los residuos parecen provenir de una distribución normal (no se rechaza H0)")
else:
    print("Los residuos no parecen provenir de una distribución normal (se rechaza H0)")


#HOMOCEDASTICIDAD
'''Visualizamos Graficamente los residuos vs los valores predichos
para interpretar la Homocedasticidad del modelo'''
plt.scatter(model_stepwise.fittedvalues, residuos)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Residuos vs. Valores Predichos')
plt.axhline(y=0, color='r', linestyle='--')  # Línea horizontal en y=0 para referencia
plt.show()


'''Calculamos los residuos y visualizamos Graficamente los residuos vs. los valores predichos'''
residuos = y_test - y_pred

plt.scatter(y_pred, residuos)
plt.xlabel('Predichos')
plt.ylabel('Residuos')
plt.title('Residuos vs. Predichos - Regresión Lineal')
plt.axhline(y=0, color='red', linestyle='--')  # Línea de referencia en y=0
plt.show()


#AUTOCORRELACION
import statsmodels.stats.api as sms

'''Calculamos el estadístico de Durbin-Watson y mostramos por pantalla el resultado
y posteriormente interpretamos el resultado para evaluar la posible Autocorrelacion de las variables'''
durbin_watson_statistic = sms.durbin_watson(model_stepwise.resid)
print("Estadístico de Durbin-Watson:", durbin_watson_statistic)

if durbin_watson_statistic < 1.5:
    print("Los residuos tienen autocorrelación positiva.")
elif durbin_watson_statistic > 2.5:
    print("Los residuos tienen autocorrelación negativa.")
else:
    print("Los residuos no tienen autocorrelación significativa.")


#MULTICOLINEALIDAD
from statsmodels.stats.outliers_influence import variance_inflation_factor

'''Ajustamos el modelo de regresión lineal utilizando todas las variables explicativas'''
model = sm.OLS(y_train, X_train_stepwise)
result = model.fit()

'''Calculamos el VIF para cada variable explicativa y mostramos los resultados
para evaluar la posible Multicolinealidad del modelo'''
vif = pd.DataFrame()
vif["Variable"] = X_train_stepwise.columns
vif["VIF"] = [variance_inflation_factor(X_train_stepwise.values, i) for i in range(X_train_stepwise.shape[1])]

print(vif)