import pandas as pd
import matplotlib.pyplot as plt
import micefinal 
import numpy as np
from scipy.stats import norm, probplot
from scipy.stats import boxcox


# Supongamos que tienes un DataFrame llamado df (con minimo de -26...)
#df_roa_descriptive =  micefinal.df_mean_roa 
#df_roa_descriptive =  pd.read_excel('datos_final.xlsx') # con este archivo analizamos la normalidad con el dato mínimo incluido
df_roa_descriptive =  pd.read_excel('datos_final_sin_minimo.xlsx') #con este archivo analizamos la normalidad sin el dato mínimo

# Supongamos que deseas analizar la variable 'variable_de_interes' en tu DataFrame
variable_de_interes = df_roa_descriptive['roa_mean']

# Estadísticas descriptivas
descripcion = variable_de_interes.describe()

# Calcula el coeficiente de asimetría y la curtosis
coef_asimetria = variable_de_interes.skew()
curtosis = variable_de_interes.kurtosis()

# Imprimir las estadísticas descriptivas
print("Estadísticas descriptivas:")
print(descripcion)
print("Coeficiente de asimetría:", coef_asimetria)
print("Curtosis:", curtosis)

# Configura el tamaño del gráfico
plt.figure(figsize=(10, 6))

# Crea el histograma
plt.hist(variable_de_interes, bins=20, color='blue', alpha=0.7, density=True)

# Obtener los parámetros de la distribución normal
mu, std = variable_de_interes.mean(), variable_de_interes.std()

# Crear un rango de valores x para la curva de la distribución normal
x = np.linspace(variable_de_interes.min(), variable_de_interes.max(), 100)

# Calcular la curva de la distribución normal usando la función de densidad de probabilidad (pdf)
y = norm.pdf(x, mu, std)

# Trazar la curva de la distribución normal en el histograma
plt.plot(x, y, 'r--', linewidth=2)

# Agrega etiquetas y título
plt.xlabel('Media ROA')
plt.ylabel('Densidad de probabilidad')
plt.title('Histograma de la media del ROA con curva de distribución normal')
plt.grid(True)

# Muestra el histograma
plt.show()

# Crea un boxplot para visualizar la distribución y los valores atípicos de la variable
plt.figure(figsize=(10, 6))
plt.boxplot(df_roa_descriptive['roa_mean'])
plt.xlabel('Roa Mean')
plt.title('Boxplot de Roa Mean')
plt.grid(True)
plt.show()

from scipy.stats import shapiro

# Realizar la prueba de Shapiro-Wilk
stat, p_valor = shapiro(variable_de_interes)

# Mostrar el resultado del p-valor
print("Resultado de la prueba de Shapiro-Wilk:")
print("Estadístico de prueba:", stat)
print("P-valor:", p_valor)

# Interpretar el resultado
alpha = 0.05
if p_valor > alpha:
    print('Los datos parecen provenir de una distribución normal (no se rechaza H0)')
else:
    print('Los datos no parecen provenir de una distribución normal (se rechaza H0)')


# Realizar la prueba de Kolmogorov-Smirnov
from scipy.stats import kstest

# Realizar la prueba de Kolmogorov-Smirnov
stat, p_valor = kstest(variable_de_interes, 'norm')

# Mostrar el resultado del p-valor
print("Resultado de la prueba de Kolmogorov-Smirnov:")
print("Estadístico de prueba:", stat)
print("P-valor:", p_valor)

# Interpretar el resultado
alpha = 0.05
if p_valor > alpha:
    print('Los datos parecen provenir de una distribución normal (no se rechaza H0)')
else:
    print('Los datos no parecen provenir de una distribución normal (se rechaza H0)')


# Gráfico Q-Q Normal Sin Tendencia
plt.figure(figsize=(8, 8))
probplot(variable_de_interes, dist="norm", plot=plt)
plt.title('Gráfico Q-Q Normal Sin Tendencia')
plt.xlabel('Cuantiles teóricos')
plt.ylabel('Cuantiles observados')
plt.grid(True)
plt.show()


# Realizar el test de Box-Cox para homocedasticidad
transformed_data, best_lambda = boxcox(variable_de_interes)

# Imprimir el mejor valor de lambda encontrado por el test de Box-Cox
print("Mejor valor de lambda encontrado por el test de Box-Cox:", best_lambda)

# Gráfico de los datos transformados vs. lambda
plt.figure(figsize=(10, 6))
plt.plot(np.arange(-2, 2, 0.1), [boxcox(variable_de_interes, lmbda=l)[0].mean() for l in np.arange(-2, 2, 0.1)], 'r--', label='Media')
plt.plot(np.arange(-2, 2, 0.1), [boxcox(variable_de_interes, lmbda=l)[0].std() for l in np.arange(-2, 2, 0.1)], 'b--', label='Desviación Estándar')
plt.axvline(x=best_lambda, color='g', linestyle='--', label='Mejor Lambda')
plt.xlabel('Valor de lambda')
plt.ylabel('Media / Desviación Estándar')
plt.title('Efecto de lambda en la media y la desviación estándar')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de los datos transformados
plt.figure(figsize=(10, 6))
plt.hist(transformed_data, bins=20, color='blue', alpha=0.7, density=True)
plt.xlabel('Datos Transformados')
plt.ylabel('Densidad de probabilidad')
plt.title('Histograma de los datos transformados')
plt.grid(True)
plt.show()