import micefinal
import pandas as pd
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame llamado df
df_roa_descriptive =  micefinal.df_mean_roa

# Supongamos que deseas analizar la variable 'variable_de_interes' en tu DataFrame
variable_de_interes = df_roa_descriptive['roa_mean']
print(variable_de_interes)

# Estadísticas descriptivas
descripcion = variable_de_interes.describe()
# Imprimir las estadísticas descriptivas
print(descripcion)
df_descripción = pd.DataFrame(descripcion)
print(df_descripción)

# Configura el tamaño del gráfico
plt.figure(figsize=(10, 6))

# Crea el histograma
plt.hist(variable_de_interes, bins=20, color='blue', alpha=0.7)

# Agrega etiquetas y título
plt.xlabel('Media ROA')
plt.ylabel('Frecuencia')
plt.title('Histograma de la media del ROA')
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