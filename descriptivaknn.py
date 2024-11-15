import KNN
import pandas as pd
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame llamado df
df_roa_descriptive =  KNN.df_mean_roa

# Supongamos que deseas analizar la variable 'variable_de_interes' en tu DataFrame
variable_de_interes = KNN.df_mean_roa['roa_mean']

# Estadísticas descriptivas
descripcion = variable_de_interes.describe()

# Imprimir las estadísticas descriptivas
print(descripcion)
df_descripción = pd.DataFrame(descripcion)
print(df_descripción)

# Configura el tamaño del gráfico
plt.figure(figsize=(10, 6))

# Crea el histograma
plt.hist(descripcion, bins=20, color='blue', alpha=0.7)

# Agrega etiquetas y título
plt.xlabel('Media ROA')
plt.ylabel('Frecuencia')
plt.title('Histograma de la media del ROA')

# Muestra el histograma
plt.show()