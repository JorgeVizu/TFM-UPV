import pandas as pd
import numpy as np
import filtrarNulos
# importing the KNN from fancyimpute library
from fancyimpute import KNN

# Seleccionar solo las columnas que deseas imputar
datos_a_imputar = filtrarNulos.df_filtrado[filtrarNulos.columnas_interes]

# printing the dataframe
print(datos_a_imputar)
  
# calling the KNN class
knn_imputer = KNN()
# imputing the missing value with knn imputer
df = knn_imputer.fit_transform(datos_a_imputar)

# Convertir los datos imputados de nuevo a un DataFrame
datos_imputados_df = pd.DataFrame(df, columns=filtrarNulos.columnas_interes)

# printing dataframe
print(datos_imputados_df)

# Especifica el nombre del archivo Excel donde deseas guardar los datos
nombre_archivo = 'knn.xlsx'

# Guarda el DataFrame en un archivo Excel sin incluir el índice
datos_imputados_df.to_excel(nombre_archivo, index=False)


'''
calculamos la media del ROA para cada bodega en el dataframe con los datos imputados mediante MICE.
'''
df_mean_roa = pd.DataFrame(datos_imputados_df)

# Calcula la media en cada fila de las columnas especificadas
roa_mean = df_mean_roa[filtrarNulos.columnas_interes].mean(axis=1)

# Agrega la columna de medias al DataFrame
df_mean_roa['roa_mean'] = roa_mean

# Encuentra el valor mínimo en la columna 'roa_mean'
minimo = df_mean_roa['roa_mean'].min()

# Encuentra el valor máximo en la columna 'roa_mean'
maximo = df_mean_roa['roa_mean'].max()

print("Valor mínimo en 'roa_mean':", minimo)
print("Valor máximo en 'roa_mean':", maximo)


# Encuentra el índice de la fila donde está el valor máximo en la columna 'roa_mean'
indice_maximo = df_mean_roa['roa_mean'].idxmax()
print("Índice de la fila con el valor máximo en 'roa_mean':", indice_maximo)

# Guarda el DataFrame en un archivo Excel sin incluir el índice
Roa_mean_knn = 'Roa_mean_knn.xlsx'
df_mean_roa.to_excel(Roa_mean_knn, index=False)