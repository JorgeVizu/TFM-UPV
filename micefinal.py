from fancyimpute import IterativeImputer
import pandas as pd

'''
Aplicamos el algoritmo MICE para el tratamiento de los datos faltantes
sobre el df_filtrado con menos del 20% de nulos y una vez eliminados los outliers con el bucle: 1546 bodegas y una vez realizado el tratamiento de outliers 1230 bodegas'''

# Especifica la ruta del archivo Excel
#ruta_archivo_excel = 'archivo_ROA_depurado.xlsx'
#ruta_archivo_excel = 'datos_sin_atipicos.xlsx'
ruta_archivo_excel = 'datos_sin_atipicos_con_resto_columnas.xlsx'

# Carga el archivo Excel en un DataFrame
datos_df = pd.read_excel(ruta_archivo_excel)

# Especificar las columnas de interés
columnas_interes = ['ROA 2021', 'ROA 2020', 'ROA 2019', 'ROA 2018', 'ROA 2017', 'ROA 2016', 'ROA 2015', 'ROA 2014', 'ROA 2013', 'ROA 2012']

# Seleccionar solo las columnas que deseas imputar
datos_a_imputar = datos_df[columnas_interes]

# Crear un IterativeImputer con el método MICE
imputer = IterativeImputer()

# Imputar los datos faltantes
datos_imputados_array = imputer.fit_transform(datos_a_imputar)

# Convertir los datos imputados de nuevo a un DataFrame
datos_imputados_df = pd.DataFrame(datos_imputados_array, columns=columnas_interes)

# Actualizar las columnas imputadas en el DataFrame original
datos_df[columnas_interes] = datos_imputados_df

# Especifica el nombre del archivo Excel donde deseas guardar los datos
#nombre_archivo = 'mice_final.xlsx'
nombre_archivo = 'mice_sin_atipicos.xlsx'

# Guarda el DataFrame en un archivo Excel sin incluir el índice
datos_imputados_df.to_excel(nombre_archivo, index=False)


'''
calculamos la media del ROA para cada bodega en el dataframe con los datos imputados mediante MICE.
'''
df_mean_roa = pd.DataFrame(datos_imputados_df)

# Calcula la media en cada fila de las columnas especificadas
roa_mean = df_mean_roa[columnas_interes].mean(axis=1)

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

# Encuentra el índice de la fila donde está el valor mínimo en la columna 'roa_mean'
indice_minimo = df_mean_roa['roa_mean'].idxmin()
print("Índice de la fila con el valor mínimo en 'roa_mean':", indice_minimo)

# Guarda el DataFrame en un archivo Excel sin incluir el índice
#Roa_mean = 'Roa_mean.xlsx'
Roa_mean = 'Roa_mean_sin_atipicos.xlsx'
df_mean_roa.to_excel(Roa_mean, index=False)

