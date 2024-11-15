import pandas as pd

'''
con este código calculamos las filas que presentan más de 2 nulos
y las eliminamos.
'''

# Cargar el archivo xlsx en un DataFrame
df = pd.read_excel('fusionconNAN.xlsx')

# Especificar las columnas de interés
columnas_interes = ['ROA 2021', 'ROA 2020', 'ROA 2019', 'ROA 2018', 'ROA 2017', 'ROA 2016', 'ROA 2015', 'ROA 2014', 'ROA 2013', 'ROA 2012']

# Contar el número de valores nulos en las columnas de interés para cada fila
num_nulos_por_fila = df[columnas_interes].isnull().sum(axis=1)

# Filtrar las filas con más de 2 valores nulos
filas_con_mas_de_dos_nulos = df[num_nulos_por_fila > 2]

# Obtener el número de filas con más de 2 valores nulos
num_filas_con_mas_de_dos_nulos = len(filas_con_mas_de_dos_nulos)

# Imprimir el resultado
print("Número de filas con más de 2 valores nulos en las columnas de interés:", num_filas_con_mas_de_dos_nulos)

# Eliminar las filas con más de dos datos nulos del DataFrame
df_filtrado = df.drop(filas_con_mas_de_dos_nulos.index)

# Guardar el DataFrame filtrado en un nuevo archivo xlsx
df_filtrado.to_excel('archivo_filtrado.xlsx', index=False)

# Eliminar las columnas restantes a las columnas de interés del dataframe
df_filtrado = df_filtrado[columnas_interes]
df_filtrado.to_excel('archivo_filtrado_solo_columnas_de_interes.xlsx', index=False )


'''Una vez tenemos el archivo filtrado de datos faltantes
imprimimos por pantalla la lista de bodegas definitivas con menos de un 20% de faltantes
obteniendo una lista de 1546 bodegas en total'''
#imprimimos el dataframe  con los nombres de las bodegas
Nombre = "Nombre"
df_nombres_bodegas = (df_filtrado[Nombre])
print(df_nombres_bodegas)

#Lista de las páginas web de las bodegas
lista_bodegas = list(df_nombres_bodegas)
print(lista_bodegas)

