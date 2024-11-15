import pandas as pd

# Leer los archivos xlsx
#df1 = pd.read_excel('bodegasconweb10años.xlsx')
#df2 = pd.read_excel('scrap_binario.xlsx')

# Unir los DataFrames usando la columna clave
#df_merged = pd.merge(df1, df2, on='Nombre')

# Guardar el DataFrame unido en un nuevo archivo xlsx
#df_merged.to_excel('fusion.xlsx', index=False)

'''Con el siguiente fragmento de código, fusionamos el archivo obtenido en micefinal.py "Roa_mean_sin_atipicos" (con la imputación MICE 
y el cálculo de la media del ROA) con el archivo obtenido en pcaoutliersprueba.py "datos_sin_atipicos_con resto_columnas" (sin outliers)'''

# Leer los archivos xlsx
df1 = pd.read_excel('Roa_mean_sin_atipicos.xlsx')
#df2 = pd.read_excel('datos.xlsx')
df2 = pd.read_excel('datos_sin_atipicos_con_resto_columnas.xlsx')

# Unir los DataFrames usando la columna clave
df_merged = pd.merge(df1, df2, on='Nombre')

# Guardar el DataFrame unido en un nuevo archivo xlsx
df_merged.to_excel('datos_fusion.xlsx', index=False)

'''Una vez tenemos el archivo filtrado de datos faltantes 
imprimimos por pantalla la lista de bodegas definitivas con menos de un 20% de faltantes
obteniendo una lista de 1546 bodegas en total'''
#imprimimos el dataframe  con los nombres de las bodegas
Nombre = "Nombre"
df_nombres_bodegas = (df_merged[Nombre])
print(df_nombres_bodegas)

#Lista de las páginas web de las bodegas
lista_bodegas = list(df_nombres_bodegas)
print(lista_bodegas)
