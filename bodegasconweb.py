import pandas as pd

# Nombre del archivo Excel
archivo_excel = 'SABI_Export_bodegas.xls'

# Nombre de la hoja y de la columna en la que deseas buscar datos faltantes
hoja_deseada = 'Resultados'  # Reemplaza con el nombre de tu hoja
columna_deseada = 'Web'  # Reemplaza con el nombre de tu columna
Nombre = 'Nombre'

# Leer el archivo Excel
df = pd.read_excel(archivo_excel, sheet_name=hoja_deseada)

# Eliminar filas con datos faltantes en la columna deseada
df = df.dropna(subset=[columna_deseada])

#imprimimos el dataframe  con los nombres de las bodegas
df_nombres_bodegas = (df[Nombre])
print(df_nombres_bodegas)


#df_bodegas_con_web = df[columna_deseada].tolist()
#df_bodegas_con_web = list(df[columna_deseada])

#Lista de las páginas web de las bodegas
lista_bodegas_con_web = list(df_nombres_bodegas)
print(lista_bodegas_con_web)

#Guardar el DataFrame modificado de nuevo en el archivo Excel
df.to_excel('bodegasconweb.xlsx', index=False)