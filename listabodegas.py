import pandas as pd

# Nombre del archivo Excel
archivo_excel = 'SABI_Export_bodegas.xls'

# Nombre de la hoja y de la columna que deseas extraer
hoja_deseada = 'Resultados'  # Reemplaza con el nombre de tu hoja
columna_deseada = 'Nombre'  # Reemplaza con el nombre de tu columna

# Leer el archivo Excel y crear una lista con los elementos de la columna
df = pd.read_excel(archivo_excel, sheet_name=hoja_deseada)

# Imprimir el dataframe de las bodegas
df_bodegas = df[columna_deseada]
print(df_bodegas)

#lista_bodegas = df[columna_deseada].tolist()
#lista_bodegas = list(df[columna_deseada])

# Imprimir la lista de bodegas
#print(lista_bodegas)



