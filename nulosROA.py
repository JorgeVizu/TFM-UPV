import pandas as pd

'''
con este código reemplazamos los n.d por NAN  en un determinado número de columnas 
y posteriormente calcularemos el número de filas con valores nulos 
y lo mostraremos por pantalla y lo guardamos en otro excel
'''

# Leer el archivo xlsx en un DataFrame
#df = pd.read_excel('fusion.xlsx')
df = pd.read_excel('fusion_sinbinarizar.xlsx')

# Reemplazar "n.d." por NaN (valores nulos de Pandas)
df.replace("n.d.", float("nan"), inplace=True)

# Convertir las columnas a tipo de datos 'object'
df = df.infer_objects(copy=False)

# Seleccionar las columnas de interés
columnas_de_interes = ['ROA 2021', 'ROA 2020', 'ROA 2019', 'ROA 2018', 'ROA 2017', 'ROA 2016', 'ROA 2015', 'ROA 2014', 'ROA 2013', 'ROA 2012']

# Calcular el número de filas con valores nulos en las columnas de interés
num_filas_con_nulos = df[columnas_de_interes].isnull().any(axis=1).sum()

# Imprimir el resultado
print("Número de filas con valores nulos en las columnas de interés:", num_filas_con_nulos)

# Guarda los datos en un archivo Excel
df.to_excel('fusionconNAN.xlsx', index=False)