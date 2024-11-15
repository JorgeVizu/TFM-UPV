
'''
eliminamos aquellas columnas que tienen datos categóricos
'''

# Selecciona las columnas que contienen datos categóricos no numéricos
#columnas_no_numericas = datos_imputados_df.select_dtypes(exclude=['number']).columns

# Elimina las columnas no numéricas del DataFrame
#df_sin_columnas_no_numericas = datos_imputados_df.drop(columns=columnas_no_numericas)

#df_sin_columnas_no_numericas.to_excel('archivo_sin_categoricos.xlsx', index=False)
