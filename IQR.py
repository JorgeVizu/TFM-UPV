import pandas as pd
import matplotlib.pyplot as plt

def identificar_outliers_iqr(column):
    # Calcular el rango intercuartil (IQR)
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir los límites inferior y superior para identificar outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identificar outliers
    outliers_lower = column[column < lower_bound]
    outliers_upper = column[column > upper_bound]
    
    # Verificar si hay outliers antes de construir el DataFrame
    if outliers_lower.empty and outliers_upper.empty:
        return pd.DataFrame({'Outliers inferiores': [], 'Outliers superiores': []})
    
    # Ajustar la longitud de los arrays para que tengan la misma cantidad de elementos
    min_length = min(len(outliers_lower), len(outliers_upper))
    outliers_lower = outliers_lower[:min_length]
    outliers_upper = outliers_upper[:min_length]
    
    # Crear un DataFrame con los outliers inferiores y superiores
    outliers_df = pd.DataFrame({'Outliers inferiores': outliers_lower.values, 'Outliers superiores': outliers_upper.values})
    
    return outliers_df