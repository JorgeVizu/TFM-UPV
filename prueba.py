import pandas as pd
from pytrends.request import TrendReq

# Término específico para comparar
termino_de_comparacion = "bodegas Torre Oria"

# Lista de términos que vamos a comparar con respecto al término de comparación
lista_de_terminos = ["bodega Pago de Tharsys", "bodegas Murviedro", "bodegas Nodus"]
#lista_de_terminos = ["bodega Pago de Tharsys", "bodegas Murviedro", "bodegas Nodus", "bodegas Ramírez de la Piscina", "bodegas Vereda Real", "bodegas Alejandro", "bodegas Cueva", "bodega Se biran", "Esnac bodegas", "bodega Rafael Cambra", "bodegas Iranzo", "bodega Masia de la Hoya", "bodega Mamerto de la Vara", "bodegas La Torre", "bodegas Pinoso", "bodegas Gutierrez de la Vega", "bodegas Francisco Gomez", "bodegas Casa Corredor", "bodega El Mollet", "bodega Sierra Norte", "bodegas Vivanza", "bodegas Xalo", "bodegas Godelleta", "bodega Sant Pere Moixent", "bodegas Familia bastida", "bodegas Cerrogallina", "bodegas Pigar", "bodegas Artadi", "bodegas Volver", "bodega Vinival", "bodega Vegamar", "bodega Torre Enmedio", "bodegas Sierra Salinas", "bodegas Sierra de Cabreras", "bodegas Proexa", "bodegas Pedro Moreno 1940", "bVC bodegas", "bodega La Alcublana", "bodega Monovar", "bodega Mitos", "bodegas Los Frailes", "bodegas Karmin", "bodegas Jimenez Vila", "bodegas Hispanosuizas", "bodegas Fuso", "bodegas Enguera", "bodegas Alfori", "bodegas Enrique Mendoza", "bodegas bataller", "bodegas bleda", "bodegas Primitivo Quiles", "Pasiego bodegas", "bodegas Nuestra Senora del Socorro", "bodega Flors", "bodega Virtudes", "bodega Mustiguillo", "bodega Santa Catalina", "bodega Cooperativa de Castalla", "bodega Demoya", "bodega Aran Leon", "bodega Dussart Pedron", "bodegas Arraez", "bodegas Antonio Alcaraz"]


# Crea una instancia de pytrends
pytrends = TrendReq(hl='es-ES', tz=360)

# Creamos un dataframe para posteriormente almacenar los datos
df_data = pd.DataFrame()

# Bucle para comparar cada término de la lista con respecto al término de comparación
for termino in lista_de_terminos:
    # Definimos la búsqueda
    pytrends.build_payload([termino, termino_de_comparacion], cat=0, timeframe='today 1-m', geo='ES', gprop='')
    
    # Realizamos la búsqueda
    datos_busqueda = pytrends.interest_over_time()

    # Imprimos los datos de búsqueda y las diferentes comparaciones
    print("Comparación para:", termino)
    print(datos_busqueda)
    print("\n")

    # Agregamos los datos de búsqueda al DataFrame
    df_data[f'{termino}_vs_{termino_de_comparacion}'] = datos_busqueda[termino]
    df_data[f'{termino_de_comparacion}_vs_{termino}'] = datos_busqueda[termino_de_comparacion]


# Guarda los datos en un archivo CSV
df_data.to_csv("prueba.csv", index=False)
