from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import time
import bodegasconweb
import filtrarNulos
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

class Trending():
    
    def find_top(self):
        print("estoy probando")
        # Lista de términos que deseas comparar
        lista_de_terminos = bodegasconweb.lista_bodegas_con_web
        #lista_de_terminos = filtrarNulos.lista_bodegas
        print(lista_de_terminos)
        
        # Creamos una instancia de pytrends
        pytrends = TrendReq(hl='es-ES', tz=360)

        # Diccionario para almacenar los volúmenes de búsqueda de cada término
        volumenes_de_busqueda = {}

        # Bucle para obtener el volumen de búsqueda de cada término
        for termino in lista_de_terminos:
            try:
                # Realizamos la búsqueda
                pytrends.build_payload([termino], cat=0, timeframe='today 1-m', geo='ES', gprop='')
                datos_busqueda = pytrends.interest_over_time()

                # Verificamos si el término está presente en los datos antes de acceder a él
                if termino in datos_busqueda.columns:
                    # Calculamos el volumen de búsqueda promedio
                    volumen_promedio = datos_busqueda[termino].mean()

                    # Almacenamos el volumen de búsqueda en el diccionario
                    volumenes_de_busqueda[termino] = volumen_promedio
                else:
                    print(f"El término '{termino}' no está presente en los datos de búsqueda.")
            except TooManyRequestsError as e:
                print(f"Error 429: Demasiadas solicitudes. Esperando y reintentando...")
                time.sleep(60)
        
        # Término con el mayor volumen de búsqueda:
        termino_mas_buscado = max(volumenes_de_busqueda, key=volumenes_de_busqueda.get)

        # Imprimimos el término más buscado
        print(f"El término más buscado es: {termino_mas_buscado}")
        return termino_mas_buscado