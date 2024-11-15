from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import time
import bodegasconweb
import fusion
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

class Trending():

    def find_top(self):
        print("estoy probando")
        #Lista de términos que deseas comparar
        #lista_de_terminos = [" bodega Pago de Tharsys", "bodegas Murviedro", "bodegas Nodus", "bodegas Alejandro", "bodegas Cueva", "bodega Rafael Cambra", "bodegas La Torre", "bodegas Francisco Gomez", "bodega Sierra Norte", "bodegas Xalo", "bodegas Artadi", "bodegas Volver", "bodegas Los Frailes", "bodegas Enguera", "bodegas Enrique Mendoza", "bodegas bleda", "bodega Flors", "bodega Mustiguillo", "bodega Santa Catalina", "bodegas Arraez"]
        #lista_de_terminos = bodegasconweb.lista_bodegas_con_web
        lista_de_terminos = fusion.lista_bodegas
       
       
        #Creamos una instancia de pytrends
        pytrends = TrendReq(hl='es-ES', tz=360)

        """
        testing for error management
        """

        #pytrends = TrendReq(hl='en-US', tz=360, timeout=(20,40), proxies=['https://20.205.61.143:80',], retries=2, backoff_factor=0.1, requests_args={'verify':False})
        #pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), retries=2, backoff_factor=0.1, requests_args={'verify':False})

        #Diccionario para almacenar los volúmenes de búsqueda de cada término
        volumenes_de_busqueda = {}

        #Bucle para obtener el volumen de búsqueda de cada término
        for termino in lista_de_terminos:
            
#Posteriormente, obtenemos los datos de búsqueda:
#Google Trends tiene limitaciones con la cantidad de solicitudes que se pueden hacer en un tiempo específico.
#Mediante el siguiente Try except limitamos la frecuencia de las solicitudes con un "time Sleep de 60 segundos"
        
        #Realizamos la búsqueda
            try:
                pytrends.build_payload([termino], cat=0, timeframe='today 1-m', geo='ES', gprop='')
                datos_busqueda = pytrends.interest_over_time()

            except TooManyRequestsError as e:
                print(f"Error 429: Demasiadas solicitudes. Esperando y reintentando...")
                time.sleep(60)
                datos_busqueda = pytrends.interest_over_time()

            # Verificamos si el término está presente en los datos antes de acceder a él
            if termino in datos_busqueda.columns:
        # Calculamos el volumen de búsqueda promedio
                volumen_promedio = datos_busqueda[termino].mean()

        # Almacenamos el volumen de búsqueda en el diccionario
                volumenes_de_busqueda[termino] = volumen_promedio
            else:
                print(f"El término '{termino}' no está presente en los datos de búsqueda.")

        #Término con el mayor volumen de búsqueda:
        termino_mas_buscado = max(volumenes_de_busqueda, key=volumenes_de_busqueda.get)

        #Imprimimos el término más buscado
        print(f"El término más buscado es: {termino_mas_buscado}")
        return termino_mas_buscado