import pandas as pd
from pytrends.request import TrendReq
from trending import Trending
import bodegasconweb
import filtrarNulos
import fusion

class Bodegas():
    #__my_lista_de_terminos = ["bodega Pago de Tharsys", "bodegas Murviedro", "bodegas Nodus"]
    #__my_lista_de_terminos = filtrarNulos.lista_bodegas
    __my_lista_de_terminos = fusion.lista_bodegas
        
    def __init__(self):
        trending = Trending()
        self.most_searched = trending.find_top()
        self.my_lista_de_terminos = Bodegas.__my_lista_de_terminos
        self.pytrends = TrendReq(hl='es-ES', tz=360)

    def _compare_terms(self, term, reference_term):
        self.pytrends.build_payload([term, reference_term], cat=0, timeframe='today 1-m', geo='ES', gprop='')
        data_search = self.pytrends.interest_over_time()
        return data_search

    def get_popularity(self):
        reference_term = self.most_searched
        df_data = pd.DataFrame()

        for term in self.my_lista_de_terminos:
            try:
                data_search = self._compare_terms(term, reference_term)

                print("Comparación para:", term)
                print(data_search)
                print("\n")

                df_data[f'{term}_vs_{reference_term}'] = data_search[term]
                df_data[f'{reference_term}_vs_{term}'] = data_search[reference_term]

            except Exception as e:
                print(f"Error durante la comparación para {term}: {e}")

        df_data.to_csv("terceraprueba.csv", index=False)

