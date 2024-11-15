import requests

def consultar_estadisticas_instagram(query, client_id, token):
    # URL base de la API de Social Blade
    url_base = 'https://matrix.sbapis.com/b/'
    endpoint = 'instagram/statistics'

    # Método 1: Parámetros de consulta en la URL
    url = f'{url_base}{endpoint}?query={query}&clientid={client_id}&token={token}'
    respuesta = requests.get(url)
    if respuesta.status_code == 200:
        datos = respuesta.json()
        print('Método 1 - Parámetros de consulta:')
        print('Seguidores:', datos['statistics']['followers'])
        print('Publicaciones:', datos['statistics']['posts'])
    else:
        print('Error al realizar la solicitud:', respuesta.status_code)

    # Método 2: Encabezados de solicitud
    url = f'{url_base}{endpoint}?query={query}'
    headers = {
        'ClientID': client_id,
        'Token': token
    }
    respuesta = requests.get(url, headers=headers)
    if respuesta.status_code == 200:
        datos = respuesta.json()
        print('\nMétodo 2 - Encabezados de solicitud:')
        print('Seguidores:', datos['statistics']['followers'])
        print('Publicaciones:', datos['statistics']['posts'])
    else:
        print('Error al realizar la solicitud:', respuesta.status_code)

# Parámetros de consulta
query = 'bodegas murviedro'
client_id = 'cli_807c6031ed578ae5373e2f54'
token = '0e02af31883607906a9435930e415613fc7de7aae237c00d2ff730629e5b4c9251183872115f6d9298593fd8de998e33f027824503f997af34a6770485449703'

# Realizar la consulta a la API de Social Blade
consultar_estadisticas_instagram(query, client_id, token)