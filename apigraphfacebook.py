import requests

def obtener_datos_facebook(page_id, access_token):
    url = f"https://graph.facebook.com/{page_id}?fields=fan_count,posts&access_token={access_token}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("Número de seguidores:", data['fan_count'])
        print("Últimas publicaciones:")
        for post in data['posts']['data']:
            print(post.get('message', 'No hay mensaje'))
    else:
        print("Error al realizar la solicitud:", response.status_code)

# ID de la página de Facebook de la bodega
page_id = 'bodegasmurviedro'

# Token de acceso con los permisos necesarios
access_token = '3931038020505871'

# Realizar la solicitud a la API de Facebook Graph
obtener_datos_facebook(page_id, access_token)