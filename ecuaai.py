import requests

def obtener_imagenes(access_key, collection_id, total_images, images_per_page=30):
    url = f"https://api.unsplash.com/collections/{collection_id}/photos"
    headers = {"Authorization": f"Client-ID {access_key}"}
    all_photos = []

    for page in range(1, (total_images // images_per_page) + 2):
        params = {"per_page": images_per_page, "page": page}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            photos = response.json()
            print(len(photos))
            all_photos.extend(photos)

            if len(all_photos) >= total_images:
                break
        else:
            print(f"Error en la solicitud: {response.status_code}")
            break

    return all_photos

# Uso de la función
access_key = "nH8j13Vrx-gsXfeCy4lzGYkY3j3sHGiNIg6qdiblt_s"
collection_id = "N4wDOieIAwg"
total_images = 80

fotos = obtener_imagenes(access_key, collection_id, total_images)

# for foto in fotos:
#     print(foto['urls']['regular'])

import pandas as pd

# Suponiendo que 'fotos' es una lista de diccionarios con información de las fotos
df = pd.DataFrame(fotos)

# Selecciona solo las columnas relevantes, por ejemplo, 'id' y 'urls'
df = df[['id', 'urls']]

# Elimina filas con datos faltantes
df.dropna(inplace=True)

# Guarda el DataFrame limpio en un archivo CSV
df.to_csv('datos_limpieza.csv', index=False)


