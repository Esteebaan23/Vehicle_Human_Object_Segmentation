import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directorios
image_dir = r'...'  # Directorio con las imágenes originales
json_dir = r'...'  # Directorio con las anotaciones JSON
mask_dir = r'...'  # Directorio de salida para las máscaras

# Crear el directorio de salida si no existe
os.makedirs(mask_dir, exist_ok=True)

# Mapeo de clases a valores (puedes ajustar los valores si es necesario)
class_mapping = {
    "vehicles": 1,  # Estructuras asignadas al valor 1
    "human": 2,         # Coches asignados al valor 2
    "Structures": 3
}

def create_mask(image_size, objects):
    """
    Crear una máscara a partir de anotaciones.
    """
    mask = Image.new("L", image_size, 0)  # Imagen en escala de grises (0 para fondo)
    draw = ImageDraw.Draw(mask)

    for obj in objects:
        label = obj.get('label')  # Nombre de la clase
        polygon = obj.get('polygon', [])  # Lista de puntos del polígono
        class_value = class_mapping.get(label, 0)  # Valor correspondiente a la clase

        # Validar que el polígono no esté vacío
        if polygon:
            # Convertir coordenadas a enteros
            polygon = [tuple(map(int, punto)) for punto in polygon]
            # Dibujar el polígono en la máscara
            draw.polygon(polygon, outline=class_value, fill=class_value)

    return mask


def plot_images_with_masks(image_paths, mask_paths):
    """
    Mostrar imágenes originales con sus máscaras correspondientes.
    """
    plt.figure(figsize=(15, 10))
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        if i >= 10:  # Mostrar solo las primeras 10 imágenes
            break

        # Cargar imagen y máscara
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Plot imagen original
        plt.subplot(10, 2, 2 * i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Imagen Original")

        # Plot máscara
        plt.subplot(10, 2, 2 * i + 2)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title("Máscara")

    plt.tight_layout()
    plt.show()

# Almacenar rutas de imágenes y máscaras para visualización posterior
image_paths = []
mask_paths = []

# Procesar cada archivo JSON
for json_file in tqdm(os.listdir(json_dir)):
    if not json_file.endswith(".json"):
        continue

    # Leer anotaciones JSON
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Validar contenido del JSON
    image_base_name = json_file.replace("_gtFine_polygons.json", "")
    image_name = image_base_name + ".png"
    objects = data.get('objects', [])
    if not image_name or not objects:
        print(f"Archivo JSON inválido: {json_file}")
        continue

    # Ruta a la imagen original
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Imagen no encontrada: {image_path}")
        continue

    # Abrir la imagen para obtener su tamaño
    with Image.open(image_path) as img:
        image_size = img.size  # (ancho, alto)

    # Crear la máscara
    mask = create_mask(image_size, objects)

    # Guardar la máscara
    mask_output_path = os.path.join(mask_dir, image_base_name + ".png")  # Cambia extensión a PNG
    mask.save(mask_output_path)

    # Almacenar rutas para visualización
    image_paths.append(image_path)
    mask_paths.append(mask_output_path)

# Mostrar imágenes con máscaras
plot_images_with_masks(image_paths, mask_paths)
