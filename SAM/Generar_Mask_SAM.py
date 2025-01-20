import os
import json
import numpy as np
from PIL import Image, ImageDraw

# Configuración
ANNOTATIONS_PATH = r"..."  # Ruta al archivo JSON con las anotaciones
IMAGES_PATH = r"..."  # Carpeta donde están las imágenes originales
OUTPUT_MASKS_PATH = r"..."  # Carpeta base para guardar máscaras
CLASS_NAME = "vehicles"  # Cambia esto para la clase que quieres procesar

# Crear carpeta de salida para la clase si no existe
class_output_path = os.path.join(OUTPUT_MASKS_PATH, CLASS_NAME)
os.makedirs(class_output_path, exist_ok=True)

# Obtener lista de archivos JSON
annotation_files = [f for f in os.listdir(ANNOTATIONS_PATH) if f.endswith(".json")]

# Procesar cada archivo de anotación
for annotation_file in annotation_files:
    # Cargar el archivo JSON
    with open(os.path.join(ANNOTATIONS_PATH, annotation_file), "r") as f:
        annotation = json.load(f)

    # Obtener el nombre base de la imagen asociada
    base_name = annotation_file.replace("_gtFine_polygons.json", "")
    image_name = f"{base_name}.png"
    image_path = os.path.join(IMAGES_PATH, image_name)

    # Cargar tamaño de la imagen original
    if not os.path.exists(image_path):
        print(f"Imagen no encontrada para {annotation_file}. Saltando...")
        continue

    with Image.open(image_path) as img:
        width, height = img.size

    # Crear una máscara vacía (fondo = 0)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Dibujar polígonos para la clase especificada
    for obj in annotation.get("objects", []):
        if obj["label"] == CLASS_NAME:
            polygon = [(x, y) for x, y in obj["polygon"]]
            draw.polygon(polygon, outline=1, fill=1)  # Clase = 1 en la máscara

    # Guardar la máscara como PNG
    mask_output_path = os.path.join(class_output_path, f"{base_name}_{CLASS_NAME}_mask.png")
    mask.save(mask_output_path)

    print(f"Máscara generada: {mask_output_path}")

print(f"Generación de máscaras para la clase '{CLASS_NAME}' completada.")