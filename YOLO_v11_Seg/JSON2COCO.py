import json
import glob
from pathlib import Path

# Define rutas
input_path = "...*.json"
output_path = "..."

# Asegúrate de que la carpeta de salida exista
Path(output_path).mkdir(parents=True, exist_ok=True)

# Define categorías
categories = [
    {"id": 1, "name": "vehicles"},
    {"id": 2, "name": "human"},
    {"id": 3, "name": "Structures"}
]
category_to_id = {cat["name"]: cat["id"] for cat in categories}

# Procesa cada archivo en la carpeta de entrada
files = glob.glob(input_path)

for file in files:
    with open(file, "r") as f:
        data = json.load(f)
    
    # Inicializa las estructuras COCO para este archivo
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    # Información de la imagen
    file_name = Path(file).name
    image_id = 1  # ID único para la imagen (local a este archivo)
    annotation_id = 1  # ID único para cada anotación (local a este archivo)

    image_info = {
        "id": image_id,
        "file_name": file_name,
        "height": data.get("imageHeight", 1),
        "width": data.get("imageWidth", 1)
    }
    coco_data["images"].append(image_info)

    # Procesa cada objeto anotado
    for obj in data.get("objects", []):
        label = obj.get("label")
        if label not in category_to_id:
            print(f"Etiqueta desconocida '{label}' en archivo {file_name}")
            continue

        # Convierte el polígono en una lista plana
        polygon = obj.get("polygon", [])
        segmentation = [coord for point in polygon for coord in point]

        # Calcula el bounding box [x_min, y_min, width, height]
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        x_min, y_min = min(x_coords), min(y_coords)
        bbox_width = max(x_coords) - x_min
        bbox_height = max(y_coords) - y_min
        bbox = [x_min, y_min, bbox_width, bbox_height]

        # Añade la anotación
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_to_id[label],
            "segmentation": [segmentation],
            "area": bbox_width * bbox_height,
            "bbox": bbox,
            "iscrowd": 0
        }
        coco_data["annotations"].append(annotation)
        annotation_id += 1

    # Guarda los datos COCO en un archivo individual
    output_file = Path(output_path) / f"{Path(file).stem}_COCO.json"
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Archivo convertido: {output_file}")

print("Conversión completada para todos los archivos.")
