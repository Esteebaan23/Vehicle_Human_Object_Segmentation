import json
import glob
from pathlib import Path

# Dimensiones fijas de las imágenes
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 320

# Define las rutas
input_path = ".../*.json"
output_path = "..."

# Asegúrate de que la carpeta de salida exista
Path(output_path).mkdir(parents=True, exist_ok=True)

# Procesa cada archivo COCO en la carpeta de entrada
files = glob.glob(input_path)

if not files:
    print("No se encontraron archivos COCO en la carpeta de entrada.")
else:
    for file in files:
        with open(file, "r") as f:
            coco_data = json.load(f)

        # Verifica que las claves principales existan
        if "images" not in coco_data or "annotations" not in coco_data or "categories" not in coco_data:
            print(f"El archivo {file} no tiene el formato COCO esperado. Se omite.")
            continue

        # Crear un archivo YOLO por cada imagen
        for image_info in coco_data["images"]:
            image_id = image_info["id"]
            file_name = Path(image_info["file_name"]).stem
            yolo_file = Path(output_path) / f"{file_name}.txt"

            # Obtener anotaciones correspondientes a esta imagen
            annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

            with open(yolo_file, "w") as yolo_f:
                for ann in annotations:
                    category_id = ann["category_id"]
                    segmentations = ann.get("segmentation", [])

                    # El formato COCO almacena el segmento como una lista plana
                    if not segmentations or not isinstance(segmentations[0], list):
                        print(f"Segmentación inválida en el archivo {file_name}. Se omite esta anotación.")
                        continue

                    # Normalizar las coordenadas del polígono
                    normalized_segmentation = []
                    for point_idx in range(0, len(segmentations[0]), 2):
                        x = segmentations[0][point_idx] / IMAGE_WIDTH
                        y = segmentations[0][point_idx + 1] / IMAGE_HEIGHT
                        normalized_segmentation.extend([x, y])

                    # Escribir en formato YOLO Segmentación: <class> <x1> <y1> <x2> <y2> ... <xn> <yn>
                    segmentation_str = " ".join([f"{coord:.6f}" for coord in normalized_segmentation])
                    yolo_f.write(f"{category_id - 1} {segmentation_str}\n")

            print(f"Archivo YOLO Segmentación creado: {yolo_file}")

print("Conversión completada para todos los archivos.")

