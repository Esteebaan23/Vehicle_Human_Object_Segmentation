import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Cargar el modelo
model = YOLO("best_todo.pt")

# Ruta de la carpeta de imágenes
image_folder = r"..."
output_folder = r"..."
os.makedirs(output_folder, exist_ok=True)

image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Colores para cada clase
class_colors = {
    0: [0, 0, 255],  # Rojo para la clase 0
    1: [0, 255, 0],  # Verde para la clase 1
    2: [255, 0, 0]   # Azul para la clase 2
}

# Procesar las imágenes
for image_path in image_paths:
    # Leer y redimensionar la imagen
    image = cv2.imread(image_path)
    original_image = image.copy()
    resized_image = cv2.resize(image, (640, 320))

    # Inferencia
    results = model(resized_image)

    # Comprobar si se detectaron objetos
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # Obtener las máscaras
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []  # Obtener las clases
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []  # Obtener las cajas

        segmentation_overlay = np.zeros_like(resized_image, dtype=np.uint8)

        # Iterar sobre los objetos detectados
        for mask, cls, box in zip(masks, classes, boxes):
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min

            # Verificar si el tamaño del objeto es menor a 24x24 píxeles
            if width < 400 and height < 100:
                mask = mask.astype(bool)
                color = class_colors.get(int(cls), [255, 255, 255])  # Color predeterminado blanco si no se encuentra la clase
                segmentation_overlay[mask] = color  # Aplicar la máscara sobre el overlay

        # Redimensionar la máscara al tamaño original de la imagen
        segmentation_overlay = cv2.resize(segmentation_overlay, (original_image.shape[1], original_image.shape[0]))

        # Combinar la imagen original con la máscara
        combined_image = cv2.addWeighted(original_image, 0.7, segmentation_overlay, 0.3, 0)

        # Redimensionar a 256x256 para guardar
        output_image = cv2.resize(combined_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Crear el nombre del archivo de salida
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_folder, f"{name}_YOLO.png")

        # Guardar la imagen combinada
        cv2.imwrite(output_path, output_image)

        # Mostrar el resultado (opcional)
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


