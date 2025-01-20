import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
import torch
from transformers import SamProcessor, SamModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuración del modelo
model_path = "trained_model.pth"  # Ruta donde guardaste el modelo entrenado
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.to(device)
model.eval()

# Configuración de dispositivo

# Función para cargar imágenes desde una carpeta
def load_images_from_folder(folder_path, size=(256, 256)):
    image_filenames = sorted(os.listdir(folder_path))
    images = []
    filenames = []
    
    for filename in image_filenames:
        image_path = os.path.join(folder_path, filename)
        try:
            image = Image.open(image_path).resize(size)
            images.append(np.array(image))
            filenames.append(filename)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    
    return images, filenames




# Carpeta de imágenes para inferencia
input_folder = r"..."  # Cambiar a la carpeta de imágenes
output_folder = r"..." # Carpeta para guardar los overlays

# Cargar imágenes
images, filenames = load_images_from_folder(input_folder)



