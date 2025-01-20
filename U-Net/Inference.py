import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

# Definición del modelo UNet
class UNet(torch.nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        self.final = torch.nn.Conv2d(64, n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(torch.nn.functional.max_pool2d(enc1, 2))
        enc3 = self.enc3(torch.nn.functional.max_pool2d(enc2, 2))
        enc4 = self.enc4(torch.nn.functional.max_pool2d(enc3, 2))

        bottleneck = self.bottleneck(torch.nn.functional.max_pool2d(enc4, 2))

        dec4 = self.dec4(torch.cat([torch.nn.functional.interpolate(bottleneck, scale_factor=2), enc4], dim=1))
        dec3 = self.dec3(torch.cat([torch.nn.functional.interpolate(dec4, scale_factor=2), enc3], dim=1))
        dec2 = self.dec2(torch.cat([torch.nn.functional.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.dec1(torch.cat([torch.nn.functional.interpolate(dec2, scale_factor=2), enc1], dim=1))

        return self.final(dec1)

# Guardar resultados con transparencia
# Guardar resultados con transparencia
def save_results(image_path, preds, class_colors, output_dir):
    image = Image.open(image_path).convert("RGB")
    preds_colored = np.zeros((*preds.shape, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        preds_colored[preds == class_id] = color

    overlay = Image.fromarray(preds_colored).convert("RGBA")
    base_image = image.convert("RGBA")
    blended = Image.blend(base_image, overlay, alpha=0.5)

    # Redimensionar a 256x256 usando LANCZOS
    blended = blended.resize((256, 256), Image.Resampling.LANCZOS)

    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Construir el nombre del archivo de salida
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_Unet.png")

    # Guardar la imagen sin bordes blancos
    blended.save(output_path, "PNG")


# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model_path = "unet_model.pth"
output_dir = r"..."

model = UNet(n_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Ruta de la carpeta de imágenes
image_dir = r"..."
image_paths = glob.glob(os.path.join(image_dir, "*.png"))

class_colors = {
    0: (0, 0, 0),       # Fondo
    1: (255, 0, 0),     # Clase 1
    2: (0, 255, 0),     # Clase 2
    3: (0, 0, 255)      # Clase 3
}

# Inferencia sobre imágenes en la carpeta
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    save_results(image_path, preds, class_colors, output_dir)

