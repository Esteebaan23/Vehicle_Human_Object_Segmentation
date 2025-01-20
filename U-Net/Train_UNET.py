import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# Dataset personalizado para imágenes y máscaras
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.png', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset y DataLoader
image_dir = '...'
mask_dir = '...'
val_image_dir = '...'
val_mask_dir = '...'

dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Ajustado para evitar problemas de memoria
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Definición del modelo UNet
class UNet(nn.Module):
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

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        dec4 = self.dec4(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='nearest'), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, scale_factor=2, mode='nearest'), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='nearest'), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='nearest'), enc1], dim=1))

        return self.final(dec1)

# Configuración del modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = UNet(n_classes=4).to(device)

# Configuración de optimizador y función de pérdida
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Entrenamiento
num_epochs = 2
accumulation_steps = 4  # Para acumulación de gradientes
def train():
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

        # Validación
        eval_metrics = evaluate_model(model, val_dataloader, device)
        print(f"Validation Metrics: {eval_metrics}")

    print("Entrenamiento finalizado.")

train()

# Evaluación
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(masks.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }

# Evaluar y guardar modelo
eval_metrics = evaluate_model(model, val_dataloader, device)
print("Final Metrics:", eval_metrics)
model_path = "unet_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")





