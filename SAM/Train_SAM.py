import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor, SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Función para cargar imágenes y máscaras desde carpetas separadas y redimensionarlas
def load_images_and_masks(image_dir, mask_dir, size=(256, 256)):
    images = []
    masks = []

    image_filenames = sorted(os.listdir(image_dir))

    for image_filename in image_filenames:
        image_path = os.path.join(image_dir, image_filename)
        # Construir el nombre de la máscara a partir del nombre de la imagen
        mask_filename = os.path.splitext(image_filename)[0] + "_vehicles_mask.png"
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(f"Mask not found for image {image_filename}. Skipping...")
            continue

        image = Image.open(image_path).resize(size)
        mask = Image.open(mask_path).resize(size, Image.NEAREST)

        images.append(np.array(image))
        masks.append(np.array(mask))

    return images, masks

# Rutas a las carpetas de imágenes y máscaras
train_image_dir = "..."
train_mask_dir = "..."
val_image_dir = "..."
val_mask_dir = "..."

# Cargar imágenes y máscaras
train_images, train_masks = load_images_and_masks(train_image_dir, train_mask_dir)
val_images, val_masks = load_images_and_masks(val_image_dir, val_mask_dir)

# Crear dataset como lista de diccionarios
def create_dataset(images, masks):
    return [{'image': img, 'label': mask} for img, mask in zip(images, masks)]

train_dataset_list = create_dataset(train_images, train_masks)
val_dataset_list = create_dataset(val_images, val_masks)

# Mostrar una imagen y su máscara
def show_image_and_mask(example):
    fig, ax = plt.subplots()
    ax.imshow(example['image'])
    mask = example['label']
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.axis("off")
    plt.show()

# Mostrar ejemplo del dataset
example = train_dataset_list[0]
show_image_and_mask(example)

# Obtener bounding box de una máscara
def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:  # Verificar si la máscara está vacía
        return [0, 0, 0, 0]

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return [x_min, y_min, x_max, y_max]

# Dataset para SAM
class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        prompt = get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

# Procesador y datasets
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=train_dataset_list, processor=processor)
val_dataset = SAMDataset(dataset=val_dataset_list, processor=processor)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Modelo SAM
model = SamModel.from_pretrained("facebook/sam-vit-base")
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

# Configuración de entrenamiento
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Entrenamiento
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}, Mean Loss: {mean(epoch_losses):.4f}')

# Guardar el modelo
model_save_path = "trained_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Evaluación
model.eval()
accuracies, precisions, recalls, f1_scores = [], [], [], []
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        inputs = batch["pixel_values"].to(device)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        outputs = model(pixel_values=inputs, multimask_output=False)
        predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
        predicted_masks = (predicted_masks > 0.5).cpu().numpy().astype(np.uint8)
        ground_truth_masks = ground_truth_masks.cpu().numpy().astype(np.uint8)

        predicted_masks_flat = predicted_masks.flatten()
        ground_truth_masks_flat = ground_truth_masks.flatten()

        accuracies.append(accuracy_score(ground_truth_masks_flat, predicted_masks_flat))
        precisions.append(precision_score(ground_truth_masks_flat, predicted_masks_flat))
        recalls.append(recall_score(ground_truth_masks_flat, predicted_masks_flat))
        f1_scores.append(f1_score(ground_truth_masks_flat, predicted_masks_flat))

print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f}")



