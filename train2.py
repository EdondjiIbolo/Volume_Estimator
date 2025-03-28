import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import re
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm

# ===========================
# 1️⃣ Calcular escala automáticamente
# ===========================
def calcular_escala(imagen_path, ancho_real_cm=10.0):
    try:
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            print(f"⚠️ No se pudo cargar la imagen: {imagen_path}")
            return 1.0
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        bordes = cv2.Canny(gris, 50, 150)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            print(f"⚠️ No se encontraron contornos en {imagen_path}")
            return 1.0
        
        contorno_mayor = max(contornos, key=cv2.contourArea)
        _, _, w, _ = cv2.boundingRect(contorno_mayor)
        return w / ancho_real_cm if w > 0 else 1.0  # Evita divisiones por cero
    except Exception as e:
        print(f"⚠️ Error al calcular la escala: {e}")
        return 1.0

# ===========================
# 2️⃣ Configuración
# ===========================
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0001  # Reducido para evitar NaN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 3️⃣ Transformaciones para imágenes
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===========================
# 4️⃣ Dataset personalizado
# ===========================
class VolumeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Asegurar que las columnas son numéricas
        self.data["volume_ml"] = pd.to_numeric(self.data["volume_ml"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors='coerce').fillna(1.0)
        self.data["zoom_mm"] = pd.to_numeric(self.data["zoom_mm"], errors='coerce').fillna(1.0)
        
        # Normalización
        min_val, max_val = self.data["volume_ml"].min(), self.data["volume_ml"].max()
        if max_val - min_val > 0:
            self.data["volume_ml"] = (self.data["volume_ml"] - min_val) / (max_val - min_val + 1e-8)
        else:
            print("⚠️ No se pudo normalizar los volúmenes, todos los valores son iguales.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.normpath(os.path.join(self.img_dir, os.path.basename(self.data.iloc[idx, 0])))

        if not os.path.exists(img_name):
            raise FileNotFoundError(f"⚠️ Imagen no encontrada: {img_name}")
        
        try:
            image = Image.open(img_name).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"⚠️ Error al abrir la imagen {img_name}: {e}")

        volume = self.data.iloc[idx]["volume_ml"]
        escala = calcular_escala(img_name)
        escala = max(escala, 1e-8)  # Evitar valores cero

        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        escala = torch.tensor(escala, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, volume, escala

# ===========================
# 5️⃣ Cargar los datos
# ===========================
train_dataset = VolumeDataset("dataset/volumenes.csv", "dataset/train", transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===========================
# 6️⃣ Modelo EfficientNet
# ===========================
class VolumeEstimator(nn.Module):
    def __init__(self):
        super(VolumeEstimator, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features + 1, 1)

    def forward(self, x, escala):
        x = self.model.forward_features(x)
        x = x.mean([2, 3])
        escala = escala.view(-1, 1)
        x = torch.cat((x, escala), dim=1)
        return self.model.classifier(x)

# ===========================
# 7️⃣ Inicializar modelo, pérdida y optimizador
# ===========================
model = VolumeEstimator().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ===========================
# 8️⃣ Entrenamiento
# ===========================
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    for images, volumes, escalas in train_loader:
        images, volumes, escalas = images.to(DEVICE), volumes.to(DEVICE), escalas.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, escalas)
        loss = criterion(outputs, volumes)
        
        if torch.isnan(loss):
            print("⚠️ Pérdida NaN detectada, interrumpiendo entrenamiento.")
            break

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Época [{epoch+1}/{EPOCHS}], Pérdida: {epoch_loss/len(train_loader):.4f}")

# ===========================
# 9️⃣ Guardar modelo
# ===========================
torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, "models/modelo_volumen.pth")
print("✅ Modelo guardado exitosamente.")
