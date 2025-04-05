import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm  # Para usar EfficientNet

# ===========================
# 1️⃣ Configuración
# ===========================
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2️⃣ Transformaciones para imágenes
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===========================
# 3️⃣ Dataset personalizado con escala
# ===========================
class VolumeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        volume = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)  # Volumen real
        escala = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)  # Escala

        if self.transform:
            image = self.transform(image)

        return image, volume, escala

# ===========================
# 4️⃣ Cargar los datos
# ===========================
train_dataset = VolumeDataset("dataset/volumenes.csv", "dataset/train", transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===========================
# 5️⃣ Definir el modelo EfficientNet
# ===========================
class VolumeEstimator(nn.Module):
    def __init__(self):
        super(VolumeEstimator, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features + 1, 1)  # Agregamos la escala como entrada adicional

    def forward(self, x, escala):
        x = self.model.forward_features(x)  # Extrae features de la imagen
        escala = escala.view(-1, 1)  # Ajusta la escala para concatenarla
        x = torch.cat((x.mean([2, 3]), escala), dim=1)  # Concatenar features con escala
        return self.model.classifier(x)

# ===========================
# 6️⃣ Inicializar el modelo, pérdida y optimizador
# ===========================
model = VolumeEstimator().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===========================
# 7️⃣ Entrenamiento
# ===========================
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for images, volumes, escalas in train_loader:
        images, volumes, escalas = images.to(DEVICE), volumes.to(DEVICE).unsqueeze(1), escalas.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, escalas)
        loss = criterion(outputs, volumes)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Época [{epoch+1}/{EPOCHS}], Pérdida: {epoch_loss/len(train_loader):.4f}")

# ===========================
# 8️⃣ Guardar el modelo
# ===========================
torch.save(model.state_dict(), "models/modelo_volumen.pth")
print("Modelo guardado exitosamente.")
