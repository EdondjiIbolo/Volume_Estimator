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
EPOCHS = 40  # Aumentamos las épocas para mejor convergencia
LR = 0.0005  # Reducimos el learning rate para mayor estabilidad
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2️⃣ Transformaciones mejoradas
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),  # Rotaciones leves para mejorar generalización
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Variaciones de color
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===========================
# 3️⃣ Dataset personalizado con escala normalizada
# ===========================
class VolumeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.escala_max = self.data.iloc[:, 1].max()  # Obtener el valor máximo de la escala

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        volume = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)  # Volumen real
        escala = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)  # Escala

        # Normalizar la escala entre 0 y 1
        escala /= self.escala_max  

        if self.transform:
            image = self.transform(image)

        return image, volume, escala

# ===========================
# 4️⃣ Cargar los datos
# ===========================
train_dataset = VolumeDataset("dataset/volumenes.csv", "dataset/train", transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===========================
# 5️⃣ Definir el modelo EfficientNet mejorado
# ===========================
class VolumeEstimator(nn.Module):
    def __init__(self):
        super(VolumeEstimator, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True)
        in_features = self.model.num_features  # Obtiene el número correcto de características (1280 en EfficientNet-B0)

        self.pool = nn.AdaptiveAvgPool2d(1)  
        self.flatten = nn.Flatten()  

        # Nueva capa totalmente conectada
        self.fc = nn.Sequential(
            nn.Linear(in_features + 1, 512),  # Ahora usa 1280 + 1 en lugar de 512 + 1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, escala):
        x = self.model.forward_features(x)  
        x = self.pool(x)  
        x = self.flatten(x)  

        escala = escala.view(-1, 1)  
        x = torch.cat((x, escala), dim=1)  # Ahora la dimensión debería coincidir (batch_size x 1281)

        return self.fc(x)

# ===========================
# 6️⃣ Inicializar el modelo, pérdida y optimizador mejorado
# ===========================
model = VolumeEstimator().to(DEVICE)
criterion = nn.HuberLoss(delta=1.0)  # Mejor manejo de valores extremos
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)  # AdamW para mayor estabilidad

# ===========================
# 7️⃣ Entrenamiento mejorado
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
torch.save({"model_state_dict": model.state_dict()}, "models/modelo_volumen_v2.pth")
print("Modelo mejorado guardado exitosamente.")
