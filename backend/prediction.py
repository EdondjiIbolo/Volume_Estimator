
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch.nn as nn
# Definir el modelo como lo hiciste antes
class VolumeEstimator(nn.Module):
    def __init__(self):
        super(VolumeEstimator, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=False)
        in_features = self.model.num_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(in_features + 1, 512),
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
        x = torch.cat((x, escala), dim=1)
        return self.fc(x)

# Cargar el modelo
def load_model():
    model = VolumeEstimator()
    checkpoint = torch.load("models/modelo_volumen_v2.pth", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# Cargar el modelo una vez
model = load_model()

# Función predict_volume ajustada para trabajar con Flask
def volume_prediction(image, zoom_mm):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)  # Añadir dimensión de batch
    escala = torch.tensor([zoom_mm / 500.0], dtype=torch.float32)  # Normalizar escala

    with torch.no_grad():
        output = model(image, escala)
    
    predicted_volume = output.item()  # Convertir a valor escalar
    return round(predicted_volume, 2)
