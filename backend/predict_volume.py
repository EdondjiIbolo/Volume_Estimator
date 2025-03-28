import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Cargar el modelo entrenado
model = torch.load("models/modelo_volumen.pth", map_location=torch.device("cpu"))
model.eval()

# Transformaciones para la imagen (deben coincidir con las usadas en el entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_volume(image, zoom_mm):
    image = transform(image).unsqueeze(0)  # Añadir dimensión de batch

    with torch.no_grad():
        output = model(image)

    predicted_volume = output.item()  # Convertir a valor escalar

    # Ajustar el volumen en base al zoom (ejemplo: simple proporcionalidad)
    adjusted_volume = predicted_volume * (50 / zoom_mm)  # Ajusta según la escala

    return round(adjusted_volume, 2)
