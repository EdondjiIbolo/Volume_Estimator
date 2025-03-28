import torch
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from PIL import Image
import os

# ===========================
# 1️⃣ Cargar el modelo entrenado
# ===========================
class VolumeEstimator(nn.Module):
    def __init__(self):
        super(VolumeEstimator, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=False)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features + 1, 1)

    def forward(self, x, escala):
        x = self.model.forward_features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        escala = escala.view(-1, 1)
        x = torch.cat((x, escala), dim=1)
        return self.model.classifier(x)

# Configurar dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo y pesos entrenados
model = VolumeEstimator().to(DEVICE)
checkpoint = torch.load("models/modelo_volumen.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ===========================
# 2️⃣ Transformaciones para imágenes
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def transform_image(image_path):
    """Carga y transforma la imagen para el modelo."""
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(DEVICE)  # Agregar dimensión de batch
    except Exception as e:
        print(f"Error al procesar {image_path}: {e}")
        return None  # Retornar None para evitar fallos

# ===========================
# 3️⃣ Probar imágenes
# ===========================
test_dir = "dataset/test"

# Filtrar solo archivos de imagen
imagenes_prueba = [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

for imagen in imagenes_prueba:
    ruta_imagen = os.path.join(test_dir, imagen)

    # Definir escala correcta (ajústala según valores reales si es necesario)
    escala = torch.tensor([250.0], dtype=torch.float32, device=DEVICE)

    # Transformar imagen
    imagen_tensor = transform_image(ruta_imagen)
    if imagen_tensor is None:
        continue  # Saltar a la siguiente imagen si hubo un error

    # Realizar la predicción 
    with torch.no_grad():
        volumen_predicho = model(imagen_tensor, escala).item()  # Extraer valor numérico

    print(f"Imagen: {imagen} → Volumen estimado: {volumen_predicho:.2f} ml")