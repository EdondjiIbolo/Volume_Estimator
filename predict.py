import torch
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from PIL import Image
import os
import csv

# ===========================
# 1️⃣ Definir el modelo correctamente
# ===========================
class VolumeEstimator(nn.Module):
    def __init__(self):
        super(VolumeEstimator, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=False)  # Evita descargar pesos nuevos
        in_features = self.model.num_features  # Corregido a num_features (1280)

        self.pool = nn.AdaptiveAvgPool2d(1)  # Consistente con el entrenamiento
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
        x = self.pool(x)  # Pooling adaptativo
        x = self.flatten(x)  # Convertir a vector de características
        escala = escala.view(-1, 1)  # Asegurar dimensiones correctas
        x = torch.cat((x, escala), dim=1)
        return self.fc(x)

# ===========================
# 2️⃣ Cargar modelo y pesos correctamente
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VolumeEstimator().to(DEVICE)

# Cargar pesos correctamente
try:
    checkpoint = torch.load("models/modelo_volumen_v2.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

model.eval()

# ===========================
# 3️⃣ Transformaciones de imagen
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
        print(f"❌ Error al procesar {image_path}: {e}")
        return None  # Evitar fallos en caso de error

# ===========================
# 4️⃣ Normalización de escala (Importante)
# ===========================
ESCALA_MAX = 500.0  # Ajustar según los datos de entrenamiento
def normalizar_escala(escala):
    return torch.tensor([escala / ESCALA_MAX], dtype=torch.float32, device=DEVICE)

# ===========================
# 5️⃣ Probar imágenes y guardar resultados en CSV
# ===========================
test_dir = "dataset/test"
output_csv = "dataset/estimated_vol.csv"

# Filtrar solo archivos de imagen
imagenes_prueba = [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

# Crear archivo CSV con encabezados
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Imagen", "Volumen Estimado (ml)"])

    for imagen in imagenes_prueba:
        ruta_imagen = os.path.join(test_dir, imagen)

        # Definir escala correcta y normalizar
        escala = normalizar_escala(250.0)  # Ajusta el valor según corresponda

        # Transformar imagen
        imagen_tensor = transform_image(ruta_imagen)
        if imagen_tensor is None:
            continue  # Saltar si hay error en la imagen

        # Realizar la predicción 
        with torch.no_grad():
            volumen_predicho = model(imagen_tensor, escala).item()

        # Escribir el resultado en el CSV
        writer.writerow([imagen, f"{volumen_predicho:.1f}"])

        print(f"🖼️ Imagen: {imagen} → 📏 Volumen estimado: {volumen_predicho:.1f} ml")

print(f"✅ Resultados guardados en {output_csv}")
