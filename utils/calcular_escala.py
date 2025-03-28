import cv2
import numpy as np

def calcular_escala(imagen_path, tamano_real_cm=1.0):
    """Calcula la escala en píxeles/cm a partir de un objeto de referencia."""
    imagen = cv2.imread(imagen_path)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Detectar bordes
    bordes = cv2.Canny(gris, 50, 150)

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None  # No se detectó un objeto

    # Seleccionar el objeto de referencia (ejemplo: el más grande)
    contorno_mayor = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)

    # Calcular escala
    escala = w / tamano_real_cm  # píxeles/cm
    return escala

# Ejemplo de uso
imagen_prueba = "dataset/train/img_001.jpg"
escala = calcular_escala(imagen_prueba)
print(f"Escala estimada: {escala:.2f} píxeles/cm")
