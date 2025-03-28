import json
import pandas as pd
import os
import re

def json_to_csv(json_path, csv_path):
    """Convierte un archivo JSON a CSV, asegurando que la carpeta de destino exista y limpiando los datos."""
    
    # Crear la carpeta si no existe
    carpeta_destino = os.path.dirname(csv_path)
    if carpeta_destino and not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    # Leer JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Renombrar 'zoom' a 'zoom_mm', 'volume' a 'volume_ml' y limpiar valores numéricos
    for entry in data:
        entry["zoom_mm"] = float(re.sub(r"[^\d.]", "", str(entry.pop("zoom"))))  # Extraer solo el número
        entry["volume_ml"] = float(re.sub(r"[^\d.]", "", str(entry.pop("volume"))))  # Extraer solo el número

    # Guardar en CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"✅ Archivo CSV guardado en: {csv_path}")

# Ejecutar conversión
json_to_csv("utils/datos.json", "dataset/volumenes.csv")
