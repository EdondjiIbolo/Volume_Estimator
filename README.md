# 📌 Estimación de Volumen con Visión Artificial

Este proyecto utiliza **EfficientNet** para estimar el volumen de materiales a partir de imágenes, teniendo en cuenta la escala y el zoom de la cámara.

---

## 📂 Estructura del Proyecto

```
📦 proyecto_volumen
├── 📂 dataset
│   ├── 📂 train  # Imágenes de entrenamiento
│   ├── 📂 test   # Imágenes de prueba
│   ├── 📜 volumenes.csv  # Datos de entrenamiento con escala y zoom
├── 📂 models
│   ├── 📜 modelo_volumen.pth  # Modelo entrenado
├── 📂 scripts
│   ├── 📜 train.py  # Script de entrenamiento
│   ├── 📜 predict.py  # Predicción de volumen
│   ├── 📜 json_to_csv.py  # Conversión de JSON a CSV
├── 📜 requirements.txt  # Dependencias
├── 📜 README.md  # Instrucciones
```

---

## 🚀 Instalación del Entorno

### 1️⃣ Configurar entorno en macOS (Silicon M2)

```bash
# Instalar Homebrew si no lo tienes
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar Python y pip
brew install python

# Crear entorno virtual
python3 -m venv env
source env/bin/activate  # Activar entorno
```

### 2️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Entrenamiento del Modelo

Ejecuta el siguiente comando para entrenar el modelo:

```bash
python scripts/train.py
```

✅ Esto generará el archivo `modelo_volumen.pth` en la carpeta `models/`.

📌 **Estructura del CSV (`volumenes.csv`)**:

| imagen   | volumen_real_ml | escala | zoom_mm |
| -------- | --------------- | ------ | ------- |
| img1.jpg | 50.2            | 1.2    | 4.5     |
| img2.jpg | 30.7            | 1.0    | 3.2     |

---

## 🔍 Predicción de Volumen

Para predecir el volumen de nuevas imágenes:

```bash
python scripts/predict.py
```

El resultado mostrará el volumen estimado de cada imagen en la carpeta `test/`.

---

## 🔄 Conversión de JSON a CSV

Si tienes datos en formato JSON, conviértelos a CSV con:

```bash
python scripts/json_to_csv.py
```

Esto generará un archivo CSV compatible con el entrenamiento del modelo.

---

## 📌 Notas Finales

- **Asegúrate de tomar fotos con la misma distancia y ángulo.**
- **El zoom se mide en mm**, no en factores (ej. x1.0, x2.0).
- **Cuida la iluminación y nitidez para mejores resultados.**
