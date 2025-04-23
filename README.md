# 🧠 Callus Volume Estimator

A machine learning-based application to estimate the **volume of callus-like masses inside a flask** using images. The model is powered by **EfficientNet** and exposed through a **Flask** API, with a user-friendly **Vite/React** frontend.

---

## 📸 Project Overview

This project combines deep learning with computer vision to estimate the volume of cell aggregations or callus structures, using image input taken from different angles. Inspired by biological growth experiments, it aims to provide fast and approximate volume predictions from photographic data.

---

## 🔧 Tech Stack

- **Frontend**: Vite + React
- **Backend**: Python + Flask
- **Model**: EfficientNet (trained on synthetic data resembling callus in flasks)
- **Image Input**: PNG/JPG files captured with consistent background and scale (e.g., measurement cup in frame)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- pip / venv
- (optional) CUDA-compatible GPU for model inference acceleration

### Backend Setup (Flask API)

```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py



Aquí tienes la traducción al inglés de tu README:

---

# 📌 Volume Estimation with Computer Vision

This project uses **EfficientNet** to estimate the volume of materials from images, taking into account the camera's scale and zoom.

---

## 📂 Project Structure

```

📦 volume_project
├── 📂 dataset
│ ├── 📂 train # Training images
│ ├── 📂 test # Test images
│ ├── 📜 volumenes.csv # Training data with scale and zoom info
├── 📂 models
│ ├── 📜 modelo_volumen_v2.pth # Trained model
├── 📂 utils
│ ├── 📜 convert_to_csv.py # Convert JSON to CSV
│ ├── 📜 datos.json # Image info (filename, volume, scale)
├── 📂 scripts
│ ├── 📜 test.py # Training script
│ ├── 📜 predict.py # Volume prediction
├── 📜 requirements.txt # Dependencies
├── 📜 README.md # Instructions

````

---

## 🚀 Environment Setup

### 1️⃣ Setup on macOS (Silicon M2)

```bash
# Install Homebrew if you don’t have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and pip
brew install python

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Activate the environment
````

### 2️⃣ Install dependencies

```bash
pip install -r prerequisites.txt
```

---

## 📊 Generate the CSV Table

📌 **CSV Structure (`volumenes.csv`)**:  
Run the following command to generate the training data:

```bash
python utils/convert_to_csv.py
```

This will generate `volumenes.csv` in the `dataset/` folder. The table will look like this:

| image    | volumen_ml | zoom_mm |
| -------- | ---------- | ------- |
| img1.jpg | 50.2       | 4.5     |
| img2.jpg | 30.7       | 3.2     |

---

## 🏋️‍♂️ Model Training

Run the following command to train the model:

```bash
python test.py
```

✅ This will generate the `modelo_volumen_v2.pth` file in the `models/` folder.

---

## 🔍 Volume Prediction

To predict the volume of new images:

```bash
python predict.py
```

The output will show the estimated volume of each image in the `test/` folder.

---

## 🔄 Convert JSON to CSV

If you have data in JSON format, convert it to CSV with:

```bash
python scripts/json_to_csv.py
```

This will generate a CSV file compatible with the model training.

---

## 📌 Final Notes

- **Make sure photos are taken from the same distance and angle.**
- **Zoom is measured in mm**, not in factors (e.g., x1.0, x2.0).
- **Good lighting and image clarity improve accuracy.**

🔹 2️⃣ Make sure VS Code uses the correct environment:  
Press `Ctrl + Shift + P` in VS Code.  
Type and select **"Python: Select Interpreter"**.  
Choose the one that looks like `env/Scripts/python.exe` (Windows) or `env/bin/python` (Mac/Linux).

To activate backend:

```bash
source env/Scripts/activate
python ./backend/app.py
```

To train:

```bash
python test.py
```

To predict:

```bash
python predict.py
```

---


git add . 
git commit -m "new changes"
git push origin main