# 📌 Volume Estimation with Computer Vision

This project uses **EfficientNet** to estimate the volume of materials from images, taking into account the camera’s scale and zoom.

---

## 📂 Project Structure

```
📦 volume_project
├── 📂 dataset
│   ├── 📂 train         # Training images
│   ├── 📂 test          # Testing images
│   ├── 📜 volumes.csv   # Training data with scale and zoom
├── 📂 models
│   ├── 📜 volume_model_v2.pth  # Trained model
├── 📂 utils
│   ├── 📜 convert_to_csv.py    # Convert JSON to CSV
│   ├── 📜 data.json            # Image info (name, volume, scale)
├── 📜 test.py       # Training script
├── 📜 predict.py       # Volume prediction script
├── 📜 requirements.txt  # Dependencies
├── 📜 README.md         # Instructions
```

---

## 🚀 Environment Setup

### 1️⃣ Set up environment on macOS (Apple Silicon M2)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and pip
brew install python

# Create a virtual environment
python3 -m venv env
source env/bin/activate  # Activate the environment
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Generate the Value Table

📌 **CSV structure (`volumes.csv`)**:  
Run the following command to generate the CSV file:

```bash
python utils/convert_to_csv.py
```

This will generate a `volumes.csv` file inside the `dataset` folder with data like this:

| image    | volume_ml | zoom_mm |
| -------- | --------- | ------- |
| img1.jpg | 50.2      | 4.5     |
| img2.jpg | 30.7      | 3.2     |

---

## 🏋️‍♂️ Model Training

Run the following command to train the model:

```bash
python scripts/test.py
```

✅ This will generate the file `volume_model_v2.pth` inside the `models/` folder.

---

## 🔍 Volume Prediction

To predict the volume of new images:

```bash
python scripts/pred.py
```

The result will show the estimated volume of each image in the `test/` folder.

---

## 🔄 JSON to CSV Conversion

If you have data in JSON format, convert it to CSV with:

```bash
python scripts/json_to_csv.py
```

This will generate a CSV file compatible with the model training.

---

## 📌 Final Notes

- **Make sure to take photos from the same distance and angle.**
- **Zoom is measured in millimeters**, not in zoom factors (e.g., x1.0, x2.0).
- **Good lighting and sharpness are important for better results.**

---

🔹 **Ensure VS Code uses the correct Python environment**:  
Press `Ctrl + Shift + P` in VS Code,  
Search for and select **"Python: Select Interpreter"**,  
Choose the one that looks like `env\Scripts\python.exe` (on Windows) or `env/bin/python` (on Mac/Linux).

---

✅ For training → `test.py`  
✅ For prediction → `predict.py`
