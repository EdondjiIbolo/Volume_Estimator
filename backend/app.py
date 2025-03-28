from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
import io
# from predict_volume import predict_volume

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
   
    if "image" not in request.files or "zoom" not in request.form:
        return jsonify({"error": "Missing image or zoom level"}), 400
    print('received')
    return jsonify({"success": "Image Received","data":3}) , 200
    image_file = request.files["image"]
    zoom_level = float(request.form["zoom"])

    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    # Llamar a la función de predicción
    # estimated_volume = predict_volume(image, zoom_level)

    # return jsonify({"volume": estimated_volume})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
