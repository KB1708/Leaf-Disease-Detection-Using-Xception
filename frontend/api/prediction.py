import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
import os
import requests

# --- IMPORTANT: Paste the GitHub Release URL you copied here ---
MODEL_DOWNLOAD_URL = "https://github.com/KB1708/Leaf-Disease-Detection-Using-Xception/releases/download/v1.0.0/leaf_disease_model.keras"

# Vercel's serverless functions can only write to the /tmp directory
MODEL_DESTINATION_PATH = '/tmp/leaf_disease_model.keras'
CLASS_NAMES_PATH = '/tmp/class_names.json' # We will download this too

def download_file(url, destination):
    """Downloads a file from a URL to a local destination if it doesn't exist."""
    if not os.path.exists(destination):
        print(f"Downloading {os.path.basename(destination)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"{os.path.basename(destination)} already exists. Skipping download.")

# --- Download and Load Model and Class Names ---
# The download will only happen during a "cold start" of the function.
# Subsequent requests will be much faster as the files will already be in /tmp.
download_file(MODEL_DOWNLOAD_URL, MODEL_DESTINATION_PATH)

# We also need the class names JSON file. Create a release for it too.
# For now, let's create it manually. In a real project, you'd download this too.
class_names_data = {"0": "Apple___Apple_scab", "1": "Apple___Black_rot", "2": "Apple___Cedar_apple_rust", "3": "Apple___healthy", "4": "Blueberry___healthy", "5": "Cherry_(including_sour)___Powdery_mildew", "6": "Cherry_(including_sour)___healthy", "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "8": "Corn_(maize)___Common_rust_", "9": "Corn_(maize)___Northern_Leaf_Blight", "10": "Corn_(maize)___healthy", "11": "Grape___Black_rot", "12": "Grape___Esca_(Black_Measles)", "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "14": "Grape___healthy", "15": "Orange___Haunglongbing_(Citrus_greening)", "16": "Peach___Bacterial_spot", "17": "Peach___healthy", "18": "Pepper,_bell___Bacterial_spot", "19": "Pepper,_bell___healthy", "20": "Potato___Early_blight", "21": "Potato___Late_blight", "22": "Potato___healthy", "23": "Raspberry___healthy", "24": "Soybean___healthy", "25": "Squash___Powdery_mildew", "26": "Strawberry___Leaf_scorch", "27": "Strawberry___healthy", "28": "Tomato___Bacterial_spot", "29": "Tomato___Early_blight", "30": "Tomato___Late_blight", "31": "Tomato___Leaf_Mold", "32": "Tomato___Septoria_leaf_spot", "33": "Tomato___Spider_mites Two-spotted_spider_mite", "34": "Tomato___Target_Spot", "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "36": "Tomato___Tomato_mosaic_virus", "37": "Tomato___healthy"}
with open(CLASS_NAMES_PATH, 'w') as f:
    json.dump(class_names_data, f)

model = tf.keras.models.load_model(MODEL_DESTINATION_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

# --- Preprocessing and Prediction functions (these remain the same) ---
def preprocess_image(image_bytes: bytes, target_size: tuple):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_prediction(image_bytes: bytes):
    processed_image = preprocess_image(image_bytes, target_size=(224, 224))
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[str(predicted_index)]
    confidence = float(np.max(predictions[0]))
    return {"predicted_class": predicted_class, "confidence": round(confidence, 4)}