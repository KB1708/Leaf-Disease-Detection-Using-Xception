import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
import os

# --- Build reliable paths to model files ---
# This ensures the paths work correctly in Vercel's environment.
# os.path.dirname(__file__) gets the directory where this script is located.
base_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.join(base_dir, 'models/leaf_disease_model.keras')
CLASS_NAMES_PATH = os.path.join(base_dir, 'models/class_names.json')

# --- Load Model and Class Names ---
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

# --- Preprocessing Function ---
def preprocess_image(image_bytes: bytes, target_size: tuple):
    """
    Preprocesses the uploaded image bytes to the model's expected format.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Prediction Function ---
def get_prediction(image_bytes: bytes):
    """
    Takes image bytes, preprocesses the image, and returns the prediction.
    """
    # Preprocess the image to the size our model expects (224x224)
    processed_image = preprocess_image(image_bytes, target_size=(224, 224))
    
    # Get model predictions
    predictions = model.predict(processed_image)
    
    # Find the class with the highest probability
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[str(predicted_index)] # Ensure index is a string for JSON lookup
    confidence = float(np.max(predictions[0]))
    
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    }