import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# --- Build the path to the TFLite model file ---
base_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.join(base_dir, 'leaf_disease_model.tflite')

# --- Class names ---
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Tomato_Bacterial_spot",
    "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus", "Tomato_healthy"
]

# --- Load the TFLite model's interpreter ---
# The full tensorflow-cpu package provides the tf.lite module
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details from the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Preprocessing Function ---
def preprocess_image(image_bytes: bytes, target_size: tuple):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Prediction Function ---
def get_prediction(image_bytes: bytes):
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    processed_image = preprocess_image(image_bytes, target_size=(height, width))

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]))

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    }