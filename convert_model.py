import tensorflow as tf
import os

# --- IMPORTANT: Make sure this path points to your Keras model file ---
# Assuming your model is in the 'models' folder at the root for conversion
keras_model_path = os.path.join('models', 'leaf_disease_model.keras')

print(f"Loading Keras model from: {keras_model_path}")
model = tf.keras.models.load_model(keras_model_path)

# Create a TensorFlow Lite converter object from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply default optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Perform the conversion
tflite_model = converter.convert()

# Save the new, lightweight .tflite model
tflite_model_path = 'leaf_disease_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"SUCCESS: Model converted and saved to {tflite_model_path}")