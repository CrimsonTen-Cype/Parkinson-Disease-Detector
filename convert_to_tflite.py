import tensorflow as tf
import os

h5_model_path = 'parkinsons_model.h5'
tflite_model_path = 'model.tflite'

if not os.path.exists(h5_model_path):
    print(f"Error: {h5_model_path} not found.")
    exit(1)

print(f"Loading Keras model from {h5_model_path}...")
model = tf.keras.models.load_model(h5_model_path)

print("Converting to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print(f"Saving TFLite model to {tflite_model_path}...")
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Conversion successful!")
