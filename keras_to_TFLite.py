import tensorflow as tf
from tensorflow import keras
import os

# Load the Keras model
model = keras.models.load_model(os.path.abspath(os.getcwd())+"/model_result/V9_40ep_custom_aug.h5")

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open(os.path.abspath(os.getcwd())+"/model_result/EfficientNetB0_V9.tflite", 'wb') as f:
    f.write(tflite_model)
