import tensorflow as tf
from PIL import Image
import numpy as np
import sys

IMG_SIZE = 160
MODEL_PATH = "../models/cats_vs_dogs_mobilenetv2.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(img_path):
    arr = preprocess_image(img_path)
    pred = model.predict(arr, verbose=0)
    return "Dog 🐶" if pred[0][0] > 0.5 else "Cat 🐱"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_local.py path_to_image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"Prediction for {img_path}: {predict(img_path)}")
