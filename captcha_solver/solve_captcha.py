import os
import cv2
import numpy as np
from PIL import Image
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "model/captcha_solver.h5"
CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMG_WIDTH, IMG_HEIGHT = 100, 50
MAX_CAPTCHA_LEN = 5

model = load_model(MODEL_PATH)

def preprocess_image(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    img = np.array(pil_img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img

def decode_prediction(pred):
    pred = pred.reshape((MAX_CAPTCHA_LEN, len(CHAR_SET)))
    text = "".join([CHAR_SET[np.argmax(c)] for c in pred])
    return text.strip()

def solve_captcha(img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)

    pred = pred.reshape((MAX_CAPTCHA_LEN, len(CHAR_SET)))  

    captcha_text = decode_prediction(pred)
    return captcha_text

if __name__ == "__main__":
    img_path = "test_images/cap.jpeg"

    print("Predicted CAPTCHA:", solve_captcha(img_path))
