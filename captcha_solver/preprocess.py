import os
import cv2
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.preprocessing.image import img_to_array

DATA_DIR = "data/"
IMG_WIDTH, IMG_HEIGHT = 100, 50
CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_CAPTCHA_LEN = 5

def load_data():
    images, labels = [], []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".jpeg"):
            img_path = os.path.join(DATA_DIR, filename)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img_to_array(img) / 255.0 
            images.append(img)

            try:
                label = filename.split("_")[1].split(".")[0]
                if len(label) < MAX_CAPTCHA_LEN:
                    label = label.ljust(MAX_CAPTCHA_LEN, " ")
                labels.append(label)
            except IndexError:
                print(f"Skipping file {filename} (Invalid filename format)")

    return np.array(images), np.array(labels)

def encode_labels(labels):
    encoded_labels = np.zeros((len(labels), MAX_CAPTCHA_LEN, len(CHAR_SET)))

    for i, label in enumerate(labels):
        for j, char in enumerate(label):
            if j < MAX_CAPTCHA_LEN and char in CHAR_SET:
                idx = CHAR_SET.index(char)
                encoded_labels[i, j, idx] = 1

    return encoded_labels

if __name__ == "__main__":
    images, labels = load_data()
    labels_encoded = encode_labels(labels)

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/images.npy", images)
    np.save("data/processed/labels.npy", labels_encoded)

    print(f"Saved {len(images)} images and labels.")
