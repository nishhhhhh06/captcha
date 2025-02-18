import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

X = np.load("data/processed/images.npy")
y = np.load("data/processed/labels.npy")

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(50, 100, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(5 * 62, activation="sigmoid"), 
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = build_model()
model.fit(X, y.reshape(y.shape[0], -1), batch_size=32, epochs=20, validation_split=0.1)
model.save("model/captcha_solver.h5")
