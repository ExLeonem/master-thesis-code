import tensorflow as tf
from tensorflow import keras
import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Conv2D, MaxPool2D, Dropout, Flatten, Dense, Softmax



""""
    Simple CNN model for testing purposes.
"""
def default_model():
    input_shape = (28, 28, 1)
    return Sequential([
        Conv2D(128, 4,activation="relu", input_shape=input_shape),
        MaxPool2D(),
        Dropout(.2),
        Conv2D(64, 3, activation="relu"),
        MaxPool2D(),
        Dropout(.2),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

