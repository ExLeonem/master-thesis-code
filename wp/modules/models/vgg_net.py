import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Softmax


def vgg11(input_shape=(32, 32, 3), num_output=10):
    return Sequential([
    Conv2D(64, 3, activation="relu", padding="same", input_shape=input_shape),
    MaxPooling2D(),
    Conv2D(128, 3, activation="relu", padding="same"),
    MaxPooling2D(),
    Conv2D(256, 3, activation="relu", padding="same"),
    Conv2D(256, 3, activation="relu", padding="same"),
    Dropout(0.25),
    MaxPooling2D(),
    Conv2D(512, 3, activation="relu", padding="same"),
    Conv2D(512, 3, activation="relu", padding="same"),
    MaxPooling2D(),
    Conv2D(512, 3, activation="relu", padding="same"),
    Conv2D(512, 3, activation="relu", padding="same"),
    Flatten(),
    Dense(4096, activation="relu"),
    Dropout(0.5),
    Dense(4096, activation="relu"),
    Dense(num_output),
    Softmax()
])


def vgg16(input_shape=(32, 32, 3), num_output=10):
    return Sequential([
        Conv2D(64, 3, activation="relu", padding="same", input_shape=input_shape),
        Conv2D(64, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        Conv2D(128, 3, activation="relu", padding="same"),
        Conv2D(128, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        Conv2D(256, 3, activation="relu", padding="same"),
        Conv2D(256, 3, activation="relu", padding="same"),
        Conv2D(256, 3, activation="relu", padding="same"),
        Dropout(0.25),
        MaxPooling2D(),
        Conv2D(512, 3, activation="relu", padding="same"),
        Conv2D(512, 3, activation="relu", padding="same"),
        Conv2D(512, 3, activation="relu", padding="same"),
        MaxPooling2D(),
        Conv2D(512, 3, activation="relu", padding="same"),
        Conv2D(512, 3, activation="relu", padding="same"),
        Conv2D(512, 3, activation="relu", padding="same"),
        Dropout(.25),
        MaxPooling2D(),
        Flatten(),
        Dense(4096, activation="relu"),
        Dense(4096, activation="relu"),
        Dense(num_output),
        Softmax()
    ])



