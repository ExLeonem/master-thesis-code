import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Dropout



fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images/255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images/255.0

model = Sequential([
    Conv2D(128, (3, 3), input_shape=(28, 28, 1), activation="relu"),
    MaxPool2D(2, 2),
    Dropout(0.2),
    Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation="relu"),
    MaxPool2D(2, 2),
    Dropout(0.2),
    Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
    MaxPool2D(2, 2),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=10)
model.evaluate(test_images, test_labels)

y_samples = np.stack([model(test_images, training=True) for sample in range(100)])
y_samples_mean = y_samples.mean(axis=0)
y_samples_std = y_samples.std(axis=0)

print(f'Mean = {np.round(y_samples_mean[:1], 2)}')
print(f'Std = {np.round(y_samples_std[:1], 2)}')
