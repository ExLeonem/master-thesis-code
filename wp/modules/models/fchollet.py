import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten



class FcholletCNN(Model):
    """
        CNN used for benchmarking in paper
        'Bayesian deep learning and a probabilistic perspective of generalization'
    """

    def __init__(self, output=128):
        super(FcholletCNN, self).__init__()
        self._name = "Fchollet-CNN"

        self.conv = Sequential([
            Conv2D(32, 3, activation=tf.nn.relu, padding="same"),
            Conv2D(64, 3, activation=tf.nn.relu, padding="same"),
            MaxPooling2D(),
            Dropout(.25)
        ], name="Convolutions")

        self.flatten = Flatten()
        self.linear = Sequential([
            Dense(128, activation=tf.nn.relu),
            Dropout(.5),
            Dense(output, activation="softmax")
        ], name="Linear")

    
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.flatten(x)
        return self.linear(x)
