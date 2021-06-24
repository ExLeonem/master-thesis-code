import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten


def fchollet_cnn(input_shape=(28, 28, 1), output=128):
    return Sequential([
        Conv2D(32, 3, activation=tf.nn.relu, padding="same", input_shape=input_shape),
        Conv2D(64, 3, activation=tf.nn.relu, padding="same"),
        MaxPooling2D(),
        Dropout(.25),
        Flatten(),
        Dense(128, activation=tf.nn.relu),
        Dropout(.5),
        Dense(output, activation="softmax")        
    ])



class FcholletCNN(Model):
    """
        CNN used for benchmarking in paper
        'Bayesian deep learning and a probabilistic perspective of generalization'
    """

    def __init__(self, output=128):
        super(FcholletCNN, self).__init__()
        self._name = "Fchollet-CNN"
        self.conv1 = Conv2D(32, 3, activation=tf.nn.relu, padding="same")
        self.conv2 = Conv2D(64, 3, activation=tf.nn.relu, padding="same")
        self.max_pool = MaxPooling2D()
        self.dp1 = Dropout(.25)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation=tf.nn.relu)
        self.dp2 = Dropout(.5)
        self.dense2 = Dense(output, activation="softmax")

    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dp1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dp2(x)
        return self.dense2(x)
