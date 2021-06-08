import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense



class CnnFchollet(Model):
    """
        
    """

    def init(self):
        super().init()
        
        self.conv = Sequential(
            Conv2D(32, 3, activation=tf.nn.relu"), # padding=0
            Conv2D(64, 3, activation=tf.nn.relu),
            MaxPooling2D(),
            Dropout(.25)
        )

        self.linear = Sequential(
            Dense(14*14*64*128, activation="relu"),
            Dropout(.5),
            Dense(128, 10)
        )

    
    def call(self, inputs):
        x = self.conv(inputs)
        x = x.view(-1, 14*14*64)
        return self.linear(x)
