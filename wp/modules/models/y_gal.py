import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation


# hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))

def ygal_cnn(number_of_datapoints, input_shape=(28, 28, 1), output=128):
    """
        
    """

    nb_filters = 32
    nb_conv = 4
    nb_pool = 2
    
    c = 3.5
    weight_decay = c / float(number_of_datapoints)

    return Sequential([
        Conv2D(nb_filters, nb_conv, activation="relu", input_shape=input_shape),
        Conv2D(nb_filters, nb_conv, activation="relu"),
        MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(128, kernel_regularizer=l2(weight_decay), activation="relu"),
        Dropout(0.5),
        Dense(output),
        Activation("softmax")
    ])