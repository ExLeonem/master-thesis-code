
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Softmax


def dnn_dense_dropout(
    input_dim, 
    output_dim, 
    n_layers=1,
    layer_nodes=8,
    multiplier=1,
    dropout_after_n=2,
    dropout_percentage=.25,
    name="model"
):
    """
        Simple deep neural network using only dense
        and dropout layers for bayesian approximation.

        Parameters:
            input_dim (int): Number of features going into the network.
            output_dim (int): Number of classes/outputs expected from the network.
            n_layers (int): The numbers of Dense/Dropout layers of the network. (default=1)
            layer_nodes (int): The neurons per dense layer. (default=8)
            multiplier (int): Sequentially multiplies the layer number of neurons. Except the last layer. (default=1)
            dropout_after_n (int): Applying dropout after each n-th layer. (default=2)
            dropout_percentage (float): With which probability to apply dropout? (default=.25)
            name (str): The name of the model. (default='model')

        Returns:
            (keras.Sequential) the dnn.
    """

    model = Sequential(name=name)
    for i in range(n_layers):
        
        if i == 0:
            model.add(Dense(layer_nodes, input_dim=input_dim, activation="relu"))
            continue

        layer = None
        if (i+1) % dropout_after_n == 0  and i != n_layers-1:
            layer = Dropout(dropout_percentage)
        
        elif i == n_layers-1:
            layer = Dense(output_dim)

        else:
            layer = Dense(layer_nodes*multiplier, activation="relu")
            multiplier *= multiplier

        model.add(layer)
    
    model.add(Softmax())
    return model