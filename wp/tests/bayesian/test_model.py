
import pytest
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Model as TfModel
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

from modules.wrapper import Model
from modules.active_learning import Config, TrainConfig


def mock_inputs(samples=10):
    return np.random.randn(samples, 28, 28, 1)

def mock_targets(num_targets=10, unique_targets=3):
    labels = np.linspace(0, unqiue_targets-1, unique_targets)
    return np.random.choice(labels, num_targets)


# # Mock of a sequential model
# mock_seq_model = Sequential([
#     Conv2D(3, 3, activation="relu"),
#     Flatten(),
#     Dense(3, activation="softmax")
# ])


class MockModel(TfModel):
    """ Mocking a basic model, subclass model """

    def __init__(self):
        super(MockModel, self).__init__()

    def call(self, inputs, training=False):
        return inputs
        
        

class InvalidMockModel():
    """ Using a model of invalid type """

    def call(self, inputs, training=False):
        return inputs



class TestModel:

    def test_tensorflow_subclass_model(self):
        config = TrainConfig()
        model = Model(MockModel(), config)

        inputs = np.random.randn(10, 28, 28, 1)
        
        # assert result is None

    # def test_tensorflow_seq_model(self):
    #     mock_seq_model = Sequential([
    #         Conv2D(3, 3, activation="relu"),
    #         Flatten(),
    #         Dense(3, activation="softmax")
    #     ])


    def test_query_config_params_valid(self):

        config = Config(
            query={"sample_size": 25}
        )
        model = Model(MockModel(), config)

        query_config = model.get_query_config()
        assert query_config != {}

    
    def test_fit_config_params_valid(self):
        config = Config(
            fit={"epochs": 100}
        )
        model = Model(MockModel(), config)

        fit_config = model.get_fit_config()
        assert fit_config != {}

    
    def test_eval_config_params_valid(self):
        config = Config(
            eval={"sample_size": 25}
        )
        model = Model(MockModel(), config)
        eval_config = model.get_eval_config()

        assert eval_config != {} and "sample_size" in eval_config