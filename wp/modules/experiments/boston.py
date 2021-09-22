import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(TF_PATH)

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
# from tf_al_mp.wrapper import MomentPropagation

from models import dnn_dense_dropout, setup_growth, disable_tf_logs
from utils import setup_logger


# Set initial Seed
SEED = 841312


# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data(test_split=.4)
dataset = Dataset(x_train, y_train, test=(x_test, y_test), init_size=20)

num_features = x_train.shape[1]

print(x_train.shape)


# Create and configure model
disable_tf_logs()
setup_growth()


sample_size = 100
config = Config(
    fit={"epochs": 100, "batch_size": 15},
    query={"sample_size": sample_size},
    eval={"batch_size": 900, "sample_size": sample_size}
)
base_model = dnn_dense_dropout(num_features, 1, n_layers=4)
mc_model = McDropout(base_model, config)
mc_model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_squared_error"]
)


mc_model.fit(x_train[:10], y_train[:10])
print("Eval")
print(mc_model.evaluate(x_test, y_test))
