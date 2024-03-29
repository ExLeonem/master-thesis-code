import argparse
import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al_mp")
sys.path.append(TF_PATH)

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al_mp.wrapper import MomentPropagation

from models import dnn_dense_dropout, setup_growth, disable_tf_logs
from utils import setup_logger


# Paths 
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")


# https://janakiev.com/blog/keras-iris/
# Set initial random seed 
SEED = 841312
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Load data and preprocess
iris = load_iris()
x = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Encode
enc = OneHotEncoder()
y_encoded = enc.fit_transform(y[:, np.newaxis]).toarray()
scaler = StandardScaler()

# Theoretically do not scale (preprocess)
# x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=.4)
n_features = x.shape[1]
n_classes = y_encoded.shape[1]
dataset = Dataset(x_train, y_train, test=(x_test, y_test), init_size=3)


# Create and configure models
disable_tf_logs()
setup_growth()

config = Config(fit={"epochs": 100, "batch_size": 5})
n_features = x.shape[1]
n_classes = len(np.unique(y))
base_model = dnn_dense_dropout(n_features, n_classes, n_layers=3)
print(base_model.summary())

mp_model = MomentPropagation(base_model, config)
mp_model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["accuracy"]
)

all_seeds = [SEED] + list(np.random.randint(1000, 10000, 9))
print("Seeds:", all_seeds)
experiments = ExperimentSuit(
    [mp_model],
    [
        "random",
        "max_entropy",
        "bald",
        "std_mean",
        "max_var_ratio"
    ],
    dataset,
    max_rounds=50,
    no_save_state=True,
    seed=all_seeds
)

experiments.start()