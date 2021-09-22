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


"""
    
"""
# Set base path
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")


# Setup following: https://ermlab.com/en/blog/data-science/breast-cancer-classification-using-scikit-learn-and-keras/
# Set initial seed
SEED = 841312
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Load and create dataset
dia = load_diabetes()
x = dia.data
y = dia.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)
dataset = Dataset(x_train, y_train, test=(x_test, y_test), init_size=20)


# Create model
disable_tf_logs()
setup_growth()

sample_size = 100
config = Config(
    fit={"epochs": 100, "batch_size": 5},
    query={"sample_size": sample_size},
    eval={"sample_size": sample_size}
)
n_features = x_train.shape[1]
base_model = dnn_dense_dropout(n_features, 1, n_layers=2)
mc_model = McDropout(base_model, config)
mc_model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["accuracy"]
)

mc_model.fit(x_train, y_train)
mc_model.evaluate(x_test, y_test)

# METRICS_PATH = os.path.join(BASE_PATH, "metrics", "iris_temp")
# metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

# all_seeds = [SEED] + list(np.random.randint(1000, 10000, 9))
# print("Seeds:", all_seeds)
# experiments = ExperimentSuit(
#     [mc_model],
#     [
#         "random",
#         "max_entropy",
#         "bald",
#         "std_mean",
#         "max_var_ratio"
#     ],
#     dataset,
#     max_rounds=50,
#     no_save_state=True,
#     seed=all_seeds,
#     metrics_handler=metrics_handler
# )

# experiments.start()