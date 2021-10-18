import os, sys, math, gc, time

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(MODULES_PATH)
sys.path.append(TF_PATH)


from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
# from tf_al_mp.wrapper import MomentPropagation

from models import fchollet_cnn, setup_growth, disable_tf_logs
from utils import setup_logger


verbose = True

# Synthetic dataset
inputs = np.random.randn(100, 10)
targets = np.random.randn(100)
x_test = np.random.randn(50, 10)
y_test = np.random.randn(50)
dataset = Dataset(inputs, targets, test=(x_test, y_test), init_size=5)


# Model
setup_growth()

num_classes = 1
batch_size = 900
sample_size = 25

base_model = Sequential([
    Dense(num_classes, activation="relu"),
    Dropout(.25),
    Dense(num_classes*2),
    Dense(1),
    ReLU()
])


config = Config(
    fit={"epochs": 200, "batch_size": batch_size},
    query={"sample_size": sample_size},
    eval={"batch_size": batch_size, "sample_size": sample_size}
)
mc_model = McDropout(base_model, config=config, classification=False, verbose=verbose)

optimizer = "adam"
loss = "mean_squared_error"
mc_model.compile(optimizer=optimizer, loss=loss)


# Active Learning
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = os.path.join(BASE_PATH, "..", "..", "metrics")
metrics_handler = ExperimentSuitMetrics(os.path.join(METRICS_PATH, "debug_regression"))


step_size = 10
query_fns = [
    AcquisitionFunction("random", batch_size=batch_size, verbose=verbose),
    AcquisitionFunction("max_entropy", batch_size=batch_size, verbose=verbose)
]

experiments = ExperimentSuit(
    mc_model, 
    query_fns,
    dataset,
    step_size=step_size,
    no_save_state=True,
    metrics_handler=metrics_handler,
    verbose=verbose
)
experiments.start()