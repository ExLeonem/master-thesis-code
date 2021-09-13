import argparse
import os, sys, time

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow.keras as keras

# from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay, AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from tensorflow.keras.optimizers import Adam

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
from tf_al_mp.wrapper import MomentPropagation

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

from data import BenchmarkData, DataSetType
from models import fchollet_cnn, setup_growth, disable_tf_logs

BASE_PATH = os.path.join(BASE_PATH, "..", "..")
DATASET_PATH = os.path.join(BASE_PATH, "datasets")


# Setup logger
c_dir_path = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(c_dir_path, "..", "logs")

# Pool/Dataset parameters
val_set_size = 100
test_set_size = 10_000
initial_pool_size = 300

# Split data into (x, 10K, 100) = (train/test/valid)
mnist = BenchmarkData(DataSetType.MNIST, os.path.join(DATASET_PATH, "mnist"), dtype=np.float32)
x_train, x_test, y_train, y_test = train_test_split(mnist.inputs, mnist.targets, test_size=test_set_size)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_set_size)

dataset = Dataset(
    x_train, y_train, 
    val=(x_val, y_val), 
    test=(x_test, y_test), 
    init_size=initial_pool_size
)

# Active Learning parameters
step_size = 1000
learning_rate = 0.001
verbose = False
sample_size = 100

# Configure Tensorflow
disable_tf_logs()
setup_growth()

# Define Models
num_classes = len(np.unique(mnist.targets))
base_model = fchollet_cnn(output=num_classes)
# base_model = ygal_cnn(initial_pool_size, output=num_classes)

def reset_step(self, pool, dataset):
    """
        Overwrite reset function after each acquisiton iteration.

        Parameters:
            pool (Pool): Pool of labeled datapoints.
            dataset (Dataset): dataset object containing train, test and eval sets.
    """
    # number_samples = pool.get_length_labeled()
    self.model = fchollet_cnn(output=num_classes)
    # self.load_weights()

# MC Dropout Model
batch_size = 20

early_stopping = keras.callbacks.EarlyStopping(
    monitor="sparse_categorical_accuracy",
    min_delta=0.01,
    patience=5
)
fit_params = {"epochs": 300, "batch_size": batch_size, "callbacks": [early_stopping]}

mc_config = Config(
    fit=fit_params,
    query={"sample_size": 25},
    eval={"batch_size": 900, "sample_size": 25}
)

setattr(McDropout, "reset", reset_step)
mc_model = McDropout(base_model, config=mc_config, verbose=verbose)
mc_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Moment Propagation
mp_config = Config(
    fit={"epochs": 100, "batch_size": batch_size},
    eval={"batch_size": 900}
)
# setattr(MomentPropagation, "reset", reset_step)
mp_model = MomentPropagation(base_model, mp_config, verbose=verbose)
mp_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Setup metrics handler
METRICS_PATH = os.path.join(BASE_PATH, "metrics", "ygal_sgd_modified")
metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

# Setup experiment Suit
models = [mc_model]
query_fns = [
    AcquisitionFunction("random", batch_size=900, verbose=verbose),
    # AcquisitionFunction("max_entropy", batch_size=900, verbose=verbose),
    # AcquisitionFunction("bald", batch_size=900, verbose=verbose),
    # AcquisitionFunction("max_var_ratio", batch_size=900, verbose=verbose),
    # AcquisitionFunction("std_mean", batch_size=900, verbose=verbose)
]

# 
experiments = ExperimentSuit(
    models,
    query_fns,
    dataset,
    step_size=step_size,
    limit=100,
    # runs=4,   
    no_save_state=True,
    metrics_handler=metrics_handler,
    verbose=verbose
)
experiments.start()