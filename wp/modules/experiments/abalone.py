import os, sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow.keras as keras



BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

TF_AL_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(TF_AL_PATH)

TF_MP_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al_mp")
sys.path.append(TF_MP_PATH)

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
from tf_al.utils import gen_seeds
from tf_al_mp.wrapper import MomentPropagation

from models import setup_growth, disable_tf_logs, dnn_dense_dropout
from utils import setup_logger


# # Set initial seeds
seeds = gen_seeds(5)
np.random.seed(seeds[0])
tf.random.set_seed(seeds[0])


# Sex		nominal			M, F, and I (infant)
# Length		continuous	mm	Longest shell measurement
# Diameter	continuous	mm	perpendicular to length
# Height		continuous	mm	with meat in shell
# Whole weight	continuous	grams	whole abalone
# Shucked weight	continuous	grams	weight of meat
# Viscera weight	continuous	grams	gut weight (after bleeding)
# Shell weight	continuous	grams	after being dried
# Rings		integer			+1.5 gives the age in years


# Load and prepare data (https://archive.ics.uci.edu/ml/datasets/Abalone)
DATASET_PATH = os.path.join(BASE_PATH, "..", "..", "datasets", "abalone")
columns = [
    "sex", 
    "length", 
    "diameter", 
    "height", 
    "whole_weight", 
    "shucked_weight", 
    "viscera_weight", 
    "shell_weight", 
    "rings"
]

data = pd.read_csv(os.path.join(DATASET_PATH, "abalone.data"), sep=",")
data.columns = columns

encoder = LabelEncoder()
inputs = data[columns[:-1]]
inputs["sex"] = encoder.fit_transform(inputs["sex"].to_numpy())
inputs = inputs.to_numpy()
targets = data["rings"].to_numpy()

num_input_features = len(columns[:-1])
num_classes = len(np.unique(targets))


x_train, x_test, y_train, y_test = train_test_split(inputs, targets)
initial_pool_size = 100
dataset = Dataset(
    x_train, y_train,
    test=(x_test, y_test),
    init_size=initial_pool_size
)

# Active Learning parameters
step_size = 10
batch_size = 10
verbose = True
sample_size = 25

# Configure Tensorflow
disable_tf_logs()
setup_growth()

# Define Models and compilation parameter
num_classes = len(np.unique(targets))
base_model = dnn_dense_dropout(num_input_features, num_classes, 3)
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = [keras.metrics.SparseCategoricalAccuracy()]

# --------------- ------
# MC Dropout Model
fit_params = {"epochs": 200, "batch_size": batch_size}
mc_config = Config(
    fit=fit_params,
    query={"sample_size": sample_size},
    eval={"batch_size": 900, "sample_size": sample_size}
)
mc_model = McDropout(base_model, config=mc_config, verbose=verbose)
mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# Moment Propagation Model
mp_config = Config(
    fit={"epochs": 100, "batch_size": batch_size},
    eval={"batch_size": 900}
)
mp_model = MomentPropagation(base_model, mp_config, verbose=verbose)
mp_model.compile(optimizer=optimizer, loss=loss,  metrics=metrics)
# -----------------------------------------------------------------

# Setup metrics handler
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
METRICS_PATH = os.path.join(BASE_PATH, "metrics", "abalone_local")
print(METRICS_PATH)
metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

# Setup experiment Suit
models = [mc_model]
query_fns = [
    AcquisitionFunction("random", batch_size=900, verbose=verbose),
    AcquisitionFunction("bald", batch_size=900, verbose=verbose),
    AcquisitionFunction("max_entropy", batch_size=900, verbose=verbose),
    AcquisitionFunction("max_var_ratio", batch_size=900, verbose=verbose),
    AcquisitionFunction("std_mean", batch_size=900, verbose=verbose)
]


experiments = ExperimentSuit(
    models,
    query_fns,
    dataset,
    step_size=step_size,
    max_rounds=100,
    # seed=seeds,
    no_save_state=True,
    metrics_handler=metrics_handler,
    verbose=verbose,
)
experiments.start()