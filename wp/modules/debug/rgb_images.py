import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras

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

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# inputs = np.random.randn(100, 28, 28, 3)
# classes = list(range(5))
# num_classes = len(classes)
# targets = np.random.choice(classes, 100)

# x_test = np.random.randn(50, 28, 28, 3)
# y_test = np.random.choice(classes, 50)
y_test = y_test.flatten()
y_train = y_train.flatten()

dataset = Dataset(x_train, y_train, test=(x_test, y_test), init_size=20)
num_classes = len(list(np.unique(y_train)))


# Model
disable_tf_logs()
setup_growth()

batch_size = 10
sample_size = 25

base_model = fchollet_cnn(input_shape=(32, 32, 3), output=num_classes)
config = Config(
    fit={"epochs": 200, "batch_size": batch_size},
    query={"sample_size": sample_size},
    eval={"batch_size": batch_size, "sample_size": sample_size}
)
mc_model = McDropout(base_model, config=config, verbose=verbose)

optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = [keras.metrics.SparseCategoricalAccuracy()]
mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# Active Learning
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
    verbose=verbose
)
experiments.start()