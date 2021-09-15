import os, sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
from tf_al_mp.wrapper import MomentPropagation

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

from models import fchollet_cnn, setup_growth, disable_tf_logs

# Set a seed
SEED = 84123
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Create dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

init_pool_size = 20
dataset = Dataset(
    x_train, y_train,
    test=(x_test, y_test),
    init_size=init_pool_size
)


# Model params
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = [keras.metrics.SparseCategoricalAccuracy()]

# Create McModel
num_classes = 10
base_model = fchollet_cnn(output=num_classes)
batch_size = 10
sample_size = 25

config = Config(
    fit={"epochs": 200, "batch_size": batch_size},
    query={"sample_size": sample_size},
    eval={"batch_size": batch_size, "sample_size": sample_size}
)
mc_model = McDropout(base_model, config=config)
mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def reset_step(self, pool, dataset):
        """
            Overwrite reset function after each acquisiton iteration.

            Parameters:
                pool (Pool): Pool of labeled datapoints.
                dataset (Dataset): dataset object containing train, test and eval sets.
        """
        self._model = fchollet_cnn(output=num_classes)
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

setattr(McDropout, "reset", reset_step)


# Create and run active learning loop
METRICS_PATH = os.path.join(BASE_PATH, "..", "..", "metrics", "cifar_10")
metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

models = [mc_model]
query_fns = [
    AcquisitionFunction("random", batch_size=900),
    AcquisitionFunction("max_entropy", batch_size=900),
    AcquisitionFunction("max_var_ratio", batch_size=900),
    AcquisitionFunction("bald", batch_size=900),
    AcquisitionFunction("std_mean", batch_size=900)
]

step_size = 10
max_rounds = 100
experiments = ExperimentSuit(
    models,
    query_fns,
    dataset,
    seed=SEED,
    step_size=step_size,
    no_save_state=True,
    max_rounds=max_rounds,
    metrics_handler=metrics_handler
)
experiments.start()