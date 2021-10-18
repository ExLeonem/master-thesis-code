import argparse
import copy
import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

from models import fchollet_cnn, setup_growth, disable_tf_logs
from utils import setup_logger


BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

# Pool/Dataset parameters
val_set_size = 100
test_set_size = 10_000
initial_pool_size = 20

# Split data into (x, 10K, 100) = (train/test/valid)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Concat and normalize data
x_stack = np.expand_dims(np.vstack([x_train, x_test]), axis=-1).astype(np.float32)
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(x_stack)

inputs = datagen.standardize(x_stack)
targets = np.hstack([y_train, y_test])
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_set_size)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_set_size)

disable_tf_logs()
setup_growth()

# seeds = gen_seeds(10)
# seeds = []
seeds = [20432, 10942, 83152, 59138, 49976, 10109, 74983, 66781, 93135]
print("Initial seeds {}".format(seeds))
first_seed = seeds[0]
np.random.seed(first_seed)
tf.random.set_seed(first_seed)

num_classes = len(np.unique(targets))
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = [keras.metrics.SparseCategoricalAccuracy()]

step_size = 1
batch_size = 10
verbose = False
sample_size = 25
fit_params = {"epochs": 200, "batch_size": batch_size}
base_model = fchollet_cnn(output=num_classes)
mc_config = Config(
    fit=fit_params,
    query={"sample_size": sample_size},
    eval={"batch_size": 900, "sample_size": sample_size}
)

mc_model = McDropout(base_model, config=mc_config)
mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print(mc_model.evaluate(x_test, y_test, batch_size=900))


# first_seed = seeds[1]
# np.random.seed(first_seed)
# tf.random.set_seed(first_seed)

# keras.backend.clear_session()
# model = keras.models.clone_model(base_model)
# model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

first_seed = seeds[0]
np.random.seed(first_seed)
tf.random.set_seed(first_seed)
o_model = copy.copy(mc_model)
print(o_model.evaluate(x_test, y_test, batch_size=900))