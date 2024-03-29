import os, sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(TF_PATH)

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.utils import gen_seeds
from tf_al.wrapper import McDropout
# from tf_al_mp.wrapper import MomentPropagation

from models import fchollet_cnn, setup_growth, disable_tf_logs, vgg11

# Set a seed
# seeds = gen_seeds(5)
seeds = [68919, 81488, 47908, 52279, 18232]
print("Initial seeds: ", seeds)
np.random.seed(seeds[0])
tf.random.set_seed(seeds[0])


# Create dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(x_train)
x_train = datagen.standardize(x_train.astype(np.float32))

datagen.fit(x_test)
x_test = datagen.standardize(x_test.astype(np.float32))

y_train = y_train.flatten()
y_test = y_test.flatten()

init_pool_size = 20_000
dataset = Dataset(
    x_train, y_train,
    test=(x_test, y_test),
    init_size=init_pool_size
)

# Model params
disable_tf_logs()
setup_growth()

verbose = True
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = [keras.metrics.SparseCategoricalAccuracy()]

# Create McModel
num_classes = len(list(np.unique(y_test)))
input_shape = tuple(x_train.shape[1:])
base_model = vgg11(input_shape, num_classes)
batch_size = 1000
sample_size = 25

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', patience=3)
]

# base_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# base_model.fit(x_train, y_train, epochs=200, batch_size=1000, callbacks=callbacks)
# print(base_model.evaluate(x_test, y_test))

config = Config(
    fit={"epochs": 200, "batch_size": batch_size, "callbacks": callbacks},
    query={"sample_size": sample_size},
    eval={"batch_size": batch_size, "sample_size": sample_size}
)
mc_model = McDropout(base_model, config=config, verbose=verbose)
mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# Create and run active learning loop
METRICS_PATH = os.path.join(BASE_PATH, "..", "..", "metrics", "cifar_10")
metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

models = [mc_model]
query_fns = [
    AcquisitionFunction("random", batch_size=900, verbose=verbose),
    AcquisitionFunction("max_entropy", batch_size=900, verbose=verbose),
    AcquisitionFunction("max_var_ratio", batch_size=900, verbose=verbose),
    AcquisitionFunction("bald", batch_size=900, verbose=verbose),
    AcquisitionFunction("std_mean", batch_size=900, verbose=verbose)
]

step_size = 100
max_rounds = 200
experiments = ExperimentSuit(
    models,
    query_fns,
    dataset,
    seed=seeds,
    step_size=step_size,
    no_save_state=True,
    max_rounds=max_rounds,
    metrics_handler=metrics_handler,
    verbose=verbose
)
experiments.start()