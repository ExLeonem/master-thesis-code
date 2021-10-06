

import os, sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from hyperopt import tpe, fmin, hp, space_eval

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(TF_PATH)

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.utils import gen_seeds
from tf_al.wrapper import McDropout
# from tf_al_mp.wrapper import MomentPropagation

from models import fchollet_cnn, setup_growth, disable_tf_logs
from utils import search_initial_pool_size

# Set a seed
# seeds = gen_seeds(5)
# print("Initial seeds: ", seeds)
# np.random.seed(seeds[0])
# tf.random.set_seed(seeds[0])


# Create dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(x_train)
x_train = datagen.standardize(x_train.astype(np.float32))

datagen.fit(x_test)
x_test = datagen.standardize(x_test.astype(np.float32))

y_train = y_train.flatten()
y_test = y_test.flatten()

init_pool_size = 20
dataset = Dataset(
    x_train, y_train,
    test=(x_test, y_test),
    init_size=init_pool_size
)

# Model params
disable_tf_logs()
setup_growth()

optimizer = "adam"
loss_fn = "sparse_categorical_crossentropy"
metrics = [keras.metrics.SparseCategoricalAccuracy()]

# Create McModel
num_classes = len(list(np.unique(y_test)))
input_shape = tuple(x_train.shape[1:])

# config = Config(
#     fit={"epochs": 200, "batch_size": 10},
#     evaluate={"batch_size": 10}
# )
# mc_model = McDropout(base_model, config=config)


indices = np.arange(0, len(x_train))
best_score = 100
def objective(space):
    global best_score
    global optimizer
    global loss_fn
    global metrics
    global x_train
    global y_train
    global x_test
    global y_test

    base_model = fchollet_cnn(input_shape=input_shape, output=num_classes)
    base_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    size = space["init_size"]
    selected = np.random.choice(indices, size)
    x_train_sub = x_train[selected]
    y_train_sub = y_train[selected]
    base_model.fit(x_train_sub, y_train_sub, epochs=200, batch_size=space["batch_size"], verbose=0)
    loss, acc = base_model.evaluate(x_test, y_test, verbose=0)
    if loss < best_score:
        best_score = loss
    
    return loss


space = {
    "init_size": hp.choice("p_size", np.arange(10, 1000, 10)),
    "batch_size": hp.choice("p_batch", np.arange(10, 100))
}


best = fmin(objective, space, algo=tpe.suggest, max_evals=10)
print(best)
print(space_eval(space, best))