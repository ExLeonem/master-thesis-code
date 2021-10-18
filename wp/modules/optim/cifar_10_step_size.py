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

from tf_al import Config, Dataset, ActiveLearningLoop, AcquisitionFunction
from tf_al.wrapper import McDropout
# from tf_al_mp.wrapper import MomentPropagation

from models import fchollet_cnn, setup_growth, disable_tf_logs


# Create dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(x_train)
x_train = datagen.standardize(x_train.astype(np.float32))

datagen.fit(x_test)
x_test = datagen.standardize(x_test.astype(np.float32))

y_train = y_train.flatten()
y_test = y_test.flatten()

init_pool_size = 840
dataset = Dataset(
    x_train, y_train,
    test=(x_test, y_test),
    init_size=init_pool_size
)

# Model params
disable_tf_logs()
setup_growth()

verbose = False
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = [keras.metrics.SparseCategoricalAccuracy()]

# Create McModel
num_classes = len(list(np.unique(y_test)))
input_shape = tuple(x_train.shape[1:])
base_model = fchollet_cnn(input_shape=input_shape, output=num_classes)
batch_size = 32
sample_size = 25

config = Config(
    fit={"epochs": 200, "batch_size": batch_size},
    query={"sample_size": sample_size},
    eval={"batch_size": batch_size, "sample_size": sample_size}
)
mc_model = McDropout(base_model, config=config, verbose=verbose)
mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Create and run active learning loop


models = [mc_model]
step_size = 10
max_rounds = 100
best_loss = 500
def objective(space):
    global dataset
    global acf
    global mc_model
    global best_loss

    acl = ActiveLearningLoop(
        mc_model,
        dataset,
        "random",
        step_size=space["step_size"],
        max_rounds=3
    )

    loss = None
    for metrics in acl:
        loss = metrics["eval"]["sparse_categorical_crossentropy"]

    if loss < best_loss:
        best_loss = loss
    
    mc_model.reset(None, None)
    return loss
    


space = {
    "step_size": hp.choice("al_step_size", np.arange(10, 200, 10))
}


best = fmin(objective, space,algo=tpe.suggest, max_evals=10)
print(best)
print(space_eval(space, best))