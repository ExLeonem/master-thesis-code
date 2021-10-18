import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(MODULES_PATH)
sys.path.append(TF_PATH)

from tf_al import Config, Pool
from tf_al.wrapper import McDropout
from tf_al.utils import setup_logger
from tf_al.stats.sampling import Accuracy, Loss, AUC

from models import fchollet_cnn, setup_growth, disable_tf_logs


# Dataset
val_set_size = 100
test_set_size = 10_000
initial_pool_size = 20
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
inputs = np.expand_dims(np.vstack([x_train, x_test])/255., axis=-1)
targets = np.hstack([y_train, y_test])
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_set_size)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_set_size)
pool = Pool(x_train, y_train)
pool.init(initial_pool_size)

# Model
disable_tf_logs()
setup_growth()
sample_size = 25
batch_size = 10
num_classes = len(np.unique(y_train))
base_model = fchollet_cnn(output=num_classes)


fit_params = {"epochs": 1, "batch_size": 900}
mc_config = Config(
    fit=fit_params,
    query={"sample_size": sample_size},
    eval={"batch_size": 900, "sample_size": sample_size}
)
mc_model = McDropout(base_model, config=mc_config)


# optimizer = "adam"
# loss = "sparse_categorical_crossentropy"
# metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.SparseCategoricalAccuracy()]
# mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

optimizer = "adam"
loss = "sparse_categorical_crossentropy"
acc = keras.metrics.SparseCategoricalAccuracy()
metrics = [acc]
mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Fit and evaluate
x_lab, y_lab = pool.get_labeled_data()
pred = mc_model(x_lab)
res = np.mean(pred, axis=1)


# mc_model.fit(x_train, y_train)

# mc_model.eval_metrics = [
#     Accuracy("sparse_categorical_accuracy"),
#     Loss("sparse_categorical_crossentropy"),
# ]


# eval_result = mc_model.evaluate(x_lab, y_lab)
# print(eval_result)

