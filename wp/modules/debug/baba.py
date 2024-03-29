import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
import tensorflow.keras as keras


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(TF_PATH)

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
# from tf_al_mp.wrapper import MomentPropagation

from models import dnn_dense_dropout, setup_growth, disable_tf_logs


SEED = 841312
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Load data and preprocess
iris = load_iris()
x = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Encode
enc = OneHotEncoder()
y_encoded = enc.fit_transform(y[:, np.newaxis]).toarray()
scaler = StandardScaler()

# Theoretically do not scale (preprocess)
# x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=.4)
n_features = x.shape[1]
n_classes = y_encoded.shape[1]
dataset = Dataset(x_train, y_train, test=(x_test, y_test), init_size=3)



# Create and configure models
disable_tf_logs()
setup_growth()

sample_size = 100
config = Config(
    fit={"epochs": 120, "batch_size": 1},
    query={"sample_size": sample_size},
    eval={"batch_size": 900, "sample_size": sample_size}
)
base_model = dnn_dense_dropout(n_features, n_classes, n_layers=2)
mc_model = McDropout(base_model, config)
mc_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])


baba = mc_model.get_query_fn("baba")
output = baba(x_train[:10])


print(output)
