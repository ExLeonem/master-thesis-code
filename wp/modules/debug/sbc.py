import os, sys

import numpy as np
from numpy.core.defchararray import center
from sklearn.model_selection import train_test_split
from sklearn.cluster import k_means
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

from tf_al import Config, Dataset, Pool, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
from tf_al.utils import gen_seeds
from tf_al_mp.wrapper import MomentPropagation
from tf_al.pool_init import sbc

from models import fchollet_cnn, setup_growth, disable_tf_logs
from utils import setup_logger


test_set_size = 10_000
initial_pool_size = 20

# Split data into (x, 10K, 100) = (train/test/valid)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Concat and normalize data
x_stack = np.expand_dims(np.vstack([x_train, x_test]), axis=-1).astype(np.float32)
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(x_stack)

# inputs = datagen.standardize(x_stack)
inputs = np.vstack([x_train, x_test])
targets = np.hstack([y_train, y_test])
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_set_size)

x_sub = x_train/255
# x_sub = x_sub+1
# x_sub= x_sub.reshape(len(x_sub), 28*28)
y_sub = y_train



initial_set_size = 20
res = sbc(initial_set_size, x_sub)
print(res)
print(np.unique(res))
# num_classes = len(np.unique(y_train))
# centeroids, label, inertia = k_means(x_sub, initial_set_size)

# print("k-means-labels: ", label)

# indices_selected = []
# for idx in range(initial_set_size):
#     distance = np.sum(np.abs(x_sub - centeroids[idx]), axis=-1)
#     min_sample_index = np.argmin(distance, axis=0)
#     indices_selected.append(min_sample_index)


# print(indices_selected)
# print(np.unique(indices_selected))
    



# num_classes = len(np.unique(y_sub))
# model = SpectralClustering(n_clusters=num_classes)
# labels = model.fit_predict(x_sub, y_sub)
# print(labels)