import argparse
import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(TF_PATH)

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
# from tf_al_mp.wrapper import MomentPropagation

from models import dnn_dense_dropout, setup_growth, disable_tf_logs
from utils import setup_logger


"""
    
"""
# Set base path
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")


# Setup following: https://ermlab.com/en/blog/data-science/breast-cancer-classification-using-scikit-learn-and-keras/
# Set initial seed
SEED = 841312
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Load data
breast = load_breast_cancer()
x = breast["data"]
y = breast["target"]
names = breast["target_names"]
feature_names = breast["feature_names"]

# Encode
print(y.shape)
print(x.shape)
print(feature_names)





