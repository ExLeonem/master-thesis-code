import os, importlib, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

import mp.MomentPropagation as mp
import data.mnist as mnist_loader
from models import default_model, setup_growth


from acl import ActiveLearning
from active_learning import TrainConfig, Config, Metrics, aggregates_per_key
from bayesian import McDropout, MomentPropagation


if __name__ == "__main__":

    # Common paths
    BASE_PATH = os.path.join(os.getcwd(), "..")
    MODULE_PATH = os.path.join(BASE_PATH, "modules")
    DS_PATH = os.path.join(BASE_PATH, "datasets")

    # Set gpu growth (needed  for moment propagation)
    setup_growth()

    # Create deep learning models
    tf_base_model = default_model()
    mp = mp.MP()
    mp_m = mp.create_MP_Model(model=tf_base_model, use_mp=True, verbose=True)

    # Load Data
    mnist_path = os.path.join(DS_PATH, "mnist")
    inputs, targets = mnist_loader.load(mnist_path)

    # Select only first and second class
    selector = (targets==0) | (targets==1)
    new_inputs = inputs[selector].astype("float32")/255.0
    new_targets = targets[selector]

    # Create splits
    # x_train, x_test, y_train, y_test = train_test_split(new_inputs, new_targets)
    # x_test, x_val, y_test, y_val = train_test_split(x_test, y_test)

    metrics = ["binary_accuracy", "val_loss"]

    
    # ACL configuration
    train_config = TrainConfig(
        batch_size=60,
        epochs=10,
        metrics=["binary_accuracy"]
    )

    acq_config = Config(
        name="std_mean",
        pseudo=True
    )

    model_name = "mp"
    acq_name = "max_entropy"

    # Active learning models
    mp_model = MomentPropagation(tf_base_model)
    dp_model = McDropout(tf_base_model)

    # Active learning loop
    active_learning = ActiveLearning(
        mp_model,
        np.expand_dims(new_inputs, axis=-1), labels=new_targets, 
        train_config=train_config,
        eval_config=train_config,
        acq_name=acq_name,
        debug=False
    )

    history = active_learning.start(step_size=40)

    METRICS_PATH = os.path.join(BASE_PATH, "metrics")
    metrics = Metrics(METRICS_PATH, keys=["iteration", "train_time", "query_time", "eval_loss", "eval_acc", "loss", "binary_accuracy"])
    metrics.write(model_name + "_" + acq_name, history)