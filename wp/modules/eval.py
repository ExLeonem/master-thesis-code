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
from wrapper import McDropout, MomentPropagation


def select_model(model_switch, base_model):
    """
        Select metrics and create a specific bayesian model from the given base model.

        Parameters:
            model_switch (str): The model type. One of ["dp", "mp"]
            base_model (tf.Model): Tensorflow model

        Returns:
            ((model, list())) The BayesianModel and specific model metrics for the metric writer.
    """

    metrics = None 
    model = None
    if model_switch == "dp":
        model = McDropout(tf_base_model)
        metrics = ['loss', 'binary_accuracy']

    elif model_switch == "mp":
        model = MomentPropagation(tf_base_model)
        metrics = [
            'loss', 'tf_op_layer_Sigmoid_1_loss', 
            'tf_op_layer_Mul_8_loss', 'tf_op_layer_Sigmoid_1_binary_accuracy', 
            'tf_op_layer_Mul_8_binary_accuracy'
        ]
    else:
        raise ValueError("There is nothing specified for model type {}.".format(model_switch))


    return model, metrics



if __name__ == "__main__":

    tf.random.set_seed(2)

    # Common paths
    BASE_PATH = os.path.join(os.getcwd(), "..")
    MODULE_PATH = os.path.join(BASE_PATH, "modules")
    DS_PATH = os.path.join(BASE_PATH, "datasets")

    # Set gpu growth (needed  for moment propagation)
    setup_growth()

    # Create deep learning models
    tf_base_model = default_model()

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

    # ACL configuration
    train_config = TrainConfig(
        batch_size=100,
        epochs=10,
        metrics=["binary_accuracy"]
    )

    acq_config = Config(
        name="std_mean",
        pseudo=True
    )

    model_name = "dp"
    acq_name = "max_entropy"

    # Create model for active learning
    model, metrics_to_write = select_model(model_name, tf_base_model)


    # Active learning loop
    active_learning = ActiveLearning(
        model,
        np.expand_dims(new_inputs, axis=-1), labels=new_targets, 
        train_config=train_config,
        eval_config=train_config,
        acq_name=acq_name,
        debug=False
    )

    history = active_learning.start(limit_iter=10, step_size=40)

    METRICS_PATH = os.path.join(BASE_PATH, "metrics")
    metrics = Metrics(METRICS_PATH, keys=["iteration", "train_time", "query_time"] + metrics_to_write)
    metrics.write(model_name + "_" + acq_name, history)