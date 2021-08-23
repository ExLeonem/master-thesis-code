from copy import deepcopy
import argparse
import os, sys, importlib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from active_learning import TrainConfig, Config, Pool, AcquisitionFunction, Dataset, ExperimentSuit, ExperimentSuitMetrics
from wrapper import McDropout, MomentPropagation, BayesModel
from data import BenchmarkData, DataSetType
from models import default_model, setup_growth
from utils import setup_logger

import tensorflow as tf
import logging


if __name__ == "__main__":

    # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
    # Disable tensorflow loggin
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    dir_path = os.path.dirname(os.path.realpath(__file__))
    BASE_PATH = os.path.join(dir_path, "..")
    DATASET_PATH = os.path.join(BASE_PATH, "datasets")

    # Setup Logger
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(current_dir, "logs")
    logger = setup_logger(True, path=log_path)

    setup_growth()

    # Load and transform mnist dataset
    num_classes = 10
    mnist = BenchmarkData(DataSetType.MNIST, os.path.join(DATASET_PATH, "mnist"), classes=num_classes)
    x_train, x_test, y_train, y_test = train_test_split(mnist.inputs, mnist.targets)

    dataset = Dataset(x_train, y_train, test=(x_test, y_test), init_size=10)
    base_model = default_model(output_shape=num_classes)
    model_config = Config(
        fit={"epochs": 10, "batch_size": 10},
        eval={"batch_size": 900}
    )

    METRICS_PATH = os.path.join(BASE_PATH, "metrics")
    metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

    # Create Models
    mc_model = McDropout(base_model, model_config)
    mc_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # mp_model = MomentPropagation(base_model, model_config)
    # mp_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    models = [mc_model]
    query_fns = [
        AcquisitionFunction("random", batch_size=900), 
        # AcquisitionFunction("max_entropy", batch_size=900)
    ]

    experiments = ExperimentSuit(
        models, 
        query_fns, 
        dataset, 
        step_size=50, 
        limit=5,
        runs=4,
        acceptance_timeout=2, 
        metrics_handler=metrics_handler, 
        verbose=True
    )

    experiments.start()

    # acl = ActiveLearningLoop(mc_model, dataset, "random", step_size=50, limit=10)
    # acl.run()
    # o_loop = deepcopy(acl)

    