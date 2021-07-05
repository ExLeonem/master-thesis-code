
import argparse
import os, sys, importlib
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from active_learning import TrainConfig, Config, Metrics, Pool, AcquisitionFunction, Dataset, ActiveLearningLoop
from bayesian import McDropout, MomentPropagation, BayesModel
from data import BenchmarkData, DataSetType
from models import default_model, setup_growth
from utils import setup_logger

import tensorflow as tf
import logging


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    BASE_PATH = os.path.join(dir_path, "..")
    DATASET_PATH = os.path.join(BASE_PATH, "datasets")

    # Setup Logger
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(current_dir, "logs")
    logger = setup_logger(True, path=log_path)

    # Load and transform mnist dataset
    num_classes = 10
    mnist = BenchmarkData(DataSetType.MNIST, os.path.join(DATASET_PATH, "mnist"), classes=num_classes)
    x_train, x_test, y_train, y_test = train_test_split(mnist.inputs, mnist.targets)
    
    setup_growth()

    dataset = Dataset(mnist.inputs, mnist.targets, init_size=10)
    query_fn = AcquisitionFunction("random")

    base_model = default_model(output_shape=num_classes)
    mc_model = McDropout(base_model)
    mc_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    logger.info("Dataset size: {}".format(len(dataset.x_train)))

    acl = ActiveLearningLoop(mc_model, dataset, "random", step_size=50, limit=10)
    acl.run()
    
    
    

    