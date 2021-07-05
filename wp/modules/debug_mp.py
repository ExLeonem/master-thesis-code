import argparse
import os, sys, importlib
import time
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from active_learning import TrainConfig, Config, Metrics, Pool, AcquisitionFunction
from bayesian import McDropout, MomentPropagation, BayesModel
from data import BenchmarkData, DataSetType
from models import default_model, setup_growth
from utils import setup_logger

import tensorflow as tf
import logging



def select_model(model_name, base_model, **kwargs):
    """
        Select metrics and create a specific bayesian model from the given base model.

        Parameters:
            model_name (str): The model type. One of ["dp", "mp"]
            base_model (tf.Model): Tensorflow model

        Returns:
            ((model, list())) The BayesianModel and specific model metrics for the metric writer.
    """
    metrics = None 
    model = None
    if model_name == "dp":
        model = McDropout(base_model, **kwargs)
        metrics = ['loss', 'binary_accuracy']

    elif model_name == "mp":
        pass
    else:
        raise ValueError("There is nothing specified for model type {}.".format(model_switch))

    return model, metrics


def keys_to_dict(**kwargs):
    return kwargs



if __name__ == "__main__":
    """
        Execute a manually constructed active learning loop.
        Several runtime parameters are parsable to the script.

        Argv:
            -m, --model (str) The name of the model to use. One of ['mp', 'dp']
            -c, --class_count (int) The number of classes to use from targets
            -acq, --acquisition_function (str): The acquisition function to use for evaluation. One of ['max_entropy', 'bald', 'max_var_ratio', 'std_mean', 'random']
            -s, --step_size (int): The step size to iterate the dataset over

    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    BASE_PATH = os.path.join(dir_path, "..")
    DATASET_PATH = os.path.join(BASE_PATH, "datasets")

    # Parse script arguments
    parser = argparse.ArgumentParser(description="Active learning parser")
    parser.add_argument("-c", "--class_count", default=3, type=int)
    parser.add_argument("-aqf", "--acquisition_function", default="max_entropy")
    parser.add_argument("-aqb", "--acquisition_batch_size", default=900, type=int)
    parser.add_argument("-s", "--step_size", default=10, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--tf_seed", default=None, type=int)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-l", "--limit", default=None, type=int)
    parser.add_argument("-d", "--debug", default=False, action="store_true")
    parser.add_argument("-i", "--initial_size", default=1, type=int)
    # parser.add_argument("")
    args = parser.parse_args()

    # Runtime arguments
    num_classes = args.class_count
    acq_function_name = args.acquisition_function
    acq_batch_size = args.acquisition_batch_size
    step_size = args.step_size
    epochs = args.epochs
    seed = args.seed
    tf_seed = args.tf_seed
    limit = args.limit

    # Setup Logger
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(current_dir, "logs")
    logger = setup_logger(args.debug, file="debug_mp.log", path=log_path)

    # Set seeds for reproducability?
    if not (seed is None):
        np.random.seed(seed)

    if not (tf_seed is None):
        tf.random.set_seed(tf_seed)

    # Load and transform mnist dataset
    mnist = BenchmarkData(DataSetType.MNIST, os.path.join(DATASET_PATH, "mnist"), classes=num_classes)
    x_train, x_test, y_train, y_test = train_test_split(mnist.inputs, mnist.targets)

    # Setup active learning specifics
    acquisition = AcquisitionFunction(acq_function_name, batch_size=acq_batch_size, verbose=True)
    pool = Pool(x_train, y_train)
    pool.init(args.initial_size)

    # Setup the Model
    setup_growth()
    loss = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
    metrics = "binary_accuracy" if num_classes == 2 else "accuracy"
    config = TrainConfig(
        batch_size=900,
        optimizer="adadelta",
        loss=loss,
        metrics=[metrics]
    )

    # model, metrics = select_model(model_name, base_model, is_binary=(num_classes == 2))
    # model = McDropout(base_model, is_binary=(num_classes == 2))

    # Active learning loop
    save_state_exists = False
    iterator = tqdm(range(0, pool.get_length_unlabeled()), step_size), leave=True)
    acl_history = []
    it = 0
    for i in iterator:
        loop_start = time.time()

        tf.keras.backend.clear_session()
        if not (limit is None) and it > limit:
            # Defined iteration limit reached
            break
        
        # Create base model and fit the base model
        base_model = BayesModel(default_model(output_classes=num_classes), config)
        if not save_state_exists:
            base_model.save_weights()
        else:
            base_model.load_weights()
        base_model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=config["metrics"])


        logger.info("---------")
        # Fit the base model
        start = time.time()
        logger.info("(Start) Train")
        lab_inputs, lab_targets = pool.get_labeled_data()
        history = base_model.fit(lab_inputs, lab_targets, batch_size=10, epochs=epochs, verbose=False)
        end = time.time()
        logger.info("Metrics: {}".format(history.history))
        logger.info("(Finish) Train (%.2fs)" % (end-start))

        # Create MP model
        model = MomentPropagation(base_model.get_model(), is_binary=(num_classes==2), verbose=False)
        model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=["binary_accuracy"])

        # Selected datapoints and labels
        logger.info("(Start) Acqusition")
        start = time.time()
        indices, _pred = acquisition(model, pool, num=step_size)
        end = time.time()
        logger.info("(Finish) Acquisition (%.2fs)" % (end-start))
        
        # Update pools
        logger.info("(Start) Update pool")
        start = time.time()
        pool.annotate(indices)
        lab_pool_size = pool.get_length_labeled()
        end = time.time()
        logger.info("(Finish) Update pool (%.2fs)" % (end-start))

        # Evaluate model
        logger.info("(Start) Evaluate")
        start = time.time()
        eval_result = model.evaluate(x_test, y_test, batch_size=config["batch_size"], verbose=False)
        eval_metrics = model.map_eval_values(eval_result)
        logger.info("(Finish) Evaluate (%.2fs)" % (end-start))


        loop_end = time.time()
        acl_history.append(keys_to_dict(
            iteration=it,
            labeled_size=lab_pool_size,
            time=(loop_end-loop_start),
            **eval_metrics
        ))

        logger.info("Iteration {}".format(i))
        logger.info("Metrics: {}".format(str(eval_metrics)))
        logger.info("Labeled_size: {}".format(lab_pool_size))
        it += 1

    METRICS_PATH = os.path.join(BASE_PATH, "metrics")
    metrics = Metrics(METRICS_PATH, keys=["iteration", "labeled_size", "loss", "time", "accuracy"])
    model_name = str(model.get_model_type()).split(".")[1].lower() 
    metrics.write("moment_propagation" + "_" + acq_function_name, acl_history)


    # Check length of pool and initial data
    # print("Initial data: {}".format(len(x_train)))
    # print("Labeled data: {}".format(len(labeled_pool)))

