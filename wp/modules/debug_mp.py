import argparse
import os, sys, importlib
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from active_learning import TrainConfig, Config, Metrics, LabeledPool, UnlabeledPool, AcquisitionFunction
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


def init_pools(unlabeled_pool, labeled_pool, targets, num_init_per_target=10):
    """
        Initialize the pool with randomly selected values.

        Paramters:
            unlabeled_pool (UnlabeldPool): Pool that holds information about unlabeld datapoints.
            labeled_pool (LabeledPool): Pool that holds information about labeled datapoints.
            targest (numpy.ndarray): The labels of input values.
            num_init_per_target (int): The initial labels used per target. 

        Todo:
            - Make it work for labels with additional dimensions (e.g. bag-of-words, one-hot vectors)
    """

    # unlabeled_pool.update(indices)
    # labeled_pool[indices] = labels

    # Use initial target values?
    if num_init_per_target <= 0:
        return

    # Select 'num_init_per_target' per unique label 
    unique_targets = np.unique(targets)
    for idx in range(len(unique_targets)):

        # Select indices of labels for unique label[idx]
        with_unique_value = targets == unique_targets[idx]
        indices_of_label = np.argwhere(with_unique_value)

        # Set randomly selected labels
        selected_indices = np.random.choice(indices_of_label.flatten(), num_init_per_target, replace=True)

        # WILL NOT WORK FOR LABELS with more than 1 dimension
        unlabeled_pool.update(selected_indices)
        new_labels = np.full(len(selected_indices), unique_targets[idx])
        labeled_pool[selected_indices] = new_labels


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
    parser.add_argument("-c", "--class_count", default=2, type=int)
    parser.add_argument("-aqf", "--acquisition_function", default="max_entropy")
    parser.add_argument("-aqb", "--acquisition_batch_size", default=900, type=int)
    parser.add_argument("-s", "--step_size", default=1, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--tf_seed", default=None, type=int)
    parser.add_argument("-e", "--epochs", default=5, type=int)
    parser.add_argument("-l", "--limit", default=None, type=int)
    parser.add_argument("-d", "--debug", default=False, type=bool)
    parser.add_argument("-i", "--initial_size", default=2, type=int)
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
    acquisition = AcquisitionFunction(acq_function_name, batch_size=acq_batch_size)
    labeled_pool = LabeledPool(x_train)
    unlabeled_pool = UnlabeledPool(x_train)
    init_pools(unlabeled_pool, labeled_pool, y_train, num_init_per_target=args.step_size)

    # Setup the Model
    setup_growth()
    config = TrainConfig(
        batch_size=step_size
    )

    # model, metrics = select_model(model_name, base_model, is_binary=(num_classes == 2))
    # model = McDropout(base_model, is_binary=(num_classes == 2))

    # Active learning loop
    save_state_exists = False
    iterator = tqdm(range(0, len(unlabeled_pool), step_size), leave=True)
    acl_history = []
    it = 0
    for i in iterator:

        tf.keras.backend.clear_session()
        if not (limit is None) and it > limit:
            # Defined iteration limit reached
            break
        
        # Create base model and fit the base model
        base_model = BayesModel(default_model(), config)
        if not save_state_exists:
            base_model.save_weights()
        else:
            base_model.load_weights()
        base_model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=["binary_accuracy"])


        logger.info("---------")
        # Fit the base model
        lab_inputs, lab_targets = labeled_pool[:]
        history = base_model.fit(lab_inputs, lab_targets, batch_size=config["batch_size"], epochs=epochs, verbose=False)
        logger.info("Training: {}".format(history.history))

        # Create MP model
        model = MomentPropagation(base_model.get_model(), is_binary=(num_classes==2))
        model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=["binary_accuracy"])

        # Selected datapoints and labels
        indices, _pred = acquisition(model, unlabeled_pool, num=step_size)
        labels = y_train[indices]
        
        # Update pools
        lab_pool_size = len(labeled_pool)
        unlabeled_pool.update(indices)
        labeled_indices = unlabeled_pool.get_labeled_indices()
        labeled_pool[indices] = labels

        # Evaluate model
        eval_result = model.evaluate(x_test, y_test, batch_size=config["batch_size"], verbose=False)
        eval_metrics = model.map_eval_values(eval_result)
        acl_history.append(keys_to_dict(
            iteration=it,
            labeled_size=lab_pool_size,
            **eval_metrics
        ))

        logger.info("Iteration {}".format(i))
        logger.info("Metrics: {}".format(str(eval_metrics)))
        logger.info("Labeled_size: {}".format(len(labeled_pool)))
        it += 1

    METRICS_PATH = os.path.join(BASE_PATH, "metrics")
    metrics = Metrics(METRICS_PATH, keys=["iteration", "labeled_size", "loss", "accuracy"])
    model_name = str(model.get_model_type()).split(".")[1].lower() 
    metrics.write("moment_propagation" + "_" + acq_function_name, acl_history)


    # Check length of pool and initial data
    # print("Initial data: {}".format(len(x_train)))
    # print("Labeled data: {}".format(len(labeled_pool)))

