import argparse
import os, sys, importlib
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from active_learning import TrainConfig, Config, Metrics, LabeledPool, UnlabeledPool, AcquisitionFunction
from bayesian import McDropout, MomentPropagation
from data import BenchmarkData, DataSetType
from models import default_model, setup_growth
from utils import setup_logger, init_pools

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
        model = MomentPropagation(base_model, **kwargs)
        metrics = [
            'loss', 'tf_op_layer_Sigmoid_1_loss', 
            'tf_op_layer_Mul_8_loss', 'tf_op_layer_Sigmoid_1_binary_accuracy', 
            'tf_op_layer_Mul_8_binary_accuracy'
        ]
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
    parser.add_argument("-m", "--model", default="dp", help="Select the model the run an experiment for.")
    parser.add_argument("-c", "--class_count", default=2, type=int, help="Select the num of classes to run the experiment for.")
    parser.add_argument("-aqf", "--acquisition_function", default="bald", help="Select an aquisition function to execute. One of ['max_entropy', 'bald', 'std_mean', 'max_var_ratio']")
    parser.add_argument("-aqb", "--acquisition_batch_size", default=900, type=int, help="Set an batch size for the acquisition function iterations.")
    parser.add_argument("-s", "--step_size", default=1200, type=int, help="Set a step size. How many datapoints to add per iteration to pool of labeled data.")
    parser.add_argument("--n_times", default=10, type=int, help="The number of predictions to do for mc dropout predictions. (default=10)")
    parser.add_argument("--seed", default=None, type=int, help="Setting a seed for random processes.")
    parser.add_argument("-e", "--epochs", default=5, type=int, help="The number of epochs to fit the network. (default=5)")
    parser.add_argument("-l", "--limit", default=None, type=int, help="A limit for the iteration to do.")
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Output logging messages?")
    parser.add_argument("-i", "--initial_size", default=10, type=int, help="The initial size of the pool of labeled data. (default=10)")
    # parser.add_argument("")
    args = parser.parse_args()

    # Runtime arguments
    model_name = args.model
    num_classes = args.class_count
    acq_function_name = args.acquisition_function
    acq_batch_size = args.acquisition_batch_size
    step_size = args.step_size
    epochs = args.epochs
    seed = args.seed
    limit = args.limit

    # Setup Logger
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(current_dir, "logs")
    logger = setup_logger(args.debug, file="debug.log", path=log_path)

    # Set seeds for reproducability?
    if not (seed is None):
        np.random.seed(seed)
        tf.random.set_seed(seed)


    # Load and transform mnist dataset
    mnist = BenchmarkData(DataSetType.MNIST, os.path.join(DATASET_PATH, "mnist"), classes=num_classes)
    x_train, x_test, y_train, y_test = train_test_split(mnist.inputs, mnist.targets)

    # Setup active learning specifics
    acquisition = AcquisitionFunction(acq_function_name, batch_size=acq_batch_size)
    labeled_pool = LabeledPool(x_train)
    unlabeled_pool = UnlabeledPool(x_train)
    init_pools(unlabeled_pool, labeled_pool, y_train, num_init_per_target=args.initial_size)

    # Setup/configure the Model
    setup_growth()
    loss = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    config = TrainConfig(
        batch_size=5,
        loss=loss
    )

    output_classes = num_classes if num_classes != 2 else 1
    base_model = default_model(output_classes=output_classes)


    model, metrics = select_model(model_name, base_model, is_binary=(num_classes == 2))
    # model = McDropout(base_model, is_binary=(num_classes == 2))
    model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=["accuracy"])
    model.save_weights()

    # Active learning loop
    iterator = tqdm(range(0, len(unlabeled_pool), step_size), leave=True)
    acl_history = []
    it = 0
    for i in iterator:

        tf.keras.backend.clear_session()
        if not (limit is None) and it > limit:
            # Defined iteration limit reached
            break
        model.load_weights()
        logger.info("---------")

        # Fit the model
        lab_inputs, lab_targets = labeled_pool[:]
        history = model.fit(lab_inputs, lab_targets, batch_size=config["batch_size"], epochs=epochs, verbose=False)
        logger.info("Training: {}".format(history.history))

        # Selected datapoints and labels
        indices, _pred = acquisition(model, unlabeled_pool, num=step_size, n_times=args.n_times)
        labels = y_train[indices]
        
        # Update pools
        lab_pool_size = len(labeled_pool)
        unlabeled_pool.update(indices)
        labeled_indices = unlabeled_pool.get_labeled_indices()
        labeled_pool[indices] = labels

        # Evaluate model
        eval_result = model.evaluate(x_test[:100], y_test[:100], batch_size=config["batch_size"], verbose=False)
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
        # break

    # Write metrics
    METRICS_PATH = os.path.join(BASE_PATH, "metrics")
    metrics = Metrics(METRICS_PATH, keys=["iteration", "labeled_size"] + metrics)
    model_name = str(model.get_model_type()).split(".")[1].lower() 
    metrics.write(model_name + "_" + acq_function_name, acl_history)


    # Check length of pool and initial data
    print("Initial data: {}".format(len(x_train)))
    print("Labeled data: {}".format(len(labeled_pool)))

