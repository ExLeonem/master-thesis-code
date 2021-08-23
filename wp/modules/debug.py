import argparse
import time
import os, sys, importlib
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from active_learning import TrainConfig, Config, Metrics, AcquisitionFunction, Pool
from wrapper import McDropout, MomentPropagation
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
        metrics = ['loss', 'accuracy']

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
    parser.add_argument("-c", "--class_count", default=10, type=int, help="Select the num of classes to run the experiment for.")
    parser.add_argument("-aqf", "--acquisition_function", default="max_entropy", help="Select an aquisition function to execute. One of ['max_entropy', 'bald', 'std_mean', 'max_var_ratio']")
    parser.add_argument("-aqb", "--acquisition_batch_size", default=900, type=int, help="Set an batch size for the acquisition function iterations.")
    parser.add_argument("-s", "--step_size", default=10, type=int, help="Set a step size. How many datapoints to add per iteration to pool of labeled data.")
    parser.add_argument("--n_times", default=100, type=int, help="The number of predictions to do for mc dropout predictions. (default=10)")
    parser.add_argument("--seed", default=None, type=int, help="Setting a seed for random processes.")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="The number of epochs to fit the network. (default=5)")
    parser.add_argument("-l", "--limit", default=4, type=int, help="A limit for the iteration to do.")
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Output logging messages?")
    parser.add_argument("-i", "--initial_size", default=1, type=int, help="The initial size of the pool of labeled data. (default=10)")
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
    acquisition = AcquisitionFunction(acq_function_name, batch_size=acq_batch_size, verbose=True)
    pool = Pool(x_train, y_train)
    pool.init(args.initial_size)

    # Setup/configure the Model
    setup_growth()
    loss = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    config = TrainConfig(
        batch_size=10,
        loss=loss,
        optimizer="adadelta"
    )

    output_classes = num_classes if num_classes != 2 else 1
    base_model = default_model(output_classes=output_classes)


    model, metrics = select_model(model_name, base_model, is_binary=(num_classes == 2))
    # model = McDropout(base_model, is_binary=(num_classes == 2))
    model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=["accuracy"])
    
    if not model.has_save_state():
        logger.warn("Creating initial save state.")
        model.save_weights()

    # Active learning loop
    iterator = tqdm(range(0, pool.get_length_unlabeled(), step_size), leave=True)
    acl_history = []
    it = 0
    for i in iterator:
        loop_start = time.time()

        tf.keras.backend.clear_session()
        if not (limit is None) and it > limit:
            # Defined iteration limit reached
            break
        model.load_weights()
        logger.info("---------")

        # Fit the model
        logger.info("(Start) Train")
        start = time.time()
        lab_inputs, lab_targets = pool.get_labeled_data()
        history = model.fit(lab_inputs, lab_targets, batch_size=config["batch_size"], epochs=epochs, verbose=False)
        end = time.time()
        logger.info("Training: {}".format(history.history))
        logger.info("(Finish) Train (%.2fs)" % (end-start))
        
        # Needs to be moved: in last iteration no aquisition
        # Selected datapoints and labels
        logger.info("(Start) Acquisition")
        start = time.time()
        indices, _pred = acquisition(model, pool, num=step_size, sample_size=args.n_times)
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
        logger.info("(Start) Evaluation")
        start = time.time()
        eval_result = model.evaluate(x_test, y_test, sample_size=args.n_times, batch_size=1000)
        eval_metrics = model.map_eval_values(eval_result)
        end = time.time()
        logger.info("Metrics: {}".format(str(eval_metrics)))
        logger.info("(End) Eval (%.2fs)" % (end-start))

        # Accumulate history
        loop_end = time.time()
        acl_history.append(keys_to_dict(
            iteration=it,
            labeled_size=lab_pool_size,
            time=(loop_end-loop_start),
            **eval_metrics
        ))

        logger.info("Iteration {}".format(i))
        logger.info("Labeled_size: {}".format(lab_pool_size))
        it += 1
        # break

    # Write metrics
    METRICS_PATH = os.path.join(BASE_PATH, "metrics")
    metrics = Metrics(METRICS_PATH, keys=["iteration", "labeled_size", "time"] + metrics)
    model_name = str(model.get_model_type()).split(".")[1].lower() 
    metrics.write(model_name + "_" + acq_function_name, acl_history)


    # Check length of pool and initial data
    print("Initial data: {}".format(len(x_train)))
    print("Labeled data: {}".format(pool.get_length_labeled()))