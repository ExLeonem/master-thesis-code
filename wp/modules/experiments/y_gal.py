import argparse
import os, sys, time

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow.keras as keras

# from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay, AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from tensorflow.keras.optimizers import Adam


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

from active_learning import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from bayesian import McDropout, MomentPropagation
from data import BenchmarkData, DataSetType
from models import fchollet_cnn, ygal_cnn, setup_growth, disable_tf_logs
from utils import setup_logger, init_pools


def keys_to_dict(**kwargs):
    return kwargs


if __name__ == "__main__":
    """
        Perform the experiment of the paper:
        Deep Bayesian active learning with image data. (Yarin Gal)

        - Experiments repeated 3 times
        - Results averaged

        Pool/Dataset parameters
        - Initial training set: 20 data points
        - Validation set: 100 data points (for weight decay optimization)
        - Test set: 10K data points
        - Rest: Used as pool set

        Active learning parameters
        - acquisition process repeated 100-times
        - (step size) 10 points per run, that maximised acquisition function (over pool set)
    """

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Gal Experiment")
    parser.add_argument("-n", "--name", default="gal", help="")
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Activate debug output?")
    parser.add_argument("-s", "--seed", default=1, type=int, help="Seed for random number generation process.")
    parser.add_argument("-e", "--epochs", default=20, type=int, help="How many epochs the network to train.")
    parser.add_argument("-p", "--prediction-runs", default=10, type=int, help="How often to sample from posterior distribution.")
    parser.add_argument("-a", "--acquisition", default="max_entropy", help="The aquisition function to use for the experiment")
    args = parser.parse_args()

    # Paths
    BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    DATASET_PATH = os.path.join(BASE_PATH, "datasets")

    # Setup logger
    c_dir_path = os.path.dirname(os.path.realpath(__file__))
    logs_path = os.path.join(c_dir_path, "..", "logs")
    logger = setup_logger(args.debug, "Y.Gal. Log", file="gal_experiment.log", path=logs_path)

    logger.info("----------")
    logger.info("------------------------")

    # seed = 10
    # if not (args.seed is None):
    #     print("Settings seed {}".format(args.seed))
    #     np.random.seed(args.seed)
    #     tf.random.set_seed(args.seed)

    # Pool/Dataset parameters
    val_set_size = 100
    train_set_size = 40_000
    test_set_size = 10_000
    initial_pool_size = 20

    # Split data into (x, 10K, 100) = (train/test/valid)
    mnist = BenchmarkData(DataSetType.MNIST, os.path.join(DATASET_PATH, "mnist"), dtype=np.float32)
    x_train, x_test, y_train, y_test = train_test_split(mnist.inputs, mnist.targets, test_size=test_set_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_set_size)

    dataset = Dataset(
        x_train, y_train, 
        val=(x_val, y_val), 
        test=(x_test, y_test), 
        init_size=initial_pool_size
    )

    # Active Learning parameters
    step_size = 10
    batch_size = 128
    learning_rate = 0.001
    verbose = False
    sample_size = 100

    # Configure Tensorflow
    disable_tf_logs()
    setup_growth()

    # Define Models
    num_classes = len(np.unique(mnist.targets))
    # base_model = fchollet_cnn(output=num_classes)
    base_model = ygal_cnn(initial_pool_size, output=num_classes)

    def reset_step(self, pool, dataset):
        """
            Overwrite reset function after each acquisiton iteration.

            Parameters:
                pool (Pool): Pool of labeled datapoints.
                dataset (Dataset): dataset object containing train, test and eval sets.
        """
        number_samples = pool.get_length_labeled()
        self.model = ygal_cnn(number_samples, output=num_classes)
        # self.load_weights()

    # MC Dropout Model    
    fit_params = {"epochs": 100, "batch_size": batch_size}
    loss = SparseCategoricalCrossentropy(reduction=Reduction.SUM)

    mc_config = Config(
        fit=fit_params,
        eval={"batch_size": 900, "sample_size": 25}
    )

    setattr(McDropout, "reset", reset_step)
    mc_model = McDropout(base_model, config=mc_config, verbose=verbose)
    mc_model.compile(optimizer="adam", loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # mc_config_2 = Config(
    #     fit=fit_params,
    #     eval={"batch_size": 900, "sample_size": 5}
    # )
    # mc_model_2 = McDropout(base_model, mc_config_2, name="sample_size_5", verbose=verbose)
    # mc_model_2.compile(optimizer="adam", loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # mc_config_3 = Config(
    #     fit=fit_params,
    #     eval={"batch_size": 900, "sample_size":15}
    # )
    # mc_model_3 = McDropout(base_model, mc_config_3, name="sample_size_15", verbose=verbose)
    # mc_model_3.compile(optimizer="adam", loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # mc_config_4 = Config(
    #     fit=fit_params,
    #     eval={"batch_size": 900, "sample_size": 25}
    # )
    # mc_model_4 = McDropout(base_model, mc_config_4, name="sample_size_25", verbose=verbose)
    # mc_model_4.compile(optimizer="adam", loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # Moment Propagation
    mp_config = Config(
        fit={"epochs": 100, "batch_size": batch_size},
        eval={"batch_size": 900}
    )
    setattr(MomentPropagation, "reset", reset_step)
    mp_model = MomentPropagation(base_model, mp_config, verbose=verbose)
    mp_model.compile(optimizer="adam", loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # Setup metrics handler
    METRICS_PATH = os.path.join(BASE_PATH, "metrics", "y_gal_detailed")
    metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

    # Setup experiment Suit
    # models = [mc_model, mp_model]
    models = [mc_model]
    # models = [mc_model, mc_model_2, mc_model_3, mc_model_4]
    query_fns = [
        AcquisitionFunction("random", batch_size=900, verbose=verbose),
        AcquisitionFunction("max_entropy", batch_size=900, verbose=verbose),
        AcquisitionFunction("bald", batch_size=900, verbose=verbose),
        AcquisitionFunction("max_var_ratio", batch_size=900, verbose=verbose),
        AcquisitionFunction("std_mean", batch_size=900, verbose=verbose)
    ]

    # 
    experiments = ExperimentSuit(
        models,
        query_fns,
        dataset,
        step_size=step_size,
        limit=100,
        runs=4,
        no_save_state=True,
        metrics_handler=metrics_handler,
        verbose=verbose
    )
    experiments.start()