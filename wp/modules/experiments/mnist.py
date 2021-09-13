import argparse
import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras

# from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay, AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from tensorflow.keras.optimizers import Adam


from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
from tf_al_mp.wrapper import MomentPropagation

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

from models import fchollet_cnn, setup_growth, disable_tf_logs
from utils import setup_logger


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
    parser.add_argument("-s", "--seed", default=83152, type=int, help="Seed for random number generation process.")
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

    SEED = args.seed
    if SEED is not None:
        print("Settings seed {}".format(args.seed))
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

    # Pool/Dataset parameters
    val_set_size = 100
    test_set_size = 10_000
    initial_pool_size = 20

    # Split data into (x, 10K, 100) = (train/test/valid)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    inputs = np.expand_dims(np.vstack([x_train, x_test])/255., axis=-1)
    targets = np.hstack([y_train, y_test])
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_set_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_set_size)

    dataset = Dataset(
        x_train, y_train, 
        val=(x_val, y_val), 
        test=(x_test, y_test), 
        init_size=initial_pool_size
    )

    # Active Learning parameters
    step_size = 10
    batch_size = 10
    learning_rate = 0.001
    verbose = False
    sample_size = 25

    # Configure Tensorflow
    disable_tf_logs()
    setup_growth()

    # Define Models
    num_classes = len(np.unique(targets))
    base_model = fchollet_cnn(output=num_classes)
    # base_model = ygal_cnn(initial_pool_size, output=num_classes)

    # ---------------------
    # MC Dropout   
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="sparse_categorical_accuracy",
        min_delta=0.01,
        patience=10,
        restore_best_weights=True
    )
    fit_params = {"epochs": 200, "batch_size": batch_size}
    # loss = SparseCategoricalCrossentropy(reduction=Reduction.SUM)

    mc_config = Config(
        fit=fit_params,
        query={"sample_size": sample_size},
        eval={"batch_size": 900, "sample_size": sample_size}
    )

    mc_model = McDropout(base_model, config=mc_config, verbose=verbose)
    # optimizer = keras.optimizers.SGD(
    #     lr=learning_rate,
    #     momentum=0.9
    # )

    optimizer = "adam"
    loss = "sparse_categorical_crossentropy"
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
    mc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def reset_step(self, pool, dataset):
        """
            Overwrite reset function after each acquisiton iteration.

            Parameters:
                pool (Pool): Pool of labeled datapoints.
                dataset (Dataset): dataset object containing train, test and eval sets.
        """
        self._model = fchollet_cnn(output=num_classes)
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # number_samples = pool.get_length_labeled()
        # self.model = ygal_cnn(number_samples, output=num_classes)
        # self.load_weights()
    setattr(McDropout, "reset", reset_step)

    # ---------------------
    # Moment Propagation
    mp_config = Config(
        fit={"epochs": 100, "batch_size": batch_size},
        eval={"batch_size": 900}
    )
    mp_model = MomentPropagation(base_model, mp_config, verbose=verbose)
    mp_model.compile(optimizer=optimizer, loss=loss,  metrics=metrics)

    def reset_step(self, pool, dataset):
        """
            Reset The moment propagation model and freshly start training.

            Parameters:
                pool (Pool): Pool of labeled datapoints
                dataset (Dataset): dataset object containing train,t est and eval sets.
        """
        # print(dir(self))
        self._model = fchollet_cnn(output=num_classes)
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.__mp_model = self._create_mp_model(self._model)

    setattr(MomentPropagation, "reset", reset_step)

    # Setup metrics handler
    METRICS_PATH = os.path.join(BASE_PATH, "metrics", "temp")
    metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

    # Setup experiment Suit
    models = [mc_model, mp_model]
    # models = [mp_model]
    query_fns = [
        AcquisitionFunction("random", batch_size=900, verbose=verbose),
        AcquisitionFunction("max_entropy", batch_size=900, verbose=verbose),
        AcquisitionFunction("max_var_ratio", batch_size=900, verbose=verbose),
        AcquisitionFunction("bald", batch_size=900, verbose=verbose),
        AcquisitionFunction("std_mean", batch_size=900, verbose=verbose)
    ]

    # 
    experiments = ExperimentSuit(
        models,
        query_fns,
        dataset,
        step_size=step_size,
        # runs=2,
        max_rounds=100,
        seed=SEED,
        no_save_state=True,
        metrics_handler=metrics_handler,
        verbose=verbose
    )
    experiments.start()