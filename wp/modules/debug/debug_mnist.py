import argparse
import os, sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)

TF_PATH = os.path.join(BASE_PATH, "..", "..", "tf_al")
sys.path.append(TF_PATH)

from tf_al import Config, Dataset, ExperimentSuitMetrics, ExperimentSuit, AcquisitionFunction
from tf_al.wrapper import McDropout
# from tf_al_mp.wrapper import MomentPropagation

from models import fchollet_cnn, setup_growth, disable_tf_logs
from utils import setup_logger



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

    # Paths
    BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

    SEED = 83152
    print("Settings initial seed {}".format(SEED))
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Pool/Dataset parameters
    val_set_size = 100
    test_set_size = 10_000
    initial_pool_size = 20

    # Split data into (x, 10K, 100) = (train/test/valid)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Concat and normalize data
    x_stack = np.expand_dims(np.vstack([x_train, x_test]), axis=-1).astype(np.float32)
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    datagen.fit(x_stack)

    inputs = datagen.standardize(x_stack)
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
    verbose = False
    sample_size = 25

    # Configure Tensorflow
    disable_tf_logs()
    setup_growth()

    # Define Models
    num_classes = len(np.unique(targets))
    base_model = fchollet_cnn(output=num_classes)

    # ---------------------
    # MC Dropout   
    fit_params = {"epochs": 200, "batch_size": batch_size}
    mc_config = Config(
        fit=fit_params,
        query={"sample_size": sample_size},
        eval={"batch_size": 900, "sample_size": sample_size}
    )

    mc_model = McDropout(base_model, config=mc_config)

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
        
    setattr(McDropout, "reset", reset_step)

    # ---------------------
    # Moment Propagation
    # mp_config = Config(
    #     fit={"epochs": 100, "batch_size": batch_size},
    #     eval={"batch_size": 900}
    # )
    # mp_model = MomentPropagation(base_model, mp_config, verbose=verbose)
    # mp_model.compile(optimizer=optimizer, loss=loss,  metrics=metrics)

    # def reset_step(self, pool, dataset):
    #     """
    #         Reset The moment propagation model and freshly start training.

    #         Parameters:
    #             pool (Pool): Pool of labeled datapoints
    #             dataset (Dataset): dataset object containing train,t est and eval sets.
    #     """
    #     # print(dir(self))
    #     self._model = fchollet_cnn(output=num_classes)
    #     self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #     self.__mp_model = self._create_mp_model(self._model)

    # setattr(MomentPropagation, "reset", reset_step)

    # Setup metrics handler
    METRICS_PATH = os.path.join(BASE_PATH, "metrics", "debug_mnist_baba")
    metrics_handler = ExperimentSuitMetrics(METRICS_PATH)

    # Setup experiment Suit
    # models = [mc_model, mp_model]
    models = [mc_model]
    query_fns = [
        AcquisitionFunction("baba", batch_size=900, verbose=verbose),
        # AcquisitionFunction("max_entropy", batch_size=900, verbose=verbose),
        # AcquisitionFunction("max_var_ratio", batch_size=900, verbose=verbose),
        # AcquisitionFunction("bald", batch_size=900, verbose=verbose),
        # AcquisitionFunction("std_mean", batch_size=900, verbose=verbose)
    ]

    # 
    experiments = ExperimentSuit(
        models,
        query_fns,
        dataset,
        step_size=step_size,
        max_rounds=100,
        seed=[SEED, 20432, 10942],
        no_save_state=True,
        metrics_handler=metrics_handler,
        verbose=verbose
    )
    experiments.start()