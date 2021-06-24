import argparse
import os, sys, time

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
# import tensorflow_addons as tfa


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)


from active_learning import TrainConfig, LabeledPool, UnlabeledPool, Metrics, AcquisitionFunction
from bayesian import McDropout
from data import BenchmarkData, DataSetType
from models import fchollet_cnn, setup_growth
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
    parser.add_argument("-s", "--seed", default=None, type=int, help="Seed for random number generation process.")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="How many epochs the network to train.")
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

    # Parameter
    initial_pool_size = 20
    val_set_size = 100
    test_set_size = 10_000
    step_size = 10
    batch_size = 10
    verbose = False

    # Split data into (x, 10K, 100) = (train/test/valid)
    mnist = BenchmarkData(DataSetType.MNIST, os.path.join(DATASET_PATH, "mnist"), dtype=np.float32, classes=3)
    x_train, x_test, y_train, y_test = train_test_split(mnist.inputs, mnist.targets, test_size=test_set_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_set_size)
    labeled_pool = LabeledPool(x_train)
    unlabeled_pool = UnlabeledPool(x_train)
    init_pools(unlabeled_pool, labeled_pool, y_train, 20)

    # Setup model
    setup_growth()
    base_model = fchollet_cnn(output=len(np.unique(mnist.targets)))
    model = McDropout(base_model, verbose=verbose)

    # step = tf.Variable(0, trainable=False)
    # schedule = tf.optimizers.schedules.PiecewiseConstantDecay([1407*20, 1407*30], [1e-3, 1e-4, 1e-5])
    # wd = lambda: 1e-1 * schedule(step)
    # optimizer = tfa.optimizers.AdamW(weight_decay=wd)
    
    model.compile(optimizer="adadelta", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.save_weights()

    # Active learning loop
    num_iterations = 100
    acquisition = AcquisitionFunction(args.acquisition, batch_size=1000, verbose=verbose)
    iterator = tqdm(range(num_iterations), leave=True)
    acl_history = []
    it = 0
    logger.info("Start loop")
    for i in iterator:
        logger.info("----------")
        loop_start = time.time()

        # Reset
        tf.keras.backend.clear_session()
        model.load_weights()

        # Fit model
        logger.info("(Start) Train Model")
        start = time.time()
        lab_inputs, lab_targets = labeled_pool[:]
        h = model.fit(
            lab_inputs, lab_targets, 
            batch_size=batch_size, epochs=args.epochs, verbose=verbose,
            validation_data=(x_val, y_val)
        )
        end = time.time()
        logger.info("History: {}".format(h.history))
        logger.info("--(Finish) Train Model (%.1fs)" % (end-start))

        # Last iteration? No need for new acquisition
        if i < num_iterations -1:
            logger.info("(Start) Acquisiton")

            # Acquisition process
            start = time.time()
            indices, _pred = acquisition(model, unlabeled_pool, num=step_size, sample_size=100)
            labels = y_train[indices]
            end = time.time()
            logger.info("--(Finish) Acquisiton (%.1fs)" % (end-start))

            # Update ppols
            lab_pool_size = len(labeled_pool)
            unlabeled_pool.update(indices)
            labeled_indices = unlabeled_pool.get_labeled_indices()
            labeled_pool[indices] = labels

        # Evaluate
        logger.info("(Start) Eval")
        start = time.time()
        eval_result = model.evaluate(x_test[:100], y_test[:100], sample_size=100, batch_size=900)
        eval_metrics = model.map_eval_values(eval_result)
        end = time.time()
        logger.info("Metrics: {}".format(eval_metrics))
        logger.info("--(End) Eval (%.1fs)" % (end-start))

        loop_end = time.time()
        acl_history.append(
            keys_to_dict(
                iteration=it,
                time=(loop_end-loop_start),
                labeled_size=lab_pool_size,
                **eval_metrics
            )
        )
        it += 1
    
    # Write metrics
    METRICS_PATH = os.path.join(BASE_PATH, "metrics")
    metrics = Metrics(METRICS_PATH, keys=["iteration", "time", "labeled_size"] + model.get_metric_names())
    model_name = str(model.get_model_type()).split(".")[1].lower() 
    metrics.write(model_name + "_" + args.acquisition , acl_history)