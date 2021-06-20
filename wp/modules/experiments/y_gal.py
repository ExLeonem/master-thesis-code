import argparse
import os, sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "..")
sys.path.append(MODULES_PATH)


from active_learning import TrainConfig, LabeledPool, UnlabeledPool, Metrics, AquisitionFunction
from bayesian import McDropout
from data import BenchmarkData, DataSetType
from models import FcholletCNN, setup_growth
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

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Gal Experiment")
    parser.add_argument("-n", "--name", default="gal", help="")
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Activate debug output?")
    parser.add_argument("-s", "--seed", default=None, type=int, help="Seed for random number generation process.")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="How many epochs the network to train.")
    parser.add_argument("-p", "--prediction-runs", default=10, type=int, help="How often to sample from posterior distribution.")
    args = parser.parse_args()

    # Paths
    BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    DATASET_PATH = os.path.join(BASE_PATH, "datasets")

    # Setup logger
    c_dir_path = os.path.dirname(os.path.realpath(__file__))
    logs_path = os.path.join(c_dir_path, "logs")
    logger = setup_logger(args.debug, "Y.Gal. Log", file="gal_experiment.log", path=logs_path)

    # Parameter
    initial_pool_size = 20
    val_set_size = 100
    test_set_size = 10_000
    step_size = 10
    batch_size = 10

    # Setup data pools
    benchmark_data = BenchmarkData(DataSetType.MNIST, os.path.join(DATASET_PATH, "mnist"))
    x_train, x_test, y_train, y_test = train_test_split(benchmark_data.inputs, benchmarK_data.targets, test_size=test_set_size)
    x_train, x_val, y_train, y_val = 

    # Setup model
    base_model = FcholletCNN(output=len(benchmark.targets))
    model = McDropout(base_model)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.save_weights()

    # Active learning loop
    num_iterations = 100
    iterator = tqdm(range(num_iterations), leave=True)
    for i in iterator:

        # Reset
        tf.keras.backend.clear_session()
        model.load_weights()

        # Fit model
        # lab_inputs, lab_targets = labeled_pool[:]

        break

    
