
import numpy as np
import pytest
import os
from modules.data import BenchmarkData, DataSetType

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = os.path.join(DIR_PATH, "..", "..")
MNIST_PATH = os.path.join(BASE_PATH, "datasets", "mnist")

class TestBenchmarkData:
    """
        Test functionality of BenchmarkData.

        *To perform this test data needs to be present*
    """

    def test_load_mnist_data(self):
        """ Test MNIST loader implementation """
        mnist_data = BenchmarkData(DataSetType.MNIST, MNIST_PATH)

        assert len(mnist_data.inputs) != 0
        assert len(mnist_data.targets) != 0
        assert len(mnist_data.inputs) == len(mnist_data.targets)


    def test_subset_class_loader(self):
        num_classes = 3
        mnist_data = BenchmarkData(DataSetType.MNIST, MNIST_PATH, classes=num_classes)
        unique_targets = np.unique(mnist_data.targets)
        assert len(unique_targets) == num_classes

    
    def test_load_without_path(self):
        with pytest.raises(TypeError) as e:
            mnist_data = BenchmarkData(DataSetType.MNIST)

    
    def test_load_non_existent_dataset(self):
        with pytest.raises(Exception) as e:
            BenchmarkData("Hello", MNIST_PATH)