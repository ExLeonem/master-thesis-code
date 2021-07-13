import numpy as np
from modules.active_learning import Dataset


class TestDataset:

    
    def test_flags_with_targets(self):
        inputs = np.random.randn(10, 100)
        targets = np.random.randn(10)

        dataset = Dataset(inputs, targets)
        assert True

    
    def test_default_data_split(self):
        inputs = np.random.randn(10)
        targets = np.random.choice([0, 1, 2, 3], 10)

        dataset = Dataset(inputs, targets, test_size=0.25)
        assert dataset.has_test_set()




