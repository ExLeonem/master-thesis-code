

import numpy as np
from modules.active_learning import Dataset


class TestDataset:

    def test_flags_only_inputs(self):
        """
            Test if correct flag set for only inputs dataset.
        """

        inputs = np.random.randn(10, 100)
        dataset = Dataset(inputs)
        assert not dataset.has_targets()

    
    def test_flags_with_targets(self):
        """
            Test correct flags when inputs and targets.
        """
        inputs = np.random.randn(10, 100)
        targets = np.random.randn(10)

        dataset = Dataset(inputs, targets)
        assert dataset.has_targets()

    
    def test_train_test_split(self):
        pass



