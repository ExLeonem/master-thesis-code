

import numpy as np
from modules.active_learning import PooledDataset


class TestPooledDataset:

    def test_flags_only_inputs(self):
        """
            Test if correct flag set for only inputs dataset.
        """

        inputs = np.random.randn(10, 100)
        dataset = PooledDataset(inputs)
        assert not dataset.has_targets()

    
    def test_flags_with_targets(self):
        """
            Test correct flags when inputs and targets.
        """
        inputs = np.random.randn(10, 100)
        targets = np.random.randn(10)

        dataset = PooledDataset(inputs, targets)
        assert dataset.has_targets()

    
    def test_split(self):
        pass



