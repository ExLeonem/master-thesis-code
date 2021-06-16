
import numpy as np
from modules.active_learning import DataPool, UnlabeledPool, LabeledPool
    


class TestDataPool:

    def test_get_item(self):
        data = np.random.randn(10, 10)
        pool = DataPool(data)
        assert np.all(pool[0] == data[0])


    def test_len(self):
        num_datapoints = 12
        data = np.random.randn(num_datapoints, 10)
        pool = DataPool(data)
        assert len(pool) == num_datapoints



class TestUnlabeledPool:

    def test_valid_pool_length(self):
        num_datapoints = 10
        data = np.random.randn(num_datapoints, 28, 28)
        pool = UnlabeledPool(data)

        assert len(pool) == num_datapoints

    
    def test_update_works(self):
        num_datapoints = 10
        data = np.random.randn(num_datapoints, 28, 28)
        pool = UnlabeledPool(data)
        assert not pool.is_empty()

        # Mark indices as labeled
        indices = np.linspace(0, num_datapoints-1, num_datapoints).astype(np.int32)
        pool.update(indices)
        assert pool.is_empty()


    def test_get_indices(self):
        """ Get unlabeled indices """
        num_datapoints = 10
        data = np.random.randn(num_datapoints, 28, 28)
        pool = UnlabeledPool(data)

        # Check same amount unlabeled data as datapoints
        indices = pool.get_indices()
        assert len(indices) == len(data)

        # Check
        indices_to_update = np.random.choice(indices, 2, replace=False)
        pool.update(indices_to_update)
        new_indices = pool.get_indices()
        assert len(new_indices) == (len(indices)-2) 


    def test_get_data(self):
        """ Should return all datapoints which are not labeled """
        num_datapoints = 10
        data = np.random.randn(num_datapoints, 28, 28)
        pool = UnlabeledPool(data)

        # Without update
        old_data = pool.get_data()
        assert old_data.shape == data.shape

        # With update
        indices = pool.get_indices()
        indices_to_update = np.random.choice(indices, 2, replace=False)
        pool.update(indices_to_update)
        new_data = pool.get_data()
        assert new_data.shape != data.shape
        assert len(new_data) == (len(data)-2)


    def test_get_labeled_indices(self):
        """ Inverse of function get_indices. Returns only indices of already labeld/removed indices """
        num_datapoints = 10
        data = np.random.randn(num_datapoints, 28, 28)
        pool = UnlabeledPool(data)

        # Should be empty
        labeled_indices = pool.get_labeled_indices()
        assert len(labeled_indices) == 0

        # Updated
        indices = pool.get_indices()
        indices_to_update = np.random.choice(indices, 2, replace=False)
        pool.update(indices_to_update)
        new_labeled_indices = pool.get_labeled_indices()
        assert len(new_labeled_indices) == 2


    def test_get_labeled_data(self):
        """ Should return all datapoints that are already labeled """
        num_datapoints = 10
        data = np.random.randn(num_datapoints, 28, 28)



class TestLabeledPool:
    """
        Test functionality of pool of labeled data
    """

    def test_initial_emptiness(self):
        num_samples = 10
        data = np.random.randn(num_samples, 28)
        pool = LabeledPool(data)
        assert 0 == len(pool)

    
    def test_adding_labels(self):
        num_samples = 10
        data = np.random.randn(num_samples, 28)
        pool = LabeledPool(data)
        pool[0] = 1
        assert len(pool) == 1
        

    def test_setting_batch(self):
        num_samples = 10
        inputs = np.random.randn(num_samples, 28, 28)
        targets = np.random.randint(0, 3, num_samples)
        indices = np.linspace(0, num_samples-1, num_samples).astype(np.int32)
        pool = LabeledPool(inputs)
        pool[indices[:2]] = targets[:2]
        assert len(pool) == 2

    
    def test_get_labeled_items(self):
        num_samples = 10
        inputs = np.random.randn(num_samples, 28, 28)
        targets = np.random.randint(0, 3, num_samples)
        indices = np.linspace(0, num_samples-1, num_samples).astype(np.int32)
        pool = LabeledPool(inputs)
        pool[indices[:2]] = targets[:2]

        # Check items at postion x
        (input_at_pos, target_at_pos) = pool[0]
        assert target_at_pos == targets[0]


    def test_get_indices_of_labeled_samples(self):
        num_samples = 10
        inputs = np.random.randn(num_samples, 28, 28)
        targets = np.random.randint(0, 3, num_samples)
        indices = np.linspace(0, num_samples-1, num_samples).astype(np.int32)
        pool = LabeledPool(inputs)
        pool[indices[:2]] = targets[:2]
        assert True






    
