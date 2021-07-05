
import pytest
import numpy as np
from modules.bayesian import BayesModel
from modules.active_learning import AcquisitionFunction, UnlabeledPool, LabeledPool, Pool
from modules.utils import init_pools


class MockModel:

    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, *args, **kwargs):
        return np.random.randn(self.output_shape)

    def predict(self, *args, **kwargs):
        return np.random.randn(self.output_shape)


    def get_query_fn(self, name):
        return __generic_query_fn
    
    def __generic_query_fn(self, data, **kwargs):
        return np.random.randn(len(data))



class TestAcquisitionFunction:

    def test_select_adaption(self):
        inputs = np.random.randn(10)
        targets = np.random.choice([0, 1, 2, 3], 10)

        model = BayesModel(MockModel(10))
        pool = Pool(inputs)

        acf = AcquisitionFunction("random", verbose=True)
        indices, data = acf(model, pool)
        assert len(indices) == 10

    
    def test_random_acquisition(self):
        inputs = np.random.randn(10)
        targets = np.random.choice([0, 1, 2, 3], 10)

        model = BayesModel(MockModel(10))
        pool = Pool(inputs)

        acf = AcquisitionFunction("random", verbose=True)
        indices, data = acf(model, pool, 2)
        assert len(indices) == 2


    def test_generic_selection(self):
        inputs = np.random.randn(10)
        targets = np.random.choice([0, 1, 2, 3], 10)

        model = BayesModel(MockModel(10))
        pool = Pool(inputs)

        acf = AcquisitionFunction("bald", verbose=True)
        indices, data = acf(model, pool, 3)
        assert len(indices) == 3

    
    def test_generic_selection_adapted(self):
        inputs = np.random.randn(10)
        targets = np.random.choice([0, 1, 2, 3], 10)

        model = BayesModel(MockModel(10))
        pool = Pool(inputs, targets)
        pool.init(5)

        acf = AcquisitionFunction("bald", verbose=True)
        indices, data = acf(model, pool)

        assert len(indices) == 5